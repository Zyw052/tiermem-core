# -*- coding: utf-8 -*-
"""
Stage 6 自测：方案 A（止血热修）六大核心修复专项验证。

覆盖点（与风控编号一一对应）：
1. R-01 L3 置顶记忆免死金牌：
   - 规则层 detect_conflicts 跳过 pinned=1 候选；
   - 落库层 _apply_conflict_resolution 的 SQL 带 `COALESCE(pinned,0)=0` 守卫，
     即使判官强行返回冲突也无法软废弃置顶记忆；
   - 端到端：置顶偏好被否定句“推翻”后仍保持 active。
2. R-03 真·多轮合并批处理：5 轮对话只产生 1 次事实抽取 / 1 次门禁 / 1 次批量 Embedding。
3. R-02 L1 实时落盘防蒸发：未满阈值即断电也不丢原始对话；析出后回写 l2_processed；
   idle 超时可自动刷盘。
4. R-04 严格会话隔离：默认开启，跨会话记忆物理不可见；仅 scope_key='global' 例外。
5. R-05 检索纯读：retrieve 不再同步 UPDATE，命中计数内存累加，显式刷盘后才落库。
6. R-06 HTTP 连接池复用：Session 只创建一次，客户端具备幂等 close()。

运行： python3 tests/test_stage6_hardening.py
（同时兼容 pytest 收集：所有用例均为 test_* 函数，无外部依赖）
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import models_client  # noqa: E402
from evaluator import EvaluatorGate  # noqa: E402
from retriever import MemoryRetriever  # noqa: E402
from storage import MemoryStorage  # noqa: E402
from summarizer import MemorySummarizer  # noqa: E402


# ---------------------------------------------------------------------------
# 测试桩
# ---------------------------------------------------------------------------


class CountingEmbeddingClient:
    """
    稳定伪向量 + 调用计数桩。

    - 批量调用 get_embeddings 与单条调用 get_embedding 分别计数，
      用于证明 R-03 的“单次批量”而不是 N+1 逐条；
    - 向量按文本哈希做 one-hot，保证不同事实不会被余弦误融合。
    """

    dim = 64

    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0
        self.last_batch_sizes: List[int] = []

    def _vec(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(text)))
        vec[seed % self.dim] = 1.0
        return vec

    async def get_embedding(self, text: str) -> List[float]:
        self.single_calls += 1
        return self._vec(text)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.batch_calls += 1
        self.last_batch_sizes.append(len(texts))
        return [self._vec(t) for t in texts]


class ZeroEmbeddingClient:
    """零向量桩：禁用余弦融合，便于观察冲突/废弃行为。"""

    dim = 8

    async def get_embedding(self, text: str) -> List[float]:
        return [0.0] * self.dim


def _new_storage(tmp_dir: Path) -> MemoryStorage:
    storage = MemoryStorage(tmp_dir)
    storage.init_schema()
    return storage


def _new_summarizer(
    storage: MemoryStorage,
    embedding_client: Any = None,
    config: Dict[str, Any] = None,
) -> MemorySummarizer:
    base: Dict[str, Any] = {"summary_turn_interval": 5}
    if config:
        base.update(config)
    return MemorySummarizer(
        storage=storage,
        embedding_client=embedding_client or CountingEmbeddingClient(),
        rerank_client=None,
        config=base,
    )


def _insert_atom(
    conn: Any,
    atom_id: str,
    scope_key: str,
    content: str,
    pinned: int = 0,
    importance: float = 0.6,
) -> None:
    """直接插入一条最小可用 L2 原子（补齐所有 NOT NULL 列）。"""
    now = time.time()
    with conn:
        conn.execute(
            """
            INSERT INTO l2_atoms (
                atom_id, scope_key, session_id, content,
                topics_json, key_facts_json, sentiment, importance,
                access_count, embedding_blob, source_type, source_ref,
                last_accessed_at, created_at, updated_at,
                atom_type, reinforcement_count, pinned, tier, metadata,
                status, superseded_by, superseded_at
            ) VALUES (?, ?, ?, ?, '[]', '[]', 'neutral', ?, 0, NULL, 'chat', '',
                      ?, ?, ?, 'fact', 1, ?, ?, '{}', 'active', '', 0.0)
            """,
            (
                atom_id,
                scope_key,
                scope_key,
                content,
                float(importance),
                now,
                now,
                now,
                int(pinned),
                3 if pinned else 2,
            ),
        )


# ---------------------------------------------------------------------------
# R-01：L3 置顶记忆免死金牌
# ---------------------------------------------------------------------------


async def test_r01_pinned_immortal() -> None:
    with tempfile.TemporaryDirectory(prefix="r01_pin_") as tmp:
        storage = _new_storage(Path(tmp))
        sm = _new_summarizer(storage, embedding_client=ZeroEmbeddingClient())

        # 1) 写入含偏好关键词的事实 → 自动晋升 L3（pinned=1）
        await sm.process_dialogue("s_pin", "我喜欢喝绿茶", "记住了")
        conn = storage.connect()
        old = conn.execute(
            "SELECT atom_id, content, pinned, tier, status, importance, updated_at "
            "FROM l2_atoms WHERE scope_key = ? AND content LIKE '%绿茶%'",
            ("s_pin",),
        ).fetchone()
        assert old is not None, "置顶事实未写入"
        assert int(old["pinned"]) == 1 and int(old["tier"]) == 3, dict(old)

        # 2) 规则层：pinned=1 的候选不得被判为冲突
        gate = EvaluatorGate({"evaluator": {}})
        assert gate.detect_conflicts(
            "我不喜欢喝绿茶",
            [{"atom_id": "atom_pin", "content": "我喜欢喝绿茶", "pinned": 1}],
        ) == [], "规则层极性翻转不应命中置顶记忆"
        # 对照：非置顶时同样输入应判为冲突（证明测试本身有效）
        assert gate.detect_conflicts(
            "我不喜欢喝绿茶",
            [{"atom_id": "atom_normal", "content": "我喜欢喝绿茶", "pinned": 0}],
        ), "非置顶对照用例应能识别冲突"
        print("[OK] R-01 规则层：pinned=1 候选被豁免，非置顶对照仍可命中")

        # 3) 落库层：判官强行返回冲突，也必须被 SQL 守卫拦下
        applied = sm._apply_conflict_resolution(
            [{"atom_id": str(old["atom_id"]), "fact": "我不喜欢喝绿茶", "reason": "judge_forced"}],
            {},
            time.time(),
        )
        assert applied == 0, f"置顶记忆被强行废弃了 {applied} 条"

        # 4) 端到端：真实走一遍推翻流程
        await sm.process_dialogue("s_pin", "我不喜欢喝绿茶", "已更新")
        after = conn.execute(
            "SELECT status, pinned, importance, superseded_by FROM l2_atoms WHERE atom_id = ?",
            (old["atom_id"],),
        ).fetchone()
        assert str(after["status"]) == "active", dict(after)
        assert int(after["pinned"]) == 1
        assert float(after["importance"]) == 0.9, dict(after)
        assert str(after["superseded_by"]) == "", dict(after)
        assert conn.execute(
            "SELECT COUNT(1) AS c FROM l2_atoms WHERE scope_key = ? AND status = 'superseded'",
            ("s_pin",),
        ).fetchone()["c"] == 0
        print("[OK] R-01 端到端：置顶偏好经历否定句后仍 active / pinned=1 / 重要度无损")

        storage.close()


# ---------------------------------------------------------------------------
# R-03：真·多轮合并批处理（1 次 LLM + 1 次门禁 + 1 次批量 Embedding）
# ---------------------------------------------------------------------------


async def test_r03_batch_single_pass() -> None:
    with tempfile.TemporaryDirectory(prefix="r03_batch_") as tmp:
        storage = _new_storage(Path(tmp))
        emb = CountingEmbeddingClient()
        sm = _new_summarizer(storage, embedding_client=emb)
        sm.start()
        try:
            turns = [
                ("我的手机号是13800000001", "已记录"),
                ("我的生日是3月5日", "已记录"),
                ("我的职业是程序员", "已记录"),
                ("我住在杭州市西湖区", "已记录"),
                ("我的邮箱是demo@example.com", "已记录"),
            ]
            for user, bot in turns:
                assert await sm.enqueue_dialogue("s_batch", user, bot) is True
            await asyncio.wait_for(sm._queue.join(), timeout=10)

            stats = sm.get_buffer_stats()
            assert stats["stat_llm_extract_calls"] == 1, stats
            assert stats["stat_gate_calls"] == 1, stats
            assert emb.batch_calls == 1, f"批量 Embedding 应只调用 1 次，实际 {emb.batch_calls}"
            assert emb.single_calls == 0, f"不应出现逐条 Embedding，实际 {emb.single_calls}"
            assert emb.last_batch_sizes and emb.last_batch_sizes[0] >= 1

            counts = storage.get_table_counts()
            assert counts["l1_docs"] == 5, counts
            assert counts["l2_atoms"] >= 1, counts
            # 单次批量请求覆盖同一批事实（而非 5 次）
            assert stats["stat_batches_enqueued"] == 1
            print(
                "[OK] R-03 5 轮合并：LLM=%d 门禁=%d 批量Embedding=%d(单条=%d) 原子=%d"
                % (
                    stats["stat_llm_extract_calls"],
                    stats["stat_gate_calls"],
                    emb.batch_calls,
                    emb.single_calls,
                    counts["l2_atoms"],
                )
            )
        finally:
            await sm.stop(flush=False)
            storage.close()


# ---------------------------------------------------------------------------
# R-02：L1 实时落盘 + 析出回写 + idle 超时刷盘
# ---------------------------------------------------------------------------


async def test_r03_llm_json_parse() -> None:
    """R-03 辅助：合并上下文的 LLM JSON 解析必须稳健（容忍代码块/字符串/对象混排）。"""
    with tempfile.TemporaryDirectory(prefix="r03_parse_") as tmp:
        storage = _new_storage(Path(tmp))
        sm = _new_summarizer(storage)

        raw = (
            "```json\n"
            '[{"fact":"我喜欢喝乌龙茶","importance":0.6},'
            '{"fact":"我住在上海","importance":0.5},'
            '"我的工号是A100"]\n'
            "```"
        )
        facts = sm._parse_llm_facts(raw, max_facts=10)
        assert len(facts) == 3, facts
        by_fact = {f["fact"]: f for f in facts}
        assert by_fact["我喜欢喝乌龙茶"]["pinned"] == 1, by_fact
        assert by_fact["我喜欢喝乌龙茶"]["importance"] == 0.9, by_fact
        assert by_fact["我住在上海"]["pinned"] == 0, by_fact
        assert by_fact["我的工号是A100"]["pinned"] == 0, by_fact

        # 非法 / 空输入必须安全返回空列表（由调用方回退规则抽取）
        assert sm._parse_llm_facts("not a json", 5) == []
        assert sm._parse_llm_facts("", 5) == []
        assert sm._parse_llm_facts("[1,2,3]", 5) == []
        print("[OK] R-03 LLM JSON 解析：混排格式正确归一并套用置顶规则，非法输入安全降级")
        storage.close()

async def test_r02_l1_realtime_and_idle_flush() -> None:
    with tempfile.TemporaryDirectory(prefix="r02_l1_") as tmp:
        storage = _new_storage(Path(tmp))
        sm = _new_summarizer(storage, embedding_client=CountingEmbeddingClient())

        # 1) 未达阈值（3/5 轮）→ 尚未总结，但 L1 必须已在 SQLite 中
        for user, bot in (
            ("我住在杭州", "记住了"),
            ("我的生日是3月5日", "记住了"),
            ("我的职业是程序员", "记住了"),
        ):
            await sm.enqueue_dialogue("s_l1", user, bot)

        counts = storage.get_table_counts()
        assert counts["l1_docs"] == 3, f"L1 未实时落盘: {counts}"
        assert counts["l2_atoms"] == 0, "未达阈值不应析出 L2"
        conn = storage.connect()
        metas = [
            str(r["metadata"])
            for r in conn.execute(
                "SELECT metadata FROM l1_docs WHERE scope_key = ?", ("s_l1",)
            ).fetchall()
        ]
        assert all('"l2_processed": false' in m for m in metas), metas
        print("[OK] R-02 未满 5 轮即已实时写入 L1=3 条，元数据 l2_processed=false")

        # 2) 强制冲洗 → 析出 L2 后回写 l2_processed=true
        sm.start()
        try:
            flushed = await sm.flush_session("s_l1")
            assert flushed == 3
            await asyncio.wait_for(sm._queue.join(), timeout=10)
            metas_after = [
                str(r["metadata"])
                for r in conn.execute(
                    "SELECT metadata FROM l1_docs WHERE scope_key = ?", ("s_l1",)
                ).fetchall()
            ]
            assert all('"l2_processed": true' in m for m in metas_after), metas_after
            assert storage.get_table_counts()["l2_atoms"] >= 1
            print("[OK] R-02 析出 L2 后 L1 状态回写 l2_processed=true")
        finally:
            await sm.stop(flush=False)

        # 3) idle 超时自动刷盘（1 秒静默 → 自动析出）
        storage2 = _new_storage(Path(tmp) / "idle")
        emb2 = CountingEmbeddingClient()
        sm2 = _new_summarizer(
            storage2,
            embedding_client=emb2,
            config={"summary_idle_flush_seconds": 1},
        )
        sm2.start()
        try:
            await sm2.enqueue_dialogue("s_idle", "我的车牌号是浙A12345", "已记录")
            await sm2.enqueue_dialogue("s_idle", "我养了一只猫叫豆豆", "已记录")
            assert sm2.pending_turns("s_idle") == 2
            await asyncio.sleep(2.2)
            assert sm2.pending_turns("s_idle") == 0, "idle 超时未触发刷盘"
            await asyncio.wait_for(sm2._queue.join(), timeout=10)
            assert storage2.get_table_counts()["l2_atoms"] >= 1
            print("[OK] R-02 idle 1s 静默后自动刷盘，缓冲池清空且 L2 已沉淀")
        finally:
            await sm2.stop(flush=False)
            storage2.close()

        storage.close()


# ---------------------------------------------------------------------------
# R-04：默认严格会话隔离
# ---------------------------------------------------------------------------


async def test_r04_strict_isolation() -> None:
    with tempfile.TemporaryDirectory(prefix="r04_iso_") as tmp:
        storage = _new_storage(Path(tmp))
        conn = storage.connect()
        # 先补齐扩展列（pinned/tier/status 等），再插入测试数据
        retriever = MemoryRetriever(
            storage=storage,
            embedding_client=ZeroEmbeddingClient(),
            rerank_client=None,
            config={"top_k": 10, "recall_pool_limit": 100},
        )
        retriever._ensure_storage_columns(conn)
        _insert_atom(conn, "atom_a", "s_a", "A 会话的杭州住址是西湖区")
        _insert_atom(conn, "atom_b", "s_b", "B 会话的杭州住址是滨江区")
        _insert_atom(conn, "atom_b_pin", "s_b", "B 会话置顶的私密密码", pinned=1)
        _insert_atom(conn, "atom_global", "global", "全局公告：所有会话可见的杭州规则")
        _insert_atom(conn, "atom_global_pin", "global", "全局置顶：杭州总则", pinned=1)

        # 默认配置（不显式传 strict_session_isolation）必须隔离
        assert retriever.config.get("strict_session_isolation", True) is True
        result = await retriever.retrieve("杭州", session_id="s_a", top_k=10)
        contents = [i.get("content", "") for i in result["retrieved_l2"]]
        contents += [i.get("content", "") for i in result["pinned_l3"]]

        assert any("A 会话" in c for c in contents), contents
        assert any("全局" in c for c in contents), "显式 global 记忆应允许召回"
        assert not any("B 会话" in c for c in contents), f"发生跨会话串台: {contents}"
        print(f"[OK] R-04 默认严格隔离：仅本会话 + global 可见，B 会话被物理隔离")

        # 显式关闭隔离时（兼容旧行为）应能看到其他会话
        loose = MemoryRetriever(
            storage=storage,
            embedding_client=ZeroEmbeddingClient(),
            rerank_client=None,
            config={
                "top_k": 10,
                "recall_pool_limit": 100,
                "strict_session_isolation": False,
            },
        )
        loose_result = await loose.retrieve("杭州", session_id="s_a", top_k=10)
        loose_contents = [i.get("content", "") for i in loose_result["retrieved_l2"]]
        loose_contents += [i.get("content", "") for i in loose_result["pinned_l3"]]
        assert any("B 会话" in c for c in loose_contents), loose_contents
        print("[OK] R-04 显式关闭隔离后恢复全局视图（配置开关有效）")

        await retriever.close()
        await loose.close()
        storage.close()


# ---------------------------------------------------------------------------
# R-05：检索纯读 + 异步刷盘访问计数
# ---------------------------------------------------------------------------


async def test_r05_retrieve_is_readonly() -> None:
    with tempfile.TemporaryDirectory(prefix="r05_read_") as tmp:
        storage = _new_storage(Path(tmp))
        conn = storage.connect()
        retriever = MemoryRetriever(
            storage=storage,
            embedding_client=ZeroEmbeddingClient(),
            rerank_client=None,
            config={"top_k": 5, "recall_pool_limit": 50, "access_flush_threshold": 10_000},
        )
        retriever._ensure_storage_columns(conn)
        _insert_atom(conn, "atom_r05", "s_r05", "杭州西湖区的咖啡馆推荐")
        result = await retriever.retrieve("杭州咖啡馆", session_id="s_r05", top_k=5)
        assert result["retrieved_l2"], "检索未命中，用例前置条件不成立"

        # 1) 检索后数据库仍为 0：证明没有同步写库
        row = conn.execute(
            "SELECT access_count FROM l2_atoms WHERE atom_id = ?", ("atom_r05",)
        ).fetchone()
        assert int(row["access_count"]) == 0, "检索仍在同步写库（R-05 未生效）"
        # 2) 内存计数已累加
        assert retriever._access_counter.get("atom_r05") == 1, retriever.get_access_stats()
        print(f"[OK] R-05 检索纯读：DB access_count=0，内存计数={retriever.get_access_stats()['pending_deltas']}")

        # 3) 显式刷盘后落库
        flushed = await retriever.flush_access_counters()
        assert flushed == 1, flushed
        row_after = conn.execute(
            "SELECT access_count FROM l2_atoms WHERE atom_id = ?", ("atom_r05",)
        ).fetchone()
        assert int(row_after["access_count"]) == 1, "异步刷盘未生效"

        # 4) close() 会把剩余计数刷净
        await retriever.retrieve("杭州咖啡馆", session_id="s_r05", top_k=5)
        await retriever.close()
        row_final = conn.execute(
            "SELECT access_count FROM l2_atoms WHERE atom_id = ?", ("atom_r05",)
        ).fetchone()
        assert int(row_final["access_count"]) == 2, "close() 未刷净剩余计数"
        print("[OK] R-05 异步刷盘 + close 兜底刷净均正常")
        storage.close()


# ---------------------------------------------------------------------------
# R-06：HTTP 连接池复用
# ---------------------------------------------------------------------------


async def test_r06_session_reuse() -> None:
    class FakeSession:
        """最小 aiohttp.ClientSession 替身：仅提供 closed 属性。"""

        def __init__(self) -> None:
            self.closed = False

    fake = FakeSession()
    owner = models_client._HttpSessionOwner(5.0, fake)
    first = await owner.get()
    second = await owner.get()
    assert first is second is fake, "Session 未被复用"
    await owner.close()
    assert fake.closed is False, "外部注入的 Session 不应被代为关闭"

    # 自建连接池：close 幂等
    owned = models_client._HttpSessionOwner(5.0)
    await owned.close()  # 未创建时调用应安全
    assert owned.session is None

    # 客户端具备 close()，且未配置密钥时不触网（返回安全零向量）
    client = models_client.EmbeddingClient(api_key="", dim=8)
    assert hasattr(client, "close") and callable(client.close)
    vecs = await client.get_embeddings(["a", "b"])
    assert vecs == [[0.0] * 8, [0.0] * 8], vecs
    await client.close()

    reranker = models_client.RerankClient(api_key="")
    assert hasattr(reranker, "close") and callable(reranker.close)
    await reranker.close()

    # 静态守卫：models_client 内禁止任何“每次请求新建 Session”的写法
    source = Path(models_client.__file__).read_text(encoding="utf-8")
    assert "async with aiohttp.ClientSession" not in source, (
        "models_client 仍存在每次请求新建 Session 的写法"
    )
    assert source.count("self._http = _HttpSessionOwner(") == 2, (
        "EmbeddingClient / RerankClient 均应持有持久连接池"
    )
    print("[OK] R-06 Session 只创建一次并复用，客户端 close() 幂等，静态守卫通过")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> int:
    cases = [
        test_r01_pinned_immortal,
        test_r03_batch_single_pass,
        test_r03_llm_json_parse,
        test_r02_l1_realtime_and_idle_flush,
        test_r04_strict_isolation,
        test_r05_retrieve_is_readonly,
        test_r06_session_reuse,
    ]
    for case in cases:
        asyncio.run(case())
    print(f"\n=== Stage6 止血热修专项 用例全部通过（{len(cases)} 项） ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
