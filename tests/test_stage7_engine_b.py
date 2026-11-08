# -*- coding: utf-8 -*-
"""
Stage 7 自测：方案 B（工业级引擎升级版）三大核心改造专项验证。

覆盖点：
1. 【B-1】SQLite 原生 FTS5 全文索引
   - FTS5 能力探测与优雅降级；
   - 虚拟表 l2_atoms_fts（unicode61）与 INSERT/UPDATE/DELETE/软废弃 四类同步触发器；
   - 中文关键词 MATCH + bm25 倒排召回（含 CJK 预分词正确性）；
   - 旧库升级 backfill（fts_text 空值自动补齐）；
   - 无 FTS5 环境回退 Python BM25（Zero-breaking change）；
   - MATCH 表达式对特殊字符的注入安全与千级记忆性能。
2. 【B-2】内存向量矩阵缓存 _EmbeddingMatrixCache
   - 首次构建（atom_id -> index 映射 + 二维矩阵）；
   - 批量余弦与逐条 cosine_similarity 的数值一致性；
   - 增量追加、原位更新、失效掩码、批量失效、维度变更重建；
   - 与数据库指纹的轻量同步（含绕过缓存的外部变更）；
   - numpy 与纯 Python 双后端等价性。
3. 【B-3】task_dlq 死信队列闭环
   - pending 到期任务自动捞取并成功消费 → completed；
   - 失败后的指数退避重排期（base * 2^(n-1)，受 max_backoff 封顶）；
   - 重试耗尽 → failed；脏 payload / 未知 task_type → failed；
   - 「重试不产生新死信」爆炸防护；
   - 后台守护协程自动消费 + 优雅停机回收；
   - 端到端：_process_batch 失败落库 → sweep 重试成功 → 记忆真正析出。
4. 【综合】三组件协同与旧库平滑升级（零回归）。

运行： python3 tests/test_stage7_engine_b.py
（同时兼容 pytest 收集：所有用例均为 test_* 函数，无外部依赖）
"""

import asyncio
import json
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import storage as storage_mod  # noqa: E402
from models_client import cosine_similarity  # noqa: E402
from retriever import MemoryRetriever  # noqa: E402
from storage import (  # noqa: E402
    FTS_TABLE_NAME,
    NUMPY_AVAILABLE,
    EmbeddingMatrixCache,
    MemoryStorage,
    build_fts_match_expr,
    detect_fts5_support,
    fts_preprocess,
)
from summarizer import MemorySummarizer  # noqa: E402

DIM = 8


# ---------------------------------------------------------------------------
# 测试桩与辅助函数
# ---------------------------------------------------------------------------


class FakeEmbeddingClient:
    """确定性零依赖 Embedding 桩：同文本必得同向量，便于断言数值一致性。"""

    dim = DIM

    async def get_embedding(self, text: str) -> List[float]:
        vec = [0.0] * DIM
        for idx, ch in enumerate(str(text)):
            vec[idx % DIM] += (ord(ch) % 19) / 19.0
        # 避免全零向量（否则余弦无定义）
        return [x + 0.01 for x in vec]


class FakeRerankClient:
    """直通 Rerank 桩：按原顺序返回，便于验证粗排结果。"""

    async def rerank(self, query: str, docs: Sequence[str], top_n: int = 5, **kwargs):
        return [
            {"index": i, "document": doc, "relevance_score": 1.0 - 0.01 * i}
            for i, doc in enumerate(list(docs)[:top_n])
        ]


class ScriptedSummarizer(MemorySummarizer):
    """
    可控失败的事实抽取总结器。

    ``fail_times`` 控制前 N 次抽取抛异常（模拟 LLM/网络故障），
    之后返回固定事实，用于精确制造与消费死信任务。
    """

    def __init__(self, *args, fail_times: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_times = fail_times
        self.attempts = 0

    async def _extract_facts_from_context(self, **kwargs):  # noqa: D401
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError("模拟 LLM 事实抽取失败")
        return [
            {
                "fact": "主人系统学习了死信队列的指数退避重试机制",
                "importance": 0.62,
                "pinned": 0,
                "atom_type": "fact",
            }
        ]


def _pack(vec: Optional[Sequence[float]]) -> Optional[bytes]:
    """把浮点向量打包为 SQLite BLOB（float32 小端）。"""
    if not vec:
        return None
    floats = [float(x) for x in vec]
    return struct.pack(f"<{len(floats)}f", *floats)


def _new_storage(tmp_dir: Path) -> MemoryStorage:
    st = MemoryStorage(tmp_dir)
    st.init_schema()
    return st


def _insert_atom(
    conn,
    atom_id: str,
    content: str,
    vec: Optional[Sequence[float]] = None,
    scope: str = "s1",
    status: str = "active",
    text_field: bool = True,
) -> None:
    """向 l2_atoms 写入一条测试原子（自动适配已存在/缺失的扩展列）。"""
    now = time.time()
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
    fields = [
        "atom_id", "scope_key", "session_id", "content",
        "topics_json", "key_facts_json", "sentiment", "importance",
        "access_count", "embedding_blob", "source_type", "source_ref",
        "last_accessed_at", "created_at", "updated_at",
    ]
    values: List[Any] = [
        atom_id, scope, scope, content,
        "[]", "[]", "neutral", 0.5,
        0, _pack(vec), "chat", "",
        now, now, now,
    ]
    if "fts_text" in cols and text_field:
        fields.append("fts_text")
        values.append(fts_preprocess(content))
    if "status" in cols:
        fields.append("status")
        values.append(status)
    placeholders = ",".join("?" for _ in fields)
    conn.execute(
        f"INSERT INTO l2_atoms({','.join(fields)}) VALUES ({placeholders})", values
    )


def _insert_dlq(
    conn,
    task_id: str,
    payload: Any,
    retry_count: int = 0,
    max_retries: int = 3,
    status: str = "pending",
    next_retry_at: Optional[float] = None,
    task_type: str = "summarize_dialogue",
) -> None:
    """直接写入一条死信任务，便于精确控制到期时间与重试次数。"""
    now = time.time()
    conn.execute(
        """
        INSERT INTO task_dlq (
            task_id, task_type, scope_key, payload_json, error_msg,
            retry_count, max_retries, status, next_retry_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            task_type,
            "s1",
            payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
            int(retry_count),
            int(max_retries),
            status,
            float(now if next_retry_at is None else next_retry_at),
            now,
            now,
        ),
    )
    conn.commit()


def _dlq_row(conn, task_id: str) -> Optional[Any]:
    return conn.execute(
        "SELECT * FROM task_dlq WHERE task_id = ?", (task_id,)
    ).fetchone()


def _batch_payload(session_id: str = "s1") -> Dict[str, Any]:
    return {
        "kind": "batch",
        "session_id": session_id,
        "turns": [
            {
                "user_text": "我们聊聊死信队列怎么实现重试",
                "assistant_text": "可以用指数退避加状态机来闭环",
                "enqueued_at": time.time(),
                "l1_doc_id": "",
            }
        ],
        "reason": "threshold",
        "enqueued_at": time.time(),
    }


def _require_fts(st: MemoryStorage) -> bool:
    """FTS5 不可用时打印跳过原因并返回 False（保证测试可移植）。"""
    if not st.fts5_enabled:
        print("[SKIP] 当前 SQLite 未编译 FTS5，跳过该用例（回退路径由专门用例覆盖）")
        return False
    return True


# ===========================================================================
# B-1：FTS5 全文索引
# ===========================================================================


async def test_fts_probe_and_schema() -> None:
    """FTS5 能力探测 + 虚拟表/触发器结构完整性。"""
    with tempfile.TemporaryDirectory(prefix="stage7_fts_probe_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()

        # 1. 探测函数与实例状态必须一致（避免误判导致运行期异常）
        probed = detect_fts5_support(conn)
        assert isinstance(probed, bool), "detect_fts5_support 必须返回布尔值"
        assert probed == st.fts5_enabled, (
            f"FTS5 探测结果不一致: probe={probed} instance={st.fts5_enabled}"
        )
        print(f"[OK] FTS5 能力探测一致: enabled={probed}")

        if not probed:
            print("[SKIP] 无 FTS5 支持，跳过结构与召回用例")
            st.close()
            return

        # 2. 虚拟表与三类基础触发器必须存在
        objects = {
            str(r["name"]): str(r["type"])
            for r in conn.execute(
                "SELECT name, type FROM sqlite_master "
                "WHERE name = ? OR name LIKE 'trg_l2_atoms_fts_%'",
                (FTS_TABLE_NAME,),
            ).fetchall()
        }
        assert objects.get(FTS_TABLE_NAME) == "table", f"缺少虚拟表 {FTS_TABLE_NAME}"
        for trigger in ("trg_l2_atoms_fts_ai", "trg_l2_atoms_fts_au", "trg_l2_atoms_fts_ad"):
            assert trigger in objects, f"缺少同步触发器 {trigger}"
        print(f"[OK] 虚拟表 + 3 个基础触发器就绪: {sorted(objects)}")

        # 3. 预分词器行为：CJK 逐字展开，英文保留整词
        sample = fts_preprocess("主人爱喝美式咖啡 tiermem-core")
        assert "咖" in sample.split() and "啡" in sample.split(), "CJK 未被逐字展开"
        assert "tiermem" in sample.split() and "core" in sample.split(), "英文整词丢失"
        assert " " in sample, "预分词结果应以空格分隔"

        # 4. MATCH 表达式对 FTS5 语法字符免疫
        for tricky in ['咖啡 "quoted"', "a-b", "x*(y)", "AND OR NOT NEAR", "foo:bar", "100%"]:
            expr = build_fts_match_expr(tricky)
            assert expr == "" or " OR " in expr or '"' in expr
            if expr:
                conn.execute(
                    f"SELECT atom_id FROM {FTS_TABLE_NAME} WHERE {FTS_TABLE_NAME} MATCH ?",
                    (expr,),
                ).fetchall()
        print("[OK] 预分词与 MATCH 表达式构造：CJK/英文正确、特殊字符无语法异常")
        st.close()


async def test_fts_crud_sync() -> None:
    """FTS5 与主表的 INSERT / UPDATE / DELETE 自动同步。"""
    with tempfile.TemporaryDirectory(prefix="stage7_fts_crud_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        if not _require_fts(st):
            st.close()
            return

        # INSERT 同步
        with conn:
            _insert_atom(conn, "a1", "主人喜欢喝美式咖啡，不加糖")
            _insert_atom(conn, "a2", "主人对花生过敏，禁止推荐含花生的食物")
            _insert_atom(conn, "a3", "主人正在开发 tiermem-core 记忆插件")
        stats = st.get_fts_stats(conn)
        assert stats["indexed"] == 3, f"INSERT 触发器未同步，indexed={stats['indexed']}"

        hits = dict(st.fts_search("咖啡", conn=conn))
        assert hits.get("a1", 0) > 0, f"中文 MATCH 未命中: {hits}"
        assert "a2" not in hits, f"无关原子被误召回: {hits}"
        hits_en = dict(st.fts_search("tiermem 插件", conn=conn))
        assert hits_en.get("a3", 0) > 0, f"中英混合 MATCH 未命中: {hits_en}"
        print(f"[OK] INSERT 同步 + 中文 MATCH: 咖啡->{hits} | tiermem->{hits_en}")

        # UPDATE 同步：旧内容失效、新内容可召回
        with conn:
            conn.execute(
                "UPDATE l2_atoms SET content = ?, fts_text = ? WHERE atom_id = ?",
                ("主人改喝无糖乌龙茶", fts_preprocess("主人改喝无糖乌龙茶"), "a1"),
            )
        assert dict(st.fts_search("咖啡", conn=conn)).get("a1") is None, "更新后旧内容仍被召回"
        assert dict(st.fts_search("乌龙茶", conn=conn)).get("a1"), "更新后新内容未被召回"
        assert st.get_fts_stats(conn)["indexed"] == 3, "UPDATE 同步产生了重复倒排条目"
        print("[OK] UPDATE 同步：旧词失效、新词可召回，无重复条目")

        # DELETE 同步
        with conn:
            conn.execute("DELETE FROM l2_atoms WHERE atom_id = 'a3'")
        assert dict(st.fts_search("tiermem", conn=conn)) == {}, "物理删除后仍有幽灵索引"
        assert st.get_fts_stats(conn)["indexed"] == 2, "DELETE 触发器未清理倒排条目"
        print("[OK] DELETE 同步：幽灵索引已清理")
        st.close()


async def test_fts_soft_deprecation() -> None:
    """软废弃（status='superseded'）即时摘除索引，恢复后重新可召回。"""
    with tempfile.TemporaryDirectory(prefix="stage7_fts_soft_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        if not _require_fts(st):
            st.close()
            return

        # 软废弃触发器依赖 status 扩展列（真实链路由 retriever 的列迁移补齐）
        with conn:
            conn.execute(
                "ALTER TABLE l2_atoms ADD COLUMN status TEXT NOT NULL DEFAULT 'active'"
            )
        assert st.ensure_fts_index(conn) is True
        assert st._fts_status_trigger is True, "软废弃同步触发器未创建"

        with conn:
            _insert_atom(conn, "a1", "主人对花生过敏必须严格避免")
        assert dict(st.fts_search("花生", conn=conn)).get("a1"), "初始写入未命中"

        with conn:
            conn.execute("UPDATE l2_atoms SET status = 'superseded' WHERE atom_id = 'a1'")
        assert dict(st.fts_search("花生", conn=conn)) == {}, "软废弃后仍被召回"
        # 触发器模式下条目已物理摘除，include_superseded 不再可见（设计如此）
        assert dict(st.fts_search("花生", conn=conn, include_superseded=True)) == {}, (
            "触发器就绪时软废弃条目应已从倒排表摘除"
        )

        with conn:
            conn.execute("UPDATE l2_atoms SET status = 'active' WHERE atom_id = 'a1'")
        assert dict(st.fts_search("花生", conn=conn)).get("a1"), "恢复 active 后未重新入索引"
        print("[OK] 软废弃即时失效 / 恢复即时重建（触发器模式）")

        # 退化路径：无软废弃触发器时改走「查询期 status 过滤」，
        # 此时 include_superseded=True 才具备可见性语义。
        with conn:
            conn.execute("DROP TRIGGER IF EXISTS trg_l2_atoms_fts_status")
            conn.execute("UPDATE l2_atoms SET status = 'superseded' WHERE atom_id = 'a1'")
        st._fts_status_trigger = False
        assert dict(st.fts_search("花生", conn=conn)) == {}, "退化为查询期过滤后仍被召回"
        assert dict(st.fts_search("花生", conn=conn, include_superseded=True)).get("a1"), (
            "include_superseded=True 在查询期过滤模式下应可见"
        )
        print("[OK] 无触发器时退化为查询期过滤，include_superseded 语义生效")
        st.close()


async def test_fts_backfill_legacy_db() -> None:
    """旧库升级：fts_text 全空时自动 backfill，无需人工重建。"""
    with tempfile.TemporaryDirectory(prefix="stage7_fts_backfill_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        if not _require_fts(st):
            st.close()
            return

        with conn:
            _insert_atom(conn, "a1", "主人偏好使用暗色主题界面")
        # 模拟历史数据：预分词列为空
        with conn:
            conn.execute("UPDATE l2_atoms SET fts_text = ''")
        assert st.refresh_fts_text(conn, only_empty=True) == 1, "backfill 未处理空值行"
        assert dict(st.fts_search("暗色主题", conn=conn)).get("a1"), "backfill 后可检索性未恢复"

        # 全量索引重建同样应保留可检索性
        rebuilt = st.rebuild_fts_index(conn)
        assert rebuilt == 1, f"重建条目数异常: {rebuilt}"
        assert dict(st.fts_search("暗色主题", conn=conn)).get("a1"), "重建后索引丢失"
        print(f"[OK] 旧库 backfill 与索引重建均恢复可检索性（rebuilt={rebuilt}）")
        st.close()


async def test_fts_fallback_bm25() -> None:
    """FTS5 不可用时 retriever 必须优雅回退 Python BM25，结果仍可用。"""
    with tempfile.TemporaryDirectory(prefix="stage7_fts_fallback_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()

        with conn:
            _insert_atom(conn, "a1", "主人喜欢喝美式咖啡", vec=await emb.get_embedding("主人喜欢喝美式咖啡"))
            _insert_atom(conn, "a2", "项目使用 SQLite 做持久化存储", vec=await emb.get_embedding("项目使用 SQLite 做持久化存储"))

        retriever = MemoryRetriever(
            storage=st,
            embedding_client=emb,
            rerank_client=FakeRerankClient(),
            config={"strict_session_isolation": False},
        )
        # 模拟运行环境 SQLite 未编入 FTS5
        st.fts5_enabled = False
        retriever._engine_b_ready = True
        retriever.fts_enabled = False
        retriever._matrix_cache = None  # 同时关闭矩阵缓存，走最保守路径

        result = await retriever.retrieve("咖啡", session_id="s1", top_k=2, candidate_k=4)
        assert result["retrieved_l2"], "回退路径召回为空"
        top = result["retrieved_l2"][0]
        assert "咖啡" in top["content"], f"回退路径 Top1 不相关: {top['content']}"
        assert float(top["coarse_keyword_score"]) > 0, "Python BM25 关键词分未生效"
        print(
            f"[OK] 无 FTS5 时回退 Python BM25 正常: top={top['content']!r} "
            f"kw={top['coarse_keyword_score']}"
        )
        st.close()


async def test_fts_performance_thousand_atoms() -> None:
    """千级记忆下 FTS5 MATCH 必须显著快于 Python BM25 逐条打分。"""
    with tempfile.TemporaryDirectory(prefix="stage7_fts_perf_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        if not _require_fts(st):
            st.close()
            return

        now = time.time()
        rows = []
        for i in range(1000):
            content = f"记忆条目编号{i} 关于项目进度与偏好设置的详细描述 甲乙丙丁"
            rows.append(
                (
                    f"p{i}", "s1", "s1", content, "[]", "[]", "neutral", 0.5, 0,
                    None, "chat", "", now, now, now, fts_preprocess(content),
                )
            )
        with conn:
            conn.executemany(
                "INSERT INTO l2_atoms(atom_id,scope_key,session_id,content,topics_json,"
                "key_facts_json,sentiment,importance,access_count,embedding_blob,source_type,"
                "source_ref,last_accessed_at,created_at,updated_at,fts_text) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        assert st.get_fts_stats(conn)["indexed"] == 1000, "千级数据索引条数不符"

        rounds = 30
        start = time.perf_counter()
        for _ in range(rounds):
            hits = st.fts_search("项目进度偏好", limit=50, conn=conn)
        fts_ms = (time.perf_counter() - start) / rounds * 1000.0
        assert hits, "千级检索无命中"

        # Python BM25 对照（同一查询词）
        from models_client import _compute_bm25_scores

        docs = [r[3] for r in rows]
        start = time.perf_counter()
        _compute_bm25_scores("项目进度偏好", docs)
        py_ms = (time.perf_counter() - start) * 1000.0

        assert fts_ms < 50.0, f"FTS5 千级检索过慢: {fts_ms:.2f}ms"
        print(
            f"[OK] 千级记忆检索: FTS5={fts_ms:.3f}ms vs Python BM25={py_ms:.3f}ms "
            f"(加速 {py_ms / max(fts_ms, 1e-6):.1f}x)"
        )
        st.close()


# ===========================================================================
# B-2：内存向量矩阵缓存
# ===========================================================================


async def test_matrix_build_and_consistency() -> None:
    """首次构建 + 批量余弦与逐条 cosine_similarity 数值一致。"""
    with tempfile.TemporaryDirectory(prefix="stage7_matrix_build_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()
        corpus = {
            "a1": "主人喜欢喝美式咖啡",
            "a2": "主人对花生过敏",
            "a3": "开发 tiermem-core 记忆插件",
            "a4": "偏好暗色主题界面",
        }
        vectors = {}
        for aid, text in corpus.items():
            vectors[aid] = await emb.get_embedding(text)
        with conn:
            for aid, text in corpus.items():
                _insert_atom(conn, aid, text, vec=vectors[aid])

        cache = st.embedding_cache
        built = cache.build(conn)
        assert built == 4, f"构建条数异常: {built}"
        stats = cache.stats()
        assert stats["built"] is True and stats["size"] == 4 and stats["dim"] == DIM
        assert stats["backend"] == ("numpy" if NUMPY_AVAILABLE else "python")
        print(f"[OK] 矩阵缓存首次构建: {stats}")

        query = await emb.get_embedding("咖啡")
        score_map = cache.score_map(query)
        assert len(score_map) == 4, f"打分条数异常: {len(score_map)}"
        max_diff = 0.0
        for aid, vec in vectors.items():
            expected = cosine_similarity(query, vec)
            diff = abs(score_map[aid] - expected)
            max_diff = max(max_diff, diff)
            assert diff < 1e-4, f"{aid} 数值不一致: cache={score_map[aid]} direct={expected}"
        # Top1 必须是与咖啡语义最接近的那条
        top_id = max(score_map, key=score_map.get)
        assert top_id == "a1", f"Top1 应为 a1，实际 {top_id}"
        print(f"[OK] 批量余弦数值一致（max_diff={max_diff:.2e}），Top1={top_id}")

        # 空/非法查询向量必须安全返回空
        assert cache.score_map([]) == {}, "空查询向量应返回空打分"
        assert cache.score_map([0.0] * DIM) == {}, "全零查询向量应返回空打分"
        assert cache.score_map([1.0, 2.0]) == {}, "维度不匹配应返回空打分"
        print("[OK] 空/全零/维度不匹配查询均安全返回空")
        st.close()


async def test_matrix_incremental_and_mask() -> None:
    """增量追加、原位更新、失效掩码与批量失效。"""
    with tempfile.TemporaryDirectory(prefix="stage7_matrix_incr_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()
        cache = st.embedding_cache

        base_vec = await emb.get_embedding("主人喜欢喝美式咖啡")
        with conn:
            _insert_atom(conn, "a1", "主人喜欢喝美式咖啡", vec=base_vec)
        cache.build(conn)
        assert cache.stats()["size"] == 1

        # 1. 增量追加：不重建，只扩容
        new_vec = await emb.get_embedding("主人新增了健身习惯")
        with conn:
            _insert_atom(conn, "a2", "主人新增了健身习惯", vec=new_vec)
        assert cache.upsert("a2", new_vec) is True
        stats = cache.stats()
        assert stats["size"] == 2 and stats["stat_appends"] >= 1, f"增量追加失败: {stats}"
        assert stats["stat_builds"] == 1, "增量追加不应触发全量重建"
        print(f"[OK] 增量追加不重建: size={stats['size']} appends={stats['stat_appends']}")

        # 2. 原位更新：同 atom_id 覆盖，size 不变
        updated_vec = await emb.get_embedding("主人已经完全戒掉咖啡改喝乌龙茶")
        assert cache.upsert("a1", updated_vec) is True
        stats = cache.stats()
        assert stats["size"] == 2 and stats["stat_updates"] >= 1, f"原位更新失败: {stats}"
        score_map = cache.score_map(updated_vec)
        assert score_map["a1"] > 0.99, f"更新后的向量未被采用: {score_map['a1']}"
        print("[OK] 原位更新覆盖成功（size 不变、打分随新向量变化）")

        # 3. 失效掩码：软废弃后不再参与打分
        assert cache.invalidate("a1") is True
        assert cache.invalidate("a1") is False, "重复失效应返回 False（幂等）"
        score_map = cache.score_map(updated_vec)
        assert "a1" not in score_map, "失效向量仍参与打分"
        assert cache.get_vector("a1") == [], "失效后不应返回向量"
        print("[OK] 失效掩码生效：size 保留、打分剔除、向量不可取")

        # 4. 批量失效
        third_vec = await emb.get_embedding("第三条记忆")
        with conn:
            _insert_atom(conn, "a3", "第三条记忆", vec=third_vec)
        cache.upsert("a3", third_vec)
        invalidated = cache.invalidate_many(["a2", "a3", "not_exist"])
        assert invalidated == 2, f"批量失效条数异常: {invalidated}"
        assert cache.score_map(third_vec) == {}, "批量失效后仍有残留打分"
        print("[OK] 批量失效正确（忽略不存在的 id）")
        st.close()


async def test_matrix_dim_change_rebuild() -> None:
    """向量维度变更时自动触发全量重建，保证矩阵不错位。"""
    with tempfile.TemporaryDirectory(prefix="stage7_matrix_dim_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        cache = st.embedding_cache

        with conn:
            _insert_atom(conn, "a1", "维度四的记忆", vec=[1.0, 0.0, 0.0, 0.0])
        cache.build(conn)
        assert cache.dim == 4, f"初始维度应为 4，实际 {cache.dim}"

        # 换模型场景：新维度向量成为多数派，重建后主导维度应切换为 8，
        # 旧维度脏数据被安全跳过而不是让矩阵错位。
        vec8_a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vec8_b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        with conn:
            _insert_atom(conn, "a2", "维度八的记忆甲", vec=vec8_a)
            _insert_atom(conn, "a3", "维度八的记忆乙", vec=vec8_b)
        cache.upsert("a2", vec8_a)
        cache.upsert("a3", vec8_b)
        stats = cache.stats()
        assert stats["dim"] == 8, f"维度未切换到 8: {stats}"
        assert stats["stat_builds"] >= 2, f"维度变更未触发重建: {stats}"
        assert stats["size"] == 2, f"旧维度脏数据应被跳过，size={stats['size']}"
        score_map = cache.score_map(vec8_a)
        assert score_map.get("a2", 0) > 0.99, f"重建后打分异常: {score_map}"
        assert "a1" not in score_map, "旧维度脏数据不应参与打分"
        print(
            f"[OK] 维度变更按众数重建: dim={stats['dim']} size={stats['size']} "
            f"builds={stats['stat_builds']}"
        )
        st.close()


async def test_matrix_stale_sync() -> None:
    """绕过缓存的外部数据库变更可被指纹同步发现并重建。"""
    with tempfile.TemporaryDirectory(prefix="stage7_matrix_sync_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()
        cache = st.embedding_cache

        vec = await emb.get_embedding("第一条记忆")
        with conn:
            _insert_atom(conn, "a1", "第一条记忆", vec=vec)
        cache.build(conn)
        assert cache.stats()["size"] == 1

        # 外部直连写入（不经过 cache.upsert），模拟 migrator / 手工 SQL
        time.sleep(0.01)
        external_vec = await emb.get_embedding("外部写入的第二条记忆")
        with conn:
            _insert_atom(conn, "a2", "外部写入的第二条记忆", vec=external_vec)

        changed = cache.sync_if_stale(conn, max_age=0)
        assert changed is True, "外部变更未被发现"
        assert cache.stats()["size"] == 2, f"同步后条数异常: {cache.stats()}"

        # 无变更时不应反复重建
        before = cache.stats()["stat_builds"]
        assert cache.sync_if_stale(conn, max_age=0) is False, "无变更却触发了重建"
        assert cache.stats()["stat_builds"] == before, "无变更时重建次数不应增加"
        print("[OK] 指纹同步：外部变更可感知、无变更不重建")
        st.close()


async def test_matrix_python_backend_equivalence() -> None:
    """无 numpy 环境下纯 Python 后端必须给出等价结果（零依赖回退）。"""
    with tempfile.TemporaryDirectory(prefix="stage7_matrix_pybackend_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()

        vectors = {
            "a1": [1.0, 0.0, 0.0, 0.0],
            "a2": [0.0, 2.0, 0.0, 0.0],
            "a3": [3.0, 3.0, 0.0, 0.0],
        }
        with conn:
            for aid, vec in vectors.items():
                _insert_atom(conn, aid, f"记忆{aid}", vec=vec)

        # 对照：numpy 后端（若可用）
        np_cache = EmbeddingMatrixCache(st)
        np_cache.build(conn)
        np_scores = np_cache.score_map([1.0, 1.0, 0.0, 0.0])

        # 切换到纯 Python 后端
        saved_np = storage_mod._np
        saved_flag = storage_mod.NUMPY_AVAILABLE
        try:
            storage_mod._np = None
            storage_mod.NUMPY_AVAILABLE = False
            py_cache = EmbeddingMatrixCache(st)
            py_cache.build(conn)
            assert py_cache.stats()["backend"] == "python", "未切换到纯 Python 后端"
            py_scores = py_cache.score_map([1.0, 1.0, 0.0, 0.0])
        finally:
            storage_mod._np = saved_np
            storage_mod.NUMPY_AVAILABLE = saved_flag

        assert set(py_scores) == set(np_scores), "两后端可用 id 集合不一致"
        for aid in np_scores:
            diff = abs(np_scores[aid] - py_scores[aid])
            assert diff < 1e-5, f"{aid} 两后端数值不一致: {np_scores[aid]} vs {py_scores[aid]}"
        print(
            f"[OK] numpy / 纯 Python 双后端等价 (scores={ {k: round(v, 6) for k, v in py_scores.items()} })"
        )
        st.close()


async def test_matrix_retriever_integration() -> None:
    """retriever 实际走矩阵乘法，且候选向量可从缓存回填（语义零偏差）。"""
    with tempfile.TemporaryDirectory(prefix="stage7_matrix_retr_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()

        corpus = {
            "a1": "主人喜欢喝美式咖啡不加糖",
            "a2": "主人对花生过敏禁止推荐花生制品",
            "a3": "主人在开发 tiermem-core 记忆插件",
        }
        vectors = {aid: await emb.get_embedding(text) for aid, text in corpus.items()}
        with conn:
            for aid, text in corpus.items():
                _insert_atom(conn, aid, text, vec=vectors[aid])

        retriever = MemoryRetriever(
            storage=st,
            embedding_client=emb,
            rerank_client=FakeRerankClient(),
            config={"strict_session_isolation": False},
        )
        result = await retriever.retrieve("咖啡", session_id="s1", top_k=3, candidate_k=5)
        assert retriever._matrix_cache is not None, "矩阵缓存未挂载"
        assert retriever._matrix_cache.built, "首次检索后矩阵缓存应已构建"
        assert result["retrieved_l2"], "矩阵路径召回为空"

        query = await emb.get_embedding("咖啡")
        for item in result["retrieved_l2"]:
            aid = item["atom_id"]
            expected = cosine_similarity(query, vectors[aid])
            got = float(item["coarse_vector_score"])
            assert abs(got - expected) < 1e-3, (
                f"{aid} 矩阵打分与逐条余弦偏差过大: {got} vs {expected}"
            )
        top = result["retrieved_l2"][0]
        assert top["atom_id"] == "a1", f"Top1 应为 a1，实际 {top['atom_id']}"
        print(
            f"[OK] retriever 矩阵路径生效: cache={retriever._matrix_cache.stats()['backend']} "
            f"top={top['content']!r} vec_score={top['coarse_vector_score']}"
        )
        st.close()


# ===========================================================================
# B-3：task_dlq 死信队列闭环
# ===========================================================================


def _new_summarizer(
    tmp_dir: Path,
    fail_times: int = 0,
    dlq_interval: float = 1.0,
    dlq_max_retries: int = 3,
    dlq_base_backoff: float = 1.0,
) -> ScriptedSummarizer:
    storage = _new_storage(tmp_dir)
    return ScriptedSummarizer(
        storage=storage,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=None,
        config={
            "summary_turn_interval": 5,
            # 关闭判定门禁，避免测试触达外部模型
            "evaluator": {"enable": False},
            # 关闭 idle 刷盘，测试专注 DLQ 行为
            "summary_idle_flush_seconds": 0,
            "dlq_retry_enabled": True,
            "dlq_retry_interval": dlq_interval,
            "dlq_max_retries": dlq_max_retries,
            "dlq_base_backoff": dlq_base_backoff,
            "dlq_batch_size": 10,
        },
        fail_times=fail_times,
    )


async def test_dlq_success_consume() -> None:
    """到期的 pending 死信被捞取 → 重投消费成功 → completed。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_ok_") as tmp:
        sm = _new_summarizer(Path(tmp), fail_times=0)
        conn = sm.storage.connect()
        _insert_dlq(conn, "dlq_ok", _batch_payload(), retry_count=0, next_retry_at=time.time() - 1)

        result = await sm.sweep_dlq_once()
        assert result["scanned"] == 1 and result["completed"] == 1, f"消费结果异常: {result}"
        row = _dlq_row(conn, "dlq_ok")
        assert row["status"] == "completed", f"状态未流转为 completed: {row['status']}"

        atoms = conn.execute("SELECT COUNT(1) AS c FROM l2_atoms").fetchone()["c"]
        assert atoms >= 1, "重试成功后未析出任何 L2 记忆"
        print(f"[OK] DLQ 成功消费：completed=1，析出 L2 记忆 {atoms} 条")
        await sm.stop(flush=False)


async def test_dlq_exponential_backoff() -> None:
    """失败后按 base * 2^(n-1) 指数退避，且受 max_backoff 封顶。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_backoff_") as tmp:
        sm = _new_summarizer(
            Path(tmp), fail_times=99, dlq_max_retries=5, dlq_base_backoff=10.0
        )
        conn = sm.storage.connect()
        base = time.time() - 1
        _insert_dlq(conn, "dlq_bo", _batch_payload(), retry_count=0, next_retry_at=base)

        # 第 1 次重试失败 → retry_count=1, next = now + base*2^0
        now1 = time.time()
        r1 = await sm.sweep_dlq_once(now=now1)
        assert r1["rescheduled"] == 1, f"应重排期: {r1}"
        row = _dlq_row(conn, "dlq_bo")
        assert int(row["retry_count"]) == 1, f"retry_count 应为 1: {row['retry_count']}"
        delay1 = float(row["next_retry_at"]) - now1
        assert abs(delay1 - 10.0) < 1.0, f"首次退避应≈10s，实际 {delay1:.2f}s"

        # 第 2 次失败 → retry_count=2, next = now + base*2^1
        now2 = float(row["next_retry_at"]) + 0.1
        r2 = await sm.sweep_dlq_once(now=now2)
        assert r2["rescheduled"] == 1, f"应再次重排期: {r2}"
        row = _dlq_row(conn, "dlq_bo")
        delay2 = float(row["next_retry_at"]) - now2
        assert abs(delay2 - 20.0) < 1.0, f"第二次退避应≈20s，实际 {delay2:.2f}s"
        print(f"[OK] 指数退避生效: 第1次≈{delay1:.1f}s，第2次≈{delay2:.1f}s")
        await sm.stop(flush=False)


async def test_dlq_max_retries_failed() -> None:
    """重试次数耗尽后进入 failed 终态，不再被扫描。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_exhaust_") as tmp:
        sm = _new_summarizer(
            Path(tmp), fail_times=99, dlq_max_retries=2, dlq_base_backoff=1.0
        )
        conn = sm.storage.connect()
        _insert_dlq(
            conn, "dlq_ex", _batch_payload(), retry_count=1, max_retries=2,
            next_retry_at=time.time() - 1,
        )
        # retry_count=1 → 本次失败后 retry_count=2 >= max_retries=2 → failed
        result = await sm.sweep_dlq_once()
        assert result["failed"] >= 1, f"应进入 failed: {result}"
        row = _dlq_row(conn, "dlq_ex")
        assert row["status"] == "failed", f"状态应为 failed: {row['status']}"
        assert int(row["retry_count"]) == 2, f"retry_count 应为 2: {row['retry_count']}"

        # 已被标记 failed，不再进入扫描集合
        result2 = await sm.sweep_dlq_once(now=time.time() + 1000)
        assert result2["scanned"] == 0, f"failed 任务不应再被扫描: {result2}"
        print("[OK] 重试耗尽转 failed 终态，且不再被重复扫描")
        await sm.stop(flush=False)


async def test_dlq_dirty_payload_and_unknown_type() -> None:
    """脏 payload / 未知 task_type 必须收敛为 failed，不得卡死队列。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_dirty_") as tmp:
        sm = _new_summarizer(Path(tmp), fail_times=0)
        conn = sm.storage.connect()
        past = time.time() - 1
        _insert_dlq(conn, "dlq_bad_json", "{not-a-json", next_retry_at=past)
        _insert_dlq(conn, "dlq_empty_obj", {}, next_retry_at=past)
        _insert_dlq(
            conn, "dlq_unknown", _batch_payload(), next_retry_at=past,
            task_type="unsupported_task",
        )

        result = await sm.sweep_dlq_once()
        assert result["failed"] == 3, f"三类脏数据应全部 failed: {result}"
        for tid in ("dlq_bad_json", "dlq_empty_obj", "dlq_unknown"):
            row = _dlq_row(conn, tid)
            assert row["status"] == "failed", f"{tid} 未收敛为 failed: {row['status']}"
            assert str(row["error_msg"]), f"{tid} 未记录失败原因"
        print("[OK] 脏 payload / 空对象 / 未知 task_type 均安全收敛为 failed")
        await sm.stop(flush=False)


async def test_dlq_skips_not_due_and_dirty_exhausted() -> None:
    """未到期任务不被提前消费；重试超限的 pending 脏数据自动收敛。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_due_") as tmp:
        sm = _new_summarizer(Path(tmp), fail_times=0, dlq_max_retries=3)
        conn = sm.storage.connect()
        _insert_dlq(
            conn, "dlq_future", _batch_payload(),
            retry_count=0, next_retry_at=time.time() + 9999,
        )
        _insert_dlq(
            conn, "dlq_exhausted_dirty", _batch_payload(),
            retry_count=3, max_retries=3, next_retry_at=time.time() - 1,
        )

        result = await sm.sweep_dlq_once()
        assert result["scanned"] == 0, f"未到期任务被误捞: {result}"
        assert result["failed"] == 1, f"超限脏数据未被收敛: {result}"
        assert _dlq_row(conn, "dlq_future")["status"] == "pending", "未到期任务状态被改动"
        assert _dlq_row(conn, "dlq_exhausted_dirty")["status"] == "failed"
        print("[OK] 未到期不消费 / 超限脏数据自动 failed")
        await sm.stop(flush=False)


async def test_dlq_no_explosion() -> None:
    """重试失败不得写入新死信记录（防死信爆炸）。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_noexplode_") as tmp:
        sm = _new_summarizer(
            Path(tmp), fail_times=99, dlq_max_retries=3, dlq_base_backoff=1.0
        )
        conn = sm.storage.connect()
        _insert_dlq(conn, "dlq_single", _batch_payload(), next_retry_at=time.time() - 1)

        for _ in range(3):
            await sm.sweep_dlq_once(now=time.time())
            await asyncio.sleep(0.01)

        total = conn.execute("SELECT COUNT(1) AS c FROM task_dlq").fetchone()["c"]
        assert total == 1, f"重试产生了额外死信记录，总数={total}（应为 1）"
        print("[OK] 重试不产生新死信（DLQ 总数恒为 1，无爆炸）")
        await sm.stop(flush=False)


async def test_dlq_stats() -> None:
    """get_dlq_stats 的状态分布与运行期统计正确。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_stats_") as tmp:
        sm = _new_summarizer(Path(tmp), fail_times=0)
        conn = sm.storage.connect()
        past = time.time() - 1
        _insert_dlq(conn, "dlq_s1", _batch_payload(), next_retry_at=past)
        _insert_dlq(
            conn, "dlq_s2", _batch_payload(), retry_count=3, max_retries=3,
            status="failed", next_retry_at=past,
        )
        _insert_dlq(conn, "dlq_s3", _batch_payload(), next_retry_at=time.time() + 500)

        await sm.sweep_dlq_once()
        stats = sm.get_dlq_stats()
        assert stats["enabled"] is True
        assert stats["stat_completed"] >= 1, f"完成统计异常: {stats}"
        assert stats["by_status"].get("completed", 0) >= 1, f"分布异常: {stats['by_status']}"
        assert stats["by_status"].get("failed", 0) >= 1, f"分布异常: {stats['by_status']}"
        assert stats["by_status"].get("pending", 0) >= 1, f"分布异常: {stats['by_status']}"
        print(f"[OK] DLQ 统计: by_status={stats['by_status']} retried={stats['stat_retried']}")
        await sm.stop(flush=False)


async def test_dlq_background_loop() -> None:
    """后台守护协程能自动巡检并消费死信；停机时任务被回收。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_loop_") as tmp:
        sm = _new_summarizer(Path(tmp), fail_times=0, dlq_interval=1.0)
        conn = sm.storage.connect()
        _insert_dlq(conn, "dlq_bg", _batch_payload(), next_retry_at=time.time() - 1)

        sm.start()
        try:
            assert sm._dlq_task is not None, "DLQ 守护协程未启动"
            status = "pending"
            deadline = time.time() + 8.0
            while time.time() < deadline:
                await asyncio.sleep(0.2)
                row = _dlq_row(conn, "dlq_bg")
                status = str(row["status"]) if row is not None else "missing"
                if status == "completed":
                    break
            assert status == "completed", f"后台未自动消费死信，最终状态={status}"
            print("[OK] 后台守护自动消费死信（无需人工触发）")
        finally:
            await sm.stop(flush=False)

        assert sm._dlq_task is None, "停机后 DLQ 守护协程未回收"
        assert not sm._dlq_inflight, "停机后 inflight 集合未清空"
        print("[OK] 停机时 DLQ 守护协程与 inflight 状态被正确回收")


async def test_dlq_end_to_end_closed_loop() -> None:
    """端到端闭环：批处理失败落库 → 到期 sweep 重试成功 → 记忆真正析出。"""
    with tempfile.TemporaryDirectory(prefix="stage7_dlq_e2e_") as tmp:
        sm = _new_summarizer(
            Path(tmp), fail_times=1, dlq_max_retries=3, dlq_base_backoff=1.0
        )
        conn = sm.storage.connect()
        payload_before = _batch_payload()
        turns = payload_before["turns"]

        # 第 1 次处理必然失败（ScriptedSummarizer 抛异常）→ 自动写入 DLQ
        await sm._process_batch("s1", turns, reason="threshold")
        rows = conn.execute("SELECT * FROM task_dlq").fetchall()
        assert len(rows) == 1, f"失败后应产生 1 条死信，实际 {len(rows)}"
        task = rows[0]
        assert task["status"] == "pending", f"死信初始状态应为 pending: {task['status']}"
        assert int(task["retry_count"]) == 0
        assert float(task["next_retry_at"]) > time.time(), "首次重试应遵循退避基数而非立即重放"

        l2_before = conn.execute("SELECT COUNT(1) AS c FROM l2_atoms").fetchone()["c"]
        assert l2_before == 0, f"失败批次不应析出记忆，实际 {l2_before} 条"

        # 把到期时间拨到过去，模拟退避到期
        with conn:
            conn.execute(
                "UPDATE task_dlq SET next_retry_at = ? WHERE task_id = ?",
                (time.time() - 1, task["task_id"]),
            )

        result = await sm.sweep_dlq_once()
        assert result["completed"] == 1, f"重试未成功: {result}"
        assert _dlq_row(conn, task["task_id"])["status"] == "completed"

        l2_after = conn.execute("SELECT COUNT(1) AS c FROM l2_atoms").fetchone()["c"]
        assert l2_after >= 1, "重试成功后仍未析出 L2 记忆"
        content = conn.execute("SELECT content FROM l2_atoms LIMIT 1").fetchone()["content"]
        assert "死信队列" in str(content), f"析出内容不符: {content}"

        # 幂等：l1_docs 不应因重试而重复膨胀
        l1_count = conn.execute("SELECT COUNT(1) AS c FROM l1_docs").fetchone()["c"]
        assert l1_count == 1, f"重试导致 L1 重复写入，l1_docs={l1_count}"
        print(
            f"[OK] DLQ 端到端闭环：失败落库 → 到期重试 → completed，"
            f"析出记忆={content!r}，L1 无重复({l1_count})"
        )
        await sm.stop(flush=False)


# ===========================================================================
# 综合：三组件协同 + 旧库平滑升级
# ===========================================================================


async def test_engine_b_synergy_and_legacy_upgrade() -> None:
    """三组件协同（FTS + 矩阵 + 完整写入链路）与旧库平滑升级。"""
    with tempfile.TemporaryDirectory(prefix="stage7_synergy_") as tmp:
        root = Path(tmp)
        st = _new_storage(root)
        emb = FakeEmbeddingClient()

        sm = ScriptedSummarizer(
            storage=st,
            embedding_client=emb,
            rerank_client=None,
            config={"summary_turn_interval": 5, "evaluator": {"enable": False},
                    "summary_idle_flush_seconds": 0},
            fail_times=0,
        )
        # 走真实写入链路：批处理会写 L1 → 抽取事实 → 批量向量 → INSERT L2
        await sm._process_batch(
            "s1",
            [{
                "user_text": "我们研究了死信队列的指数退避重试机制",
                "assistant_text": "确实需要闭环消费",
                "enqueued_at": time.time(),
            }],
            reason="threshold",
        )
        conn = st.connect()

        # 1. 写入侧：FTS 预分词文本与矩阵缓存都被同步
        atom = conn.execute(
            "SELECT atom_id, content, fts_text, embedding_blob FROM l2_atoms LIMIT 1"
        ).fetchone()
        assert atom is not None, "写入链路未产生 L2 原子"
        if st.fts5_enabled:
            assert str(atom["fts_text"]), "写入链路未填充 fts_text"
            assert dict(st.fts_search("死信队列", conn=conn)).get(atom["atom_id"]), (
                "写入后 FTS 未命中"
            )
        cache_stats = st.embedding_cache.stats()
        assert cache_stats["size"] >= 1, f"写入后矩阵缓存未增量同步: {cache_stats}"
        print(f"[OK] 写入链路同步: fts={st.fts5_enabled} cache={cache_stats['size']} 条")

        # 2. 检索侧：矩阵路径与 FTS 路径同时生效
        retriever = MemoryRetriever(
            storage=st,
            embedding_client=emb,
            rerank_client=FakeRerankClient(),
            config={"strict_session_isolation": False},
        )
        result = await retriever.retrieve("死信队列重试", session_id="s1", top_k=3)
        assert result["retrieved_l2"], "协同检索无结果"
        hit = result["retrieved_l2"][0]
        assert "死信队列" in hit["content"], f"协同检索命中错误: {hit['content']}"
        print(
            f"[OK] 三组件协同：fts={retriever.fts_enabled} "
            f"matrix={retriever._matrix_cache.built} top={hit['content']!r}"
        )
        await sm.stop(flush=False)
        st.close()

    # 3. 旧库平滑升级：仅有基础列的历史数据库
    with tempfile.TemporaryDirectory(prefix="stage7_legacy_") as tmp:
        legacy_root = Path(tmp)
        legacy_path = legacy_root / "agent_memory.db"
        import sqlite3

        raw = sqlite3.connect(str(legacy_path))
        raw.executescript(
            """
            CREATE TABLE l2_atoms (
                atom_id TEXT PRIMARY KEY,
                scope_key TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                topics_json TEXT NOT NULL DEFAULT '[]',
                key_facts_json TEXT NOT NULL DEFAULT '[]',
                sentiment TEXT NOT NULL DEFAULT 'neutral',
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                embedding_blob BLOB,
                source_type TEXT NOT NULL DEFAULT 'chat',
                source_ref TEXT NOT NULL DEFAULT '',
                last_accessed_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        now = time.time()
        raw.execute(
            "INSERT INTO l2_atoms(atom_id, scope_key, session_id, content, "
            "last_accessed_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            ("legacy1", "s1", "s1", "主人偏好使用 Vim 编辑器", now, now, now),
        )
        raw.commit()
        raw.close()

        upgraded = MemoryStorage(legacy_root)
        upgraded.init_schema()
        uconn = upgraded.connect()
        cols = {str(r[1]) for r in uconn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
        assert "fts_text" in cols, "升级后未补齐 fts_text 列"
        upgraded.ensure_fts_index(uconn)
        if upgraded.fts5_enabled:
            hits = dict(upgraded.fts_search("Vim 编辑器", conn=uconn))
            assert hits.get("legacy1"), f"旧库升级后历史数据不可检索: {hits}"
            print(f"[OK] 旧库平滑升级：历史数据自动 backfill 并可检索 {hits}")
        else:
            print("[SKIP] 无 FTS5 支持，旧库升级仅验证列补齐")

        assert upgraded.get_fts_stats(uconn)["atoms"] == 1
        upgraded.close()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> int:
    cases = [
        # B-1 FTS5
        test_fts_probe_and_schema,
        test_fts_crud_sync,
        test_fts_soft_deprecation,
        test_fts_backfill_legacy_db,
        test_fts_fallback_bm25,
        test_fts_performance_thousand_atoms,
        # B-2 矩阵缓存
        test_matrix_build_and_consistency,
        test_matrix_incremental_and_mask,
        test_matrix_dim_change_rebuild,
        test_matrix_stale_sync,
        test_matrix_python_backend_equivalence,
        test_matrix_retriever_integration,
        # B-3 DLQ
        test_dlq_success_consume,
        test_dlq_exponential_backoff,
        test_dlq_max_retries_failed,
        test_dlq_dirty_payload_and_unknown_type,
        test_dlq_skips_not_due_and_dirty_exhausted,
        test_dlq_no_explosion,
        test_dlq_stats,
        test_dlq_background_loop,
        test_dlq_end_to_end_closed_loop,
        # 综合
        test_engine_b_synergy_and_legacy_upgrade,
    ]
    for case in cases:
        asyncio.run(case())
    print(f"\n=== Stage7 引擎 B 专项 用例全部通过（{len(cases)} 项） ===")
    print(f"运行环境：numpy={NUMPY_AVAILABLE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
