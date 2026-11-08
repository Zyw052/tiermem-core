# -*- coding: utf-8 -*-
"""
Stage 5 自测：自作判定门禁 Evaluator Gate（廉价模型 jev）

覆盖点：
1. 配置解析：默认判官模型为 jev；凭证留空自动回退 summarizer；
2. 规则信息价值打分：寒暄/纯提问被过滤，个人事实被保留；
3. 规则冲突检测：否定极性翻转、明确推翻标记、无误判；
4. 判官模型成功路径：按模型 value_score 过滤 + 采纳合法 conflict_id；
5. 判官模型非法 JSON / 异常 / 超时 → 平滑降级为规则判定（degraded=True，不阻断）；
6. 门禁关闭时原样放行；
7. 端到端：判官不可用时 process_dialogue 仍正常写入 L2（主链路不被阻断）；
8. 端到端：冲突旧事实被软废弃（status=superseded），检索器不再召回。

运行： python3 tests/test_stage5_evaluator_gate.py
（同时兼容 pytest 收集：所有用例均为 test_* 函数，无外部依赖）
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from evaluator import EvaluatorGate  # noqa: E402
from retriever import MemoryRetriever  # noqa: E402
from storage import MemoryStorage  # noqa: E402
from summarizer import MemorySummarizer  # noqa: E402


# ---------------------------------------------------------------------------
# 测试桩
# ---------------------------------------------------------------------------


class FakeEmbeddingClient:
    """返回固定向量的 Embedding 桩，避免测试触网。"""

    dim = 8

    async def get_embedding(self, text: str):
        # 用字符和生成稳定伪向量，使相同文本向量一致
        vec = [0.0] * self.dim
        for idx, ch in enumerate(str(text)):
            vec[idx % self.dim] += (ord(ch) % 17) / 17.0
        return vec


class ZeroEmbeddingClient:
    """零向量桩：使余弦相似度不可用，仅“文本完全一致”时才融合，便于观察废弃行为。"""

    dim = 8

    async def get_embedding(self, text: str):
        return [0.0] * self.dim


def _new_storage(tmp_dir: Path) -> MemoryStorage:
    storage = MemoryStorage(tmp_dir)
    storage.init_schema()
    return storage


def _new_summarizer(
    tmp_dir: Path,
    evaluator_cfg: dict = None,
    interval: int = 5,
    embedding_client=None,
) -> MemorySummarizer:
    storage = _new_storage(tmp_dir)
    config = {
        "summary_turn_interval": interval,
        "evaluator": evaluator_cfg if evaluator_cfg is not None else {},
    }
    return MemorySummarizer(
        storage=storage,
        embedding_client=embedding_client or FakeEmbeddingClient(),
        rerank_client=None,
        config=config,
    )


def _gate(cfg: dict = None, llm_client=None) -> EvaluatorGate:
    return EvaluatorGate(
        {"evaluator": cfg if cfg is not None else {}}, llm_client=llm_client
    )


# ---------------------------------------------------------------------------
# 1. 配置解析
# ---------------------------------------------------------------------------


async def test_config_defaults_and_fallback() -> None:
    gate = _gate()
    assert gate.enabled is True
    assert gate.model_name == "jev", gate.model_name
    assert gate.min_value_score == 0.35
    assert gate.timeout_seconds == 8.0
    print("[OK] 默认判官模型为 jev，阈值/超时默认值正确")

    # 空字符串模型名 → 回退 jev
    gate2 = _gate({"model_name": "", "api_base": "", "api_key": ""})
    assert gate2.model_name == "jev"

    # 凭证留空 → 回退 summarizer 凭证
    gate3 = EvaluatorGate(
        {
            "evaluator": {"model_name": "jev-lite"},
            "summarizer": {"api_base": "https://judge.example/v1", "api_key": "sk-x"},
        }
    )
    assert gate3.api_base == "https://judge.example/v1"
    assert gate3.api_key == "sk-x"
    assert gate3.model_name == "jev-lite"
    assert gate3._build_endpoint() == "https://judge.example/v1/chat/completions"
    print("[OK] 凭证留空自动回退 summarizer，模型名可覆盖")


# ---------------------------------------------------------------------------
# 2. 规则信息价值打分
# ---------------------------------------------------------------------------


async def test_rule_value_scoring() -> None:
    gate = _gate()

    low_texts = ["你好呀", "好的谢谢", "嗯嗯呢", "哈哈哈哈哈", "晚安~"]
    for text in low_texts:
        score, reason = gate.score_value(text)
        assert score < gate.min_value_score, (text, score, reason)
    print(f"[OK] {len(low_texts)} 条寒暄/应答全部判为低价值（< {gate.min_value_score}）")

    # 纯提问、无记忆信号 → 低价值
    score, reason = gate.score_value("你吃饭了吗？")
    assert score < gate.min_value_score, (score, reason)
    print("[OK] 纯提问被判为低价值")

    high_texts = [
        "我住在杭州，我的生日是3月5日，请记住",
        "我喜欢喝绿茶，不喜欢咖啡",
        "我的手机号是13800000000，账号密码不要外传",
    ]
    for text in high_texts:
        score, reason = gate.score_value(text)
        assert score >= gate.min_value_score, (text, score, reason)
    print(f"[OK] {len(high_texts)} 条个人事实全部通过价值门禁")


# ---------------------------------------------------------------------------
# 3. 规则冲突检测
# ---------------------------------------------------------------------------


async def test_rule_conflict_detection() -> None:
    gate = _gate()
    candidates = [
        {"atom_id": "atom_tea", "content": "我喜欢喝绿茶"},
        {"atom_id": "atom_city", "content": "我住在北京"},
        {"atom_id": "atom_unrelated", "content": "我的职业是程序员"},
    ]

    # 3.1 否定极性翻转 → 冲突命中 atom_tea
    conflicts = gate.detect_conflicts("我不喜欢喝绿茶", candidates)
    ids = {c["atom_id"] for c in conflicts}
    assert "atom_tea" in ids, conflicts
    assert "atom_unrelated" not in ids
    print(f"[OK] 否定极性翻转识别成功: {conflicts[0]['reason']}")

    # 3.2 明确推翻标记 → 冲突命中 atom_city
    conflicts = gate.detect_conflicts("我已经搬家到上海了，不再是北京", candidates)
    ids = {c["atom_id"] for c in conflicts}
    assert "atom_city" in ids, conflicts
    print("[OK] 推翻标记（搬家/不再是）识别成功")

    # 3.3 无矛盾的新事实 → 不误判
    conflicts = gate.detect_conflicts("我养了一只叫豆豆的猫", candidates)
    assert conflicts == [], conflicts
    print("[OK] 无矛盾新事实不产生误判")

    # 3.4 冲突检测开关关闭 → 恒返回空
    gate_off = _gate({"conflict_check": False})
    assert gate_off.detect_conflicts("我不喜欢喝绿茶", candidates) == []
    print("[OK] conflict_check 关闭后不产生冲突")


# ---------------------------------------------------------------------------
# 4. 判官模型成功路径
# ---------------------------------------------------------------------------


async def test_llm_gate_success() -> None:
    captured = {}

    async def fake_llm(messages, model):
        captured["model"] = model
        captured["prompt"] = messages[-1]["content"]
        return json.dumps(
            {
                "keep": [
                    {"fact": "我住在杭州", "value_score": 0.95, "reason": "稳定事实"}
                ],
                "drop": [
                    {"fact": "今天天气不错", "value_score": 0.05, "reason": "废话"}
                ],
                "conflicts": [
                    {
                        "fact": "我现在住在杭州",
                        "conflict_id": "atom_city",
                        "reason": "住址变更",
                    },
                    {
                        "fact": "我现在住在杭州",
                        "conflict_id": "atom_fake_not_exist",
                        "reason": "模型臆造，应被过滤",
                    },
                ],
            },
            ensure_ascii=False,
        )

    gate = _gate(
        {
            "api_base": "https://judge.example/v1",
            "api_key": "sk-test",
            "model_name": "jev",
            "min_value_score": 0.35,
        },
        llm_client=fake_llm,
    )
    candidates = [{"atom_id": "atom_city", "content": "我住在北京"}]
    facts = [
        {"fact": "我住在杭州", "importance": 0.7, "pinned": 0, "atom_type": "fact"},
        {"fact": "今天天气不错", "importance": 0.5, "pinned": 0, "atom_type": "fact"},
    ]

    report = await gate.gate("s_llm", facts, candidates)
    assert report.source == "llm", report.source
    assert report.degraded is False
    assert captured["model"] == "jev", captured["model"]
    assert "历史事实" in captured["prompt"]

    kept_texts = [f["fact"] for f in report.kept]
    dropped_texts = [f["fact"] for f in report.dropped]
    assert kept_texts == ["我住在杭州"], kept_texts
    assert "今天天气不错" in dropped_texts, dropped_texts
    assert report.kept[0]["value_score"] == 0.95

    conflict_ids = {c["atom_id"] for c in report.conflicts}
    assert conflict_ids == {"atom_city"}, conflict_ids  # 臆造 id 被过滤
    print(f"[OK] 判官模型判定成功: kept={len(report.kept)} dropped={len(report.dropped)} conflicts={len(report.conflicts)}")
    print(f"[OK] 模型名透传正确: {captured['model']}；非法 conflict_id 已被过滤")


# ---------------------------------------------------------------------------
# 5. 判官异常降级（非法 JSON / 抛错 / 超时）
# ---------------------------------------------------------------------------


async def test_degrade_on_invalid_json() -> None:
    async def bad_json(messages, model):
        return "抱歉，我无法判断这些内容。"

    gate = _gate(
        {"api_base": "https://judge.example/v1", "api_key": "sk", "timeout_seconds": 2},
        llm_client=bad_json,
    )
    facts = [{"fact": "我住在杭州，请记住", "importance": 0.7, "pinned": 0}]
    report = await gate.gate("s_bad", facts, [])
    assert report.degraded is True, report
    assert report.source == "fallback", report.source
    assert "judge_error" in report.error, report.error
    assert len(report.kept) == 1, report.kept  # fail-open 不丢事实
    print(f"[OK] 非法 JSON 平滑降级: source={report.source} error={report.error[:40]}... 事实未丢失")


async def test_degrade_on_exception() -> None:
    async def boom(messages, model):
        raise ConnectionError("judge endpoint unreachable")

    gate = _gate(
        {"api_base": "https://judge.example/v1", "api_key": "sk"},
        llm_client=boom,
    )
    facts = [{"fact": "我喜欢喝乌龙茶", "importance": 0.8, "pinned": 1}]
    report = await gate.gate("s_exc", facts, [])
    assert report.degraded is True
    assert "ConnectionError" in report.error
    assert len(report.kept) == 1
    print("[OK] 判官连接异常平滑降级，事实未丢失")


async def test_degrade_on_timeout() -> None:
    async def slow(messages, model):
        await asyncio.sleep(1.5)
        return "{}"

    gate = _gate(
        {"api_base": "https://judge.example/v1", "api_key": "sk", "timeout_seconds": 0.2},
        llm_client=slow,
    )
    facts = [{"fact": "我的生日是3月5日，请记住", "importance": 0.9, "pinned": 1}]
    report = await gate.gate("s_slow", facts, [])
    assert report.degraded is True
    assert report.error == "judge_timeout", report.error
    assert len(report.kept) == 1
    print(f"[OK] 判官超时({gate.timeout_seconds}s)平滑降级: error={report.error}, 事实未丢失")


async def test_degrade_threshold_relaxed() -> None:
    """降级时阈值放宽：中等价值事实在降级路径保留，正常路径可能被过滤。"""

    async def boom(messages, model):
        raise RuntimeError("down")

    facts = [{"fact": "今天天气不错", "importance": 0.5, "pinned": 0}]
    # 正常阈值 0.5 → 规则分 0.5 保留；降级阈值 0.25 也保留
    gate = _gate(
        {
            "api_base": "https://judge.example/v1",
            "api_key": "sk",
            "min_value_score": 0.5,
            "fallback_min_value_score": 0.25,
        },
        llm_client=boom,
    )
    report = await gate.gate("s_th", facts, [])
    assert report.degraded is True
    assert len(report.kept) == 1, report.kept
    print("[OK] 降级路径使用放宽阈值，不误伤中等价值事实")


async def test_fail_open_false_uses_strict_threshold() -> None:
    """fail_open=False 时，降级路径仍按常规阈值严格过滤。"""

    async def boom(messages, model):
        raise RuntimeError("down")

    facts = [
        {"fact": "今天天气不错", "importance": 0.5, "pinned": 0},   # 规则分 0.5
        {"fact": "你好呀", "importance": 0.5, "pinned": 0},         # 寒暄 0.1
    ]
    gate = _gate(
        {
            "api_base": "https://judge.example/v1",
            "api_key": "sk",
            "min_value_score": 0.6,
            "fallback_min_value_score": 0.2,
            "fail_open": False,
        },
        llm_client=boom,
    )
    report = await gate.gate("s_fo", facts, [])
    assert report.degraded is True
    assert report.kept == [], [f["fact"] for f in report.kept]
    print("[OK] fail_open=False 时降级路径按常规阈值(0.6)严格过滤")

    # 对照：fail_open=True 时放宽阈值 → 0.5 分事实被保留
    gate_open = _gate(
        {
            "api_base": "https://judge.example/v1",
            "api_key": "sk",
            "min_value_score": 0.6,
            "fallback_min_value_score": 0.2,
            "fail_open": True,
        },
        llm_client=boom,
    )
    report_open = await gate_open.gate("s_fo2", facts, [])
    kept_texts = [f["fact"] for f in report_open.kept]
    assert "今天天气不错" in kept_texts, kept_texts
    assert "你好呀" not in kept_texts, kept_texts
    print("[OK] fail_open=True 时降级路径放宽阈值，仅过滤寒暄")


# ---------------------------------------------------------------------------
# 6. 门禁关闭
# ---------------------------------------------------------------------------


async def test_gate_disabled() -> None:
    gate = _gate({"enable": False})
    facts = [
        {"fact": "你好呀", "importance": 0.5, "pinned": 0},
        {"fact": "嗯嗯", "importance": 0.5, "pinned": 0},
    ]
    report = await gate.gate("s_off", facts, [])
    assert report.source == "disabled"
    assert len(report.kept) == 2
    assert report.dropped == []
    print("[OK] enable=False 时原样放行，不做任何过滤")


# ---------------------------------------------------------------------------
# 7. 端到端：判官不可用不阻断主记忆流程
# ---------------------------------------------------------------------------


async def test_e2e_fallback_does_not_block() -> None:
    async def boom(messages, model):
        raise TimeoutError("judge down")

    with tempfile.TemporaryDirectory(prefix="eval_e2e_fallback_") as tmp:
        sm = _new_summarizer(
            Path(tmp),
            evaluator_cfg={
                "api_base": "https://judge.example/v1",
                "api_key": "sk-test",
                "model_name": "jev",
                "timeout_seconds": 1,
            },
        )
        sm.evaluator._llm_client = boom

        await sm.process_dialogue(
            "s_e2e_fb", "我住在杭州，我的生日是3月5日，请记住", "好的，我记住了"
        )

        conn = sm.storage.connect()
        rows = conn.execute(
            "SELECT atom_id, content, status FROM l2_atoms WHERE scope_key = ?",
            ("s_e2e_fb",),
        ).fetchall()
        assert rows, "判官不可用时仍必须写入 L2，主链路不得阻断"
        assert sm.evaluator.stats["llm_failed"] >= 1
        assert all(str(r["status"]) == "active" for r in rows)
        print(f"[OK] 端到端降级：判官异常仍写入 {len(rows)} 条 L2，status 均为 active")
        print(f"[OK] 门禁统计: {sm.evaluator.stats}")


# ---------------------------------------------------------------------------
# 8. 端到端：冲突旧事实软废弃 + 检索不再召回
# ---------------------------------------------------------------------------


async def test_e2e_conflict_supersede_and_retrieval() -> None:
    """非置顶旧事实被推翻 → 软废弃 + 检索不再召回（置顶免死由 Stage6 专项覆盖）。"""
    with tempfile.TemporaryDirectory(prefix="eval_e2e_conflict_") as tmp:
        sm = _new_summarizer(
            Path(tmp), evaluator_cfg={}, embedding_client=ZeroEmbeddingClient()
        )

        # 第一轮：写入旧事实（刻意不含 L3 置顶关键词，确保 pinned=0 可被推翻）
        await sm.process_dialogue("s_conf", "我住在北京", "记住了")
        conn = sm.storage.connect()
        old = conn.execute(
            "SELECT atom_id, content, status, pinned FROM l2_atoms WHERE scope_key = ? AND content LIKE '%北京%'",
            ("s_conf",),
        ).fetchone()
        assert old is not None, "旧事实应已写入"
        assert str(old["status"]) == "active"
        assert int(old["pinned"]) == 0, "本用例要求旧事实为非置顶，否则应触发免死金牌"

        # 第二轮：明确推翻（搬家 + 不再是）
        await sm.process_dialogue("s_conf", "我已经搬家到上海了，不再是北京", "已更新")

        conn = sm.storage.connect()
        old_after = conn.execute(
            "SELECT status, superseded_by, importance, pinned FROM l2_atoms WHERE atom_id = ?",
            (old["atom_id"],),
        ).fetchone()
        assert str(old_after["status"]) == "superseded", dict(old_after)
        assert float(old_after["importance"]) <= 0.1
        assert int(old_after["pinned"]) == 0

        new_row = conn.execute(
            "SELECT atom_id, content, status FROM l2_atoms WHERE scope_key = ? AND content LIKE '%上海%'",
            ("s_conf",),
        ).fetchone()
        assert new_row is not None, "新事实应写入"
        assert str(old_after["superseded_by"]) == str(new_row["atom_id"]), (
            old_after["superseded_by"],
            new_row["atom_id"],
        )
        print(f"[OK] 冲突旧事实已软废弃: {old['atom_id']} -> superseded_by={new_row['atom_id']}")

        # 检索器不再召回被废弃事实
        retriever = MemoryRetriever(
            storage=sm.storage,
            embedding_client=FakeEmbeddingClient(),
            rerank_client=None,
            config={"top_k": 10, "recall_pool_limit": 100},
        )
        result = await retriever.retrieve("我住在哪个城市", session_id="s_conf", top_k=10)
        all_contents = [i.get("content", "") for i in result.get("retrieved_l2", [])]
        all_contents += [i.get("content", "") for i in result.get("pinned_l3", [])]
        assert not any("我住在北京" == c for c in all_contents), all_contents
        print(f"[OK] 检索器已排除被废弃事实；当前召回: {all_contents}")


# ---------------------------------------------------------------------------
# 9. 端到端：冲突事实被余弦融合时，不得自我废弃
# ---------------------------------------------------------------------------


async def test_e2e_merge_does_not_self_supersede() -> None:
    """新事实若被融合进旧原子（同一 atom_id），该原子必须保持 active。"""
    with tempfile.TemporaryDirectory(prefix="eval_e2e_merge_") as tmp:
        sm = _new_summarizer(Path(tmp), evaluator_cfg={})
        # 强制融合：所有文本向量相同
        class SameVecClient:
            dim = 8

            async def get_embedding(self, text: str):
                return [0.2] * self.dim

        sm.embedding_client = SameVecClient()

        await sm.process_dialogue("s_merge", "我喜欢喝绿茶", "记住了")
        await sm.process_dialogue("s_merge", "我不喜欢喝绿茶", "已更新")

        conn = sm.storage.connect()
        rows = conn.execute(
            "SELECT atom_id, content, status FROM l2_atoms WHERE scope_key = ?",
            ("s_merge",),
        ).fetchall()
        assert rows, "融合场景下仍应有原子记录"
        assert all(str(r["status"]) == "active" for r in rows), [dict(r) for r in rows]
        print(f"[OK] 融合场景未发生自我废弃，共 {len(rows)} 条原子，status 均为 active")


# ---------------------------------------------------------------------------
# 10. 配置装配：main.py 是否把 evaluator.* 正确注入 summarizer
# ---------------------------------------------------------------------------


async def test_main_config_wiring() -> None:
    """复用 Stage2 的宿主桩，验证 main.py 总装后 summarizer.evaluator 配置正确。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_stage2_hooks as stage2  # noqa: WPS433 (惰性导入，避免重复构造桩包)

    with tempfile.TemporaryDirectory(prefix="eval_cfg_wiring_") as tmp:
        plugin = stage2.make_plugin(
            Path(tmp),
            evaluator={
                "enable": True,
                "model_name": "jev-mini",
                "api_base": "",          # 留空 → 回退 summarizer.api_base
                "api_key": "",           # 留空 → 回退 summarizer.api_key
                "timeout_seconds": 3.5,
                "min_value_score": 0.4,
            },
            summarizer={
                "api_base": "https://judge.example/v1",
                "api_key": "sk-sum",
                "model": "sum-model",
            },
        )
        gate = plugin.summarizer.evaluator
        assert gate.enabled is True
        assert gate.model_name == "jev-mini", gate.model_name
        assert gate.api_base == "https://judge.example/v1", gate.api_base
        assert gate.api_key == "sk-sum"
        assert gate.timeout_seconds == 3.5
        assert gate.min_value_score == 0.4
        print(f"[OK] main.py 总装正确: {gate.describe()}")

        # 默认 model_name 留空 → 回退 jev
        plugin2 = stage2.make_plugin(Path(tmp) / "p2", evaluator={"model_name": ""})
        assert plugin2.summarizer.evaluator.model_name == "jev"
        print("[OK] model_name 留空时回退默认 jev")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> int:
    cases = [
        test_config_defaults_and_fallback,
        test_rule_value_scoring,
        test_rule_conflict_detection,
        test_llm_gate_success,
        test_degrade_on_invalid_json,
        test_degrade_on_exception,
        test_degrade_on_timeout,
        test_degrade_threshold_relaxed,
        test_fail_open_false_uses_strict_threshold,
        test_gate_disabled,
        test_e2e_fallback_does_not_block,
        test_e2e_conflict_supersede_and_retrieval,
        test_e2e_merge_does_not_self_supersede,
        test_main_config_wiring,
    ]
    for case in cases:
        asyncio.run(case())
    print(f"\n=== Stage5 判定门禁 用例全部通过（{len(cases)} 项） ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
