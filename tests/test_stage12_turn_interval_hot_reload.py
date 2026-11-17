# -*- coding: utf-8 -*-
"""
Stage 12：窗口阈值（summary_turn_interval）热更新回归测试。

背景（本次修复的 bug）：
    面板保存配置后，``apply_runtime_config()`` 只更新了插件层的
    ``self.turn_interval``，却没有把新阈值回写到**已构建**的
    ``MemorySummarizer`` 实例，导致运行中的 Worker 仍按旧阈值判定窗口，
    表现为「配置改了但总结窗口不变」的假热更新。

覆盖点：
1. 【summarizer 层】``set_turn_interval()`` 合法值生效、非法值不覆盖原值、
   ``self.config`` 同步回写；
2. 【插件层】``apply_runtime_config()`` 把新阈值回写到 ``self.summarizer``；
3. 【端到端行为】阈值 5 → 20 热更后，第 5 轮**不**触发打包、第 20 轮才触发；
4. 【反向行为】阈值 20 → 5 热更后，下一次判定即按 5 轮打包（不回滚存量缓冲）；
5. 【重载语义】``_is_reload_required()`` 与实际热更能力一致：
   ``summarizer.summary_turn_interval`` 不提示重载，凭证/模型类仍提示重载；
6. 【未就绪】summarizer 尚未装配时热更新不抛异常（延迟到构建时生效）。

运行： python -m pytest tests/test_stage12_turn_interval_hot_reload.py -q
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1) 构造最小 AstrBot 桩包（必须在 import main 之前注入 sys.path）
#    形态严格对齐真实 AstrBot 装饰器：on_llm_request / on_llm_response /
#    on_decorating_result 均为「无参工厂装饰器」。
# ---------------------------------------------------------------------------
STUB_ROOT = Path(tempfile.mkdtemp(prefix="astrbot_stub_stage12_"))

_STUB_FILES = {
    "astrbot/__init__.py": "",
    "astrbot/api/__init__.py": textwrap.dedent(
        """
        import logging

        logger = logging.getLogger("astrbot")
        """
    ),
    "astrbot/api/event.py": textwrap.dedent(
        '''
        """AstrBot 事件装饰器桩。"""


        def _make_hook_factory(hook_name):
            def factory(*args, **kwargs):
                def decorator(func):
                    setattr(func, "__astrbot_hook__", hook_name)
                    return func

                return decorator

            return factory


        on_llm_request = _make_hook_factory("on_llm_request")
        on_llm_response = _make_hook_factory("on_llm_response")
        on_decorating_result = _make_hook_factory("on_decorating_result")


        class _FilterModule:
            on_llm_request = staticmethod(on_llm_request)
            on_llm_response = staticmethod(on_llm_response)
            on_decorating_result = staticmethod(on_decorating_result)


        filter = _FilterModule()
        '''
    ),
    "astrbot/api/star.py": textwrap.dedent(
        '''
        """AstrBot Star 基类桩。"""


        class Context:
            def __init__(self, *args, **kwargs):
                pass


        class Star:
            def __init__(self, context=None):
                self.context = context


        def register(*args, **kwargs):
            def decorator(cls):
                cls.__astrbot_registered__ = True
                return cls

            return decorator
        '''
    ),
    "astrbot/core/__init__.py": "",
    "astrbot/core/utils/__init__.py": "",
    "astrbot/core/utils/astrbot_path.py": textwrap.dedent(
        f'''
        """数据目录桩：指向临时目录，避免测试污染真实插件数据。"""

        from pathlib import Path

        _DATA_ROOT = Path({str(STUB_ROOT / "plugin_data_root")!r})


        def get_astrbot_data_path():
            _DATA_ROOT.mkdir(parents=True, exist_ok=True)
            return str(_DATA_ROOT)
        '''
    ),
}

for _rel_path, _content in _STUB_FILES.items():
    _target = STUB_ROOT / _rel_path
    _target.parent.mkdir(parents=True, exist_ok=True)
    _target.write_text(_content, encoding="utf-8")

if str(STUB_ROOT) not in sys.path:
    sys.path.insert(0, str(STUB_ROOT))

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

# 清掉可能已被其它测试模块导入的 astrbot / main，确保使用本文件的桩包
for _mod in ("astrbot", "main"):
    sys.modules.pop(_mod, None)

import config_api  # noqa: E402
import main as plugin_main  # noqa: E402
from main import AgentMemoryPlugin  # noqa: E402
from storage import MemoryStorage  # noqa: E402
from summarizer import MemorySummarizer  # noqa: E402

# 测试隔离：绝不让被测插件读写真实插件的持久化目录
plugin_main.PLUGIN_NAME = "astrbot_plugin_agent_memory_test"


class FakeEmbeddingClient:
    """零向量 Embedding 客户端桩，避免测试依赖外部 API。"""

    dim = 8

    async def get_embedding(self, text: str):
        return [0.0] * self.dim


def _new_summarizer(tmp_dir: Path, interval: int = 5) -> MemorySummarizer:
    """构造一个不启动后台 Worker 的 summarizer（仅验证缓冲窗口逻辑）。"""
    storage = MemoryStorage(tmp_dir)
    storage.init_schema()
    return MemorySummarizer(
        storage=storage,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=None,
        config={"summary_turn_interval": interval},
    )


def _make_plugin(cfg: dict, data_dir: Path) -> AgentMemoryPlugin:
    """
    构造插件实例（不执行 initialize，避免触碰网络）。

    必须显式传入 ``db_path``：否则 ``_build_storage()`` 会走「默认数据目录」
    分支并 ``mkdir`` 到宿主真实的 ``plugin_data/<PLUGIN_NAME>``，
    使测试向现网数据目录写文件（本文件曾因此泄漏过 ``_test`` 目录）。
    """
    merged = dict(cfg)
    merged.setdefault("db_path", str(Path(data_dir) / "agent_memory.db"))
    return AgentMemoryPlugin(context=None, config=merged)


# ---------------------------------------------------------------------------
# 一、summarizer 层：set_turn_interval() 的安全语义
# ---------------------------------------------------------------------------
def test_set_turn_interval_accepts_valid_value():
    """合法值生效，并同步回写 self.config（状态快照不漂移）。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        sm = _new_summarizer(Path(tmp), interval=5)
        assert sm.turn_interval == 5, "初始阈值应为 5"

        assert sm.set_turn_interval(20) is True
        assert sm.turn_interval == 20, "合法值应生效"
        assert sm.config["summary_turn_interval"] == 20, "config 未同步回写"
        assert sm.get_buffer_stats()["turn_interval"] == 20, "状态快照读到旧值"

        # 数字字符串应被接受（面板提交常为字符串）
        assert sm.set_turn_interval("7") is True
        assert sm.turn_interval == 7


@pytest.mark.parametrize("bad", [None, "", "abc", 0, -3, "0", "-1", [], {}])
def test_set_turn_interval_rejects_invalid_value(bad):
    """非法值一律不覆盖原阈值，并返回 False。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        sm = _new_summarizer(Path(tmp), interval=8)
        assert sm.set_turn_interval(bad) is False
        assert sm.turn_interval == 8, f"非法值 {bad!r} 覆盖了原阈值"
        assert sm.config["summary_turn_interval"] == 8, "非法值污染了 config"


def test_set_turn_interval_reuses_parser_semantics():
    """set_turn_interval 与构造函数走同一套校验（默认回退 5 / 子节点读取）。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        # 顶层缺失、子节点提供 -> 构造函数读子节点
        storage = MemoryStorage(Path(tmp))
        storage.init_schema()
        sm = MemorySummarizer(
            storage=storage,
            embedding_client=FakeEmbeddingClient(),
            config={"summarizer": {"summary_turn_interval": 12}},
        )
        assert sm.turn_interval == 12, "构造函数未读取 summarizer 子节点"
        # 热更新接口对同一非法输入的处理与构造函数一致
        assert sm.set_turn_interval("not-a-number") is False
        assert sm.turn_interval == 12


# ---------------------------------------------------------------------------
# 二、插件层：apply_runtime_config() 必须回写 summarizer
# ---------------------------------------------------------------------------
def test_apply_runtime_config_writes_back_to_summarizer():
    """核心回归：热更新必须回写已构建的 summarizer（本次 bug 的修复点）。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        plugin = _make_plugin(
            {"summary_turn_interval": 5, "enable_memory": True}, Path(tmp)
        )
        plugin.summarizer = _new_summarizer(Path(tmp), interval=5)
        assert plugin.turn_interval == 5
        assert plugin.summarizer.turn_interval == 5

        result = plugin.apply_runtime_config({"summary_turn_interval": 20})

        assert "turn_interval" in result["applied"], result
        assert plugin.turn_interval == 20, "插件层阈值未更新"
        assert plugin.summarizer.turn_interval == 20, "summarizer 阈值未回写（bug 复现）"
        assert plugin.summarizer.config["summary_turn_interval"] == 20


def test_apply_runtime_config_invalid_value_keeps_old():
    """非法配置值不得把运行窗口打成 0/1，也不得谎报已生效。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        plugin = _make_plugin({"summary_turn_interval": 6}, Path(tmp))
        plugin.summarizer = _new_summarizer(Path(tmp), interval=6)

        result = plugin.apply_runtime_config({"summary_turn_interval": 0})

        assert plugin.turn_interval == 6, "非法值被接受（应保持原值）"
        assert plugin.summarizer.turn_interval == 6
        assert "turn_interval" not in result["applied"], "非法值不应报告为已热更"


def test_apply_runtime_config_missing_key_is_noop():
    """未提交该键时保持原值，且仍视为可用（不触发告警路径）。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        plugin = _make_plugin({"summary_turn_interval": 6}, Path(tmp))
        plugin.summarizer = _new_summarizer(Path(tmp), interval=6)

        result = plugin.apply_runtime_config({"enable_memory": True})

        assert plugin.turn_interval == 6
        assert plugin.summarizer.turn_interval == 6
        assert "turn_interval" not in result["applied"]


def test_apply_runtime_config_safe_when_summarizer_absent():
    """summarizer 尚未装配时热更新不抛异常（延迟到构建时生效）。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        plugin = _make_plugin({"summary_turn_interval": 5}, Path(tmp))
        plugin.summarizer = None

        result = plugin.apply_runtime_config({"summary_turn_interval": 30})

        assert plugin.turn_interval == 30
        assert "turn_interval" in result["applied"]
        # 未装配时不应凭空创建实例
        assert plugin.summarizer is None


def test_summarizer_skipped_list_matches_reload_semantics():
    """skipped 列表不得笼统声称 summarizer.* 全部跳过（阈值实际可热更）。"""
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        plugin = _make_plugin({"summary_turn_interval": 5}, Path(tmp))
        plugin.summarizer = _new_summarizer(Path(tmp), interval=5)

        result = plugin.apply_runtime_config(
            {
                "summary_turn_interval": 9,
                "summarizer": {"api_key": "x", "timeout_seconds": 30},
            }
        )

        applied = set(result["applied"])
        skipped = set(result["skipped"])
        assert "turn_interval" in applied
        assert "summarizer.*" not in skipped, "仍存在与实际行为矛盾的兜底表述"
        assert "summarizer.api_key" in skipped
        assert "summarizer.timeout_seconds" in skipped
        assert "summarizer.summary_turn_interval" not in skipped, (
            "可热更的阈值不应出现在 skipped 中"
        )


# ---------------------------------------------------------------------------
# 三、端到端行为：5 -> 20 热更后窗口真实变宽
# ---------------------------------------------------------------------------
async def test_window_5_to_20_no_flush_at_5_and_flush_at_20():
    """
    关键行为断言：阈值 5 热更为 20 后：
    - 第 5 轮**不得**触发打包（修复前会在此触发，因为 Worker 仍用旧阈值 5）；
    - 第 20 轮才触发打包，且批次恰好 20 轮。

    注意：这里刻意走**插件层** ``apply_runtime_config()`` 触发热更，
    以覆盖「面板保存 → 插件层 → summarizer」的完整链路；
    若只调用 ``sm.set_turn_interval()`` 则无法捕获插件层漏回写的 bug。
    """
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        plugin = _make_plugin({"summary_turn_interval": 5}, Path(tmp))
        sm = _new_summarizer(Path(tmp), interval=5)
        plugin.summarizer = sm
        sid = "session_hot_5_to_20"

        # 热更到 20（经插件层，模拟面板保存）
        result = plugin.apply_runtime_config({"summary_turn_interval": 20})
        assert "turn_interval" in result["applied"], result
        assert sm.turn_interval == 20, "插件层热更未回写 summarizer（bug 复现）"

        for i in range(1, 20):
            await sm.enqueue_dialogue(sid, f"用户第{i}轮", f"机器人第{i}轮")
            assert sm.pending_turns(sid) == i, f"第 {i} 轮后缓冲应为 {i}"
            if i == 5:
                # 旧阈值 5 在此处就会打包，修复后必须仍在缓冲池
                assert sm._queue.qsize() == 0, "第 5 轮误按旧阈值 5 触发了打包"
                assert sm.pending_turns(sid) == 5

        assert sm._queue.qsize() == 0, "第 19 轮不应触发打包"

        # 第 20 轮达到新阈值，整窗打包
        await sm.enqueue_dialogue(sid, "用户第20轮", "机器人第20轮")
        assert sm._queue.qsize() == 1, "达到新阈值 20 未触发打包"
        assert sm.pending_turns(sid) == 0, "打包后缓冲池应清空"

        batch = await asyncio.wait_for(sm._queue.get(), timeout=5)
        turns = batch.get("turns") or []
        assert len(turns) == 20, f"批次应为 20 轮，实际 {len(turns)}"


async def test_window_20_to_5_flushes_on_next_boundary():
    """
    反向热更 20 -> 5：存量缓冲不回滚，下一次判定按新阈值打包。
    已在缓冲的 7 轮在下一轮入池后（共 8 轮）立即按新阈值 5 打包。
    """
    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        sm = _new_summarizer(Path(tmp), interval=20)
        sid = "session_hot_20_to_5"

        for i in range(1, 8):
            await sm.enqueue_dialogue(sid, f"用户第{i}轮", f"机器人第{i}轮")
        assert sm.pending_turns(sid) == 7
        assert sm._queue.qsize() == 0

        # 热更到 5
        assert sm.set_turn_interval(5) is True

        # 下一轮入池后共 8 轮 >= 5，立即打包，且只取 5 轮，余 3 轮留在缓冲
        await sm.enqueue_dialogue(sid, "用户第8轮", "机器人第8轮")
        assert sm._queue.qsize() == 1, "按新阈值 5 未触发打包"
        batch = await asyncio.wait_for(sm._queue.get(), timeout=5)
        assert len(batch.get("turns") or []) == 5, "批次应恰好 5 轮"
        assert sm.pending_turns(sid) == 3, "余量应按新窗口留在缓冲池"


# ---------------------------------------------------------------------------
# 四、重载语义：RELOAD_REQUIRED_PREFIXES 与实际热更能力一致
# ---------------------------------------------------------------------------
def test_reload_required_semantics_match_hot_reload_capability():
    """阈值可热更 -> 不提示重载；凭证/模型/存储类 -> 提示重载。"""
    cases = {
        "summary_turn_interval": False,            # 顶层键：可热更
        "summarizer.summary_turn_interval": False,  # 子节点键：同样可热更
        "db_path": True,                            # 存储连接需重载
        "models.embedding.model": True,             # 客户端装配期构建
        "migration.force": True,
        "webui.register_api": True,
        "summarizer.api_key": True,
        "summarizer.model": True,
        "summarizer.timeout_seconds": True,
        "evaluator.api_base": True,
        "evaluator.model_name": True,
        "graph.enable": True,
        "evaluator.enable": False,                  # 纯阈值类可热更
        "retrieval.top_k": False,                   # 每请求现读
        "enable_memory": False,
    }
    mismatched = {
        key: config_api._is_reload_required(key)
        for key, expected in cases.items()
        if config_api._is_reload_required(key) is not expected
    }
    assert not mismatched, f"重载判定与实际热更行为不一致: {mismatched}"


def test_no_bare_prefix_false_positive():
    """裸前缀不应误伤无关键（修复前 ``db_path`` 会命中 ``db_path_extra``）。"""
    assert config_api._is_reload_required("db_path") is True
    assert config_api._is_reload_required("db_path_extra") is False
    assert config_api._is_reload_required("graph.enable") is True
    assert config_api._is_reload_required("graph") is False


def test_config_save_reports_turn_interval_without_reload(tmp_path):
    """面板保存 summary_turn_interval：落盘 + 热更，且不提示重载。"""
    cfg_path = tmp_path / "plugin_config.json"
    cfg_path.write_text(
        json.dumps(
            {"summary_turn_interval": 5, "enable_memory": True}, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory(prefix="agent_memory_s12_") as tmp:
        plugin = _make_plugin({"summary_turn_interval": 5}, Path(tmp))
        plugin.summarizer = _new_summarizer(Path(tmp), interval=5)

        manager = config_api.ConfigManager(
            plugin,
            schema_path=str(PLUGIN_ROOT / "_conf_schema.json"),
            config_path=str(cfg_path),
        )
        result = manager.save({"summary_turn_interval": 20})

    assert result["requires_reload"] is False, "阈值热更不应要求重载"
    assert "summary_turn_interval" in result["changed"]
    assert plugin.summarizer.turn_interval == 20, "保存后未回写 summarizer"


# ---------------------------------------------------------------------------
# 脚本式运行入口（与仓库其它 stage 测试保持一致的风格）
# ---------------------------------------------------------------------------
def _run_all() -> None:
    test_set_turn_interval_accepts_valid_value()
    for bad in (None, "", "abc", 0, -3):
        test_set_turn_interval_rejects_invalid_value(bad)
    test_set_turn_interval_reuses_parser_semantics()
    test_apply_runtime_config_writes_back_to_summarizer()
    test_apply_runtime_config_invalid_value_keeps_old()
    test_apply_runtime_config_missing_key_is_noop()
    test_apply_runtime_config_safe_when_summarizer_absent()
    test_summarizer_skipped_list_matches_reload_semantics()
    test_reload_required_semantics_match_hot_reload_capability()
    test_no_bare_prefix_false_positive()
    asyncio.run(test_window_5_to_20_no_flush_at_5_and_flush_at_20())
    asyncio.run(test_window_20_to_5_flushes_on_next_boundary())
    print("[OK] Stage12 窗口阈值热更新全部断言通过")


if __name__ == "__main__":
    _run_all()
