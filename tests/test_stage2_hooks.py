# -*- coding: utf-8 -*-
"""
Stage 2 自测：验证 main.py 双向钩子的装配正确性与端到端行为。

覆盖点：
1. 钩子装配：on_llm_request / on_llm_response 已由装饰器注册到插件实例；
2. 前置注入：模拟用户发起请求 → 检索被调用（query/session 正确）→ 记忆块被写入
   req.system_prompt 前部，且原始人格提示词完整保留；
3. 注入幂等：同一请求重复触发（模拟工具调用多轮）不会重复注入；
4. 后置记录：模拟 LLM 回复 → 对话被送入 summarizer 多轮缓冲池；
5. 过滤规则：流式分片 is_chunk、工具调用轮 tools_call_name 均不记录；
6. 短窗口去重：相同内容连续上报只记一次；
7. 容错降级：检索异常 / 检索超时 / retriever 缺失 / req 缺失，均静默返回 False
   且不修改 system_prompt，绝不抛出异常；
8. format_memory_block：prompt_context 优先、本地兜底组装、长度截断、空结果返回空串。

运行： python3 tests/test_stage2_hooks.py
"""

from __future__ import annotations

import asyncio
import atexit
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) 构造最小 AstrBot 桩包（必须在 import main 之前注入 sys.path）
#    桩包严格对齐真实 AstrBot 的装饰器形态：
#    on_llm_request() / on_llm_response() / on_decorating_result() 均为「无参工厂装饰器」，
#    并在被装饰函数上打 __astrbot_hook__ 标记以便断言。
# ---------------------------------------------------------------------------
STUB_ROOT = Path(tempfile.mkdtemp(prefix="astrbot_stub_stage2_"))
atexit.register(lambda: shutil.rmtree(STUB_ROOT, ignore_errors=True))

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
        """AstrBot 事件装饰器桩（形态对齐 core/star/register/star_handler.py）。"""


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

for rel_path, content in _STUB_FILES.items():
    target = STUB_ROOT / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

if str(STUB_ROOT) not in sys.path:
    sys.path.insert(0, str(STUB_ROOT))

# 插件根目录（支持直接以脚本方式运行）
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

# 清掉可能已被别处导入的 astrbot / main，确保使用本测试的桩包
for _mod in ("astrbot", "main"):
    sys.modules.pop(_mod, None)

import main as plugin_main  # noqa: E402
from main import (  # noqa: E402
    MEMORY_BLOCK_MARKER,
    AgentMemoryPlugin,
)

plugin_main.PLUGIN_NAME = "astrbot_plugin_agent_memory_test"


# ---------------------------------------------------------------------------
# 2) 测试替身
# ---------------------------------------------------------------------------
class FakeEvent:
    """模拟 AstrMessageEvent 的最小字段集。"""

    def __init__(self, session_id: str = "sess-1", text: str = "主人喜欢喝什么茶？"):
        self.session_id = session_id
        self.unified_msg_origin = f"aiocqhttp:FriendMessage:{session_id}"
        self.message_str = text


class FakeRequest:
    """模拟 ProviderRequest 的最小字段集（system_prompt 可写）。"""

    def __init__(
        self,
        prompt: str = "主人喜欢喝什么茶？",
        session_id: str = "sess-1",
        system_prompt: str = "你是薇薇安，主人的专属助理。",
    ):
        self.prompt = prompt
        self.session_id = session_id
        self.system_prompt = system_prompt
        self.contexts: list = []
        self.image_urls: list = []


class FakeResponse:
    """模拟 LLMResponse 的最小字段集。"""

    def __init__(self, text: str, is_chunk: bool = False, tools_call_name=None):
        # completion_text 在真实宿主中是 property，此处用普通属性即可等价验证
        self.completion_text = text
        self.is_chunk = is_chunk
        self.tools_call_name = list(tools_call_name or [])


class FakeRetriever:
    """可控检索器桩：可返回固定结果 / 抛异常 / 延迟以触发超时。"""

    def __init__(self, result=None, exc: Exception | None = None, delay: float = 0.0):
        self.result = result if result is not None else {}
        self.exc = exc
        self.delay = delay
        self.calls: list = []

    async def retrieve(self, query, session_id=None, top_k=5, candidate_k=20):
        self.calls.append(
            {
                "query": query,
                "session_id": session_id,
                "top_k": top_k,
                "candidate_k": candidate_k,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return self.result


SAMPLE_RESULT = {
    "pinned_l3": [
        {"content": "主人偏好龙井，不加糖。", "atom_type": "preference", "importance": 0.95},
    ],
    "retrieved_l2": [
        {"content": "上周主人提到想尝试明前龙井。", "atom_type": "episode", "importance": 0.6},
    ],
    "prompt_context": (
        "## 🧠 长期记忆上下文 (Agent Memory Context)\n"
        "### 📌 核心记忆与约束 (L3 Pinned)\n"
        "1. **[核心/preference]** 主人偏好龙井，不加糖。 *(重要度: 0.95)*\n"
        "### 🔍 相关情景记忆 (L2 Retrieved)\n"
        "1. 上周主人提到想尝试明前龙井。"
    ),
}


def make_plugin(tmp_dir: Path, retriever=None, **cfg_overrides) -> AgentMemoryPlugin:
    """构造插件实例：数据库落在临时目录，检索器替换为可控桩。"""
    config = {
        "db_path": str(tmp_dir / "agent_memory.db"),
        "enable_memory": True,
        "summary_turn_interval": 5,
        "memory_injection_enable": True,
    }
    config.update(cfg_overrides)

    plugin = AgentMemoryPlugin(context=None, config=config)
    assert plugin.storage is not None, "存储装配失败，测试前置条件不成立"
    assert plugin.summarizer is not None, "summarizer 装配失败"

    plugin.retriever = retriever if retriever is not None else FakeRetriever(SAMPLE_RESULT)
    return plugin


# ---------------------------------------------------------------------------
# 3) 用例
# ---------------------------------------------------------------------------
async def test_hooks_assembled() -> None:
    """钩子必须由真实装饰器注册，且能被宿主按约定调用。"""
    with tempfile.TemporaryDirectory(prefix="stage2_assembly_") as tmp:
        plugin = make_plugin(Path(tmp))
        try:
            assert hasattr(plugin.on_llm_request, "__astrbot_hook__"), "前置钩子未装配"
            assert getattr(plugin.on_llm_request, "__astrbot_hook__") == "on_llm_request"
            assert hasattr(plugin.on_llm_response, "__astrbot_hook__"), "后置钩子未装配"
            assert getattr(plugin.on_llm_response, "__astrbot_hook__") == "on_llm_response"

            status = plugin.get_status()
            assert status["hooks"]["on_llm_request"] is True
            assert status["hooks"]["on_llm_response"] is True
            assert status["hooks"]["injection_enabled"] is True
            print("[OK] 双向钩子已由装饰器注册, status.hooks =", status["hooks"])
        finally:
            await plugin.terminate()


async def test_prompt_injection() -> None:
    """前置钩子：检索命中 → 记忆块前置注入 System Prompt，人格提示词保留。"""
    with tempfile.TemporaryDirectory(prefix="stage2_inject_") as tmp:
        retriever = FakeRetriever(SAMPLE_RESULT)
        plugin = make_plugin(Path(tmp), retriever=retriever)
        try:
            event = FakeEvent(session_id="sess-inject", text="主人喜欢喝什么茶？")
            persona = "你是薇薇安，主人的专属助理。"
            # 忠实复刻 AstrBot 行为：req.session_id 被归一化为 event.unified_msg_origin
            req = FakeRequest(
                prompt="主人喜欢喝什么茶？",
                session_id=event.unified_msg_origin,
                system_prompt=persona,
            )

            # 走真实装饰后的钩子入口，验证整条链路
            await plugin.on_llm_request(event, req)

            assert MEMORY_BLOCK_MARKER in req.system_prompt, "记忆块未注入 system_prompt"
            assert "主人偏好龙井" in req.system_prompt, "L3 置顶记忆未注入"
            assert "明前龙井" in req.system_prompt, "L2 情景记忆未注入"
            assert persona in req.system_prompt, "原始人格提示词被破坏"

            # 注入必须在 System Prompt 前部
            assert req.system_prompt.index(MEMORY_BLOCK_MARKER) < req.system_prompt.index(
                persona
            ), "记忆块未置于 System Prompt 前部"

            # 检索入参必须与当前 Query / Session 一致
            assert len(retriever.calls) == 1, "检索应被调用一次"
            call = retriever.calls[0]
            assert call["query"] == "主人喜欢喝什么茶？", f"检索 query 错误: {call['query']}"
            assert (
                call["session_id"] == event.unified_msg_origin
            ), f"检索 session 错误: {call['session_id']}"

            print(
                "[OK] 前置注入成功 | 记忆块长度=%d | stats=%s"
                % (len(req.system_prompt), plugin.get_status()["hook_stats"])
            )

            # 幂等：同一请求重复触发（工具调用多轮）不再重复注入
            before = req.system_prompt
            await plugin.on_llm_request(event, req)
            assert req.system_prompt == before, "重复触发导致记忆块重复注入"
            assert req.system_prompt.count(MEMORY_BLOCK_MARKER) == 1, "记忆块出现多份"
            print("[OK] 注入幂等生效：同一请求仅注入一次")
        finally:
            await plugin.terminate()


async def test_turn_recording() -> None:
    """后置钩子：用户输入 + bot 回复进入多轮缓冲池，且过滤分片/工具轮/重复回调。"""
    with tempfile.TemporaryDirectory(prefix="stage2_record_") as tmp:
        plugin = make_plugin(Path(tmp))
        try:
            # 正常一轮
            event = FakeEvent(session_id="sess-rec", text="主人喜欢龙井。")
            sid = event.unified_msg_origin  # 与前置注入同源：umo 作为会话键
            resp = FakeResponse("记住了，主人喜欢龙井，不加糖。")
            await plugin.on_llm_response(event, resp)
            assert plugin.summarizer.pending_turns(sid) == 1, "正常轮次未进入缓冲池"

            # 缓冲池内实际内容校验
            buffered = plugin.summarizer._turn_buffers.get(sid, [])
            assert len(buffered) == 1
            turn = buffered[0]
            joined = str(turn)
            assert "龙井" in joined, f"缓冲轮次内容异常: {turn}"
            print(f"[OK] 后置记录成功, pending_turns={plugin.summarizer.pending_turns(sid)}, turn={turn}")

            # 流式分片：不记录
            chunk_event = FakeEvent(session_id="sess-chunk", text="分片测试")
            await plugin.on_llm_response(
                chunk_event, FakeResponse("增量片段", is_chunk=True)
            )
            assert (
                plugin.summarizer.pending_turns(chunk_event.unified_msg_origin) == 0
            ), "分片响应不应记录"

            # 工具调用轮：不记录
            tool_event = FakeEvent(session_id="sess-tool", text="工具测试")
            await plugin.on_llm_response(
                tool_event, FakeResponse("", tools_call_name=["web_search"])
            )
            assert (
                plugin.summarizer.pending_turns(tool_event.unified_msg_origin) == 0
            ), "工具调用轮不应记录"

            # 短窗口重复回调：只记一次
            await plugin.on_llm_response(event, resp)
            assert plugin.summarizer.pending_turns(sid) == 1, "重复回调导致重复记录"

            print(
                "[OK] 过滤规则生效: 分片/工具轮被跳过, 重复回调被去重 | stats=%s"
                % plugin.get_status()["hook_stats"]
            )
        finally:
            await plugin.terminate()


async def test_graceful_degradation() -> None:
    """容错：检索异常/超时/缺失/入参缺失，全部静默降级且不污染 system_prompt。"""
    with tempfile.TemporaryDirectory(prefix="stage2_degrade_") as tmp:
        event = FakeEvent(session_id="sess-degrade", text="主人喜欢什么？")
        persona = "原始人格提示词"

        # --- 1. 检索抛异常 ---
        plugin = make_plugin(Path(tmp), retriever=FakeRetriever(exc=RuntimeError("boom")))
        try:
            req = FakeRequest(session_id="sess-degrade", system_prompt=persona)
            ok = await plugin.inject_memory_into_request(event, req)
            assert ok is False, "检索异常时应返回 False"
            assert req.system_prompt == persona, "检索异常时不应修改 system_prompt"
            print("[OK] 检索异常 → 静默降级, system_prompt 未变")
        finally:
            await plugin.terminate()

        # --- 2. 检索超时 ---
        plugin = make_plugin(
            Path(tmp), retriever=FakeRetriever(result=SAMPLE_RESULT, delay=1.0)
        )
        plugin.config["retrieval"] = {"retrieve_timeout": 0.1}
        try:
            req = FakeRequest(session_id="sess-degrade", system_prompt=persona)
            ok = await plugin.inject_memory_into_request(event, req)
            assert ok is False, "检索超时应返回 False"
            assert req.system_prompt == persona, "超时不应修改 system_prompt"
            print("[OK] 检索超时(0.1s) → 静默降级")
        finally:
            await plugin.terminate()

        # --- 3. retriever 缺失 / req 缺失 / 空 query ---
        plugin = make_plugin(Path(tmp))
        plugin.retriever = None
        try:
            req = FakeRequest(session_id="sess-degrade", system_prompt=persona)
            assert await plugin.inject_memory_into_request(event, req) is False
            assert await plugin.inject_memory_into_request(event, None) is False
            assert (
                await plugin.inject_memory_into_request(FakeEvent(text=""), FakeRequest(prompt=""))
                is False
            )
            print("[OK] retriever 缺失 / req 缺失 / 空 query → 均安全返回 False")
        finally:
            await plugin.terminate()

        # --- 4. 注入开关关闭 ---
        plugin = make_plugin(Path(tmp), memory_injection_enable=False)
        try:
            req = FakeRequest(session_id="sess-degrade", system_prompt=persona)
            assert await plugin.inject_memory_into_request(event, req) is False
            assert req.system_prompt == persona
            print("[OK] memory_injection_enable=False → 跳过注入")
        finally:
            await plugin.terminate()


async def test_format_memory_block() -> None:
    """格式化：prompt_context 优先、本地兜底、长度截断、空结果返回空串。"""
    # 1. prompt_context 优先
    block = AgentMemoryPlugin.format_memory_block(SAMPLE_RESULT, max_chars=0)
    assert block.startswith(MEMORY_BLOCK_MARKER), "记忆块缺少防重标记"
    assert "主人偏好龙井" in block

    # 2. 无 prompt_context 时本地兜底组装
    raw_only = {
        "pinned_l3": [{"content": "置顶规则A", "atom_type": "rule", "importance": 0.9}],
        "retrieved_l2": [{"content": "情景记忆B"}],
    }
    block2 = AgentMemoryPlugin.format_memory_block(raw_only, max_chars=0)
    assert MEMORY_BLOCK_MARKER in block2
    assert "置顶规则A" in block2 and "情景记忆B" in block2, "本地兜底组装失败"

    # 3. 截断
    long_block = AgentMemoryPlugin.format_memory_block(
        {"prompt_context": "记" * 500}, max_chars=100
    )
    assert "已截断" in long_block, "超长记忆块未被截断"
    assert len(long_block) < 500

    # 4. 空结果
    assert AgentMemoryPlugin.format_memory_block({}) == ""
    assert AgentMemoryPlugin.format_memory_block(None) == ""
    assert AgentMemoryPlugin.format_memory_block({"pinned_l3": [], "retrieved_l2": []}) == ""

    print("[OK] format_memory_block: 优先/兜底/截断/空值 全部符合预期")


def main() -> int:
    asyncio.run(test_hooks_assembled())
    asyncio.run(test_prompt_injection())
    asyncio.run(test_turn_recording())
    asyncio.run(test_graceful_degradation())
    asyncio.run(test_format_memory_block())
    print("\n=== Stage 2 双向钩子 用例全部通过 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
