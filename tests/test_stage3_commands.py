# -*- coding: utf-8 -*-
"""
Stage 3 自测：验证动态注入位置路由与四个标准管理指令。

覆盖点：
1. 注入位置路由（三种主流大模型最佳实践）：
   - system_head：记忆块写入 System Prompt 最前端，人设完整保留，prompt 不被改写；
   - system_tail：记忆块写入 System Prompt 最末端（人设之后），prompt 不被改写；
   - user_head  ：记忆块写入用户 Query(req.prompt) 最前端，System Prompt 不被改写；
   - 旧版别名兼容：head → system_head、tail → system_tail、非法值 → system_head；
   - 幂等保护：同一请求重复注入只生效一次（防工具调用多轮重复）。
2. 管理指令处理器（cmd_status / cmd_recall / cmd_pin / cmd_migrate）：
   - 返回文本格式正确、关键字段齐全；
   - 全量无未捕获异常（含组件缺失、参数为空、源库缺失等边界）。
3. 指令组装配：/memory status|recall|pin|migrate 已按标准 @command 规范注册。

运行： python3 tests/test_stage3_commands.py
"""

from __future__ import annotations

import asyncio
import atexit
import shutil
import sqlite3
import sys
import tempfile
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) 构造最小 AstrBot 桩包（必须在 import main 之前注入 sys.path）
#    桩包严格对齐真实 AstrBot 的装饰器形态：
#    - on_llm_request / on_llm_response / on_decorating_result：无参工厂装饰器；
#    - command_group(name)：返回指令组对象，其 .command(sub) 注册子指令；
#    - command(name)：扁平指令装饰器（降级路径）。
# ---------------------------------------------------------------------------
STUB_ROOT = Path(tempfile.mkdtemp(prefix="astrbot_stub_stage3_"))
atexit.register(lambda: shutil.rmtree(STUB_ROOT, ignore_errors=True))

_EVENT_STUB = '''
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


class CommandGroupStub:
    """指令组桩：模拟 AstrBot CommandGroup 的 .command()/.group() 链式注册。"""

    def __init__(self, name):
        self.name = name
        self.children = []

    def command(self, sub_name, *args, **kwargs):
        def decorator(func):
            setattr(func, "__astrbot_hook__", "command:%s.%s" % (self.name, sub_name))
            self.children.append(sub_name)
            return func

        return decorator

    def group(self, sub_name, *args, **kwargs):
        child = CommandGroupStub("%s.%s" % (self.name, sub_name))
        self.children.append(sub_name)
        return child


def command_group(name, *args, **kwargs):
    def decorator(func):
        return CommandGroupStub(name)

    return decorator


def command(name, *args, **kwargs):
    def decorator(func):
        setattr(func, "__astrbot_hook__", "command:%s" % name)
        return func

    return decorator


class _FilterModule:
    on_llm_request = staticmethod(on_llm_request)
    on_llm_response = staticmethod(on_llm_response)
    on_decorating_result = staticmethod(on_decorating_result)
    command_group = staticmethod(command_group)
    command = staticmethod(command)


filter = _FilterModule()
'''

_STUB_FILES = {
    "astrbot/__init__.py": "",
    "astrbot/api/__init__.py": textwrap.dedent(
        """
        import logging

        logger = logging.getLogger("astrbot")
        """
    ),
    "astrbot/api/event.py": textwrap.dedent(_EVENT_STUB),
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

# 清掉可能已被别处导入的 astrbot / main / storage / migrator
for _mod in ("astrbot", "main", "storage", "migrator"):
    sys.modules.pop(_mod, None)

import main as plugin_main  # noqa: E402
from main import (  # noqa: E402
    MEMORY_BLOCK_MARKER,
    POSITION_SYSTEM_HEAD,
    POSITION_SYSTEM_TAIL,
    POSITION_USER_HEAD,
    AgentMemoryPlugin,
    _normalize_inject_position,
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
        self.message_text = text

    def plain_result(self, text: str) -> str:
        """模拟宿主消息结果包装（此处直接回传文本便于断言）。"""
        return text


class FakeRequest:
    """模拟 ProviderRequest 的最小字段集（system_prompt / prompt 均可写）。"""

    def __init__(
        self,
        prompt: str = "主人喜欢喝什么茶？",
        session_id: str = "sess-1",
        system_prompt: str = "你是薇薇安，主人的专属助理。",
    ):
        self.prompt = prompt
        self.session_id = session_id
        self.system_prompt = system_prompt


class FakeRetriever:
    """可控检索器桩：记录调用入参并返回固定结果。"""

    def __init__(self, result=None):
        self.result = result if result is not None else {}
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
        return self.result


SAMPLE_RESULT = {
    "pinned_l3": [
        {
            "atom_id": "atom_pin_1",
            "content": "主人偏好龙井，不加糖。",
            "atom_type": "preference",
            "importance": 0.95,
        }
    ],
    "retrieved_l2": [
        {
            "atom_id": "atom_l2_1",
            "content": "上周主人提到想尝试明前龙井。",
            "atom_type": "episode",
            "importance": 0.6,
            "relevance_score": 0.9182736,
        },
        {
            "atom_id": "atom_l2_2",
            "content": "主人常去南山路那家茶馆。",
            "atom_type": "episode",
            "importance": 0.5,
            "relevance_score": 0.4321000,
        },
    ],
    "prompt_context": (
        "## 🧠 长期记忆上下文 (Agent Memory Context)\n"
        "### 📌 核心记忆与约束 (L3 Pinned)\n"
        "1. **[核心/preference]** 主人偏好龙井，不加糖。 *(重要度: 0.95)*\n"
        "### 🔍 相关情景记忆 (L2 Retrieved)\n"
        "- 上周主人提到想尝试明前龙井。 *(相关度: 0.9183)*"
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


def _insert_atom(plugin: AgentMemoryPlugin, atom_id: str, content: str) -> None:
    """向 l2_atoms 插入一条最小可用记忆原子（模拟既有记忆数据）。"""
    now = time.time()
    conn = plugin.storage.connect()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO l2_atoms (
                atom_id, scope_key, session_id, content,
                topics_json, key_facts_json, sentiment, importance,
                access_count, embedding_blob, source_type, source_ref,
                last_accessed_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
            """,
            (
                atom_id,
                "sess-pin",
                "sess-pin",
                content,
                "[]",
                "[]",
                "neutral",
                0.7,
                0,
                "chat",
                "test",
                now,
                now,
                now,
            ),
        )


def _create_legacy_db(path: Path) -> None:
    """构造一个最小可用的旧版 livingmemory.db，用于迁移指令端到端验证。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY, doc_id TEXT, text TEXT, user_id TEXT,
                    created_at REAL, updated_at REAL, kb_doc_id TEXT, metadata TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE memory_atoms (
                    id INTEGER PRIMARY KEY, session_id TEXT, persona_id TEXT,
                    content TEXT, atom_type TEXT, importance REAL, confidence REAL,
                    decay_type TEXT, ttl_days REAL, reinforcement_count INTEGER,
                    entities TEXT, metadata TEXT, parent_memory_id INTEGER,
                    expires_at REAL, status TEXT, last_reinforced_at REAL,
                    event_time REAL, created_at REAL, last_accessed_at REAL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE graph_nodes (
                    id INTEGER PRIMARY KEY, node_type TEXT, node_value TEXT,
                    canonical_value TEXT, node_key TEXT, metadata TEXT,
                    created_at REAL, updated_at REAL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE graph_edges (
                    id INTEGER PRIMARY KEY, source_node_id INTEGER,
                    target_node_id INTEGER, relation_type TEXT, metadata TEXT,
                    edge_key TEXT, source_memory_id INTEGER, weight REAL,
                    status TEXT, confidence REAL, created_at REAL, updated_at REAL
                );
                """
            )

            now = time.time()
            conn.execute(
                "INSERT INTO documents (id, doc_id, text, user_id, created_at, updated_at, kb_doc_id, metadata) "
                "VALUES (1, 'doc_1', '旧版文档内容', 'u1', ?, ?, NULL, '{}')",
                (now, now),
            )
            conn.execute(
                "INSERT INTO memory_atoms (id, session_id, persona_id, content, atom_type, importance, "
                "confidence, decay_type, ttl_days, reinforcement_count, entities, metadata, parent_memory_id, "
                "expires_at, status, last_reinforced_at, event_time, created_at, last_accessed_at) "
                "VALUES (1, 'sess-legacy', '', '主人偏好龙井', 'preference', 0.9, 0.8, 'exponential', 30.0, "
                "0, '[]', '{}', 0, NULL, 'active', ?, NULL, ?, ?)",
                (now, now, now),
            )
            conn.execute(
                "INSERT INTO memory_atoms (id, session_id, persona_id, content, atom_type, importance, "
                "confidence, decay_type, ttl_days, reinforcement_count, entities, metadata, parent_memory_id, "
                "expires_at, status, last_reinforced_at, event_time, created_at, last_accessed_at) "
                "VALUES (2, 'sess-legacy', '', '主人常去南山路茶馆', 'episode', 0.4, 0.6, 'exponential', 30.0, "
                "0, '[]', '{}', 0, NULL, 'active', ?, NULL, ?, ?)",
                (now, now, now),
            )
            conn.execute(
                "INSERT INTO graph_nodes (id, node_type, node_value, canonical_value, node_key, metadata, "
                "created_at, updated_at) VALUES (1, 'concept', '龙井', '龙井', 'node_key_1', '{}', ?, ?)",
                (now, now),
            )
            conn.execute(
                "INSERT INTO graph_nodes (id, node_type, node_value, canonical_value, node_key, metadata, "
                "created_at, updated_at) VALUES (2, 'concept', '茶馆', '茶馆', 'node_key_2', '{}', ?, ?)",
                (now, now),
            )
            conn.execute(
                "INSERT INTO graph_edges (id, source_node_id, target_node_id, relation_type, metadata, edge_key, "
                "source_memory_id, weight, status, confidence, created_at, updated_at) "
                "VALUES (1, 1, 2, 'related_to', '{}', 'ek1', 1, 1.0, 'active', 0.8, ?, ?)",
                (now, now),
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3) 用例：注入位置路由
# ---------------------------------------------------------------------------
def test_position_normalization() -> None:
    """位置归一化：三态标准值 + 旧版别名 + 非法值兜底。"""
    assert _normalize_inject_position("system_head") == POSITION_SYSTEM_HEAD
    assert _normalize_inject_position("system_tail") == POSITION_SYSTEM_TAIL
    assert _normalize_inject_position("user_head") == POSITION_USER_HEAD

    # 旧版二元兼容
    assert _normalize_inject_position("head") == POSITION_SYSTEM_HEAD
    assert _normalize_inject_position("tail") == POSITION_SYSTEM_TAIL

    # 大小写 / 分隔符 / 非法值
    assert _normalize_inject_position("SYSTEM-HEAD") == POSITION_SYSTEM_HEAD
    assert _normalize_inject_position("user head") == POSITION_USER_HEAD
    assert _normalize_inject_position("bogus") == POSITION_SYSTEM_HEAD
    assert _normalize_inject_position(None) == POSITION_SYSTEM_HEAD
    assert _normalize_inject_position("") == POSITION_SYSTEM_HEAD
    print("[OK] 注入位置归一化：三态/别名/非法值 全部符合预期")


async def test_inject_position_routing() -> None:
    """三种注入位置必须改写正确字段，且保持幂等。"""
    persona = "你是薇薇安，主人的专属助理。"
    query = "主人喜欢喝什么茶？"

    # ---------------- system_head ----------------
    with tempfile.TemporaryDirectory(prefix="stage3_pos_head_") as tmp:
        plugin = make_plugin(
            Path(tmp), retrieval={"inject_position": "system_head"}
        )
        try:
            req = FakeRequest(prompt=query, system_prompt=persona)
            ok = await plugin.inject_memory_into_request(FakeEvent(text=query), req)
            assert ok is True, "system_head 注入应成功"
            assert MEMORY_BLOCK_MARKER in req.system_prompt, "记忆块未写入 system_prompt"
            assert persona in req.system_prompt, "人设提示词被破坏"
            assert req.system_prompt.index(MEMORY_BLOCK_MARKER) < req.system_prompt.index(
                persona
            ), "system_head 要求记忆块置于 System Prompt 最前端"
            assert req.prompt == query, "system_head 不应改写用户 Query"
            print("[OK] system_head：记忆块前插 System Prompt，Query 未改写")

            # 幂等：重复触发不重复注入
            before = req.system_prompt
            await plugin.inject_memory_into_request(FakeEvent(text=query), req)
            assert req.system_prompt == before, "重复触发导致重复注入"
            assert req.system_prompt.count(MEMORY_BLOCK_MARKER) == 1, "记忆块出现多份"
            print("[OK] system_head：注入幂等生效")
        finally:
            await plugin.terminate()

    # ---------------- system_tail ----------------
    with tempfile.TemporaryDirectory(prefix="stage3_pos_tail_") as tmp:
        plugin = make_plugin(
            Path(tmp), retrieval={"inject_position": "system_tail"}
        )
        try:
            req = FakeRequest(prompt=query, system_prompt=persona)
            ok = await plugin.inject_memory_into_request(FakeEvent(text=query), req)
            assert ok is True, "system_tail 注入应成功"
            assert MEMORY_BLOCK_MARKER in req.system_prompt
            assert req.system_prompt.startswith(persona), "system_tail 要求人设保持在最前"
            assert req.system_prompt.index(MEMORY_BLOCK_MARKER) > req.system_prompt.index(
                persona
            ), "system_tail 要求记忆块置于人设之后"
            tail_segment = req.system_prompt.split(persona, 1)[-1]
            assert MEMORY_BLOCK_MARKER in tail_segment, (
                "system_tail 记忆块应位于 System Prompt 末端"
            )
            assert req.prompt == query, "system_tail 不应改写用户 Query"
            print("[OK] system_tail：记忆块后插 System Prompt，Query 未改写")
        finally:
            await plugin.terminate()

    # ---------------- user_head ----------------
    with tempfile.TemporaryDirectory(prefix="stage3_pos_user_") as tmp:
        plugin = make_plugin(
            Path(tmp), retrieval={"inject_position": "user_head"}
        )
        try:
            req = FakeRequest(prompt=query, system_prompt=persona)
            ok = await plugin.inject_memory_into_request(FakeEvent(text=query), req)
            assert ok is True, "user_head 注入应成功"
            assert MEMORY_BLOCK_MARKER in req.prompt, "记忆块未写入用户 Query"
            assert req.prompt.startswith(MEMORY_BLOCK_MARKER), "user_head 要求记忆块前插 Query"
            assert query in req.prompt, "原始用户提问被破坏"
            assert req.system_prompt == persona, "user_head 不应改写 System Prompt"
            print("[OK] user_head：记忆块前插用户 Query，System Prompt 未改写")

            # 幂等
            before_prompt = req.prompt
            await plugin.inject_memory_into_request(FakeEvent(text=query), req)
            assert req.prompt == before_prompt, "重复触发导致 Query 重复注入"
            assert req.prompt.count(MEMORY_BLOCK_MARKER) == 1
            print("[OK] user_head：注入幂等生效")
        finally:
            await plugin.terminate()

    # ---------------- 旧版 head/tail 别名路由 ----------------
    with tempfile.TemporaryDirectory(prefix="stage3_pos_alias_") as tmp:
        plugin = make_plugin(Path(tmp), retrieval={"inject_position": "head"})
        try:
            assert plugin.get_status()["injection"]["position"] == POSITION_SYSTEM_HEAD
            req = FakeRequest(prompt=query, system_prompt=persona)
            await plugin.inject_memory_into_request(FakeEvent(text=query), req)
            assert MEMORY_BLOCK_MARKER in req.system_prompt, "旧版 head 别名未路由到 system_head"
            print("[OK] 旧版别名 head → system_head 路由正确")
        finally:
            await plugin.terminate()


# ---------------------------------------------------------------------------
# 4) 用例：管理指令
# ---------------------------------------------------------------------------
async def test_cmd_status() -> None:
    """cmd_status 返回体量/策略/模型/钩子四段信息，且无异常。"""
    with tempfile.TemporaryDirectory(prefix="stage3_status_") as tmp:
        plugin = make_plugin(
            Path(tmp),
            retrieval={"inject_position": "user_head"},
            summary_turn_interval=3,
        )
        try:
            _insert_atom(plugin, "atom_status_1", "主人偏好龙井。")

            text = await plugin.cmd_status(None)
            assert isinstance(text, str) and text.strip(), "status 返回为空"
            assert "分层记忆状态" in text
            assert "L1 历史" in text and "L1 待总结" in text
            assert "L2 原子" in text and "L3 置顶" in text and "关系" in text
            assert "注入位置" in text and "user_head" in text
            assert "总结阈值" in text and "3" in text
            assert "Embedding" in text and "Rerank" in text
            assert "钩子" in text and "注入" in text and "记录" in text
            print("[OK] cmd_status 文本格式正确：\n" + text)
        finally:
            await plugin.terminate()


async def test_cmd_status_without_components() -> None:
    """cmd_status 在组件缺失时仍返回可读文本，不抛异常。"""
    with tempfile.TemporaryDirectory(prefix="stage3_status_bare_") as tmp:
        plugin = make_plugin(Path(tmp))
        plugin.storage = None
        plugin.summarizer = None
        plugin.embedding_client = None
        plugin.rerank_client = None
        plugin.retriever = None
        try:
            text = await plugin.cmd_status(None)
            assert isinstance(text, str) and "分层记忆状态" in text
            assert "未装配" in text
            print("[OK] cmd_status 组件缺失时安全降级")
        finally:
            # 组件已被置空，terminate 内部对各组件均有 None 保护
            await plugin.terminate()


async def test_cmd_recall() -> None:
    """cmd_recall 打印 L3 置顶项与 L2 精排分数。"""
    with tempfile.TemporaryDirectory(prefix="stage3_recall_") as tmp:
        retriever = FakeRetriever(SAMPLE_RESULT)
        plugin = make_plugin(Path(tmp), retriever=retriever)
        try:
            text = await plugin.cmd_recall(None, "主人喜欢的茶")
            assert isinstance(text, str) and text.strip()
            assert "记忆检索测试" in text
            assert "主人喜欢的茶" in text
            assert "L3 置顶" in text
            assert "主人偏好龙井" in text
            assert "atom_pin_1" in text
            assert "L2 精排" in text
            assert "score=" in text, "L2 精排必须展示 rerank 分数"
            assert "0.9183" in text, "精排分数未正确渲染"
            assert "明前龙井" in text

            assert len(retriever.calls) == 1, "检索应被调用一次"
            assert retriever.calls[0]["query"] == "主人喜欢的茶"
            print("[OK] cmd_recall 输出正确：\n" + text)

            # 空参数 / 检索器缺失 均安全
            empty_text = await plugin.cmd_recall(None, "")
            assert "请提供检索关键词" in empty_text

            plugin.retriever = None
            no_retriever_text = await plugin.cmd_recall(None, "任意")
            assert "检索器未装配" in no_retriever_text
            print("[OK] cmd_recall 空参数/检索器缺失均安全降级")
        finally:
            await plugin.terminate()


async def test_cmd_recall_from_event() -> None:
    """cmd_recall 未显式传 query 时，应从消息文本中解析参数。"""
    with tempfile.TemporaryDirectory(prefix="stage3_recall_event_") as tmp:
        retriever = FakeRetriever(SAMPLE_RESULT)
        plugin = make_plugin(Path(tmp), retriever=retriever)
        try:
            event = FakeEvent(text="/memory recall 主人爱喝的茶")
            text = await plugin.cmd_recall(event)
            assert "主人爱喝的茶" in text, "指令参数解析失败"
            assert retriever.calls[0]["query"] == "主人爱喝的茶"
            print("[OK] cmd_recall 指令参数解析正确")
        finally:
            await plugin.terminate()


async def test_cmd_pin() -> None:
    """cmd_pin 切换 is_pinned 状态（L3 晋升/降级）并落库。"""
    with tempfile.TemporaryDirectory(prefix="stage3_pin_") as tmp:
        plugin = make_plugin(Path(tmp))
        try:
            atom_id = "atom_pin_test_1"
            _insert_atom(plugin, atom_id, "主人偏好龙井，不加糖。")

            # --- 首次：L2 → L3 晋升 ---
            text = await plugin.cmd_pin(None, atom_id)
            assert "记忆置顶" in text
            assert "已置顶" in text, f"首次置顶回显异常: {text}"
            assert atom_id in text
            assert "主人偏好龙井" in text

            conn = plugin.storage.connect()
            row = conn.execute(
                "SELECT pinned, tier FROM l2_atoms WHERE atom_id = ?", (atom_id,)
            ).fetchone()
            assert int(row["pinned"]) == 1, "pinned 未置为 1"
            assert int(row["tier"]) == 3, "tier 未晋升为 3"
            print("[OK] cmd_pin 首次操作：L2 → L3 晋升，落库正确")

            # --- 二次：L3 → L2 降级 ---
            text2 = await plugin.cmd_pin(None, atom_id)
            assert "已取消置顶" in text2, f"二次取消置顶回显异常: {text2}"
            row2 = conn.execute(
                "SELECT pinned, tier FROM l2_atoms WHERE atom_id = ?", (atom_id,)
            ).fetchone()
            assert int(row2["pinned"]) == 0, "pinned 未复位为 0"
            assert int(row2["tier"]) == 2, "tier 未降级为 2"
            print("[OK] cmd_pin 二次操作：L3 → L2 降级，落库正确")

            # --- 不存在的 ID ---
            miss_text = await plugin.cmd_pin(None, "atom_not_exists")
            assert "未找到原子记忆" in miss_text, f"缺失 ID 回显异常: {miss_text}"

            # --- 空 ID ---
            empty_text = await plugin.cmd_pin(None, "")
            assert "请提供原子记忆 ID" in empty_text
            print("[OK] cmd_pin 缺失/空 ID 均安全降级")
        finally:
            await plugin.terminate()


async def test_cmd_migrate_missing_source() -> None:
    """源库缺失时 cmd_migrate 返回友好提示，不抛异常。"""
    with tempfile.TemporaryDirectory(prefix="stage3_migrate_miss_") as tmp:
        plugin = make_plugin(Path(tmp))
        try:
            text = await plugin.cmd_migrate(None)
            assert isinstance(text, str) and "旧版记忆资产迁移" in text
            assert "不存在" in text or "不可用" in text, f"缺失源库回显异常: {text}"
            print("[OK] cmd_migrate 源库缺失安全降级")
        finally:
            await plugin.terminate()


async def test_cmd_migrate_end_to_end() -> None:
    """构造旧版库并执行真实迁移，校验统计回显与落库结果。"""
    with tempfile.TemporaryDirectory(prefix="stage3_migrate_ok_") as tmp:
        tmp_dir = Path(tmp)
        legacy_db = tmp_dir / "legacy" / "livingmemory.db"
        _create_legacy_db(legacy_db)

        plugin = make_plugin(
            tmp_dir,
            migration={
                "source_db_path": str(legacy_db),
                "force": True,
                "dry_run": False,
            },
        )
        try:
            text = await plugin.cmd_migrate(None)
            assert "旧版记忆资产迁移" in text
            assert "migrated_ok" in text, f"迁移状态异常: {text}"
            assert "L1 `1`" in text, f"L1 迁移统计异常: {text}"
            assert "L2 `2`" in text, f"L2 迁移统计异常: {text}"
            assert "✅ 数据无损迁移完成" in text

            counts = plugin.storage.get_table_counts()
            assert counts.get("l1_docs", 0) == 1, f"l1_docs 落库异常: {counts}"
            assert counts.get("l2_atoms", 0) == 2, f"l2_atoms 落库异常: {counts}"
            assert counts.get("kg_nodes", 0) == 2, f"kg_nodes 落库异常: {counts}"
            assert counts.get("kg_edges", 0) == 1, f"kg_edges 落库异常: {counts}"

            tiers = plugin.storage.get_tier_stats()
            assert tiers.get("l2_pinned", 0) >= 1, f"置顶原子统计异常: {tiers}"
            print("[OK] cmd_migrate 端到端迁移成功：\n" + text)
        finally:
            await plugin.terminate()


async def test_cmd_handlers_never_raise() -> None:
    """四个处理器在极端入参下均不抛异常。"""
    with tempfile.TemporaryDirectory(prefix="stage3_robust_") as tmp:
        plugin = make_plugin(Path(tmp))
        plugin.enable_memory = False
        try:
            assert isinstance(await plugin.cmd_status(None), str)
            assert isinstance(await plugin.cmd_recall(None, ""), str)
            assert isinstance(await plugin.cmd_pin(None, ""), str)
            assert isinstance(await plugin.cmd_migrate(None), str)
            # event 为 None 的路径
            assert isinstance(await plugin.cmd_status(), str)
            print("[OK] 四个指令处理器极端入参下零异常")
        finally:
            await plugin.terminate()


# ---------------------------------------------------------------------------
# 5) 用例：指令组装配
# ---------------------------------------------------------------------------
async def test_command_group_assembled() -> None:
    """/memory 指令组与四个子指令已按标准 @command 规范注册。"""
    with tempfile.TemporaryDirectory(prefix="stage3_cmd_asm_") as tmp:
        plugin = make_plugin(Path(tmp))
        try:
            # 指令组对象
            assert hasattr(plugin, "memory"), "memory 指令组未装配"
            group = plugin.memory
            assert getattr(group, "name", None) == "memory", "指令组名称错误"
            assert set(getattr(group, "children", [])) == {
                "status",
                "recall",
                "pin",
                "migrate",
            }, f"子指令注册不全: {getattr(group, 'children', None)}"

            # 子指令装饰标记
            for fn_name, expected in (
                ("memory_status", "command:memory.status"),
                ("memory_recall", "command:memory.recall"),
                ("memory_pin", "command:memory.pin"),
                ("memory_migrate", "command:memory.migrate"),
            ):
                fn = getattr(plugin, fn_name, None)
                assert fn is not None, f"{fn_name} 未定义"
                assert getattr(fn, "__astrbot_hook__", None) == expected, (
                    f"{fn_name} 装饰标记异常: {getattr(fn, '__astrbot_hook__', None)}"
                )

            # 宿主入口可产出消息结果
            agen = plugin.memory_status(FakeEvent(text="/memory status"))
            payload = await agen.__anext__()
            assert isinstance(payload, str) and "分层记忆状态" in payload

            agen2 = plugin.memory_pin(FakeEvent(text="/memory pin atom_x"))
            payload2 = await agen2.__anext__()
            assert isinstance(payload2, str) and "记忆置顶" in payload2

            print("[OK] 指令组 memory.{status,recall,pin,migrate} 装配正确")
        finally:
            await plugin.terminate()


def main() -> int:
    test_position_normalization()
    asyncio.run(test_inject_position_routing())
    asyncio.run(test_cmd_status())
    asyncio.run(test_cmd_status_without_components())
    asyncio.run(test_cmd_recall())
    asyncio.run(test_cmd_recall_from_event())
    asyncio.run(test_cmd_pin())
    asyncio.run(test_cmd_migrate_missing_source())
    asyncio.run(test_cmd_migrate_end_to_end())
    asyncio.run(test_cmd_handlers_never_raise())
    asyncio.run(test_command_group_assembled())
    print("\n=== Stage 3 注入位置路由 + 管理指令 用例全部通过 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
