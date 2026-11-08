# -*- coding: utf-8 -*-
"""
Stage 4 全链路端到端集成测试（End-to-End Integration Test）

本测试不再使用任何"可控桩替代核心链路"，而是把 Stage 1~3 的全部真实组件
（StorageManager / MemorySummarizer 后台 Worker / MemoryRetriever /
 main.AgentMemoryPlugin 双向钩子与指令层）串成一条完整的生产级闭环：

    ① 初始化插件实例并启动后台 Worker；
    ② 连续产生 5 轮对话（真实 after_llm 钩子），触发多轮缓冲阈值打包 kind=batch；
    ③ 等待后台 Worker 异步消费，验证 L1 原始轮次落库 + L2 原子记忆沉淀 + L3 置顶晋升；
    ④ 新一轮用户提问发起 on_llm_request，验证前置钩子执行两阶段检索，
       并把 L3 置顶 / L2 精排记忆注入到 System Prompt（默认 system_head）；
    ⑤ 执行 /memory status、/memory recall、/memory pin 指令，
       验证管理指令输出、状态切换（L2 ⇄ L3）与数据库真实联动；
    ⑥ 调用 terminate()，验证优雅停机：尾部轮次冲洗、Worker 取消、连接关闭、数据无损。

补充闭环：**冷重启持久化验证**（test_restart_persistence）——
   终止后复用同一数据库新建插件实例，新会话的提问仍能检索到重启前沉淀的记忆，
    证明沉淀数据真正落盘、不依赖进程内存。

技术要点：
- 使用「确定性哈希 Embedding 桩」替代远程 BGE-M3，保证向量相似度可复现且零网络依赖；
- Rerank 客户端显式配置空 api_key → 走本地 RRF 融合降级路径（同样是生产代码分支）；
- 通过 spy 包装 summarizer._handle_event 精确捕获 batch 事件，
  避免"Worker 消费过快导致队列瞬时为空"的竞态误判；
- 全部断言均以数据库真实查询结果为准，不依赖内部计数器。

运行： python3 tests/test_stage4_e2e.py
"""

from __future__ import annotations

import asyncio
import atexit
import re
import shutil
import sqlite3
import sys
import tempfile
import textwrap
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1) 构造最小 AstrBot 桩包（必须在 import main 之前注入 sys.path）
#    桩包严格对齐真实 AstrBot 的装饰器形态，且包含 Stage 3 所需的指令组能力：
#    - on_llm_request / on_llm_response / on_decorating_result：无参工厂装饰器；
#    - command_group(name)：返回指令组对象，其 .command(sub) 注册子指令；
#    - command(name)：扁平指令装饰器（Stage 3 降级路径）。
# ---------------------------------------------------------------------------
STUB_ROOT = Path(tempfile.mkdtemp(prefix="astrbot_stub_stage4_"))
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

for _rel_path, _content in _STUB_FILES.items():
    _target = STUB_ROOT / _rel_path
    _target.parent.mkdir(parents=True, exist_ok=True)
    _target.write_text(_content, encoding="utf-8")

if str(STUB_ROOT) not in sys.path:
    sys.path.insert(0, str(STUB_ROOT))

# 插件根目录（支持直接以脚本方式运行）
PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

# 清掉可能已被别处导入的宿主模块，确保本次使用本测试的桩包
for _mod in ("astrbot", "main", "storage", "migrator", "summarizer", "retriever", "models_client"):
    sys.modules.pop(_mod, None)

import main as plugin_main  # noqa: E402
from main import (  # noqa: E402
    MEMORY_BLOCK_MARKER,
    POSITION_SYSTEM_HEAD,
    AgentMemoryPlugin,
)
from models_client import RerankClient  # noqa: E402

plugin_main.PLUGIN_NAME = "astrbot_plugin_agent_memory_e2e_test"

# ---------------------------------------------------------------------------
# 2) 测试常量：5 轮对话脚本
#    前 3 轮含 L3 置顶关键词（喜欢 / 习惯 / 必须）→ 晋升 L3；
#    后 2 轮为中立事实陈述 → 留存 L2，供两阶段检索的"精排"链路命中。
# ---------------------------------------------------------------------------
SESSION_ID = "aiocqhttp:FriendMessage:e2e_master"
DIALOGUE_SCRIPT: List[tuple] = [
    ("主人喜欢喝龙井，不加糖。", "好的，记住了：主人喝龙井不加糖。"),
    ("上周我们讨论了西湖龙井的采摘季节。", "是的，那次聊得很详细。"),
    ("主人习惯每天晚上十点休息。", "已记录主人的作息习惯。"),
    ("明前龙井的价格今年上涨了不少。", "确实，今年行情偏高。"),
    ("主人必须每天早上喝一杯温水。", "记下了，这是硬性约束。"),
]
EXPECTED_TURNS = len(DIALOGUE_SCRIPT)  # 5 轮 → 命中 summary_turn_interval 阈值

TAIL_TURNS: List[tuple] = [
    ("主人的生日是三月十二日。", "生日快乐已记入档案。"),
    ("主人最近在读模型思维这本书。", "已记录在读书目。"),
]
"""终止前注入的尾部轮次（不足阈值），用于验证 terminate() 的 flush 冲洗能力。"""

LOBSTER_KEYWORD = "龙井"
"""贯穿对话与检索的核心关键词，用于断言记忆沉淀与召回命中。"""


# ---------------------------------------------------------------------------
# 3) 测试替身
# ---------------------------------------------------------------------------
class FakeEvent:
    """模拟 AstrMessageEvent 的最小字段集（含指令层需要的 plain_result）。"""

    def __init__(self, session_id: str = SESSION_ID, text: str = "主人喜欢喝什么茶？"):
        self.session_id = session_id
        self.unified_msg_origin = session_id
        self.message_str = text
        self.message_text = text

    def plain_result(self, text: str) -> str:
        """模拟宿主消息结果包装（直接回传文本便于断言）。"""
        return text


class FakeRequest:
    """模拟 ProviderRequest 的最小字段集（system_prompt / prompt 均可写）。"""

    def __init__(
        self,
        prompt: str = "主人喜欢喝什么茶？",
        session_id: str = SESSION_ID,
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
        self.completion_text = text
        self.is_chunk = is_chunk
        self.tools_call_name = list(tools_call_name or [])


class DeterministicEmbeddingClient:
    """
    确定性哈希 Embedding 桩：字符 unigram + bigram 哈希入桶并 L2 归一化。

    目的：让「向量语义召回」分支在完全离线的环境下也能产生稳定、可复现的相似度，
    从而真正走通 retriever 的第一阶段向量召回与 RRF 余弦打分，
    而不是退化成全零向量（那样只能验证关键词链路）。
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = int(dim)

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        cleaned = re.sub(r"\s+", "", str(text or ""))
        if not cleaned:
            return vec

        grams = list(cleaned)
        grams += [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
        for gram in grams:
            bucket = zlib.crc32(gram.encode("utf-8")) % self.dim
            vec[bucket] += 1.0

        norm = sum(v * v for v in vec) ** 0.5
        if norm > 1e-12:
            vec = [v / norm for v in vec]
        return vec

    async def get_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]


# ---------------------------------------------------------------------------
# 4) 辅助函数
# ---------------------------------------------------------------------------
def make_plugin(tmp_dir: Path, db_name: str = "agent_memory.db") -> AgentMemoryPlugin:
    """构造插件实例（真实全量装配），并把远程模型替换为确定性本地桩。"""
    config = {
        "db_path": str(tmp_dir / db_name),
        "enable_memory": True,
        "summary_turn_interval": EXPECTED_TURNS,
        "memory_injection_enable": True,
        "retrieval": {
            "top_k": 6,
            "retrieve_timeout": 8.0,
            "prompt_injection_max_chars": 4000,
            "inject_position": POSITION_SYSTEM_HEAD,
        },
    }
    plugin = AgentMemoryPlugin(context=None, config=config)
    assert plugin.storage is not None, "存储装配失败，测试前置条件不成立"
    assert plugin.summarizer is not None, "summarizer 装配失败"
    assert plugin.retriever is not None, "retriever 装配失败"

    _install_deterministic_clients(plugin)
    return plugin


def _install_deterministic_clients(plugin: AgentMemoryPlugin) -> None:
    """
    把插件内三个组件（summarizer / retriever / rerank）的模型客户端
    统一替换为确定性本地实现，保证测试离线、可复现、零外部依赖。
    """
    emb = DeterministicEmbeddingClient(dim=64)

    # Rerank 显式不配置 api_base/api_key → 命中生产代码中的本地 RRF 降级分支
    reranker = RerankClient(
        api_base="",
        api_key="",
        model="local-rrf-fallback",
        timeout=5.0,
        rrf_k=60,
        embedding_client=emb,
    )

    plugin.embedding_client = emb
    plugin.rerank_client = reranker

    plugin.summarizer.embedding_client = emb
    plugin.summarizer.rerank_client = reranker

    plugin.retriever.embedding_client = emb
    plugin.retriever.rerank_client = reranker


def _query_rows(db_path: Path, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    """只读方式查询数据库（独立连接，不干扰插件内部连接生命周期）。"""
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _count(db_path: Path, table: str) -> int:
    """统计单表行数（表不存在时返回 0，便于统计扩展表）。"""
    try:
        rows = _query_rows(db_path, f"SELECT COUNT(1) AS cnt FROM {table}")
        return int(rows[0]["cnt"]) if rows else 0
    except sqlite3.Error:
        return 0


def _snapshot_counts(db_path: Path) -> Dict[str, int]:
    """核心表体量快照，用于停机前后对比数据无损。"""
    return {
        "l1_docs": _count(db_path, "l1_docs"),
        "l2_atoms": _count(db_path, "l2_atoms"),
        "kg_nodes": _count(db_path, "kg_nodes"),
        "kg_edges": _count(db_path, "kg_edges"),
        "task_dlq": _count(db_path, "task_dlq"),
    }


async def _drain_queue(plugin: AgentMemoryPlugin, timeout: float = 60.0) -> None:
    """
    等待后台 Worker 把消费队列彻底排空。

    使用 queue.join() 而非 sleep 轮询：语义精确，且不引入时序脆弱性。
    """
    queue = plugin.summarizer._queue
    await asyncio.wait_for(queue.join(), timeout=timeout)


def _install_batch_spy(plugin: AgentMemoryPlugin, sink: List[Dict[str, Any]]) -> None:
    """
    包装 summarizer._handle_event，在真正处理前把 batch 事件抄录进 sink。

    原因：Worker 启动后消费极快，直接断言 queue.qsize() 会产生竞态误判；
    旁路抄录可以稳定地校验 "kind=batch / turn_count=5 / reason=threshold"。
    """
    original = plugin.summarizer._handle_event

    async def _spy(event):
        try:
            sink.append(dict(event) if isinstance(event, dict) else {"raw": repr(event)})
        except Exception:
            pass
        await original(event)

    plugin.summarizer._handle_event = _spy  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 5) 主用例：① ~ ⑥ 全链路闭环
# ---------------------------------------------------------------------------
async def test_full_pipeline_e2e() -> None:
    with tempfile.TemporaryDirectory(prefix="agent_memory_stage4_") as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "agent_memory.db"
        persona = "你是薇薇安，主人的专属助理。"

        # =====================================================================
        # ① 初始化插件实例并启动后台 Worker
        # =====================================================================
        plugin = make_plugin(tmp_dir)
        await plugin.initialize()

        assert plugin._initialized is True, "插件未进入已初始化状态"
        assert plugin.summarizer._running is True, "后台 Worker 未启动"
        assert plugin.summarizer._worker_task is not None, "Worker Task 未创建"
        print(
            "[① OK] 插件初始化完成 | turn_interval=%d | worker=%s | db=%s"
            % (
                plugin.turn_interval,
                plugin.summarizer._worker_task.get_name(),
                plugin.storage.db_path,
            )
        )

        batch_events: List[Dict[str, Any]] = []
        _install_batch_spy(plugin, batch_events)

        try:
            # =================================================================
            # ② 连续产生 5 轮对话（走真实 after_llm 钩子）→ 触发 batch 打包
            # =================================================================
            sid = SESSION_ID
            for idx, (user_text, bot_text) in enumerate(DIALOGUE_SCRIPT, start=1):
                event = FakeEvent(session_id=sid, text=user_text)
                resp = FakeResponse(bot_text)

                await plugin.on_llm_response(event, resp)

                pending = plugin.summarizer.pending_turns(sid)
                if idx < EXPECTED_TURNS:
                    assert pending == idx, (
                        "第 %d 轮后缓冲应为 %d，实际 %d" % (idx, idx, pending)
                    )
                    assert (
                        plugin.summarizer._stat_batches_enqueued == 0
                    ), "未达阈值却已打包，缓冲逻辑异常"
                else:
                    assert pending == 0, "达到阈值后缓冲池应被清空，实际剩余 %d 轮" % pending

            assert (
                plugin.summarizer._stat_batches_enqueued == 1
            ), "5 轮应恰好打包 1 个 batch，实际 %d" % plugin.summarizer._stat_batches_enqueued
            print(
                "[② OK] 5 轮对话已缓冲并触发打包 | batches=%d | buffered=%d"
                % (
                    plugin.summarizer._stat_batches_enqueued,
                    plugin.summarizer._stat_turns_buffered,
                )
            )

            # =================================================================
            # ③ 等待 Worker 异步消费 → L1 落库 + L2 沉淀 + L3 晋升
            # =================================================================
            await _drain_queue(plugin)

            assert batch_events, "Worker 未消费到任何事件"
            batch = batch_events[0]
            assert batch.get("kind") == "batch", "事件类型应为 batch，实际 %r" % batch.get("kind")
            assert (
                batch.get("turn_count") == EXPECTED_TURNS
            ), "batch 应含 %d 轮，实际 %r" % (EXPECTED_TURNS, batch.get("turn_count"))
            assert len(batch.get("turns") or []) == EXPECTED_TURNS
            assert batch.get("reason") == "threshold", "打包原因应为 threshold"
            assert batch.get("session_id") == sid, "batch 会话键异常"

            counts = _snapshot_counts(db_path)
            assert counts["l1_docs"] == EXPECTED_TURNS, (
                "L1 应记录 %d 条原始轮次，实际 %d" % (EXPECTED_TURNS, counts["l1_docs"])
            )
            assert counts["l2_atoms"] >= 4, "L2 原子沉淀不足，实际 %d 条" % counts["l2_atoms"]
            assert counts["task_dlq"] == 0, "正常流程不应产生死信任务"

            pinned_rows = _query_rows(
                db_path,
                "SELECT atom_id, content, pinned, tier, atom_type FROM l2_atoms "
                "WHERE pinned = 1 OR tier >= 3",
            )
            assert len(pinned_rows) >= 2, "L3 置顶晋升数量不足，实际 %d 条" % len(pinned_rows)

            lobster_pinned = [
                r for r in pinned_rows if LOBSTER_KEYWORD in str(r["content"] or "")
            ]
            assert lobster_pinned, "含偏好关键词的记忆未被晋升到 L3"

            l2_only = _query_rows(
                db_path,
                "SELECT atom_id, content FROM l2_atoms "
                "WHERE (pinned IS NULL OR (pinned = 0 AND tier < 3))",
            )
            assert l2_only, "未沉淀任何 L2（非置顶）情景记忆，精排链路无法验证"

            print(
                "[③ OK] Worker 消费完成 | L1=%d · L2=%d（其中 L3 置顶 %d） · DLQ=%d"
                % (
                    counts["l1_docs"],
                    counts["l2_atoms"],
                    len(pinned_rows),
                    counts["task_dlq"],
                )
            )

            # =================================================================
            # ④ 新提问 → on_llm_request 前置钩子：两阶段检索 + System Prompt 注入
            # =================================================================
            query = "主人喜欢喝什么茶？"
            inject_event = FakeEvent(session_id=sid, text=query)
            req = FakeRequest(prompt=query, session_id=sid, system_prompt=persona)

            await plugin.on_llm_request(inject_event, req)

            injected = req.system_prompt
            assert MEMORY_BLOCK_MARKER in injected, "前置钩子未注入记忆块"
            assert injected.count(MEMORY_BLOCK_MARKER) == 1, "记忆块被重复注入"

            # system_head 路由：记忆块必须在人格提示词之前，且人格完整保留
            assert persona in injected, "原始人格提示词被破坏"
            assert injected.index(MEMORY_BLOCK_MARKER) < injected.index(
                persona
            ), "system_head 路由下记忆块未置于 System Prompt 最前端"
            assert req.prompt == query, "system_head 路由不应改写用户 Query"

            # L3 置顶段必须注入
            assert "L3 Pinned" in injected, "缺少 L3 置顶记忆段落"
            assert LOBSTER_KEYWORD in injected, "L3 置顶记忆内容未注入"

            # L2 精排段必须注入（两阶段检索第二阶段生效）
            assert "L2 Retrieved" in injected, "缺少 L2 精排记忆段落（两阶段检索未生效）"

            # 幂等保护：同一请求重复触发不应二次注入
            await plugin.on_llm_request(inject_event, req)
            assert req.system_prompt == injected, "重复触发导致重复注入"

            stats = plugin.get_status()["hook_stats"]
            assert stats["inject_ok"] == 1, "注入成功计数异常: %r" % stats
            assert stats["inject_error"] == 0, "注入链路出现降级异常: %r" % stats
            print(
                "[④ OK] 前置注入成功 | position=%s | 记忆块 %d 字符 | stats=%s"
                % (
                    plugin._retrieval_runtime_cfg()["position"],
                    len(injected) - len(persona),
                    stats,
                )
            )

            # =================================================================
            # ⑤ 管理指令：/memory status | recall | pin
            # =================================================================
            # ---- 5.1 /memory status（真实指令入口 memory_status）----
            status_gen = plugin.memory_status(FakeEvent(session_id=sid, text="/memory status"))
            status_text = await status_gen.__anext__()
            assert isinstance(status_text, str), "status 指令未返回文本"
            assert "分层记忆状态" in status_text, "status 输出缺少标题"
            assert (
                "L2 原子 `%d`" % counts["l2_atoms"] in status_text
            ), "status 的 L2 体量与数据库不一致\n%s" % status_text
            assert (
                "L3 置顶 `%d`" % len(pinned_rows) in status_text
            ), "status 的 L3 置顶数与数据库不一致\n%s" % status_text
            assert "L1 历史 `%d`" % counts["l1_docs"] in status_text
            assert "钩子" in status_text and "模型" in status_text
            print("[⑤-1 OK] /memory status 输出正常（体量与数据库一致）")

            # ---- 5.2 /memory recall（筛参路径：从 message_str 剥离子指令）----
            recall_gen = plugin.memory_recall(
                FakeEvent(session_id=sid, text="/memory recall %s" % LOBSTER_KEYWORD)
            )
            recall_text = await recall_gen.__anext__()
            assert isinstance(recall_text, str), "recall 指令未返回文本"
            assert LOBSTER_KEYWORD in recall_text, "recall 未回显 Query 关键词"
            assert "L3 置顶" in recall_text, "recall 缺少 L3 置顶段落"
            assert "L2 精排" in recall_text, "recall 缺少 L2 精排段落"
            assert "score=" in recall_text, "recall 未输出精排分数"
            print("[⑤-2 OK] /memory recall 输出正常（含 L3 置顶 + L2 精排分数）")

            # ---- 5.3 /memory pin（L2 → L3 晋升，再回滚）----
            target_atom = str(l2_only[0]["atom_id"])
            pin_gen = plugin.memory_pin(
                FakeEvent(session_id=sid, text="/memory pin %s" % target_atom)
            )
            pin_text = await pin_gen.__anext__()
            assert isinstance(pin_text, str), "pin 指令未返回文本"
            assert "记忆置顶" in pin_text, "pin 输出缺少标题"
            assert "已置顶" in pin_text and "已取消" not in pin_text, (
                "pin 未完成 L2→L3 晋升: %s" % pin_text
            )

            pinned_after = _query_rows(
                db_path,
                "SELECT pinned, tier FROM l2_atoms WHERE atom_id = ?",
                (target_atom,),
            )
            assert pinned_after, "置顶后原子记录丢失"
            assert int(pinned_after[0]["pinned"]) == 1, "数据库 pinned 未置 1"
            assert int(pinned_after[0]["tier"]) == 3, "数据库 tier 未晋升为 3"

            # 置顶后 L3 计数应 +1（用 status 交叉验证）
            new_pinned_total = _count(
                db_path, "l2_atoms WHERE pinned = 1 OR tier >= 3"
            )
            assert new_pinned_total == len(pinned_rows) + 1, "L3 置顶计数未按预期 +1"
            status_gen2 = plugin.memory_status(FakeEvent(session_id=sid, text="/memory status"))
            status_text2 = await status_gen2.__anext__()
            assert "L3 置顶 `%d`" % new_pinned_total in status_text2, (
                "status 未反映置顶后的最新数量\n%s" % status_text2
            )

            # 再执行一次 → 取消置顶（L3 → L2 降级），验证状态切换完整可逆
            pin_gen2 = plugin.memory_pin(
                FakeEvent(session_id=sid, text="/memory pin %s" % target_atom)
            )
            pin_text2 = await pin_gen2.__anext__()
            assert "已取消置顶" in pin_text2, "pin 未完成 L3→L2 降级: %s" % pin_text2

            unpinned_after = _query_rows(
                db_path,
                "SELECT pinned, tier FROM l2_atoms WHERE atom_id = ?",
                (target_atom,),
            )
            assert int(unpinned_after[0]["pinned"]) == 0, "取消置顶后 pinned 未归零"
            assert int(unpinned_after[0]["tier"]) == 2, "取消置顶后 tier 未回落到 2"
            print(
                "[⑤-3 OK] /memory pin 状态切换可逆 | atom=%s | L3 总数 %d→%d→%d"
                % (target_atom, len(pinned_rows), new_pinned_total, len(pinned_rows))
            )

            # =================================================================
            # ⑥ terminate() 优雅停机 + 资源无损释放
            # =================================================================
            # 先注入 2 轮不足阈值的尾部对话，验证停机前 flush 冲洗能力
            for user_text, bot_text in TAIL_TURNS:
                await plugin.on_llm_response(
                    FakeEvent(session_id=sid, text=user_text), FakeResponse(bot_text)
                )
            assert (
                plugin.summarizer.pending_turns(sid) == len(TAIL_TURNS)
            ), "尾部轮次未停留在缓冲池，无法验证停机冲洗"

            before_stop = _snapshot_counts(db_path)
            await plugin.terminate()

            # 6.1 运行时资源已全部释放
            assert plugin._initialized is False, "terminate 后仍标记为已初始化"
            assert plugin.summarizer._running is False, "Worker 仍在运行"
            assert plugin.summarizer._worker_task is None, "Worker Task 未被回收"
            assert plugin.storage._conn is None, "SQLite 连接未被关闭"
            assert plugin.summarizer.pending_turns() == 0, "停机后缓冲池仍有残留轮次"
            assert plugin.summarizer._queue.qsize() == 0, "停机后消费队列仍有残留任务"

            # 6.2 尾部轮次被冲洗落库（数据无损）
            after_stop = _snapshot_counts(db_path)
            assert after_stop["l1_docs"] == before_stop["l1_docs"] + len(TAIL_TURNS), (
                "停机冲洗未把尾部轮次写入 L1：before=%d after=%d"
                % (before_stop["l1_docs"], after_stop["l1_docs"])
            )
            assert after_stop["l2_atoms"] >= before_stop["l2_atoms"], "停机过程出现 L2 数据丢失"
            assert after_stop["task_dlq"] == 0, "停机过程产生死信任务"

            # 6.3 关闭后数据库仍可被全新连接打开且数据完整（无锁表/无损坏）
            verify_counts = _snapshot_counts(db_path)
            assert verify_counts == after_stop, "关闭后重新打开的数据与停机时不一致"

            print(
                "[⑥ OK] 优雅停机完成 | 尾部冲洗 %d 轮 | L1 %d→%d · L2 %d→%d · DLQ=%d"
                % (
                    len(TAIL_TURNS),
                    before_stop["l1_docs"],
                    after_stop["l1_docs"],
                    before_stop["l2_atoms"],
                    after_stop["l2_atoms"],
                    after_stop["task_dlq"],
                )
            )
        finally:
            # 兜底：用例中途断言失败时也要释放资源，避免临时目录句柄泄漏
            if plugin.storage._conn is not None or plugin.summarizer._running:
                try:
                    await plugin.terminate()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 6) 补充闭环：冷重启后记忆仍可被检索（证明真正落盘、不依赖进程内存）
# ---------------------------------------------------------------------------
async def test_restart_persistence() -> None:
    with tempfile.TemporaryDirectory(prefix="agent_memory_stage4_restart_") as tmp:
        tmp_dir = Path(tmp)
        db_path = tmp_dir / "agent_memory.db"
        sid = SESSION_ID

        # ---- 第一进程：产生并沉淀记忆 ----
        first = make_plugin(tmp_dir)
        await first.initialize()
        try:
            for user_text, bot_text in DIALOGUE_SCRIPT:
                await first.on_llm_response(
                    FakeEvent(session_id=sid, text=user_text), FakeResponse(bot_text)
                )
            await _drain_queue(first)
            atoms_before = _count(db_path, "l2_atoms")
            assert atoms_before > 0, "第一进程未沉淀任何 L2 记忆"
        finally:
            await first.terminate()

        assert first.storage._conn is None, "第一进程连接未关闭"

        # ---- 第二进程：全新实例 + 全新连接，复用同一数据库 ----
        second = make_plugin(tmp_dir)
        await second.initialize()
        try:
            query = "主人喜欢喝什么茶？"
            req = FakeRequest(prompt=query, session_id=sid, system_prompt="你是薇薇安。")
            await second.on_llm_request(FakeEvent(session_id=sid, text=query), req)

            assert MEMORY_BLOCK_MARKER in req.system_prompt, "重启后前置注入未生效"
            assert LOBSTER_KEYWORD in req.system_prompt, (
                "重启后未能检索到上一进程沉淀的记忆（持久化失败）"
            )

            recall_text = await second.cmd_recall(FakeEvent(session_id=sid, text=query), query)
            assert "L3 置顶" in recall_text, "重启后 recall 输出异常"

            print(
                "[闭环 OK] 冷重启持久化验证通过 | 复用 %d 条 L2 记忆 | 注入块 %d 字符"
                % (atoms_before, len(req.system_prompt))
            )
        finally:
            await second.terminate()

        assert second.summarizer._running is False
        assert second.storage._conn is None
        print("[闭环 OK] 二次停机后资源同样完全释放")


# ---------------------------------------------------------------------------
# 7) 入口
# ---------------------------------------------------------------------------
def main() -> int:
    asyncio.run(test_full_pipeline_e2e())
    asyncio.run(test_restart_persistence())
    print("\n=== Stage 4 全链路端到端集成测试 全部通过（零回归） ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
