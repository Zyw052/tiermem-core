# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 插件主入口 (Stage 2 双向钩子总装)

职责：
1. 规范解析插件配置（summary_turn_interval / db_path / models / retrieval / 注入开关 ...）；
2. 全量装配核心组件：
   StorageManager → EmbeddingClient → RerankClient → MemorySummarizer → MemoryRetriever；
3. 生命周期托管：
   - initialize(): 启动 summarizer 后台 Worker；
   - terminate(): flush_all() 冲洗剩余轮次 → stop() 停 Worker → close() 关闭存储；
4. Stage 2 双向钩子：
   - on_llm_request       : 前置拦截，两阶段检索 → 注入 System Prompt 前部；
   - on_llm_response      : 后置拦截，用户输入 + 回复 → 推入多轮缓冲池；
   - on_decorating_result : 仅在缺少 on_llm_response 时作为兜底记录点；
   所有钩子全量 try-except，异常静默降级，绝不影响 AstrBot 正常对话流。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# AstrBot 框架导入（logger 提供兜底，便于脱离宿主做静态检查与单测）
# ---------------------------------------------------------------------------
try:
    from astrbot.api import logger
except Exception:  # pragma: no cover - 仅非 AstrBot 环境触发
    import logging

    logger = logging.getLogger("astrbot_plugin_agent_memory.main")

from astrbot.api.star import Context, Star, register

# Stage 2 双向钩子装饰器按可用性接入：
# - on_llm_request       : LLM 请求前置，可写 req.system_prompt（记忆注入）
# - on_llm_response      : LLM 响应后置，读取 resp.completion_text（轮次记录）
# - on_decorating_result : 发送前事件（仅作为 on_llm_response 缺失时的兜底记录点）
# 任一装饰器缺失都不会阻断插件加载，钩子按能力矩阵逐个装配。
try:
    from astrbot.api.event import filter as event_filter
except Exception:  # pragma: no cover
    event_filter = None  # type: ignore

_ON_LLM_REQUEST = (
    getattr(event_filter, "on_llm_request", None) if event_filter is not None else None
)
_ON_LLM_RESPONSE = (
    getattr(event_filter, "on_llm_response", None) if event_filter is not None else None
)
_ON_DECORATING_RESULT = (
    getattr(event_filter, "on_decorating_result", None) if event_filter is not None else None
)

# ---------------------------------------------------------------------------
# 本地模块导入（兼容包内相对导入与扁平导入两种加载方式）
# ---------------------------------------------------------------------------
try:
    from .storage import MemoryStorage
    from .models_client import EmbeddingClient, RerankClient
    from .summarizer import MemorySummarizer
    from .retriever import MemoryRetriever
except ImportError:  # pragma: no cover
    from storage import MemoryStorage  # type: ignore
    from models_client import EmbeddingClient, RerankClient  # type: ignore
    from summarizer import MemorySummarizer  # type: ignore
    from retriever import MemoryRetriever  # type: ignore


PLUGIN_NAME = "astrbot_plugin_agent_memory"
DEFAULT_TURN_INTERVAL = 5
DEFAULT_DB_FILENAME = "agent_memory.db"

# 各组件默认参数集中管理，避免散落在装配逻辑中
DEFAULT_EMBEDDING_API_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_EMBEDDING_TIMEOUT = 30.0

DEFAULT_RERANK_API_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANK_TIMEOUT = 20.0
DEFAULT_RRF_K = 60

# ---- Stage 2 钩子相关默认值 ----
DEFAULT_RETRIEVE_TIMEOUT = 3.0
"""前置检索最大等待秒数；超时即静默降级，绝不拖慢主对话。"""

DEFAULT_INJECTION_MAX_CHARS = 2000
"""注入 System Prompt 的记忆块字符上限，0 表示不限制。"""

DEFAULT_RECORD_DEDUP_WINDOW = 2.0
"""轮次记录去重时间窗（秒）：同一会话相同内容在该窗口内只记一次，防重复回调。"""

MEMORY_BLOCK_MARKER = "<!-- agent_memory:injected:v1 -->"
"""记忆注入块占位标记：用于识别并阻止同一请求内的重复注入。"""


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _as_plain(value: Any) -> Any:
    """把 AstrBotConfig 等 dict 子类递归转换为纯 dict/list，便于后续安全取值。"""
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    return value


def _first_non_empty(*values: Any, default: Any = "") -> Any:
    """返回第一个非空值（None / 空白字符串视为空），全部为空时返回 default。"""
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, str):
            if raw.strip():
                return raw.strip()
            continue
        return raw
    return default


def _safe_int(value: Any, default: int, minimum: Optional[int] = None) -> int:
    """安全转 int，非法值回退 default，可选下界钳制。"""
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None and result < minimum:
        result = minimum
    return result


def _safe_float(value: Any, default: float) -> float:
    """安全转 float，非法值回退 default。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_plugin_data_dir(plugin_name: str = PLUGIN_NAME) -> Path:
    """动态解析插件持久化数据目录，避免写死绝对路径。"""
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path

        return Path(get_astrbot_data_path()) / "plugin_data" / plugin_name
    except Exception:
        # 回退：<AstrBot 根>/data/plugin_data/<plugin_name>
        plugin_dir = Path(__file__).resolve().parent
        return plugin_dir.parent.parent / "plugin_data" / plugin_name


# ---------------------------------------------------------------------------
# Stage 2：事件/请求对象的安全字段提取（AstrBot 属性可能是 property，读取可能抛错）
# ---------------------------------------------------------------------------
def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    """安全读取对象属性：obj 为空或属性访问抛异常时返回 default。"""
    if obj is None:
        return default
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _extract_session_id(event: Any = None, req: Any = None) -> str:
    """
    提取会话标识。优先级：
    req.session_id → event.unified_msg_origin → event.session_id。
    统一为字符串，避免下游按 None 做 key。
    """
    return str(
        _first_non_empty(
            _safe_attr(req, "session_id"),
            _safe_attr(event, "unified_msg_origin"),
            _safe_attr(event, "session_id"),
            default="",
        )
    )


def _extract_user_query(event: Any = None, req: Any = None) -> str:
    """
    提取用户当前 Query。优先级：
    req.prompt（LLM 实际请求的提示词）→ event.message_str → event.message_text。
    """
    return str(
        _first_non_empty(
            _safe_attr(req, "prompt"),
            _safe_attr(event, "message_str"),
            _safe_attr(event, "message_text"),
            default="",
        )
    )


def _extract_assistant_text(resp: Any) -> str:
    """
    提取 Assistant 回复纯文本。优先 completion_text（property，内部聚合 result_chain），
    其次 completion / text 字段兜底。
    """
    return str(
        _first_non_empty(
            _safe_attr(resp, "completion_text"),
            _safe_attr(resp, "completion"),
            _safe_attr(resp, "text"),
            default="",
        )
    )


def _extract_text_from_event_result(event: Any) -> str:
    """
    兜底路径（on_decorating_result）：从 event.get_result() 的消息链中取纯文本。
    任何异常均返回空串。
    """
    try:
        getter = getattr(event, "get_result", None)
        if not callable(getter):
            return ""
        result = getter()
        if result is None:
            return ""
        chain = _safe_attr(result, "chain") or _safe_attr(result, "message_chain")
        get_plain = getattr(chain, "get_plain_text", None)
        if callable(get_plain):
            return str(_first_non_empty(get_plain(), default=""))
        # 再兜底：result_chain.get_plain_text()
        fallback_chain = _safe_attr(result, "result_chain")
        get_plain_2 = getattr(fallback_chain, "get_plain_text", None)
        if callable(get_plain_2):
            return str(_first_non_empty(get_plain_2(), default=""))
    except Exception:
        return ""
    return ""


@register(
    PLUGIN_NAME,
    "Vivian & Master",
    "智涌原生分层记忆插件 (L1工作区/L2情景/L3画像 + 独立总结模型 + 资产无缝迁移 + 专属WebUI)",
    "1.0.0",
    "",
)
class AgentMemoryPlugin(Star):
    """智涌原生分层记忆插件主类（Stage 1：多轮缓冲总结 + 组件总装 + 生命周期托管）"""

    def __init__(self, context: Context, config: Optional[dict] = None) -> None:
        super().__init__(context)

        # ---- 1. 配置规范化 ----
        self.config: Dict[str, Any] = _as_plain(config) if config is not None else {}
        if not isinstance(self.config, dict):  # 极端防御
            self.config = {}

        self.data_dir: Path = _resolve_plugin_data_dir(PLUGIN_NAME)
        self.enable_memory: bool = bool(self.config.get("enable_memory", True))
        self.turn_interval: int = self._resolve_turn_interval()
        # Stage 2：前置注入开关（与总开关独立，便于只关注入保留沉淀）
        self.enable_injection: bool = bool(
            self.config.get("memory_injection_enable", True)
        )

        # ---- 2. 组件占位（装配失败时保持为 None，插件不因此崩溃）----
        self.storage: Optional[MemoryStorage] = None
        self.embedding_client: Optional[EmbeddingClient] = None
        self.rerank_client: Optional[RerankClient] = None
        self.summarizer: Optional[MemorySummarizer] = None
        self.retriever: Optional[MemoryRetriever] = None

        self._initialized: bool = False

        # ---- Stage 2 运行时状态（观测用，不参与控制流）----
        self._last_recorded: Dict[str, tuple] = {}
        """session_id -> (fingerprint, monotonic_ts)，用于短窗口去重。"""
        self._hook_stats: Dict[str, int] = {
            "inject_ok": 0,
            "inject_skip": 0,
            "inject_error": 0,
            "record_ok": 0,
            "record_skip": 0,
            "record_error": 0,
        }

        # ---- 3. 全量装配 ----
        try:
            self._assemble()
            logger.info(
                "[agent_memory] Stage1 装配完成 | data_dir=%s | turn_interval=%d轮 | enable_memory=%s",
                self.data_dir,
                self.turn_interval,
                self.enable_memory,
            )
        except Exception as exc:
            logger.error(
                "[agent_memory] 组件装配失败，插件将以降级模式运行: %s", exc, exc_info=True
            )

    # ------------------------------------------------------------------
    # 配置解析
    # ------------------------------------------------------------------
    def _section(self, name: str) -> Dict[str, Any]:
        """安全读取配置中的子字典，非 dict 时返回空字典。"""
        value = self.config.get(name)
        return value if isinstance(value, dict) else {}

    def _resolve_turn_interval(self) -> int:
        """解析 summary_turn_interval：顶层优先，其次 summarizer 子节点，默认 5。"""
        raw = self.config.get("summary_turn_interval")
        if raw is None or raw == "":
            raw = self._section("summarizer").get("summary_turn_interval")
        return _safe_int(raw, DEFAULT_TURN_INTERVAL, minimum=1)

    def _resolve_provider_credentials(self, provider_id: Any) -> Dict[str, str]:
        """
        可选能力：当配置了 AstrBot 的 provider_id 时，尽力从宿主 Provider 实例
        中解析 api_base / api_key / model，避免用户在插件里重复填写密钥。
        任何异常或属性缺失均静默降级返回已解析到的部分。
        """
        pid = str(provider_id or "").strip()
        if not pid:
            return {}

        creds: Dict[str, str] = {}
        try:
            getter = getattr(self.context, "get_provider_by_id", None)
            if not callable(getter):
                return {}
            provider = getter(pid)
            if provider is None:
                logger.debug("[agent_memory] provider_id=%s 未找到对应 Provider，跳过。", pid)
                return {}

            # 直接属性：api_base / base_url / api_key
            for attr, key in (
                ("api_base", "api_base"),
                ("base_url", "api_base"),
                ("api_key", "api_key"),
            ):
                val = getattr(provider, attr, None)
                if isinstance(val, str) and val.strip() and key not in creds:
                    creds[key] = val.strip()

            # provider_config 兜底
            cfg = getattr(provider, "provider_config", None)
            if isinstance(cfg, dict):
                for key in ("api_base", "base_url", "api_key"):
                    val = cfg.get(key)
                    if isinstance(val, str) and val.strip():
                        target = "api_base" if key in ("api_base", "base_url") else "api_key"
                        creds.setdefault(target, val.strip())

            # 模型名：优先 get_model()
            get_model = getattr(provider, "get_model", None)
            if callable(get_model):
                try:
                    model_name = get_model()
                    if isinstance(model_name, str) and model_name.strip():
                        creds["model"] = model_name.strip()
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("[agent_memory] 解析 provider_id=%s 凭证失败: %s", pid, exc)

        return creds

    # ------------------------------------------------------------------
    # 组件装配
    # ------------------------------------------------------------------
    def _build_storage(self) -> MemoryStorage:
        """
        构建存储管理器。
        - db_path 配置为完整文件路径 → 直接使用该文件；
        - db_path 配置为目录（无后缀）→ 目录下使用默认文件名；
        - 未配置 → 使用插件数据目录下的默认数据库。
        """
        raw_db_path = str(self.config.get("db_path") or "").strip()
        if raw_db_path:
            db_file = Path(raw_db_path).expanduser()
            if db_file.suffix == "":
                db_file = db_file / DEFAULT_DB_FILENAME
            db_file.parent.mkdir(parents=True, exist_ok=True)
            storage = MemoryStorage(db_file.parent)
            # 复用 MemoryStorage 的连接管理，但把数据库文件指向用户配置路径
            storage.db_path = db_file
        else:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            storage = MemoryStorage(self.data_dir)

        storage.init_schema()
        logger.info("[agent_memory] 存储就绪: %s", storage.db_path)
        return storage

    def _build_embedding_client(self) -> EmbeddingClient:
        """构建 Embedding 客户端（优先插件配置，其次 provider_id 解析值，最后默认值）。"""
        summarizer_cfg = self._section("summarizer")
        models_cfg = self._section("models")
        emb_cfg = models_cfg.get("embedding") if isinstance(models_cfg.get("embedding"), dict) else {}
        provider_creds = self._resolve_provider_credentials(
            summarizer_cfg.get("embedding_provider_id")
        )

        api_base = _first_non_empty(
            emb_cfg.get("api_base"),
            provider_creds.get("api_base"),
            summarizer_cfg.get("embedding_api_base"),
            default=DEFAULT_EMBEDDING_API_BASE,
        )
        api_key = _first_non_empty(
            emb_cfg.get("api_key"),
            provider_creds.get("api_key"),
            summarizer_cfg.get("embedding_api_key"),
            default="",
        )
        model = _first_non_empty(
            emb_cfg.get("model"),
            provider_creds.get("model"),
            summarizer_cfg.get("embedding_model"),
            default=DEFAULT_EMBEDDING_MODEL,
        )
        dim = _safe_int(emb_cfg.get("dim"), DEFAULT_EMBEDDING_DIM, minimum=1)
        timeout = _safe_float(emb_cfg.get("timeout"), DEFAULT_EMBEDDING_TIMEOUT)

        return EmbeddingClient(
            api_base=str(api_base),
            api_key=str(api_key),
            model=str(model),
            dim=dim,
            timeout=timeout,
            send_dimensions_param=bool(emb_cfg.get("send_dimensions_param", False)),
        )

    def _build_rerank_client(self, embedding_client: EmbeddingClient) -> RerankClient:
        """构建 Rerank 客户端（未配置密钥时自动降级为本地 RRF 重排）。"""
        summarizer_cfg = self._section("summarizer")
        models_cfg = self._section("models")
        rerank_cfg = models_cfg.get("rerank") if isinstance(models_cfg.get("rerank"), dict) else {}
        provider_creds = self._resolve_provider_credentials(
            summarizer_cfg.get("rerank_provider_id")
        )

        api_base = _first_non_empty(
            rerank_cfg.get("api_base"),
            provider_creds.get("api_base"),
            summarizer_cfg.get("rerank_api_base"),
            default=DEFAULT_RERANK_API_BASE,
        )
        api_key = _first_non_empty(
            rerank_cfg.get("api_key"),
            provider_creds.get("api_key"),
            summarizer_cfg.get("rerank_api_key"),
            default="",
        )
        model = _first_non_empty(
            rerank_cfg.get("model"),
            provider_creds.get("model"),
            summarizer_cfg.get("rerank_model"),
            default=DEFAULT_RERANK_MODEL,
        )
        timeout = _safe_float(rerank_cfg.get("timeout"), DEFAULT_RERANK_TIMEOUT)
        rrf_k = _safe_int(rerank_cfg.get("rrf_k"), DEFAULT_RRF_K, minimum=1)

        return RerankClient(
            api_base=str(api_base),
            api_key=str(api_key),
            model=str(model),
            timeout=timeout,
            rrf_k=rrf_k,
            embedding_client=embedding_client,
        )

    def _build_summarizer(
        self,
        storage: MemoryStorage,
        embedding_client: EmbeddingClient,
        rerank_client: RerankClient,
    ) -> MemorySummarizer:
        """构建总结 Worker，并把 summary_turn_interval 等配置注入。"""
        summarizer_cfg = dict(self._section("summarizer"))

        # 总结 LLM 凭证：插件配置优先，其次 provider_id 解析，其次 embedding 同名配置
        provider_creds = self._resolve_provider_credentials(summarizer_cfg.get("provider_id"))
        llm_api_base = _first_non_empty(
            summarizer_cfg.get("api_base"),
            provider_creds.get("api_base"),
            default="",
        )
        llm_api_key = _first_non_empty(
            summarizer_cfg.get("api_key"),
            provider_creds.get("api_key"),
            default="",
        )
        llm_model = _first_non_empty(
            summarizer_cfg.get("model"),
            provider_creds.get("model"),
            default="",
        )

        runtime_cfg: Dict[str, Any] = {
            # 轮次窗口：达到 N 轮才打包总结
            "summary_turn_interval": self.turn_interval,
            "summarizer_queue_maxsize": _safe_int(
                self.config.get("summarizer_queue_maxsize"), 1000, minimum=1
            ),
            # 兼容 summarizer.py 现有读取路径（顶层 + summarizer 子节点）
            "llm_api_base": llm_api_base,
            "llm_api_key": llm_api_key,
            "llm_model": llm_model,
            "summarizer": {
                **summarizer_cfg,
                "api_base": llm_api_base,
                "api_key": llm_api_key,
                "model": llm_model,
                "timeout_seconds": _safe_float(
                    summarizer_cfg.get("timeout_seconds"), 45.0
                ),
                "max_concurrency": _safe_int(
                    summarizer_cfg.get("max_concurrency"), 2, minimum=1
                ),
            },
        }

        return MemorySummarizer(
            storage=storage,
            embedding_client=embedding_client,
            rerank_client=rerank_client,
            config=runtime_cfg,
        )

    def _build_retriever(
        self,
        storage: MemoryStorage,
        embedding_client: EmbeddingClient,
        rerank_client: RerankClient,
    ) -> MemoryRetriever:
        """构建混合检索器，合并 retrieval 配置与检索器内部使用的默认键。"""
        retrieval_cfg = dict(self._section("retrieval"))

        retriever_cfg: Dict[str, Any] = {
            "top_k": _safe_int(retrieval_cfg.get("top_k"), 6, minimum=1),
            "weight_vector": _safe_float(retrieval_cfg.get("weight_vector"), 0.55),
            "weight_bm25": _safe_float(retrieval_cfg.get("weight_bm25"), 0.30),
            "weight_importance": _safe_float(
                retrieval_cfg.get("weight_importance"), 0.15
            ),
            "max_pinned_l3": _safe_int(retrieval_cfg.get("max_pinned_l3"), 15, minimum=0),
            "recall_pool_limit": _safe_int(
                retrieval_cfg.get("recall_pool_limit"), 500, minimum=1
            ),
            "strict_session_isolation": bool(
                retrieval_cfg.get("strict_session_isolation", False)
            ),
        }
        # 用户配置覆盖默认值
        retriever_cfg.update(retrieval_cfg)

        return MemoryRetriever(
            storage=storage,
            embedding_client=embedding_client,
            rerank_client=rerank_client,
            config=retriever_cfg,
        )

    def _assemble(self) -> None:
        """按依赖顺序全量装配五大组件。"""
        self.storage = self._build_storage()
        self.embedding_client = self._build_embedding_client()
        self.rerank_client = self._build_rerank_client(self.embedding_client)
        self.summarizer = self._build_summarizer(
            self.storage, self.embedding_client, self.rerank_client
        )
        self.retriever = self._build_retriever(
            self.storage, self.embedding_client, self.rerank_client
        )
        logger.info(
            "[agent_memory] 组件清单: storage=%s embedding=%s rerank=%s summarizer=%s retriever=%s",
            type(self.storage).__name__,
            type(self.embedding_client).__name__,
            type(self.rerank_client).__name__,
            type(self.summarizer).__name__,
            type(self.retriever).__name__,
        )

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        """插件启用时：启动 summarizer 后台 Worker。"""
        if not self.enable_memory:
            logger.info("[agent_memory] enable_memory=False，跳过后台 Worker 启动。")
            return

        if self.summarizer is None:
            logger.error("[agent_memory] summarizer 未装配成功，无法启动后台 Worker。")
            return

        if self._initialized:
            return

        try:
            self.summarizer.start()
            self._initialized = True
            logger.info(
                "[agent_memory] 后台总结 Worker 已启动 (turn_interval=%d 轮)。",
                self.turn_interval,
            )
        except Exception as exc:
            logger.error("[agent_memory] 启动后台 Worker 失败: %s", exc, exc_info=True)

    async def terminate(self) -> None:
        """
        插件停用/重载/卸载时：
        1. flush_all() 强制冲洗缓冲池中不足 N 轮的尾部对话；
        2. stop() 停止 Worker（内部会等待队列消费完成，超时保护）；
        3. close() 关闭 SQLite 连接。
        """
        # 1 + 2：先冲洗，再停机
        if self.summarizer is not None:
            try:
                flushed = await self.summarizer.flush_all()
                if flushed:
                    logger.info(
                        "[agent_memory] 停机前冲洗 %d 个会话共 %d 轮对话。",
                        len(flushed),
                        sum(flushed.values()),
                    )
            except Exception as exc:
                logger.warning("[agent_memory] 停机前 flush_all 失败: %s", exc)

            try:
                await self.summarizer.stop()
            except Exception as exc:
                logger.error("[agent_memory] 停止 summarizer 异常: %s", exc, exc_info=True)

        # 3. 关闭存储
        if self.storage is not None:
            try:
                self.storage.close()
            except Exception as exc:
                logger.error("[agent_memory] 关闭存储异常: %s", exc, exc_info=True)

        self._initialized = False
        logger.info("[agent_memory] 插件资源已安全释放。")

    # ------------------------------------------------------------------
    # 对外稳定 API（供事件钩子 / 指令层 / WebUI 复用）
    # ------------------------------------------------------------------
    async def record_turn(
        self, session_id: str, user_text: str, bot_text: str
    ) -> bool:
        """
        记录一轮对话到会话缓冲池（达到 N 轮自动打包触发后台总结）。

        说明：缓冲池为纯内存累积，不产生任何阻塞 IO，可安全地在消息链路上调用。
        """
        if not self.enable_memory or self.summarizer is None:
            return False

        # 兼容 initialize() 未被宿主调用的场景：首次写入时惰性启动 Worker
        if not self._initialized:
            try:
                self.summarizer.start()
                self._initialized = True
                logger.info("[agent_memory] 惰性启动后台总结 Worker。")
            except Exception as exc:
                logger.warning("[agent_memory] 惰性启动 Worker 失败: %s", exc)

        try:
            return await self.summarizer.enqueue_dialogue(session_id, user_text, bot_text)
        except Exception as exc:
            logger.error("[agent_memory] record_turn 失败: %s", exc, exc_info=True)
            return False

    async def flush_session(self, session_id: str) -> int:
        """立即冲洗指定会话的剩余轮次，返回打包入队的轮次数。"""
        if self.summarizer is None:
            return 0
        try:
            return await self.summarizer.flush_session(session_id)
        except Exception as exc:
            logger.error("[agent_memory] flush_session 失败: %s", exc, exc_info=True)
            return 0

    async def retrieve_memory(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """检索分层记忆，返回 {"pinned_l3": [...], "retrieved_l2": [...], "prompt_context": str}。"""
        empty: Dict[str, Any] = {
            "pinned_l3": [],
            "retrieved_l2": [],
            "prompt_context": "",
        }
        if not self.enable_memory or self.retriever is None:
            return empty

        try:
            retrieval_cfg = self._section("retrieval")
            effective_top_k = _safe_int(
                top_k if top_k is not None else retrieval_cfg.get("top_k"),
                6,
                minimum=1,
            )
            result = await self.retriever.retrieve(
                query=query,
                session_id=session_id,
                top_k=effective_top_k,
                candidate_k=max(
                    effective_top_k * 3,
                    _safe_int(retrieval_cfg.get("candidate_k"), 20, minimum=1),
                ),
            )
            return result if isinstance(result, dict) else empty
        except Exception as exc:
            logger.error("[agent_memory] retrieve_memory 失败: %s", exc, exc_info=True)
            return empty

    def get_status(self) -> Dict[str, Any]:
        """返回插件运行状态快照（供 /memory_status 指令或 WebUI 使用）。"""
        status: Dict[str, Any] = {
            "plugin": PLUGIN_NAME,
            "enable_memory": self.enable_memory,
            "initialized": self._initialized,
            "data_dir": str(self.data_dir),
            "db_path": str(getattr(self.storage, "db_path", "")) if self.storage else "",
            "components": {
                "storage": self.storage is not None,
                "embedding_client": self.embedding_client is not None,
                "rerank_client": self.rerank_client is not None,
                "summarizer": self.summarizer is not None,
                "retriever": self.retriever is not None,
            },
            # ---- Stage 2 钩子能力矩阵与运行指标 ----
            "hooks": {
                "on_llm_request": _ON_LLM_REQUEST is not None,
                "on_llm_response": _ON_LLM_RESPONSE is not None,
                "on_decorating_result": _ON_DECORATING_RESULT is not None,
                "injection_enabled": self.enable_injection,
            },
            "hook_stats": dict(self._hook_stats),
        }
        if self.summarizer is not None:
            try:
                status["buffer"] = self.summarizer.get_buffer_stats()
            except Exception as exc:
                logger.debug("[agent_memory] 获取缓冲统计失败: %s", exc)
        if self.storage is not None:
            try:
                status["table_counts"] = self.storage.get_table_counts()
            except Exception as exc:
                logger.debug("[agent_memory] 获取表统计失败: %s", exc)
        return status

    # ------------------------------------------------------------------
    # Stage 2 核心一：前置检索注入（Prompt Injection）
    # ------------------------------------------------------------------
    def _retrieval_runtime_cfg(self) -> Dict[str, Any]:
        """汇总前置注入用到的检索侧运行时参数（带安全兜底）。"""
        cfg = self._section("retrieval")
        return {
            "timeout": _safe_float(
                cfg.get("retrieve_timeout"), DEFAULT_RETRIEVE_TIMEOUT
            ),
            "top_k": _safe_int(cfg.get("top_k"), 6, minimum=1),
            "candidate_k": _safe_int(cfg.get("candidate_k"), 20, minimum=1),
            "max_chars": _safe_int(
                cfg.get("prompt_injection_max_chars"),
                DEFAULT_INJECTION_MAX_CHARS,
                minimum=0,
            ),
            "position": str(cfg.get("inject_position") or "head").strip().lower(),
        }

    @staticmethod
    def format_memory_block(result: Any, max_chars: int = 0) -> str:
        """
        把检索结果格式化为可注入 System Prompt 的 Markdown 记忆块。

        优先使用 retriever 已生成的 prompt_context；缺失时本地组装 L3/L2 兜底。
        返回带 MEMORY_BLOCK_MARKER 的文本；无有效记忆时返回空串。
        """
        if not isinstance(result, dict):
            return ""

        text = str(result.get("prompt_context") or "").strip()

        # 兜底：本地组装（不依赖 retriever 私有方法）
        if not text:
            pinned = result.get("pinned_l3") or []
            retrieved = result.get("retrieved_l2") or []
            if not pinned and not retrieved:
                return ""
            lines = ["## 🧠 长期记忆上下文 (Agent Memory Context)"]
            if pinned:
                lines.append("### 📌 核心记忆与约束 (L3 Pinned)")
                for idx, item in enumerate(pinned, start=1):
                    if not isinstance(item, dict):
                        continue
                    content = str(item.get("content", "")).strip()
                    if not content:
                        continue
                    atom_type = str(item.get("atom_type", "preference"))
                    try:
                        importance = float(item.get("importance", 0.9))
                    except (TypeError, ValueError):
                        importance = 0.9
                    lines.append(
                        f"{idx}. **[核心/{atom_type}]** {content} *(重要度: {importance:.2f})*"
                    )
            if retrieved:
                lines.append("### 🔍 相关情景记忆 (L2 Retrieved)")
                for idx, item in enumerate(retrieved, start=1):
                    if not isinstance(item, dict):
                        continue
                    content = str(item.get("content", "")).strip()
                    if not content:
                        continue
                    lines.append(f"{idx}. {content}")
            text = "\n".join(lines).strip()

        if not text:
            return ""

        # 字符上限保护：避免污染过长导致挤占人格提示词
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n... (记忆上下文已截断)"

        return f"{MEMORY_BLOCK_MARKER}\n{text}"

    async def build_memory_prompt_block(
        self, query: str, session_id: str = ""
    ) -> str:
        """
        执行两阶段检索并返回可注入的 Markdown 记忆块。
        超时 / 异常 / 空结果统一返回空串（静默降级，绝不阻塞主对话）。
        """
        if not self.enable_memory or self.retriever is None:
            return ""

        q = str(query or "").strip()
        if not q:
            return ""

        rc = self._retrieval_runtime_cfg()
        try:
            result = await asyncio.wait_for(
                self.retriever.retrieve(
                    query=q,
                    session_id=session_id or None,
                    top_k=rc["top_k"],
                    candidate_k=max(rc["top_k"] * 3, rc["candidate_k"]),
                ),
                timeout=rc["timeout"],
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[agent_memory] 前置检索超时(%.1fs)，本次跳过记忆注入。", rc["timeout"]
            )
            self._hook_stats["inject_error"] += 1
            return ""
        except Exception as exc:
            logger.warning("[agent_memory] 前置检索异常(已降级): %s", exc)
            self._hook_stats["inject_error"] += 1
            return ""

        return self.format_memory_block(result, rc["max_chars"])

    @staticmethod
    def _apply_block_to_request(req: Any, block: str, position: str = "head") -> bool:
        """
        把记忆块写入 req.system_prompt。
        - 已含 MARKER → 视为同一请求已注入，跳过（防工具调用多轮重复注入）；
        - position=head 前置 / tail 后置。
        返回是否实际写入。
        """
        if req is None or not block:
            return False
        try:
            existing = str(_safe_attr(req, "system_prompt", "") or "")
        except Exception:
            existing = ""

        if MEMORY_BLOCK_MARKER in existing:
            return False

        if position == "tail":
            merged = f"{existing}\n\n{block}" if existing else block
        else:
            merged = f"{block}\n\n{existing}" if existing else block

        try:
            req.system_prompt = merged
            return True
        except Exception:
            # 少数宿主实现为只读属性：静默放弃
            return False

    async def inject_memory_into_request(self, event: Any = None, req: Any = None) -> bool:
        """
        前置钩子核心：抽取 Query → 检索 → 注入 System Prompt。
        全量 try-except，任何异常都不影响 AstrBot 正常对话流。
        """
        try:
            if not self.enable_memory or not self.enable_injection or req is None:
                self._hook_stats["inject_skip"] += 1
                return False

            query = _extract_user_query(event, req)
            if not query:
                self._hook_stats["inject_skip"] += 1
                return False

            session_id = _extract_session_id(event, req)
            rc = self._retrieval_runtime_cfg()
            block = await self.build_memory_prompt_block(query, session_id)
            if not block:
                self._hook_stats["inject_skip"] += 1
                return False

            ok = self._apply_block_to_request(req, block, rc["position"])
            if ok:
                self._hook_stats["inject_ok"] += 1
                logger.debug(
                    "[agent_memory] 已注入记忆块 | session=%s | chars=%d | position=%s",
                    session_id,
                    len(block),
                    rc["position"],
                )
            else:
                self._hook_stats["inject_skip"] += 1
            return ok
        except Exception as exc:  # 钩子内绝不抛出
            self._hook_stats["inject_error"] += 1
            logger.warning("[agent_memory] 注入钩子异常(已忽略): %s", exc)
            return False

    # ------------------------------------------------------------------
    # Stage 2 核心二：后置轮次记录（Turn Record）
    # ------------------------------------------------------------------
    def _should_skip_record(self, session_id: str, fingerprint: tuple) -> bool:
        """短窗口去重：同一会话相同内容在窗口内只记一次，抵御宿主重复回调。"""
        window = _safe_float(
            self._section("retrieval").get("record_dedup_window"),
            DEFAULT_RECORD_DEDUP_WINDOW,
        )
        last = self._last_recorded.get(session_id)
        if not last:
            return False
        last_fp, last_ts = last
        if last_fp != fingerprint:
            return False
        try:
            return (time.monotonic() - float(last_ts)) <= window
        except Exception:
            return False

    async def capture_turn_from_response(
        self, event: Any = None, resp: Any = None, bot_text_override: str = ""
    ) -> bool:
        """
        后置钩子核心：抽取「用户输入 + Assistant 回复」→ 推入多轮缓冲池。

        过滤规则：
        - resp.is_chunk=True：流式分片，跳过（等最终完整响应）；
        - resp.tools_call_name 非空：工具调用轮，跳过（等最终自然语言回复）。
        """
        try:
            if not self.enable_memory or self.summarizer is None:
                self._hook_stats["record_skip"] += 1
                return False

            if resp is not None:
                if bool(_safe_attr(resp, "is_chunk", False)):
                    self._hook_stats["record_skip"] += 1
                    return False
                if _safe_attr(resp, "tools_call_name", None):
                    self._hook_stats["record_skip"] += 1
                    return False

            session_id = _extract_session_id(event, None)
            user_text = _extract_user_query(event, None)
            bot_text = str(bot_text_override or "").strip() or _extract_assistant_text(resp)

            if not session_id or (not user_text and not bot_text):
                self._hook_stats["record_skip"] += 1
                return False

            fingerprint = (user_text, bot_text)
            if self._should_skip_record(session_id, fingerprint):
                self._hook_stats["record_skip"] += 1
                logger.debug("[agent_memory] 轮次重复回调已忽略 | session=%s", session_id)
                return False

            # record_turn 内部为纯内存缓冲（零 IO），await 不会阻塞对话链路
            ok = await self.record_turn(session_id, user_text, bot_text)
            if ok:
                self._hook_stats["record_ok"] += 1
                self._last_recorded[session_id] = (fingerprint, time.monotonic())
                logger.debug(
                    "[agent_memory] 已记录 1 轮 | session=%s | user=%.20s | bot=%.20s",
                    session_id,
                    user_text,
                    bot_text,
                )
            else:
                self._hook_stats["record_skip"] += 1
            return ok
        except Exception as exc:  # 钩子内绝不抛出
            self._hook_stats["record_error"] += 1
            logger.warning("[agent_memory] 轮次记录钩子异常(已忽略): %s", exc)
            return False

    # ------------------------------------------------------------------
    # Stage 2 钩子装配：双向拦截
    # 1) on_llm_request       → 前置记忆注入（写 req.system_prompt）
    # 2) on_llm_response      → 后置轮次记录（读 resp.completion_text）
    # 3) on_decorating_result → 仅在缺少 on_llm_response 时作为兜底记录点
    # 说明：三者均按宿主能力逐个装配，任一缺失不影响插件加载；
    #      钩子内部全量 try-except，异常绝不外抛污染主对话链路。
    # ------------------------------------------------------------------
    if _ON_LLM_REQUEST is not None:  # pragma: no branch

        @_ON_LLM_REQUEST()  # type: ignore[misc]
        async def on_llm_request(self, event: Any = None, req: Any = None) -> None:
            """LLM 请求前置：检索分层记忆并注入 System Prompt 前部。"""
            await self.inject_memory_into_request(event, req)

    if _ON_LLM_RESPONSE is not None:  # pragma: no branch

        @_ON_LLM_RESPONSE()  # type: ignore[misc]
        async def on_llm_response(self, event: Any = None, resp: Any = None) -> None:
            """LLM 响应后置：把「用户输入 + 机器人回复」写入轮次缓冲池。"""
            await self.capture_turn_from_response(event, resp)

    elif _ON_DECORATING_RESULT is not None:  # pragma: no cover - 兜底分支

        @_ON_DECORATING_RESULT()  # type: ignore[misc]
        async def on_decorating_result(self, event: Any = None) -> None:
            """
            兜底：宿主未提供 on_llm_response 时，从待发送消息链中取纯文本记录轮次。
            """
            try:
                bot_text = _extract_text_from_event_result(event)
                if not bot_text:
                    return
                await self.capture_turn_from_response(
                    event, None, bot_text_override=bot_text
                )
            except Exception as exc:  # pragma: no cover
                self._hook_stats["record_error"] += 1
                logger.warning(
                    "[agent_memory] on_decorating_result 兜底钩子异常(已忽略): %s", exc
                )
