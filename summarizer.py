# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 异步记忆总结与分层晋升 Worker (summarizer.py)
负责异步消费对话事件，执行：
1. L1 原始对话轮次记录
2. 关键事实提取（LLM 总结端点 + 100% 健壮规则分句回退）
3. 向量化与 L2 余弦相似度去重融合 (>0.88 融合更新，否则新建)
4. L3 置顶晋升判定（核心偏好/约束关键词或 importance >= 0.85 标记 pinned=1）
5. 全链路异常捕获与 task_dlq 死信安全兜底
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import struct
import time
import uuid
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

from astrbot.api import logger

try:
    from .models_client import EmbeddingClient, RerankClient, cosine_similarity
except ImportError:  # pragma: no cover
    from models_client import EmbeddingClient, RerankClient, cosine_similarity  # type: ignore

try:
    from .evaluator import EvaluatorGate, GateReport
except ImportError:  # pragma: no cover
    from evaluator import EvaluatorGate, GateReport  # type: ignore

try:
    from .storage import StorageManager  # type: ignore
except ImportError:
    try:
        from .storage import MemoryStorage as StorageManager
    except ImportError:  # pragma: no cover
        from storage import MemoryStorage as StorageManager  # type: ignore

# P0：系统噪声过滤（写入/抽取入口统一拦截系统注入块与后台回调噪声）
try:
    from .noise_filter import get_noise_filter
except ImportError:  # pragma: no cover
    try:
        from noise_filter import get_noise_filter  # type: ignore
    except ImportError:  # pragma: no cover - 极端兜底：无过滤器时退化为直通
        def get_noise_filter():  # type: ignore
            class _Passthrough:
                def inspect(self, text):
                    class _V:
                        original = "" if text is None else str(text)
                        cleaned_text = original
                        is_noise = False
                        hits: list = []
                        removed_chars = 0

                    return _V()

            return _Passthrough()

# 方案 B：FTS5 预分词工具（用于写入路径同步倒排索引文本）
try:
    from .storage import fts_preprocess as _fts_preprocess
except ImportError:  # pragma: no cover
    try:
        from storage import fts_preprocess as _fts_preprocess  # type: ignore
    except ImportError:

        def _fts_preprocess(text: Any) -> str:  # type: ignore
            """FTS5 不可用时的空实现，保证写入路径零依赖。"""
            return ""


L3_PIN_KEYWORDS: Tuple[str, ...] = (
    "喜欢",
    "讨厌",
    "偏好",
    "必须",
    "严禁",
    "密码",
    "习惯",
)


# ---------------------------------------------------------------------------
# 方案 C：轻量规则三元组抽取（LLM 不可用时的 100% 本地兜底）
# ---------------------------------------------------------------------------
# 每条规则为 (正则, 关系类型, 默认主语, 主语类型, 宾语类型)：
# - 正则第 1 个捕获组为 tail（实体宾语）；
# - 主语固定归一化为「主人」，与用户画像语义对齐；
# - 规则按顺序匹配，同一句可命中多条（如「我住在苏州，喜欢美式咖啡」会被拆成两句后各命中）。
_TRIPLE_RULES: Tuple[Tuple[str, str, str, str, str], ...] = (
    # 偏好 / 口味（吃掉「喝/吃/看」等动词，确保 tail 是纯实体）
    (r"(?:我|主人|用户|本人)?\s*(?:很|超|特别|最|非常|一直)?\s*(?:喜欢|喜爱|偏好|爱喝|爱吃|爱看|爱好)\s*(?:喝|吃|看|玩|听|读|用|穿)?\s*(.+)", "偏好", "主人", "person", "object"),
    # 厌恶 / 忌口
    (r"(?:我|主人|用户|本人)?\s*(?:不喜欢|讨厌|厌恶|不吃|不喝|反感|忌口)\s*(?:吃|喝|看|听|用|穿)?\s*(.+)", "厌恶", "主人", "person", "object"),
    # 居住地
    (r"(?:我|主人|用户|本人)?\s*(?:住在|居住在|居住于|定居在|定居于|落户在|老家在)\s*(.+)", "居住于", "主人", "person", "place"),
    # 姓名
    (r"(?:我的名字是|我的名字叫|我叫|本名是|昵称是)\s*(.+)", "名为", "主人", "person", "person"),
    # 职业 / 身份
    (r"(?:我的职业是|我是一名|我是一个|我的工作是|我的职位是)\s*(.+)", "职业为", "主人", "person", "role"),
    # 生日 / 出生
    (r"(?:我的生日是|我生日是|我出生于|我出生在)\s*(.+)", "生日为", "主人", "person", "date"),
    # 工号 / 编号类强标识
    (r"(?:我的工号是|工号是|我的编号是|我的身份证号是|我的手机号是)\s*(.+)", "工号为", "主人", "person", "id"),
    # 宠物 / 持有物
    (r"(?:我养了|我有一只|我有一台|我有一辆|我有一块|我有)\s*(.+)", "拥有", "主人", "person", "object"),
    # 常用工具 / 器材
    (r"(?:我使用|我常用|我用的是|我习惯用|我一直用)\s*(.+)", "使用", "主人", "person", "object"),
    # 工作单位
    (r"(?:我在|我就职于|我工作在|任职于)\s*(.+?)\s*(?:工作|上班|任职|就职)", "工作于", "主人", "person", "org"),
)

# tail 中若出现这些疑问/指代片段，说明不是稳定事实，直接丢弃
_TRIPLE_TAIL_BLOCKLIST: Tuple[str, ...] = (
    "什么",
    "谁",
    "哪里",
    "怎么",
    "为什么",
    "吗",
    "呢",
    "？",
    "?",
    "你",
    "他",
    "她",
    "它",
    "们",
)


def _normalize_fact_key(text: Any) -> str:
    """归一化事实文本作为映射键（去空白与中英文标点，统一小写）。"""
    return re.sub(r"[\s，。！？、；：,.!?;:~～\-—_\"'“”‘’()（）\[\]【】]+", "", str(text or "")).lower()


def _pack_embedding(vec: Sequence[float]) -> Optional[bytes]:
    """将浮点向量序列化为紧凑的 float32 bytes（若全零或为空则返回 None）"""
    if not vec:
        return None
    try:
        floats = [float(x) for x in vec]
        if all(abs(x) <= 1e-12 for x in floats):
            return None
        return struct.pack(f"<{len(floats)}f", *floats)
    except Exception:
        return None


def _unpack_embedding(blob: Any) -> List[float]:
    """从 SQLite BLOB (float32 bytes 或 JSON bytes/str) 安全反序列化向量"""
    if not blob:
        return []
    try:
        if isinstance(blob, (bytes, bytearray, memoryview)):
            raw = bytes(blob)
            if raw.startswith(b"["):
                return [float(x) for x in json.loads(raw.decode("utf-8"))]
            if len(raw) % 4 == 0:
                cnt = len(raw) // 4
                return list(struct.unpack(f"<{cnt}f", raw))
        if isinstance(blob, str) and blob.strip().startswith("["):
            return [float(x) for x in json.loads(blob)]
    except Exception:
        pass
    return []


class MemorySummarizer:
    """异步对话记忆总结与分层晋升后台 Worker"""

    def __init__(
        self,
        storage: StorageManager,
        embedding_client: EmbeddingClient,
        rerank_client: Optional[RerankClient] = None,
        config: Optional[ dict ] = None,
        llm_call: Optional[Callable[[List[Dict[str, str]], str], Awaitable[str]]] = None,
        evaluator_llm_client: Optional[
            Callable[[List[Dict[str, str]], str], Awaitable[str]]
        ] = None,
    ) -> None:
        self.storage = storage
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client
        self.config: Dict[str, Any] = config or {}

        # ---- P0：宿主服务商注入的 LLM 调用能力（None 时回退 HTTP / 规则）----
        # 签名：async (messages: [{"role","content"}], model: str) -> str
        self._llm_call = llm_call
        self._evaluator_llm_client = evaluator_llm_client

        # ---- 自作判定门禁（廉价模型：默认 jev）----
        # 门禁自身的所有异常都在 EvaluatorGate 内部消化，这里无需再包 try。
        try:
            self.evaluator = EvaluatorGate(
                self.config, llm_client=self._evaluator_llm_client
            )
        except Exception as exc:  # pragma: no cover - 极端配置兜底
            logger.warning(
                "[MemorySummarizer] EvaluatorGate 初始化失败，判定门禁将被禁用: %s", exc
            )
            self.evaluator = EvaluatorGate({"evaluator": {"enable": False}})

        # ---- 后台消费队列 ----
        try:
            max_queue_size = int(self.config.get("summarizer_queue_maxsize", 1000))
        except (TypeError, ValueError):
            max_queue_size = 1000
        max_queue_size = max(1, max_queue_size)
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._schema_checked: bool = False

        # ---- 多轮缓冲池 (turn window batching) ----
        # 结构: {session_id: [ {"user_text": ..., "assistant_text": ..., "enqueued_at": ...}, ... ]}
        # 说明: 所有缓冲操作均在同一事件循环内且关键区内无 await，
        #       因此天然原子，无需额外加锁。
        self._turn_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self.turn_interval: int = self._parse_turn_interval(self.config)

        # ---- 运行期统计（仅用于可观测性，不参与业务逻辑）----
        self._stat_turns_buffered: int = 0   # 累计进入缓冲池的轮次
        self._stat_batches_enqueued: int = 0  # 累计打包推入消费队列的批次数
        self._stat_turns_flushed: int = 0    # 累计通过 flush_* 强制触发的轮次
        # R-03 可观测性：批处理下每批应恰好 +1（用于验证无 N+1 风暴）
        self._stat_llm_extract_calls: int = 0      # 事实抽取调用次数
        self._stat_gate_calls: int = 0             # 判定门禁调用次数
        self._stat_embedding_batch_calls: int = 0  # 批量 Embedding 调用次数

        # ---- R-02：idle 超时刷盘（会话静默过久时主动析出，避免长期滞留内存）----
        self.idle_flush_seconds: int = self._parse_idle_flush(self.config)
        self._idle_task: Optional[asyncio.Task] = None

        # ---- R-06：总结 LLM 的持久 HTTP 连接池 ----
        self.llm_timeout: float = self._parse_llm_timeout(self.config)
        self._http: Any = None
        try:
            try:
                from .models_client import _HttpSessionOwner
            except ImportError:  # pragma: no cover - 脚本式导入
                from models_client import _HttpSessionOwner  # type: ignore
            self._http = _HttpSessionOwner(self.llm_timeout)
        except Exception as exc:  # pragma: no cover - 极端兜底
            logger.warning("[MemorySummarizer] HTTP 连接池初始化失败，将回退临时 Session: %s", exc)

        # ---- 方案 B-3：task_dlq 死信队列闭环（巡检 + 指数退避重试 + 状态流转）----
        self.dlq_enabled: bool = self._parse_bool(self.config, "dlq_retry_enabled", True)
        self.dlq_interval: float = self._parse_float(
            self.config, "dlq_retry_interval", 60.0, minimum=1.0
        )
        self.dlq_batch_size: int = self._parse_int(
            self.config, "dlq_batch_size", 10, minimum=1
        )
        self.dlq_base_backoff: float = self._parse_float(
            self.config, "dlq_base_backoff", 60.0, minimum=1.0
        )
        self.dlq_max_backoff: float = self._parse_float(
            self.config, "dlq_max_backoff", 3600.0, minimum=1.0
        )
        self.dlq_max_retries: int = self._parse_int(
            self.config, "dlq_max_retries", 3, minimum=1
        )
        self._dlq_task: Optional[asyncio.Task] = None
        self._dlq_inflight: set[str] = set()
        # 可观测性统计
        self._stat_dlq_retried: int = 0    # 累计被重新捞取的死信次数
        self._stat_dlq_completed: int = 0  # 重试成功转 completed
        self._stat_dlq_failed: int = 0     # 超过 max_retries 转 failed
        self._stat_dlq_rescheduled: int = 0  # 失败后重新排队

        # ---- 方案 C：L3 时序知识图谱抽取开关与阈值 ----
        graph_cfg: Dict[str, Any] = {}
        raw_graph_cfg = self.config.get("graph")
        if isinstance(raw_graph_cfg, dict):
            graph_cfg = raw_graph_cfg

        def _graph_setting(key: str, default: Any) -> Any:
            """图谱写侧配置读取：优先 graph 子节点，其次顶层同名键。"""
            if key in graph_cfg:
                return graph_cfg.get(key)
            return self.config.get(key, default)

        self.graph_enabled: bool = bool(_graph_setting("enable", True))
        try:
            self.graph_max_triples: int = max(
                1, int(_graph_setting("max_triples_per_fact", 6))
            )
        except (TypeError, ValueError):
            self.graph_max_triples = 6
        try:
            self.graph_confidence: float = float(
                _graph_setting("edge_confidence", 0.9)
            )
        except (TypeError, ValueError):
            self.graph_confidence = 0.9
        # 可观测性：累计写入三元组数（仅统计，不参与业务）
        self._stat_kg_triples: int = 0

    # ------------------------------------------------------------------
    # 配置解析辅助（容错：任何非法值都退回默认，绝不因配置抛异常）
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_bool(config: Dict[str, Any], key: str, default: bool) -> bool:
        """解析布尔配置，支持 true/false/1/0/yes/no/on/off 字符串。"""
        raw: Any = config.get(key)
        if raw is None or raw == "":
            summarizer_cfg = config.get("summarizer")
            if isinstance(summarizer_cfg, dict):
                raw = summarizer_cfg.get(key)
        if raw is None or raw == "":
            return default
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on", "y", "t")

    @staticmethod
    def _parse_float(
        config: Dict[str, Any], key: str, default: float, minimum: float = 0.0
    ) -> float:
        """解析浮点配置，非法/过低时安全回退。"""
        candidates: List[Any] = [config.get(key)]
        summarizer_cfg = config.get("summarizer")
        if isinstance(summarizer_cfg, dict):
            candidates.append(summarizer_cfg.get(key))
        for raw in candidates:
            if raw is None or raw == "":
                continue
            try:
                return max(minimum, float(raw))
            except (TypeError, ValueError):
                continue
        return default

    @staticmethod
    def _parse_int(
        config: Dict[str, Any], key: str, default: int, minimum: int = 0
    ) -> int:
        """解析整数配置，非法/过低时安全回退。"""
        candidates: List[Any] = [config.get(key)]
        summarizer_cfg = config.get("summarizer")
        if isinstance(summarizer_cfg, dict):
            candidates.append(summarizer_cfg.get(key))
        for raw in candidates:
            if raw is None or raw == "":
                continue
            try:
                return max(minimum, int(raw))
            except (TypeError, ValueError):
                continue
        return default

    @staticmethod
    def _parse_idle_flush(config: Dict[str, Any]) -> int:
        """
        解析 idle 刷盘阈值（秒），默认 180 秒；<=0 表示关闭该机制。

        支持顶层 ``summary_idle_flush_seconds`` 与 ``summarizer.idle_flush_seconds``。
        """
        candidates: List[Any] = [config.get("summary_idle_flush_seconds")]
        summarizer_cfg = config.get("summarizer")
        if isinstance(summarizer_cfg, dict):
            candidates.append(summarizer_cfg.get("idle_flush_seconds"))
        for raw in candidates:
            if raw is None or raw == "":
                continue
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                continue
        return 180

    @staticmethod
    def _parse_llm_timeout(config: Dict[str, Any]) -> float:
        """解析总结 LLM 超时（秒），默认 20 秒，非法值安全回退。"""
        raw: Any = config.get("llm_timeout")
        summarizer_cfg = config.get("summarizer")
        if raw in (None, "") and isinstance(summarizer_cfg, dict):
            raw = summarizer_cfg.get("timeout_seconds")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 20.0
        return value if value > 0 else 20.0

    @staticmethod
    def _coerce_turn_interval(raw: Any) -> Optional[int]:
        """
        单值校验：把任意入参转为合法的窗口阈值。

        合法条件为「可转 int 且 > 0」；其余（None / "" / 非数字 / <=0）
        一律返回 None，由调用方决定回退策略。

        :param raw: 待校验的候选值
        :return: 合法的正整数值，或 None 表示非法
        """
        if raw is None or raw == "":
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @classmethod
    def _parse_turn_interval(cls, config: Dict[str, Any]) -> int:
        """
        解析 summary_turn_interval（默认 5 轮）：
        优先读取顶层 key，其次读取 config["summarizer"]["summary_turn_interval"]，
        任何非法值（非数字/<=0）均安全回退为 5。
        """
        candidates: List[Any] = [config.get("summary_turn_interval")]
        summarizer_cfg = config.get("summarizer")
        if isinstance(summarizer_cfg, dict):
            candidates.append(summarizer_cfg.get("summary_turn_interval"))

        for raw in candidates:
            value = cls._coerce_turn_interval(raw)
            if value is not None:
                return value
        return 5

    def set_turn_interval(self, value: Any) -> bool:
        """
        热更新窗口阈值（面板保存配置时由 ``apply_runtime_config`` 调用）。

        行为约定：
        1. 复用 :meth:`_coerce_turn_interval` 的校验逻辑，非法值**不修改**
           当前阈值，返回 False，避免把运行中的窗口打成 0 或脏值；
        2. 合法值写入 ``self.turn_interval`` 并同步回 ``self.config``，
           使后续 ``get_buffer_stats()`` / 自检读到一致值；
        3. **存量缓冲不回滚**：已缓冲的轮次保留，下一轮 ``enqueue_dialogue``
           按新阈值判定，因此「调大阈值」时旧缓冲继续累积，「调小阈值」时
           若已有缓冲达到新阈值则立即打包，语义均为「下一次判定生效」。

        :param value: 候选阈值（int / 数字字符串等）
        :return: True 表示已生效；False 表示校验失败、保持原值
        """
        new_interval = self._coerce_turn_interval(value)
        if new_interval is None:
            logger.warning(
                "[MemorySummarizer] 拒绝非法 summary_turn_interval=%r，保持原值 %d",
                value,
                self.turn_interval,
            )
            return False

        old_interval = self.turn_interval
        self.turn_interval = new_interval
        # config 同步回写：面板/状态指令读取的是同一份快照，避免新旧值漂移
        try:
            self.config["summary_turn_interval"] = new_interval
        except Exception as exc:  # pragma: no cover - 极端只读映射兜底
            logger.debug("[MemorySummarizer] 回写 config 失败（忽略）: %s", exc)

        if old_interval != new_interval:
            logger.info(
                "[MemorySummarizer] summary_turn_interval 热更新: %d -> %d 轮",
                old_interval,
                new_interval,
            )
        return True

    def start(self) -> None:
        """启动后台异步消费 Task"""
        if self._running and self._worker_task and not self._worker_task.done():
            return
        self._running = True
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="agent_memory_summarizer_worker"
        )
        # R-02：启动 idle 超时刷盘守护（0 表示禁用）
        if self.idle_flush_seconds > 0 and (
            self._idle_task is None or self._idle_task.done()
        ):
            self._idle_task = asyncio.create_task(
                self._idle_sweeper_loop(), name="agent_memory_idle_sweeper"
            )
        # 方案 B-3：启动死信队列巡检重试守护（形成真正的 DLQ 闭环）
        if self.dlq_enabled and (self._dlq_task is None or self._dlq_task.done()):
            self._dlq_task = asyncio.create_task(
                self._dlq_retry_loop(), name="agent_memory_dlq_retry"
            )
        logger.info(
            "[MemorySummarizer] Async worker started (idle_flush=%ss, dlq_retry=%s).",
            self.idle_flush_seconds,
            f"{self.dlq_interval:.0f}s" if self.dlq_enabled else "off",
        )
    async def stop(self, flush: bool = True, drain_timeout: float = 5.0) -> None:
        """
        优雅停止后台异步消费 Task。

        :param flush: 停止前是否先强制冲洗所有会话的剩余轮次（默认 True，避免丢记忆）
        :param drain_timeout: 冲洗后等待队列消费完成的最大秒数
        """
        # 1. 先冲洗缓冲池，保证最后不足 N 轮的对话也能进入消费队列
        if flush:
            try:
                flushed = await self.flush_all()
                total = sum(flushed.values())
                if total:
                    logger.info(
                        "[MemorySummarizer] stop(): flushed %d pending turn(s) from %d session(s).",
                        total,
                        len(flushed),
                    )
            except Exception as exc:
                logger.warning("[MemorySummarizer] flush_all during stop failed: %s", exc)

        # 2. 尽力消费完队列中遗留任务（超时不阻塞卸载流程）
        if self._worker_task is not None and self._running:
            try:
                if not self._queue.empty():
                    await asyncio.wait_for(self._queue.join(), timeout=float(drain_timeout))
            except asyncio.TimeoutError:
                logger.warning(
                    "[MemorySummarizer] Queue drain timeout(%.1fs), %d task(s) left.",
                    float(drain_timeout),
                    self._queue.qsize(),
                )
            except Exception as exc:
                logger.warning("[MemorySummarizer] Queue drain error: %s", exc)

        # 3. 取消 Worker
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("[MemorySummarizer] Error while stopping worker: %s", exc)
            finally:
                self._worker_task = None

        # 4. 取消 idle 刷盘守护
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("[MemorySummarizer] Error while stopping idle sweeper: %s", exc)
            finally:
                self._idle_task = None

        # 4.1 方案 B-3：取消死信队列巡检守护
        if self._dlq_task is not None:
            self._dlq_task.cancel()
            try:
                await self._dlq_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning("[MemorySummarizer] Error while stopping dlq retry loop: %s", exc)
            finally:
                self._dlq_task = None
        self._dlq_inflight.clear()

        # 5. 释放 HTTP 连接池（R-06）
        await self.close_http()

        logger.info("[MemorySummarizer] Async worker stopped.")

    async def close_http(self) -> None:
        """释放总结 LLM 连接池与判官连接池（幂等，可重复调用）。"""
        if self._http is not None:
            try:
                await self._http.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("[MemorySummarizer] 关闭总结 HTTP 连接池异常: %s", exc)
        evaluator = getattr(self, "evaluator", None)
        if evaluator is not None and hasattr(evaluator, "close"):
            try:
                await evaluator.close()
            except Exception as exc:  # pragma: no cover
                logger.warning("[MemorySummarizer] 关闭判官 HTTP 连接池异常: %s", exc)

    async def _idle_sweeper_loop(self) -> None:
        """
        R-02：会话静默超时刷盘守护。

        会话在 ``idle_flush_seconds`` 内没有任何新轮次时，主动把缓冲池中的
        尾部轮次打包析出，避免“聊了 3 轮就再也没说话”导致记忆长期滞留内存。
        """
        interval = max(1.0, float(self.idle_flush_seconds))
        tick = max(1.0, min(30.0, interval / 2.0))
        while self._running:
            try:
                await asyncio.sleep(tick)
                if not self._running:
                    break
                now = time.time()
                stale_sessions = [
                    sid
                    for sid, turns in list(self._turn_buffers.items())
                    if turns
                    and (now - float(turns[0].get("enqueued_at") or now)) >= interval
                ]
                for sid in stale_sessions:
                    flushed = await self.flush_session(sid)
                    if flushed:
                        logger.info(
                            "[MemorySummarizer] idle %.0fs 触发刷盘 session=%s turns=%d",
                            interval,
                            sid,
                            flushed,
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - 守护线程绝不崩溃
                logger.warning("[MemorySummarizer] idle sweeper 异常（已忽略）: %s", exc)

    # ------------------------------------------------------------------
    # 方案 B-3：task_dlq 死信队列闭环（巡检 → 重投 → 指数退避 → 终态）
    # ------------------------------------------------------------------
    async def _dlq_retry_loop(self) -> None:
        """
        死信队列后台巡检协程。

        此前 task_dlq 是「只写不读」的黑洞，失败任务永久埋葬。本协程按
        ``dlq_retry_interval`` 周期扫描到期死信并尝试重新消费，使其形成闭环。
        """
        tick = max(1.0, min(30.0, self.dlq_interval / 2.0))
        logger.info(
            "[MemorySummarizer] DLQ 重试守护已启动 (interval=%.0fs, batch=%d, "
            "base_backoff=%.0fs, max_retries=%d)",
            self.dlq_interval,
            self.dlq_batch_size,
            self.dlq_base_backoff,
            self.dlq_max_retries,
        )
        while self._running:
            try:
                await asyncio.sleep(tick)
                if not self._running:
                    break
                result = await self.sweep_dlq_once()
                if result.get("scanned"):
                    logger.info(
                        "[MemorySummarizer] DLQ 巡检: scanned=%d completed=%d "
                        "rescheduled=%d failed=%d skipped=%d",
                        result["scanned"],
                        result["completed"],
                        result["rescheduled"],
                        result["failed"],
                        result["skipped"],
                    )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - 守护协程绝不崩溃
                logger.warning("[MemorySummarizer] DLQ 重试守护异常（已忽略）: %s", exc)

    def _fetch_due_dlq(self, now: float, limit: int) -> List[sqlite3.Row]:
        """拉取到期且仍有重试余量的 pending 死信任务（按到期时间升序）。"""
        if self.storage is None or not hasattr(self.storage, "connect"):
            return []
        try:
            conn = self.storage.connect()
            return conn.execute(
                """
                SELECT task_id, task_type, scope_key, payload_json, error_msg,
                       retry_count, max_retries, status, next_retry_at
                FROM task_dlq
                WHERE status = 'pending'
                  AND next_retry_at <= ?
                  AND retry_count < max_retries
                ORDER BY next_retry_at ASC
                LIMIT ?
                """,
                (float(now), int(limit)),
            ).fetchall()
        except Exception as exc:
            logger.debug("[MemorySummarizer] 拉取 DLQ 任务失败（忽略）: %s", exc)
            return []

    def _finalize_dlq(
        self,
        task_id: str,
        status: str,
        error_msg: Optional[str] = None,
        retry_count: Optional[int] = None,
        next_retry_at: Optional[float] = None,
    ) -> bool:
        """把单条死信任务推进到终态或新的等待态（幂等，失败仅记日志）。"""
        if self.storage is None or not hasattr(self.storage, "connect"):
            return False
        try:
            conn = self.storage.connect()
            now = time.time()
            sets = ["status = ?", "updated_at = ?"]
            params: List[Any] = [str(status), now]
            if retry_count is not None:
                sets.append("retry_count = ?")
                params.append(int(retry_count))
            if next_retry_at is not None:
                sets.append("next_retry_at = ?")
                params.append(float(next_retry_at))
            if error_msg is not None:
                sets.append("error_msg = ?")
                params.append(str(error_msg)[:1000])
            params.append(str(task_id))
            with conn:
                conn.execute(
                    f"UPDATE task_dlq SET {', '.join(sets)} WHERE task_id = ?", params
                )
            return True
        except Exception as exc:
            logger.warning("[MemorySummarizer] 更新 DLQ 状态失败 task=%s: %s", task_id, exc)
            return False

    def _fail_exhausted_dlq(self, now: Optional[float] = None) -> int:
        """
        兜底清理：把「仍为 pending 但重试次数已耗尽」的历史脏数据标记为 failed，
        避免其永远停留在扫描条件之外。返回处理条数。
        """
        if self.storage is None or not hasattr(self.storage, "connect"):
            return 0
        try:
            conn = self.storage.connect()
            with conn:
                cur = conn.execute(
                    """
                    UPDATE task_dlq
                    SET status = 'failed', updated_at = ?
                    WHERE status = 'pending' AND retry_count >= max_retries
                    """,
                    (float(now if now is not None else time.time()),),
                )
                count = int(cur.rowcount or 0)
            if count:
                self._stat_dlq_failed += count
                logger.info("[MemorySummarizer] %d 条超限 DLQ 任务已标记 failed", count)
            return count
        except Exception as exc:
            logger.debug("[MemorySummarizer] 清理超限 DLQ 失败（忽略）: %s", exc)
            return 0

    def _reschedule_dlq(self, row: sqlite3.Row, error_msg: str, now: float) -> str:
        """
        重试失败后重新排期：指数退避 or 直接终态。

        :return: ``"rescheduled"`` 或 ``"failed"``
        """
        task_id = str(row["task_id"])
        retry_count = int(row["retry_count"] or 0) + 1
        max_retries = int(row["max_retries"] or self.dlq_max_retries)
        if retry_count >= max_retries:
            self._finalize_dlq(
                task_id, "failed", error_msg=error_msg, retry_count=retry_count
            )
            self._stat_dlq_failed += 1
            logger.warning(
                "[MemorySummarizer] DLQ 达到最大重试次数，标记 failed task=%s retries=%d err=%s",
                task_id,
                retry_count,
                error_msg[:200],
            )
            return "failed"

        # 指数退避：base * 2^(retry_count-1)，并受 max_backoff 封顶
        delay = min(
            self.dlq_max_backoff,
            self.dlq_base_backoff * (2 ** (retry_count - 1)),
        )
        self._finalize_dlq(
            task_id,
            "pending",
            error_msg=error_msg,
            retry_count=retry_count,
            next_retry_at=now + delay,
        )
        self._stat_dlq_rescheduled += 1
        logger.info(
            "[MemorySummarizer] DLQ 重试失败，%.0fs 后再试 task=%s retries=%d/%d err=%s",
            delay,
            task_id,
            retry_count,
            max_retries,
            error_msg[:200],
        )
        return "rescheduled"

    async def sweep_dlq_once(self, now: Optional[float] = None) -> Dict[str, int]:
        """
        执行一轮死信队列巡检与消费（可由守护协程周期调用，也可手动/测试触发）。

        状态流转：
        - 消费成功 → ``completed``
        - 失败且仍有重试余量 → ``pending`` + 指数退避的 ``next_retry_at``
        - 失败且重试耗尽 → ``failed``

        :return: {"scanned","completed","rescheduled","failed","skipped"}
        """
        result = {
            "scanned": 0,
            "completed": 0,
            "rescheduled": 0,
            "failed": 0,
            "skipped": 0,
        }
        current = float(now if now is not None else time.time())

        # 0. 历史超限脏数据先收敛到 failed 终态
        result["failed"] += self._fail_exhausted_dlq(current)

        rows = self._fetch_due_dlq(current, self.dlq_batch_size)
        for row in rows:
            task_id = str(row["task_id"])
            if task_id in self._dlq_inflight:
                result["skipped"] += 1
                continue
            self._dlq_inflight.add(task_id)
            result["scanned"] += 1
            self._stat_dlq_retried += 1
            try:
                try:
                    payload = json.loads(str(row["payload_json"] or "{}"))
                except Exception as parse_exc:
                    self._finalize_dlq(
                        task_id, "failed", error_msg=f"payload 解析失败: {parse_exc}"
                    )
                    result["failed"] += 1
                    self._stat_dlq_failed += 1
                    continue

                if not isinstance(payload, dict) or not payload:
                    self._finalize_dlq(task_id, "failed", error_msg="payload 非法或为空")
                    result["failed"] += 1
                    self._stat_dlq_failed += 1
                    continue

                task_type = str(row["task_type"] or "")
                if task_type != "summarize_dialogue":
                    self._finalize_dlq(
                        task_id, "failed", error_msg=f"不支持的任务类型: {task_type}"
                    )
                    result["failed"] += 1
                    self._stat_dlq_failed += 1
                    continue

                # 关键：raise_on_error=True，让失败上抛给本方法统一做退避排期，
                # 避免 _process_batch 内部再写一条新 DLQ 造成「死信爆炸」。
                await self._handle_event(payload, raise_on_error=True)

                self._finalize_dlq(task_id, "completed")
                result["completed"] += 1
                self._stat_dlq_completed += 1
                logger.info("[MemorySummarizer] DLQ 重试成功 task=%s", task_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                outcome = self._reschedule_dlq(row, str(exc), current)
                result["failed" if outcome == "failed" else "rescheduled"] += 1
            finally:
                self._dlq_inflight.discard(task_id)

        return result

    def get_dlq_stats(self) -> Dict[str, Any]:
        """返回死信队列分布与运行期重试统计（可观测性）。"""
        stats: Dict[str, Any] = {
            "enabled": bool(self.dlq_enabled),
            "interval": self.dlq_interval,
            "base_backoff": self.dlq_base_backoff,
            "max_backoff": self.dlq_max_backoff,
            "max_retries": self.dlq_max_retries,
            "batch_size": self.dlq_batch_size,
            "inflight": len(self._dlq_inflight),
            "stat_retried": self._stat_dlq_retried,
            "stat_completed": self._stat_dlq_completed,
            "stat_rescheduled": self._stat_dlq_rescheduled,
            "stat_failed": self._stat_dlq_failed,
            "by_status": {},
        }
        if self.storage is not None and hasattr(self.storage, "connect"):
            try:
                conn = self.storage.connect()
                rows = conn.execute(
                    "SELECT status, COUNT(1) AS c FROM task_dlq GROUP BY status"
                ).fetchall()
                stats["by_status"] = {str(r["status"]): int(r["c"]) for r in rows}
            except Exception as exc:
                logger.debug("[MemorySummarizer] 统计 DLQ 失败（忽略）: %s", exc)
        return stats

    async def enqueue_dialogue(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> bool:
        """
        将单轮对话实时落盘 L1 并累积进会话缓冲池。

        行为：
        - R-02：先同步写入 SQLite ``l1_docs``，未满阈值即断电/重启也不丢原始对话；
        - 每轮按 session_id 累积到内存缓冲池；
        - 仅当该会话累积轮次达到 summary_turn_interval 时，才把这 N 轮
          打包成一个 batch 事件推入后台总结消费队列，减少 LLM 调用次数；
        - 不足 N 轮的尾部对话由 flush_session() / flush_all() / idle 超时强制打包。

        :param assistant_text: 机器人回复（即 bot_text）
        :return: True 表示该轮已被接受（进入缓冲或成功打包入队）
        """
        sid = str(session_id or "").strip()
        user = str(user_text or "").strip()
        bot = str(assistant_text or "").strip()
        if not sid or (not user and not bot):
            return False

        # ---- P0：系统噪声拦截（写入入口）----
        # 宿主机/子代理/其它插件会往会话里注入系统块（后台任务回调、项目大盘、
        # RAG 标记等）。这些内容必须在落库前剔除，否则会污染 L1/L2 并被反复召回。
        try:
            nf = get_noise_filter()
            u_verdict = nf.inspect(user)
            b_verdict = nf.inspect(bot)
            if u_verdict.is_noise and b_verdict.is_noise:
                logger.debug(
                    "[MemorySummarizer] 整轮判定为系统噪声，已丢弃 session=%s hits=%s",
                    sid,
                    sorted(set(u_verdict.hits) | set(b_verdict.hits)),
                )
                return False
            user = "" if u_verdict.is_noise else u_verdict.cleaned_text
            bot = "" if b_verdict.is_noise else b_verdict.cleaned_text
            if not user and not bot:
                return False
        except Exception as exc:  # pragma: no cover - 过滤器异常绝不阻断写入
            logger.debug("[MemorySummarizer] 噪声过滤异常（忽略）: %s", exc)

        # R-02：先实时落盘 L1 原始轮次，保证“未满 N 轮就断电/重启”也不丢对话
        l1_doc_id = self._record_l1_dialogue(sid, user, bot, time.time())

        turn = {
            "user_text": user,
            "assistant_text": bot,
            "enqueued_at": time.time(),
            "l1_doc_id": l1_doc_id,
        }

        # 关键区内无 await，事件循环内天然原子
        buffer = self._turn_buffers.setdefault(sid, [])
        buffer.append(turn)
        self._stat_turns_buffered += 1

        # 未达阈值：安静地留在缓冲池里等待下一轮
        if len(buffer) < self.turn_interval:
            logger.debug(
                "[MemorySummarizer] Buffered turn %d/%d for session=%s",
                len(buffer),
                self.turn_interval,
                sid,
            )
            return True

        # 达到阈值：整窗打包送入后台消费队列
        turns = buffer[: self.turn_interval]
        del buffer[: self.turn_interval]
        if not buffer:
            self._turn_buffers.pop(sid, None)

        return self._pack_and_enqueue(sid, turns, reason="threshold")

    def _pack_and_enqueue(
        self, session_id: str, turns: List[Dict[str, Any]], reason: str = "threshold"
    ) -> bool:
        """
        将若干轮对话打包为一个 batch 事件推入后台消费队列。

        :param reason: "threshold"（达到轮次阈值）或 "flush"（强制冲洗）
        :return: True 表示成功入队；队列满等异常时降级写 DLQ 并返回 False
        """
        if not turns:
            return True

        batch_event: Dict[str, Any] = {
            "kind": "batch",
            "session_id": session_id,
            "turns": turns,
            "turn_count": len(turns),
            "reason": reason,
            "enqueued_at": time.time(),
        }

        try:
            self._queue.put_nowait(batch_event)
            self._stat_batches_enqueued += 1
            if reason == "flush":
                self._stat_turns_flushed += len(turns)
            logger.debug(
                "[MemorySummarizer] Batch enqueued session=%s turns=%d reason=%s",
                session_id,
                len(turns),
                reason,
            )
            return True
        except asyncio.QueueFull:
            logger.warning(
                "[MemorySummarizer] Summary queue full, degrading batch to DLQ for session=%s turns=%d",
                session_id,
                len(turns),
            )
            self._record_dlq(batch_event, "QueueFull: summary queue reached maxsize")
            return False
        except Exception as exc:
            logger.error("[MemorySummarizer] Failed to enqueue batch: %s", exc)
            self._record_dlq(batch_event, f"EnqueueFailed: {exc}")
            return False

    async def flush_session(self, session_id: str) -> int:
        """
        立即冲洗指定会话缓冲池中的所有剩余轮次（不足 summary_turn_interval 的尾部）。

        :return: 本次实际打包入队的轮次数（0 表示该会话无待总结内容）
        """
        sid = str(session_id or "").strip()
        if not sid:
            return 0

        turns = self._turn_buffers.pop(sid, None)
        if not turns:
            return 0

        ok = self._pack_and_enqueue(sid, turns, reason="flush")
        if not ok:
            # 入队失败（队列满）时，把轮次放回缓冲池，避免静默丢数据
            self._turn_buffers.setdefault(sid, []).extend(turns)
            return 0

        logger.info(
            "[MemorySummarizer] flush_session(%s): %d turn(s) flushed.", sid, len(turns)
        )
        return len(turns)

    async def flush_all(self) -> Dict[str, int]:
        """
        立即冲洗所有会话的剩余轮次（会话结束 / 插件关闭 / 服务重启前调用）。

        :return: {session_id: 本次打包入队的轮次数}
        """
        results: Dict[str, int] = {}
        for sid in list(self._turn_buffers.keys()):
            count = await self.flush_session(sid)
            if count:
                results[sid] = count
        if results:
            logger.info(
                "[MemorySummarizer] flush_all(): %d session(s), %d turn(s) flushed.",
                len(results),
                sum(results.values()),
            )
        return results

    def pending_turns(self, session_id: Optional[str] = None) -> int:
        """查询缓冲池中尚未总结的轮次数（session_id 为空时统计全部会话）。"""
        if session_id is not None:
            return len(self._turn_buffers.get(str(session_id).strip(), []))
        return sum(len(turns) for turns in self._turn_buffers.values())

    def get_buffer_stats(self) -> Dict[str, Any]:
        """返回缓冲池与打包统计快照，供状态指令 / 健康检查使用。"""
        return {
            "turn_interval": self.turn_interval,
            "active_sessions": len(self._turn_buffers),
            "pending_turns": self.pending_turns(),
            "pending_by_session": {
                sid: len(turns) for sid, turns in self._turn_buffers.items()
            },
            "queue_size": self._queue.qsize(),
            "queue_maxsize": self._queue.maxsize,
            "stat_turns_buffered": self._stat_turns_buffered,
            "stat_batches_enqueued": self._stat_batches_enqueued,
            "stat_turns_flushed": self._stat_turns_flushed,
            "stat_llm_extract_calls": self._stat_llm_extract_calls,
            "stat_gate_calls": self._stat_gate_calls,
            "stat_embedding_batch_calls": self._stat_embedding_batch_calls,
            "idle_flush_seconds": self.idle_flush_seconds,
            "running": self._running,
        }

    async def _worker_loop(self) -> None:
        """后台循环取出队列中的对话事件并执行总结处理，包含异常捕获与死信降级"""
        while self._running:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[MemorySummarizer] Queue get error: %s", exc)
                await asyncio.sleep(0.5)
                continue

            try:
                await self._handle_event(event)
            except asyncio.CancelledError:
                self._record_dlq(event, "WorkerCancelledDuringProcess")
                raise
            except Exception as exc:
                logger.error(
                    "[MemorySummarizer] Unexpected error in worker loop for session=%s: %s",
                    event.get("session_id", ""),
                    exc,
                    exc_info=True,
                )
                self._record_dlq(event, str(exc))
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

    async def _handle_event(
        self, event: Dict[str, Any], raise_on_error: bool = False
    ) -> None:
        """
        分发单个队列事件。

        - kind == "batch"：R-03 修复——整批 N 轮合并为**一个 Context Window**，
          只调用 1 次事实抽取 LLM、1 次判定门禁、1 次批量 Embedding，
          彻底消除原先 ``for turn in turns`` 的 N+1 调用风暴；
        - 其他格式（兼容历史的单轮事件）：按单轮批处理路径处理。

        :param raise_on_error: 方案 B-3 死信重试专用——为 True 时异常向上抛出，
            由 :meth:`sweep_dlq_once` 统一做退避排期，避免重复写入新的 DLQ 记录。
        """
        if not isinstance(event, dict):
            logger.warning("[MemorySummarizer] Skip invalid event type: %s", type(event))
            return

        if event.get("kind") != "batch":
            await self._process_batch(
                session_id=event.get("session_id", ""),
                turns=[
                    {
                        "user_text": event.get("user_text", ""),
                        "assistant_text": event.get("assistant_text", ""),
                        "enqueued_at": event.get("enqueued_at", time.time()),
                        "l1_doc_id": event.get("l1_doc_id", ""),
                    }
                ],
                reason="single",
                raise_on_error=raise_on_error,
            )
            return

        session_id = str(event.get("session_id", "") or "").strip()
        turns = event.get("turns")
        if not session_id or not isinstance(turns, list) or not turns:
            logger.warning("[MemorySummarizer] Skip malformed batch event: %r", event)
            return

        await self._process_batch(
            session_id=session_id,
            turns=turns,
            reason=str(event.get("reason", "threshold") or "threshold"),
            raise_on_error=raise_on_error,
        )

    def _ensure_storage_columns(self, conn: sqlite3.Connection) -> None:
        """幂等检查并补齐 l1_docs / l2_atoms 的扩展列（如 pinned, tier 等）"""
        if self._schema_checked:
            return
        extended_cols: Dict[str, List[Tuple[str, str]]] = {
            "l1_docs": [
                ("session_id", "TEXT NOT NULL DEFAULT ''"),
                ("persona_id", "TEXT NOT NULL DEFAULT ''"),
                ("importance", "REAL NOT NULL DEFAULT 0.5"),
                ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
            ],
            "l2_atoms": [
                ("persona_id", "TEXT NOT NULL DEFAULT ''"),
                ("atom_type", "TEXT NOT NULL DEFAULT 'fact'"),
                ("decay_type", "TEXT NOT NULL DEFAULT 'exponential'"),
                ("ttl_days", "REAL NOT NULL DEFAULT 30.0"),
                ("reinforcement_count", "INTEGER NOT NULL DEFAULT 0"),
                ("pinned", "INTEGER NOT NULL DEFAULT 0"),
                ("tier", "INTEGER NOT NULL DEFAULT 2"),
                ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
                # 自作判定门禁：冲突废弃标记
                ("status", "TEXT NOT NULL DEFAULT 'active'"),
                ("superseded_by", "TEXT NOT NULL DEFAULT ''"),
                ("superseded_at", "REAL NOT NULL DEFAULT 0.0"),
                # 方案 B-1：FTS5 预分词文本列（触发器据此维护倒排索引）
                ("fts_text", "TEXT NOT NULL DEFAULT ''"),
            ],
        }
        for table, cols in extended_cols.items():
            cur = conn.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cur.fetchall()}
            for col_name, col_def in cols:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")
        self._schema_checked = True

    def _record_l1_dialogue(
        self, session_id: str, user_text: str, assistant_text: str, now: float
    ) -> str:
        """
        记录 L1 原始对话轮次（R-02：实时落盘，异常安全不阻断主链路）。

        返回 l1_docs.doc_id；写入失败时返回空字符串（调用方仍可继续缓冲/总结）。
        """
        doc_id = f"l1_{uuid.uuid4().hex[:16]}"
        dialogue_content = f"User: {user_text}\nAssistant: {assistant_text}".strip()
        title = (user_text or assistant_text)[:48].replace("\n", " ")

        try:
            if hasattr(self.storage, "record_l1_dialogue") and callable(
                getattr(self.storage, "record_l1_dialogue")
            ):
                self.storage.record_l1_dialogue(session_id, user_text, assistant_text)
                return doc_id

            conn = self.storage.connect()
            self._ensure_storage_columns(conn)
            meta_json = json.dumps(
                {
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                    # 是否已析出为 L2 原子（析出后置 True）
                    "l2_processed": False,
                },
                ensure_ascii=False,
            )
            with conn:
                conn.execute(
                    """
                    INSERT INTO l1_docs (
                        doc_id, scope_key, doc_type, title, content,
                        version, updated_at, created_at,
                        session_id, importance, metadata
                    ) VALUES (?, ?, 'dialogue_turn', ?, ?, 1, ?, ?, ?, 0.5, ?)
                    """,
                    (
                        doc_id,
                        session_id,
                        title,
                        dialogue_content,
                        now,
                        now,
                        session_id,
                        meta_json,
                    ),
                )
        except Exception as exc:
            logger.error(
                "[MemorySummarizer] L1 实时落盘失败 session=%s（不影响后续总结）: %s",
                session_id,
                exc,
                exc_info=True,
            )
            return ""
        return doc_id

    def _mark_l1_processed(self, doc_ids: Sequence[str], now: float) -> int:
        """
        R-02：L2 析出完成后回写 L1 状态（metadata.l2_processed=True）。

        意义：既能观测“哪些原始轮次已完成沉淀”，也避免重复析出；
        失败仅记日志，绝不影响已落库的 L2 记忆。
        """
        ids = [str(d) for d in (doc_ids or []) if str(d or "").strip()]
        if not ids:
            return 0
        updated = 0
        try:
            conn = self.storage.connect()
            self._ensure_storage_columns(conn)
            with conn:
                for doc_id in ids:
                    row = conn.execute(
                        "SELECT metadata FROM l1_docs WHERE doc_id = ?", (doc_id,)
                    ).fetchone()
                    if row is None:
                        continue
                    try:
                        meta = json.loads(row["metadata"] or "{}")
                        if not isinstance(meta, dict):
                            meta = {}
                    except Exception:
                        meta = {}
                    if meta.get("l2_processed") is True:
                        continue
                    meta["l2_processed"] = True
                    meta["l2_processed_at"] = now
                    conn.execute(
                        "UPDATE l1_docs SET metadata = ?, updated_at = ? WHERE doc_id = ?",
                        (json.dumps(meta, ensure_ascii=False), now, doc_id),
                    )
                    updated += 1
        except Exception as exc:
            logger.debug("[MemorySummarizer] 回写 L1 析出状态失败（忽略）: %s", exc)
        return updated

    @staticmethod
    def _evaluate_pin_and_score(
        fact_text: str, base_score: Optional[float] = None
    ) -> Tuple[int, float, str]:
        """
        评估事实的重要度评分与 L3 置顶标识：
        若包含偏好/习惯/核心约束关键词或评分 >= 0.85，则标记 pinned=1，否则为 0。
        """
        has_pin_kw = any(kw in fact_text for kw in L3_PIN_KEYWORDS)
        score = float(base_score) if base_score is not None else (0.90 if has_pin_kw else 0.65)
        score = max(0.0, min(1.0, score))
        if has_pin_kw and score < 0.85:
            score = 0.90
        pinned = 1 if (has_pin_kw or score >= 0.85) else 0
        atom_type = "preference" if has_pin_kw else "fact"
        return pinned, score, atom_type

    def _rule_extract_facts(
        self, user_text: str, assistant_text: str, max_facts: int = 5
    ) -> List[Dict[str, Any]]:
        """
        100% 健壮的规则分句与关键事实抽取回退器：
        当未配置 LLM 或 LLM 调用异常时，对用户输入与关键回复进行分句、清洗与打分。

        :param max_facts: 单次最多产出事实条数（批处理时按轮数适度放宽）
        """
        try:
            limit = max(1, int(max_facts))
        except (TypeError, ValueError):
            limit = 5
        candidates: List[str] = []
        raw_parts = re.split(r"[。！？!?\n；;]+", (user_text or "").strip())
        for part in raw_parts:
            cleaned = part.strip()
            if len(cleaned) >= 3:
                candidates.append(cleaned)

        # 若 user_text 过短或未切出句子，但整体非空，保留完整 user_text
        if not candidates and user_text and user_text.strip():
            candidates.append(user_text.strip())

        # 若助手回复中包含明确记忆确认或偏好总结关键词，也提取核心句
        if not candidates and assistant_text and assistant_text.strip():
            for part in re.split(r"[。！？!?\n；;]+", assistant_text.strip()):
                cleaned = part.strip()
                if len(cleaned) >= 4:
                    candidates.append(cleaned)

        results: List[Dict[str, Any]] = []
        seen = set()
        for sent in candidates[:limit]:
            if sent in seen:
                continue
            seen.add(sent)
            pinned, score, atom_type = self._evaluate_pin_and_score(sent)
            results.append(
                {
                    "fact": sent,
                    "importance": score,
                    "pinned": pinned,
                    "atom_type": atom_type,
                }
            )
        return results

    async def _extract_facts(
        self, user_text: str, assistant_text: str
    ) -> List[Dict[str, Any]]:
        """单轮兼容入口：包装为 Context Window 后转交批量提取。"""
        context = f"[第1轮] 用户: {user_text}\n[第1轮] 助手: {assistant_text}"
        return await self._extract_facts_from_context(
            context=context,
            merged_user=user_text,
            merged_assistant=assistant_text,
            max_facts=5,
        )

    def _parse_llm_facts(
        self, content: str, max_facts: int = 5
    ) -> List[Dict[str, Any]]:
        """
        解析总结 LLM 返回的 JSON 事实数组（容错 markdown 代码块与前后废话）。

        统一收敛字符串项与对象项两种返回格式，并强制套用 L3 置顶打分规则；
        任何解析失败都返回空列表，由调用方优雅回退到规则抽取。
        """
        text = str(content or "")
        if not text.strip():
            return []
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []

        facts: List[Dict[str, Any]] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                fact_text = item.strip()
                pinned, score, atom_type = self._evaluate_pin_and_score(fact_text)
            elif isinstance(item, dict) and item.get("fact"):
                fact_text = str(item["fact"]).strip()
                raw_imp = item.get("importance")
                try:
                    base = float(raw_imp) if raw_imp is not None else None
                except (TypeError, ValueError):
                    base = None
                pinned, score, atom_type = self._evaluate_pin_and_score(fact_text, base)
            else:
                continue
            if not fact_text:
                continue
            entry: Dict[str, Any] = {
                "fact": fact_text,
                "importance": score,
                "pinned": pinned,
                "atom_type": atom_type,
            }
            # 方案 C：模型若直接给出结构化三元组，原样透传给图谱写入层
            if isinstance(item, dict) and item.get("triples"):
                entry["triples"] = item.get("triples")
            facts.append(entry)
        try:
            limit = max(1, int(max_facts))
        except (TypeError, ValueError):
            limit = 5
        return facts[:limit]

    async def _extract_facts_from_context(
        self,
        context: str,
        merged_user: str,
        merged_assistant: str,
        max_facts: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        从一个完整 Context Window 中一次性提取关键事实（R-03 核心）。

        - 配置了总结 LLM 时：**仅 1 次** /chat/completions 调用，上下文内含全部 N 轮；
        - 未配置或发生任何异常时：100% 优雅回退到规则分句（使用合并后的原文）；
        - 该函数在任何异常下都不会抛给调用方，保证记忆主链路不阻断。
        """
        summarizer_cfg = self.config.get("summarizer", {})
        if not isinstance(summarizer_cfg, dict):
            summarizer_cfg = {}

        api_base = str(
            self.config.get("llm_api_base")
            or summarizer_cfg.get("api_base")
            or ""
        ).strip().rstrip("/")
        api_key = str(
            self.config.get("llm_api_key")
            or summarizer_cfg.get("api_key")
            or ""
        ).strip()
        model = str(
            self.config.get("llm_model")
            or summarizer_cfg.get("model")
            or "gpt-4o-mini"
        ).strip()
        timeout_sec = float(summarizer_cfg.get("timeout_seconds", 20.0))

        # 抽取提示词（宿主 Provider 与手动 HTTP 两条路径共用，保证行为一致）
        extract_prompt = (
            "请从以下多轮对话中提取值得长期记住的客观事实、用户偏好、习惯或约束。"
            "返回 JSON 数组，每项格式为: "
            '{"fact": "简洁事实陈述", "importance": 0.0到1.0之间的小数, '
            '"triples": [{"head": "主体", "relation": "关系", "tail": "客体"}]}。'
            "triples 为可选字段，用于构建知识图谱（例如 "
            '{"head":"主人","relation":"偏好","tail":"美式咖啡"}）。\n'
            f"{context}"
        )

        # ---- P0 一级：宿主服务商 LLM（provider_id 选完即通电）----
        if self._llm_call is not None:
            try:
                content = await self._llm_call(
                    [{"role": "user", "content": extract_prompt}], model
                )
                llm_facts = self._parse_llm_facts(str(content or ""), max_facts)
                if llm_facts:
                    return llm_facts
                logger.debug(
                    "[MemorySummarizer] 宿主 Provider 抽取结果为空，降级到下一级。"
                )
            except Exception as exc:
                logger.warning(
                    "[MemorySummarizer] 宿主 Provider 抽取失败，降级到下一级: %s", exc
                )

        if api_base and api_key and aiohttp is not None:
            endpoint = (
                api_base
                if api_base.endswith("/chat/completions")
                else f"{api_base}/chat/completions"
            )
            prompt = (
                "请从以下多轮对话中提取值得长期记住的客观事实、用户偏好、习惯或约束。"
                "返回 JSON 数组，每项格式为: "
                '{"fact": "简洁事实陈述", "importance": 0.0到1.0之间的小数, '
                '"triples": [{"head": "主体", "relation": "关系", "tail": "客体"}]}。'
                "triples 为可选字段，用于构建知识图谱（例如 "
                '{"head":"主人","relation":"偏好","tail":"美式咖啡"}）。\n'
                f"{context}"
            )
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            try:
                client_timeout = aiohttp.ClientTimeout(total=timeout_sec)
                # R-06：复用持久连接池；仅连接池不可用时才回退临时 Session
                if self._http is not None:
                    session = await self._http.get()
                    async with session.post(
                        endpoint, headers=headers, json=payload, timeout=client_timeout
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            content = (
                                data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                            llm_facts = self._parse_llm_facts(content, max_facts)
                            if llm_facts:
                                return llm_facts
                else:  # pragma: no cover - 连接池初始化失败的极端兜底
                    async with aiohttp.ClientSession(timeout=client_timeout) as session:
                        async with session.post(
                            endpoint, headers=headers, json=payload
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                content = (
                                    data.get("choices", [{}])[0]
                                    .get("message", {})
                                    .get("content", "")
                                )
                                llm_facts = self._parse_llm_facts(content, max_facts)
                                if llm_facts:
                                    return llm_facts
            except Exception as exc:
                logger.warning(
                    "[MemorySummarizer] LLM fact extraction failed, falling back to rules: %s",
                    exc,
                )

        return self._rule_extract_facts(
            merged_user, merged_assistant, max_facts=max_facts
        )

    # ------------------------------------------------------------------
    # 方案 C：L3 时序知识图谱写入（实体三元组抽取 + 落库 + 软废弃联动）
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_triple_tail(raw: str, max_len: int = 24) -> str:
        """
        清洗三元组宾语：去标点/空白/量词尾缀，过滤疑问句与指代片段。

        返回空串表示该候选不是稳定事实（调用方应丢弃）。
        """
        tail = str(raw or "").strip()
        if not tail:
            return ""
        # 截断到首个句读/连接词，避免把整句都当成实体
        tail = re.split(r"[。！？!?；;，,、\n]|(?:并且|而且|还有|然后|所以|但是|因为)", tail)[0]
        tail = tail.strip(" \t\r\n\"'“”‘’()（）[]【】《》<>")
        # 去掉常见量词/语气尾缀
        tail = re.sub(r"(?:的|了|啊|呀|吧|哦|啦|呢|嘛|哈)+$", "", tail)
        if not tail:
            return ""
        # 疑问 / 指代 / 过长 → 不是稳定实体
        if any(bad in tail for bad in _TRIPLE_TAIL_BLOCKLIST):
            return ""
        if len(tail) > max_len:
            return ""
        return tail

    def _rule_extract_triples(self, text: str) -> List[Dict[str, str]]:
        """
        规则层三元组抽取：从一条事实文本中提取 ``(head, relation, tail)``。

        设计目标：零依赖、可解释、可控误报。宁可少抽，也不写脏数据到图谱。
        同一句可能命中多条规则（如「我是程序员，喜欢美式咖啡」），逐条独立抽取。
        """
        content = str(text or "").strip()
        if not content:
            return []

        triples: List[Dict[str, str]] = []
        seen: set[Tuple[str, str, str]] = set()
        # 按句读切分，逐句匹配，降低跨句误抽
        segments = [seg for seg in re.split(r"[。！？!?\n；;]+", content) if seg.strip()]
        if not segments:
            segments = [content]

        for seg in segments:
            for pattern, relation, head, head_type, tail_type in _TRIPLE_RULES:
                match = re.search(pattern, seg)
                if not match:
                    continue
                tail = self._normalize_triple_tail(match.group(1))
                if not tail or tail == head:
                    continue
                key = (head, relation, tail)
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    {
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                        "head_type": head_type,
                        "tail_type": tail_type,
                    }
                )
                if len(triples) >= self.graph_max_triples:
                    return triples
        return triples

    def _coerce_llm_triples(
        self, raw_triples: Any
    ) -> List[Dict[str, str]]:
        """
        归一化 LLM 返回的三元组（支持 dict / list / "head|relation|tail" 字符串）。

        任何非法结构一律静默丢弃，绝不因模型幻觉格式污染图谱。
        """
        if not raw_triples:
            return []
        items: List[Any]
        if isinstance(raw_triples, dict):
            items = [raw_triples]
        elif isinstance(raw_triples, (list, tuple)):
            items = list(raw_triples)
        else:
            return []

        out: List[Dict[str, str]] = []
        for item in items:
            head = relation = tail = ""
            head_type = tail_type = "concept"
            if isinstance(item, dict):
                head = str(item.get("head") or item.get("subject") or "").strip()
                relation = str(item.get("relation") or item.get("predicate") or "").strip()
                tail = str(item.get("tail") or item.get("object") or "").strip()
                head_type = str(item.get("head_type") or "concept")[:24]
                tail_type = str(item.get("tail_type") or "concept")[:24]
            elif isinstance(item, str):
                parts = [p.strip() for p in re.split(r"[|｜,，:：\t]", item) if p.strip()]
                if len(parts) >= 3:
                    head, relation, tail = parts[0], parts[1], parts[2]
            if not head or not relation or not tail:
                continue
            tail_clean = self._normalize_triple_tail(tail)
            if not tail_clean or tail_clean == head:
                continue
            out.append(
                {
                    "head": head[:32],
                    "relation": relation[:32],
                    "tail": tail_clean,
                    "head_type": head_type or "concept",
                    "tail_type": tail_type or "concept",
                }
            )
            if len(out) >= self.graph_max_triples:
                break
        return out

    def _upsert_kg_triples_for_atom(
        self,
        session_id: str,
        atom_id: str,
        fact_text: str,
        now: float,
        llm_triples: Any = None,
    ) -> int:
        """
        将一条记忆原子关联的三元组写入 kg_nodes / kg_edges。

        抽取来源优先级：LLM 结构化三元组 → 规则兜底，合并去重后落库。
        异常安全：图谱写入失败只记日志，绝不影响 L2 原子入库主链路。

        :return: 实际写入/强化成功的三元组条数。
        """
        if not self.graph_enabled:
            return 0
        if not hasattr(self.storage, "upsert_kg_triple"):
            return 0

        candidates: List[Dict[str, str]] = []
        candidates.extend(self._coerce_llm_triples(llm_triples))
        candidates.extend(self._rule_extract_triples(fact_text))

        if not candidates:
            return 0

        written = 0
        seen: set[Tuple[str, str, str]] = set()
        for triple in candidates:
            head = str(triple.get("head") or "").strip()
            relation = str(triple.get("relation") or "").strip()
            tail = str(triple.get("tail") or "").strip()
            if not head or not relation or not tail:
                continue
            key = (head, relation, tail)
            if key in seen:
                continue
            seen.add(key)
            try:
                result = self.storage.upsert_kg_triple(
                    scope_key=session_id,
                    head=head,
                    relation=relation,
                    tail=tail,
                    atom_id=atom_id,
                    fact_text=fact_text,
                    now=now,
                    head_type=str(triple.get("head_type") or "concept"),
                    tail_type=str(triple.get("tail_type") or "concept"),
                    confidence=self.graph_confidence,
                )
                if result:
                    written += 1
            except Exception as exc:  # pragma: no cover - 存储层已容错，这里双保险
                logger.debug(
                    "[MemorySummarizer] 图谱三元组写入异常（忽略） atom=%s triple=%s: %s",
                    atom_id,
                    key,
                    exc,
                )
            if len(seen) >= self.graph_max_triples:
                break

        self._stat_kg_triples += written
        if written:
            logger.debug(
                "[MemorySummarizer] 图谱写入 atom=%s triples=%d", atom_id, written
            )
        return written

    def _upsert_l2_atom(
        self,
        session_id: str,
        fact_item: Dict[str, Any],
        embedding_vec: Sequence[float],
        source_doc_id: str,
        now: float,
    ) -> Tuple[str, bool]:
        """
        将单条事实向量与现有 L2 记忆比对余弦相似度：
        - 若最大余弦相似度 > 0.88（或文本完全一致），执行融合更新并强化记忆；
        - 否则插入新的 L2/L3 记忆原子。
        返回 (atom_id, is_merged)
        """
        fact_text = str(fact_item["fact"]).strip()

        # ---- P0：L2 写入二次兜底 ----
        # 即使上游（LLM 抽取 / 规则抽取）产出的是噪声片段，也绝不允许落库。
        try:
            verdict = get_noise_filter().inspect(fact_text)
            if verdict.is_noise:
                logger.debug(
                    "[MemorySummarizer] 丢弃噪声事实 session=%s hits=%s",
                    session_id,
                    verdict.hits,
                )
                return "", False
            fact_text = verdict.cleaned_text
            if not fact_text:
                return "", False
        except Exception as exc:  # pragma: no cover
            logger.debug("[MemorySummarizer] 事实噪声过滤异常（忽略）: %s", exc)

        importance = float(fact_item["importance"])
        pinned = int(fact_item["pinned"])
        atom_type = str(fact_item.get("atom_type", "fact"))
        tier = 3 if pinned == 1 else 2

        conn = self.storage.connect()
        self._ensure_storage_columns(conn)

        cur = conn.execute(
            """
            SELECT atom_id, content, key_facts_json, importance,
                   access_count, reinforcement_count, pinned, embedding_blob
            FROM l2_atoms
            WHERE scope_key = ?
            ORDER BY updated_at DESC
            LIMIT 200
            """,
            (session_id,),
        )
        rows = cur.fetchall()

        best_match_row: Optional[sqlite3.Row] = None
        best_sim: float = 0.0
        has_valid_vec = bool(embedding_vec) and any(abs(x) > 1e-12 for x in embedding_vec)

        for row in rows:
            existing_content = str(row["content"] or "").strip()
            if existing_content == fact_text:
                best_match_row = row
                best_sim = 1.0
                break
            if has_valid_vec and row["embedding_blob"]:
                existing_vec = _unpack_embedding(row["embedding_blob"])
                if existing_vec:
                    sim = cosine_similarity(embedding_vec, existing_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_match_row = row

        emb_blob = _pack_embedding(embedding_vec)

        if best_match_row is not None and best_sim > 0.88:
            atom_id = str(best_match_row["atom_id"])
            old_imp = float(best_match_row["importance"] or 0.5)
            new_imp = min(1.0, max(old_imp, importance) + 0.02)
            old_pinned = int(best_match_row["pinned"] or 0)
            new_pinned = 1 if (old_pinned == 1 or pinned == 1 or new_imp >= 0.85) else 0
            new_tier = 3 if new_pinned == 1 else 2
            new_access = int(best_match_row["access_count"] or 0) + 1
            new_reinf = int(best_match_row["reinforcement_count"] or 0) + 1

            try:
                facts_list = json.loads(best_match_row["key_facts_json"] or "[]")
                if not isinstance(facts_list, list):
                    facts_list = []
            except Exception:
                facts_list = []
            if fact_text not in facts_list:
                facts_list.append(fact_text)

            # 方案 B-1：融合更新时同步预分词文本，触发器会据此刷新倒排索引
            merged_content = (
                fact_text
                if len(fact_text) >= len(str(best_match_row["content"] or ""))
                else str(best_match_row["content"] or "")
            )
            merged_fts_text = self._build_fts_text(merged_content, facts_list)

            with conn:
                conn.execute(
                    """
                    UPDATE l2_atoms
                    SET content = ?,
                        key_facts_json = ?,
                        importance = ?,
                        access_count = ?,
                        reinforcement_count = ?,
                        pinned = ?,
                        tier = ?,
                        embedding_blob = COALESCE(?, embedding_blob),
                        last_accessed_at = ?,
                        updated_at = ?,
                        fts_text = ?
                    WHERE atom_id = ?
                    """,
                    (
                        merged_content,
                        json.dumps(facts_list, ensure_ascii=False),
                        new_imp,
                        new_access,
                        new_reinf,
                        new_pinned,
                        new_tier,
                        emb_blob,
                        now,
                        now,
                        merged_fts_text,
                        atom_id,
                    ),
                )
            # 方案 B-2：融合后把最新向量同步进内存矩阵（增量覆盖，无重建）
            self._sync_embedding_cache(atom_id, embedding_vec)
            return atom_id, True

        atom_id = f"atom_{uuid.uuid4().hex[:16]}"
        meta_json = json.dumps(
            {"pinned": bool(pinned), "tier": tier, "atom_type": atom_type},
            ensure_ascii=False,
        )
        # 方案 B-1：新原子写入预分词文本，INSERT 触发器据此建立倒排条目
        new_fts_text = self._build_fts_text(fact_text, [fact_text])
        with conn:
            conn.execute(
                """
                INSERT INTO l2_atoms (
                    atom_id, scope_key, session_id, content,
                    topics_json, key_facts_json, sentiment, importance,
                    access_count, embedding_blob, source_type, source_ref,
                    last_accessed_at, created_at, updated_at,
                    atom_type, reinforcement_count, pinned, tier, metadata,
                    fts_text
                ) VALUES (?, ?, ?, ?, '[]', ?, 'neutral', ?, 1, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    atom_id,
                    session_id,
                    session_id,
                    fact_text,
                    json.dumps([fact_text], ensure_ascii=False),
                    importance,
                    emb_blob,
                    atom_type,
                    source_doc_id,
                    now,
                    now,
                    now,
                    atom_type,
                    pinned,
                    tier,
                    meta_json,
                    new_fts_text,
                ),
            )
        # 方案 B-2：新原子增量追加进内存矩阵（均摊 O(1)，不触发全量重建）
        self._sync_embedding_cache(atom_id, embedding_vec)
        return atom_id, False

    # ------------------------------------------------------------------
    # 方案 B：引擎增强所需的写入侧维护辅助
    # ------------------------------------------------------------------
    def _build_fts_text(self, content: Any, key_facts: Any = None) -> str:
        """
        生成 FTS5 预分词文本；FTS5 不可用时返回空串（写入零额外开销）。
        """
        try:
            if not getattr(self.storage, "fts5_enabled", False):
                return ""
            facts_text = ""
            if isinstance(key_facts, (list, tuple)):
                facts_text = " ".join(str(x) for x in key_facts)
            elif key_facts:
                facts_text = str(key_facts)
            return _fts_preprocess(f"{content or ''} {facts_text}".strip())
        except Exception as exc:  # pragma: no cover - 预分词失败不应影响入库
            logger.debug("[MemorySummarizer] 生成 fts_text 失败（忽略）: %s", exc)
            return ""

    def _sync_embedding_cache(self, atom_id: str, vector: Sequence[float]) -> None:
        """
        把写入/更新后的向量同步进内存矩阵缓存（失败仅记日志，不影响入库）。
        """
        cache = getattr(self.storage, "embedding_cache", None)
        if cache is None or not vector:
            return
        try:
            cache.upsert(atom_id, vector)
        except Exception as exc:  # pragma: no cover
            logger.debug("[MemorySummarizer] 同步向量矩阵缓存失败（忽略）: %s", exc)

    # ------------------------------------------------------------------
    # 自作判定门禁：冲突候选拉取与旧事实废弃
    # ------------------------------------------------------------------
    async def _run_evaluator_gate(
        self,
        session_id: str,
        facts: List[Dict[str, Any]],
        now: float,
    ) -> Optional[GateReport]:
        """
        执行自作判定门禁。

        双重容错：
        1. EvaluatorGate 内部已处理模型超时/异常，降级为规则判定；
        2. 这里再做一层 try-except：门禁自身崩溃时返回 None，
           调用方将按“不过滤、原样入库”处理，保证记忆主链路永不被阻断。
        """
        try:
            candidates = self._fetch_conflict_candidates(session_id)
            report = await self.evaluator.gate(session_id, facts, candidates)
            if report.degraded:
                logger.warning(
                    "[MemorySummarizer] 判定门禁降级运行 session=%s reason=%s report=%s",
                    session_id,
                    report.error,
                    report.summary(),
                )
            return report
        except Exception as exc:
            logger.error(
                "[MemorySummarizer] 判定门禁异常，已 fail-open 放行 session=%s: %s",
                session_id,
                exc,
                exc_info=True,
            )
            return None

    def _fetch_conflict_candidates(
        self, session_id: str, limit: int = 40
    ) -> List[Dict[str, Any]]:
        """
        拉取当前作用域内仍有效（status != superseded）的历史原子，供判官做冲突比对。

        异常安全：任何失败都返回空列表，冲突检测自动退化为“不检测”，
        绝不影响正常入库流程。
        """
        candidates: List[Dict[str, Any]] = []
        try:
            conn = self.storage.connect()
            self._ensure_storage_columns(conn)
            cur = conn.execute(
                """
                SELECT atom_id, content, importance, pinned, status
                FROM l2_atoms
                WHERE scope_key = ? AND COALESCE(status, 'active') != 'superseded'
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (session_id, int(max(1, limit))),
            )
            for row in cur.fetchall():
                content = str(row["content"] or "").strip()
                if not content:
                    continue
                candidates.append(
                    {
                        "atom_id": str(row["atom_id"]),
                        "content": content,
                        "importance": float(row["importance"] or 0.5),
                        "pinned": int(row["pinned"] or 0),
                    }
                )
        except Exception as exc:
            logger.debug("[MemorySummarizer] 冲突候选拉取失败（忽略）: %s", exc)
            return []
        return candidates

    def _apply_conflict_resolution(
        self,
        conflicts: Sequence[Dict[str, Any]],
        fact_atom_map: Dict[str, str],
        now: float,
    ) -> int:
        """
        将判官认定的冲突旧事实标记为 superseded（软废弃），并降低其重要度。

        - 不物理删除，保留可追溯性；
        - superseded_by 尽量指向本轮新事实的 atom_id；
        - 异常安全：失败仅记日志，不阻断入库。
        返回成功废弃条数。
        """
        if not conflicts:
            return 0

        applied = 0
        deprecated_ids: List[str] = []
        # 本轮新建/融合出的原子集合：冲突废弃必须跳过它们，避免“自我废弃”
        protected_ids = {str(v) for v in fact_atom_map.values() if v}
        try:
            conn = self.storage.connect()
            self._ensure_storage_columns(conn)
            with conn:
                for item in conflicts:
                    if not isinstance(item, dict):
                        continue
                    atom_id = str(item.get("atom_id") or "").strip()
                    if not atom_id:
                        continue
                    if atom_id in protected_ids:
                        # 该旧原子本轮已被融合更新（内容已更新为新事实），无需废弃
                        logger.debug(
                            "[MemorySummarizer] 冲突目标本轮已被融合更新，跳过废弃 atom=%s",
                            atom_id,
                        )
                        continue
                    # R-01：置顶记忆（pinned=1）拥有免死金牌，判官/规则均无权软废弃
                    if int(item.get("pinned") or 0) == 1:
                        logger.info(
                            "[MemorySummarizer] 冲突目标为 L3 置顶记忆，已拒绝废弃 atom=%s reason=%s",
                            atom_id,
                            item.get("reason", ""),
                        )
                        continue
                    replacement = fact_atom_map.get(
                        _normalize_fact_key(item.get("fact", "")), ""
                    )
                    cur = conn.execute(
                        """
                        UPDATE l2_atoms
                        SET status = 'superseded',
                            superseded_by = ?,
                            superseded_at = ?,
                            pinned = 0,
                            tier = 2,
                            importance = MIN(COALESCE(importance, 0.5), 0.1),
                            updated_at = ?
                        WHERE atom_id = ?
                          AND COALESCE(status, 'active') != 'superseded'
                          AND COALESCE(pinned, 0) = 0
                        """,
                        (replacement, now, now, atom_id),
                    )
                    if cur.rowcount:
                        applied += 1
                        deprecated_ids.append(atom_id)
                        logger.info(
                            "[MemorySummarizer] 冲突废弃旧事实 atom=%s reason=%s replacement=%s",
                            atom_id,
                            item.get("reason", ""),
                            replacement or "-",
                        )
        except Exception as exc:
            logger.error(
                "[MemorySummarizer] 冲突废弃落库失败（已忽略，不影响入库）: %s",
                exc,
                exc_info=True,
            )

        # 方案 C：软废弃联动失效图谱关系边（边标记 invalidated，权重归零，不物理删除）
        if deprecated_ids:
            try:
                if hasattr(self.storage, "invalidate_kg_edges_for_atoms"):
                    invalidated = self.storage.invalidate_kg_edges_for_atoms(
                        deprecated_ids, now
                    )
                    if invalidated:
                        logger.info(
                            "[MemorySummarizer] 图谱边联动失效 atoms=%d edges=%d",
                            len(deprecated_ids),
                            invalidated,
                        )
            except Exception as kg_exc:  # pragma: no cover - 双保险
                logger.warning(
                    "[MemorySummarizer] 图谱边联动失效异常（已忽略）: %s", kg_exc
                )
        return applied

    @staticmethod
    def _build_merged_context(
        turns: Sequence[Dict[str, Any]]
    ) -> Tuple[str, str, str]:
        """
        把 N 轮对话合并为一个统一 Context Window（R-03）。

        返回 ``(merged_user, merged_assistant, context)``：
        - merged_user / merged_assistant 用「。」拼接，供规则分句回退精确保留每轮语义；
        - context 带轮次标号，供 LLM 一次性阅读完整对话。
        """
        user_parts: List[str] = []
        assistant_parts: List[str] = []
        context_lines: List[str] = []
        for idx, turn in enumerate(turns, start=1):
            user = str(turn.get("user_text") or "").strip()
            assistant = str(turn.get("assistant_text") or "").strip()
            if user:
                user_parts.append(user)
            if assistant:
                assistant_parts.append(assistant)
            context_lines.append(f"[第{idx}轮] 用户: {user or '（空）'}")
            if assistant:
                context_lines.append(f"[第{idx}轮] 助手: {assistant}")
        return "。".join(user_parts), "。".join(assistant_parts), "\n".join(context_lines)

    async def _embed_texts_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        单次批量获取 Embedding（R-03：消除逐条请求）。

        仅当客户端不支持 ``get_embeddings`` 或批量调用异常时，才降级为逐条请求
        （此时向量质量不变，只是延迟变差，属于可用性兜底）。
        """
        clean = [str(t or "").strip() for t in texts]
        if not clean:
            return []
        if self.embedding_client is None:
            return [[] for _ in clean]

        batch_api = getattr(self.embedding_client, "get_embeddings", None)
        if callable(batch_api):
            try:
                self._stat_embedding_batch_calls += 1
                vectors = await batch_api(clean)
                if isinstance(vectors, list) and len(vectors) == len(clean):
                    return vectors
                logger.warning(
                    "[MemorySummarizer] 批量 Embedding 数量不匹配 (got=%s want=%d)，降级逐条。",
                    len(vectors) if isinstance(vectors, list) else type(vectors).__name__,
                    len(clean),
                )
            except Exception as exc:
                logger.warning(
                    "[MemorySummarizer] 批量 Embedding 失败，降级逐条: %s", exc
                )

        fallback: List[List[float]] = []
        for text in clean:
            try:
                fallback.append(await self.embedding_client.get_embedding(text))
            except Exception as exc:
                logger.debug("[MemorySummarizer] 单条 Embedding 失败（忽略）: %s", exc)
                fallback.append([])
        return fallback

    async def process_dialogue(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> None:
        """
        单轮兼容入口（保持既有调用契约，内部统一走批处理路径）。

        异常全部在 ``_process_batch`` 内消化并落 DLQ，本方法不会向外抛出。
        """
        await self._process_batch(
            session_id=session_id,
            turns=[
                {
                    "user_text": user_text,
                    "assistant_text": assistant_text,
                    "enqueued_at": time.time(),
                }
            ],
            reason="single",
        )

    async def _process_batch(
        self,
        session_id: str,
        turns: Sequence[Dict[str, Any]],
        reason: str = "threshold",
        raise_on_error: bool = False,
    ) -> None:
        """
        批处理核心（R-03）：N 轮对话 → 1 个 Context Window → 1 次 LLM + 1 次门禁 + 1 次批量向量。

        处理链路：
        1. L1：复用 enqueue 阶段已实时落盘的 doc_id，缺失时补写（兼容直连调用）；
        2. 合并 N 轮为统一上下文，**只调用一次** ``_extract_facts_from_context``；
        3. **只调用一次** ``_run_evaluator_gate``（价值过滤 + 冲突检测）；
        4. **只调用一次** 批量 Embedding，再逐条做 L2 相似度融合/新建；
        5. 冲突软废弃（置顶记忆受免死金牌保护）；
        6. 回写 L1 析出状态。
        全链路异常兜底：失败整批写入 task_dlq，绝不崩溃 Worker。
        """
        clean_session = str(session_id or "").strip()
        norm_turns: List[Dict[str, Any]] = []
        for turn in turns or []:
            if not isinstance(turn, dict):
                continue
            user = str(turn.get("user_text") or "").strip()
            assistant = str(turn.get("assistant_text") or "").strip()
            if not user and not assistant:
                continue
            norm_turns.append(
                {
                    "user_text": user,
                    "assistant_text": assistant,
                    "enqueued_at": float(turn.get("enqueued_at") or time.time()),
                    "l1_doc_id": str(turn.get("l1_doc_id") or "").strip(),
                }
            )

        if not clean_session or not norm_turns:
            return

        now = time.time()
        payload: Dict[str, Any] = {
            "kind": "batch",
            "session_id": clean_session,
            "turns": norm_turns,
            "reason": reason,
            "enqueued_at": now,
        }

        try:
            # ---- 1. L1：复用实时落盘记录，缺失则补写 ----
            l1_doc_ids: List[str] = []
            for turn in norm_turns:
                doc_id = turn.get("l1_doc_id") or ""
                if not doc_id:
                    doc_id = self._record_l1_dialogue(
                        clean_session,
                        turn["user_text"],
                        turn["assistant_text"],
                        now,
                    )
                    turn["l1_doc_id"] = doc_id
                if doc_id:
                    l1_doc_ids.append(doc_id)

            # ---- 2. 合并上下文 + 单次事实抽取 ----
            merged_user, merged_assistant, context = self._build_merged_context(norm_turns)
            max_facts = max(5, min(2 * len(norm_turns), 20))
            self._stat_llm_extract_calls += 1
            facts = await self._extract_facts_from_context(
                context=context,
                merged_user=merged_user,
                merged_assistant=merged_assistant,
                max_facts=max_facts,
            )
            if not facts:
                self._mark_l1_processed(l1_doc_ids, now)
                return

            # ---- 3. 单次判定门禁 ----
            self._stat_gate_calls += 1
            gate_report = await self._run_evaluator_gate(clean_session, facts, now)
            if gate_report is not None:
                facts = gate_report.kept

            valid_facts = [
                item for item in facts if str(item.get("fact", "")).strip()
            ]
            if not valid_facts:
                logger.debug(
                    "[MemorySummarizer] 门禁过滤后无有效事实 session=%s turns=%d dropped=%s",
                    clean_session,
                    len(norm_turns),
                    len(gate_report.dropped) if gate_report else 0,
                )
                self._mark_l1_processed(l1_doc_ids, now)
                return

            # ---- 4. 单次批量向量化 ----
            fact_texts = [str(item.get("fact", "")).strip() for item in valid_facts]
            embeddings = await self._embed_texts_batch(fact_texts)

            # ---- 5. 逐条 L2 融合/新建 ----
            fact_atom_map: Dict[str, str] = {}
            source_doc_id = l1_doc_ids[0] if l1_doc_ids else ""
            for idx, item in enumerate(valid_facts):
                fact_text = fact_texts[idx]
                embedding_vec = embeddings[idx] if idx < len(embeddings) else []
                atom_id, merged = self._upsert_l2_atom(
                    session_id=clean_session,
                    fact_item=item,
                    embedding_vec=embedding_vec,
                    source_doc_id=source_doc_id,
                    now=now,
                )
                if not atom_id:
                    # P0：噪声事实在写入层被拦截，跳过图谱与映射，避免空 atom_id 下游副作用
                    continue
                fact_atom_map[_normalize_fact_key(fact_text)] = atom_id
                # 方案 C：图谱三元组写入（失败不阻断 L2 入库主链路）
                try:
                    self._upsert_kg_triples_for_atom(
                        session_id=clean_session,
                        atom_id=atom_id,
                        fact_text=fact_text,
                        now=now,
                        llm_triples=item.get("triples"),
                    )
                except Exception as kg_exc:  # pragma: no cover - 双保险
                    logger.warning(
                        "[MemorySummarizer] 图谱写入异常（已忽略） atom=%s: %s",
                        atom_id,
                        kg_exc,
                    )
                logger.debug(
                    "[MemorySummarizer] Processed fact session=%s atom=%s merged=%s pinned=%s value_score=%s",
                    clean_session,
                    atom_id,
                    merged,
                    item.get("pinned", 0),
                    item.get("value_score"),
                )

            # ---- 6. 冲突软废弃（置顶免死）----
            if gate_report is not None and gate_report.conflicts:
                applied = self._apply_conflict_resolution(
                    gate_report.conflicts, fact_atom_map, now
                )
                logger.info(
                    "[MemorySummarizer] 判定门禁冲突处理 session=%s detected=%s applied=%s",
                    clean_session,
                    len(gate_report.conflicts),
                    applied,
                )

            # ---- 7. 回写 L1 析出状态 ----
            self._mark_l1_processed(l1_doc_ids, now)

            logger.debug(
                "[MemorySummarizer] Batch done session=%s turns=%d facts=%d reason=%s",
                clean_session,
                len(norm_turns),
                len(valid_facts),
                reason,
            )
        except asyncio.CancelledError:
            # 交给 _worker_loop 统一记录一次 DLQ，避免重复写入
            raise
        except Exception as exc:
            logger.error(
                "[MemorySummarizer] process batch error for session=%s: %s",
                clean_session,
                exc,
                exc_info=True,
            )
            if raise_on_error:
                # 方案 B-3：死信重试路径必须抛出，由 sweep_dlq_once 做退避排期，
                # 否则每次重试失败都会再写一条新 DLQ，形成死信爆炸。
                raise
            self._record_dlq(payload, str(exc))

    def _record_dlq(self, payload: Dict[str, Any], error_msg: str) -> None:
        """异常安全降级：将失败任务写入 SQLite task_dlq 表，若数据库不可用则退化为错误日志"""
        now = time.time()
        task_id = f"dlq_{uuid.uuid4().hex[:16]}"
        scope_key = str(payload.get("session_id", ""))
        try:
            payload_json = json.dumps(
                payload, ensure_ascii=False, default=str, allow_nan=False
            )
        except Exception as serial_exc:
            logger.error(
                "[MemorySummarizer] DLQ payload 序列化失败，任务将仅记录日志 (%s): %s",
                serial_exc,
                error_msg,
            )
            return
        try:
            if self.storage is not None and hasattr(self.storage, "connect"):
                conn = self.storage.connect()
                with conn:
                    conn.execute(
                        """
                        INSERT INTO task_dlq (
                            task_id, task_type, scope_key, payload_json,
                            error_msg, retry_count, max_retries, status,
                            next_retry_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 0, ?, 'pending', ?, ?, ?)
                        """,
                        (
                            task_id,
                            "summarize_dialogue",
                            scope_key,
                            payload_json,
                            str(error_msg)[:1000],
                            int(self.dlq_max_retries),
                            # 首次重试同样遵循退避基数，避免瞬时故障被立即重放
                            now + float(self.dlq_base_backoff),
                            now,
                            now,
                        ),
                    )
                logger.info(
                    "[MemorySummarizer] 任务已写入 DLQ task=%s retry_at=+%.0fs err=%s",
                    task_id,
                    self.dlq_base_backoff,
                    str(error_msg)[:200],
                )
                return
        except Exception as db_exc:
            logger.error(
                "[MemorySummarizer] Failed to persist task to task_dlq (%s), original error: %s",
                db_exc,
                error_msg,
            )
