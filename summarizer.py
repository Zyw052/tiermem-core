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
import logging
import re
import sqlite3
import struct
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover
    logger = logging.getLogger("astrbot_plugin_agent_memory.summarizer")

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


L3_PIN_KEYWORDS: Tuple[str, ...] = (
    "喜欢",
    "讨厌",
    "偏好",
    "必须",
    "严禁",
    "密码",
    "习惯",
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
    ) -> None:
        self.storage = storage
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client
        self.config: Dict[str, Any] = config or {}

        # ---- 自作判定门禁（廉价模型：默认 jev）----
        # 门禁自身的所有异常都在 EvaluatorGate 内部消化，这里无需再包 try。
        try:
            self.evaluator = EvaluatorGate(self.config)
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

    @staticmethod
    def _parse_turn_interval(config: Dict[str, Any]) -> int:
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
            if raw is None or raw == "":
                continue
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 5

    def start(self) -> None:
        """启动后台异步消费 Task"""
        if self._running and self._worker_task and not self._worker_task.done():
            return
        self._running = True
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="agent_memory_summarizer_worker"
        )
        logger.info("[MemorySummarizer] Async worker started.")

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
        logger.info("[MemorySummarizer] Async worker stopped.")

    async def enqueue_dialogue(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> bool:
        """
        将单轮对话累积进会话缓冲池（零阻塞、零 IO）。

        行为：
        - 每轮先按 session_id 累积到内存缓冲池；
        - 仅当该会话累积轮次达到 summary_turn_interval 时，才把这 N 轮
          打包成一个 batch 事件推入后台总结消费队列，减少 LLM 调用次数；
        - 不足 N 轮的尾部对话由 flush_session() / flush_all() 强制打包，
          避免会话结束或插件关闭时丢失记忆。

        :param assistant_text: 机器人回复（即 bot_text）
        :return: True 表示该轮已被接受（进入缓冲或成功打包入队）
        """
        sid = str(session_id or "").strip()
        user = str(user_text or "").strip()
        bot = str(assistant_text or "").strip()
        if not sid or (not user and not bot):
            return False

        turn = {
            "user_text": user,
            "assistant_text": bot,
            "enqueued_at": time.time(),
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

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """
        分发单个队列事件：
        - kind == "batch"：逐轮调用 process_dialogue 处理同一会话的 N 轮对话；
          单轮失败不阻断整批，失败轮次汇总后统一降级写入 DLQ；
        - 其他格式（兼容历史的单轮事件）：直接按单轮处理。
        """
        if not isinstance(event, dict):
            logger.warning("[MemorySummarizer] Skip invalid event type: %s", type(event))
            return

        if event.get("kind") != "batch":
            await self.process_dialogue(
                session_id=event.get("session_id", ""),
                user_text=event.get("user_text", ""),
                assistant_text=event.get("assistant_text", ""),
            )
            return

        session_id = str(event.get("session_id", "") or "").strip()
        turns = event.get("turns")
        if not session_id or not isinstance(turns, list) or not turns:
            logger.warning("[MemorySummarizer] Skip malformed batch event: %r", event)
            return

        succeeded = 0
        failed_turns: List[Dict[str, Any]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            try:
                await self.process_dialogue(
                    session_id=session_id,
                    user_text=turn.get("user_text", ""),
                    assistant_text=turn.get("assistant_text", ""),
                )
                succeeded += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "[MemorySummarizer] Batch turn failed session=%s: %s",
                    session_id,
                    exc,
                    exc_info=True,
                )
                failed_turns.append(turn)

        if failed_turns:
            self._record_dlq(
                {
                    "session_id": session_id,
                    "turns": failed_turns,
                    "kind": "batch_partial_failure",
                    "enqueued_at": event.get("enqueued_at", time.time()),
                },
                f"BatchPartialFailure: {len(failed_turns)}/{len(turns)} turn(s) failed",
            )

        logger.debug(
            "[MemorySummarizer] Batch done session=%s ok=%d/%d reason=%s",
            session_id,
            succeeded,
            len(turns),
            event.get("reason", "threshold"),
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
        """记录 L1 原始对话轮次（优先调用 storage 方法，否则安全写入 l1_docs 表）"""
        doc_id = f"l1_{uuid.uuid4().hex[:16]}"
        dialogue_content = f"User: {user_text}\nAssistant: {assistant_text}".strip()
        title = (user_text or assistant_text)[:48].replace("\n", " ")

        if hasattr(self.storage, "record_l1_dialogue") and callable(
            getattr(self.storage, "record_l1_dialogue")
        ):
            self.storage.record_l1_dialogue(session_id, user_text, assistant_text)
            return doc_id

        conn = self.storage.connect()
        self._ensure_storage_columns(conn)
        meta_json = json.dumps(
            {"user_text": user_text, "assistant_text": assistant_text},
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
        return doc_id

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
        self, user_text: str, assistant_text: str
    ) -> List[Dict[str, Any]]:
        """
        100% 健壮的规则分句与关键事实抽取回退器：
        当未配置 LLM 或 LLM 调用异常时，对用户输入与关键回复进行分句、清洗与打分。
        """
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
        for sent in candidates[:5]:
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
        """
        提取关键事实：
        若配置了总结 LLM API (llm_api_base & llm_api_key)，则调用 OpenAI 兼容总结端点；
        未配置或发生任何异常时，100% 优雅回退到 _rule_extract_facts。
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

        if api_base and api_key and aiohttp is not None:
            endpoint = (
                api_base
                if api_base.endswith("/chat/completions")
                else f"{api_base}/chat/completions"
            )
            prompt = (
                "请从以下对话中提取值得长期记住的客观事实、用户偏好、习惯或约束。"
                "返回 JSON 数组，每项格式为: "
                '{"fact": "简洁事实陈述", "importance": 0.0到1.0之间的小数}。\n'
                f"用户: {user_text}\n助手: {assistant_text}"
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
                async with aiohttp.ClientSession(timeout=client_timeout) as session:
                    async with session.post(endpoint, headers=headers, json=payload) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            content = (
                                data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                            match = re.search(r"\[.*\]", content, re.DOTALL)
                            if match:
                                parsed = json.loads(match.group(0))
                                if isinstance(parsed, list) and parsed:
                                    llm_facts: List[Dict[str, Any]] = []
                                    for item in parsed:
                                        if isinstance(item, str) and item.strip():
                                            f_text = item.strip()
                                            p, s, atype = self._evaluate_pin_and_score(f_text)
                                            llm_facts.append(
                                                {
                                                    "fact": f_text,
                                                    "importance": s,
                                                    "pinned": p,
                                                    "atom_type": atype,
                                                }
                                            )
                                        elif isinstance(item, dict) and item.get("fact"):
                                            f_text = str(item["fact"]).strip()
                                            raw_imp = item.get("importance")
                                            p, s, atype = self._evaluate_pin_and_score(
                                                f_text,
                                                float(raw_imp) if raw_imp is not None else None,
                                            )
                                            llm_facts.append(
                                                {
                                                    "fact": f_text,
                                                    "importance": s,
                                                    "pinned": p,
                                                    "atom_type": atype,
                                                }
                                            )
                                    if llm_facts:
                                        return llm_facts
            except Exception as exc:
                logger.warning(
                    "[MemorySummarizer] LLM fact extraction failed, falling back to rules: %s",
                    exc,
                )

        return self._rule_extract_facts(user_text, assistant_text)

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
                        updated_at = ?
                    WHERE atom_id = ?
                    """,
                    (
                        fact_text if len(fact_text) >= len(str(best_match_row["content"] or "")) else best_match_row["content"],
                        json.dumps(facts_list, ensure_ascii=False),
                        new_imp,
                        new_access,
                        new_reinf,
                        new_pinned,
                        new_tier,
                        emb_blob,
                        now,
                        now,
                        atom_id,
                    ),
                )
            return atom_id, True

        atom_id = f"atom_{uuid.uuid4().hex[:16]}"
        meta_json = json.dumps(
            {"pinned": bool(pinned), "tier": tier, "atom_type": atom_type},
            ensure_ascii=False,
        )
        with conn:
            conn.execute(
                """
                INSERT INTO l2_atoms (
                    atom_id, scope_key, session_id, content,
                    topics_json, key_facts_json, sentiment, importance,
                    access_count, embedding_blob, source_type, source_ref,
                    last_accessed_at, created_at, updated_at,
                    atom_type, reinforcement_count, pinned, tier, metadata
                ) VALUES (?, ?, ?, ?, '[]', ?, 'neutral', ?, 1, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
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
                ),
            )
        return atom_id, False

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
                        """,
                        (replacement, now, now, atom_id),
                    )
                    if cur.rowcount:
                        applied += 1
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
        return applied

    async def process_dialogue(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> None:
        """
        处理单轮对话事件：
        1. L1 记录原始对话轮次；
        2. 提取关键句/事实（LLM 总结或规则分句优雅回退）；
        3. 调用 embedding_client.get_embedding(fact) 向量化并与 L2 记忆余弦相似度 (>0.88) 融合去重；
        4. 包含偏好/习惯/核心约束关键词或评分 >= 0.85 时标记 pinned=1 晋升 L3；
        5. 全流程 try-except 兜底，异常记录 error 并写入 DLQ，绝不崩溃 Worker。
        """
        clean_session = str(session_id or "").strip()
        clean_user = str(user_text or "").strip()
        clean_assistant = str(assistant_text or "").strip()
        if not clean_session or (not clean_user and not clean_assistant):
            return

        now = time.time()
        payload = {
            "session_id": clean_session,
            "user_text": clean_user,
            "assistant_text": clean_assistant,
            "processed_at": now,
        }

        try:
            # 1. L1 记录原始对话轮次
            l1_doc_id = self._record_l1_dialogue(
                clean_session, clean_user, clean_assistant, now
            )

            # 2. 提取关键事实（LLM 或规则回退）
            facts = await self._extract_facts(clean_user, clean_assistant)
            if not facts:
                return

            # 2.5 自作判定门禁：信息价值过滤 + 冲突检测（失败自动降级，不阻断）
            gate_report = await self._run_evaluator_gate(clean_session, facts, now)
            if gate_report is not None:
                facts = gate_report.kept

            if not facts:
                logger.debug(
                    "[MemorySummarizer] 判定门禁过滤后无有效事实 session=%s（dropped=%s）",
                    clean_session,
                    len(gate_report.dropped) if gate_report else 0,
                )
                return

            # 3. 逐条事实向量化、L2 余弦相似度去重融合与 L3 置顶判定
            fact_atom_map: Dict[str, str] = {}
            for item in facts:
                fact_text = str(item.get("fact", "")).strip()
                if not fact_text:
                    continue

                if self.embedding_client is not None:
                    embedding_vec = await self.embedding_client.get_embedding(fact_text)
                else:
                    embedding_vec = []

                atom_id, merged = self._upsert_l2_atom(
                    session_id=clean_session,
                    fact_item=item,
                    embedding_vec=embedding_vec,
                    source_doc_id=l1_doc_id,
                    now=now,
                )
                fact_atom_map[_normalize_fact_key(fact_text)] = atom_id
                logger.debug(
                    "[MemorySummarizer] Processed fact session=%s atom=%s merged=%s pinned=%s value_score=%s",
                    clean_session,
                    atom_id,
                    merged,
                    item.get("pinned", 0),
                    item.get("value_score"),
                )

            # 3.5 应用冲突判定：把被推翻的旧事实软废弃（superseded）
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
        except Exception as exc:
            logger.error(
                "[MemorySummarizer] process_dialogue error for session=%s: %s",
                clean_session,
                exc,
                exc_info=True,
            )
            self._record_dlq(payload, str(exc))

    def _record_dlq(self, payload: Dict[str, Any], error_msg: str) -> None:
        """异常安全降级：将失败任务写入 SQLite task_dlq 表，若数据库不可用则退化为错误日志"""
        now = time.time()
        task_id = f"dlq_{uuid.uuid4().hex[:16]}"
        scope_key = str(payload.get("session_id", ""))
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
                        ) VALUES (?, ?, ?, ?, ?, 0, 3, 'pending', ?, ?, ?)
                        """,
                        (
                            task_id,
                            "summarize_dialogue",
                            scope_key,
                            json.dumps(payload, ensure_ascii=False),
                            str(error_msg)[:1000],
                            now + 60.0,
                            now,
                            now,
                        ),
                    )
                return
        except Exception as db_exc:
            logger.error(
                "[MemorySummarizer] Failed to persist task to task_dlq (%s), original error: %s",
                db_exc,
                error_msg,
            )
