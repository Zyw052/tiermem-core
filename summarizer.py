# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 异步记忆总结与分层晋升 Worker (summarizer.py)
第一阶段：核心异步队列、后台 Worker 循环与死信安全降级骨架。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover
    logger = logging.getLogger("astrbot_plugin_agent_memory.summarizer")

try:
    from .models_client import EmbeddingClient, RerankClient
except ImportError:  # pragma: no cover
    from models_client import EmbeddingClient, RerankClient  # type: ignore

try:
    from .storage import StorageManager  # type: ignore
except ImportError:
    try:
        from .storage import MemoryStorage as StorageManager
    except ImportError:  # pragma: no cover
        from storage import MemoryStorage as StorageManager  # type: ignore


class MemorySummarizer:
    """异步对话记忆总结与分层晋升后台 Worker"""

    def __init__(
        self,
        storage: StorageManager,
        embedding_client: EmbeddingClient,
        rerank_client: RerankClient,
        config: dict = None,
    ) -> None:
        self.storage = storage
        self.embedding_client = embedding_client
        self.rerank_client = rerank_client
        self.config: Dict[str, Any] = config or {}

        max_queue_size = int(self.config.get("summarizer_queue_maxsize", 1000))
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running: bool = False

    def start(self) -> None:
        """启动后台异步消费 Task"""
        if self._running and self._worker_task and not self._worker_task.done():
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop(), name="agent_memory_summarizer_worker")
        logger.info("[MemorySummarizer] Async worker started.")

    async def stop(self) -> None:
        """优雅停止后台异步消费 Task"""
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
        极速将对话事件入队（零阻塞）。
        若队列已满则降级写入死信队列或记录告警，绝不阻塞主对话链路。
        """
        event = {
            "session_id": str(session_id or "").strip(),
            "user_text": str(user_text or "").strip(),
            "assistant_text": str(assistant_text or "").strip(),
            "enqueued_at": time.time(),
        }
        if not event["session_id"] or (not event["user_text"] and not event["assistant_text"]):
            return False

        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning(
                "[MemorySummarizer] Dialogue queue full, degrading to DLQ for session=%s",
                event["session_id"],
            )
            self._record_dlq(event, "QueueFull: dialogue queue reached maxsize")
            return False
        except Exception as exc:
            logger.error("[MemorySummarizer] Enqueue failed: %s", exc)
            return False

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
                await self.process_dialogue(
                    session_id=event.get("session_id", ""),
                    user_text=event.get("user_text", ""),
                    assistant_text=event.get("assistant_text", ""),
                )
            except asyncio.CancelledError:
                self._record_dlq(event, "WorkerCancelledDuringProcess")
                raise
            except Exception as exc:
                logger.error(
                    "[MemorySummarizer] process_dialogue failed for session=%s: %s",
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

    async def process_dialogue(
        self, session_id: str, user_text: str, assistant_text: str
    ) -> None:
        """
        处理单轮对话事件的方法桩（后续阶段注入 LLM 抽取、向量化与 L1/L2/L3 晋升逻辑）。
        """
        logger.debug(
            "[MemorySummarizer] Stub process_dialogue called for session=%s (user_len=%d, assistant_len=%d)",
            session_id,
            len(user_text),
            len(assistant_text),
        )

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
