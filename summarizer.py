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

        max_queue_size = int(self.config.get("summarizer_queue_maxsize", 1000))
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._schema_checked: bool = False

    def start(self) -> None:
        """启动后台异步消费 Task"""
        if self._running and self._worker_task and not self._worker_task.done():
            return
        self._running = True
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="agent_memory_summarizer_worker"
        )
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
        if not event["session_id"] or (
            not event["user_text"] and not event["assistant_text"]
        ):
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

            # 3. 逐条事实向量化、L2 余弦相似度去重融合与 L3 置顶判定
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
                logger.debug(
                    "[MemorySummarizer] Processed fact session=%s atom=%s merged=%s pinned=%s",
                    clean_session,
                    atom_id,
                    merged,
                    item.get("pinned", 0),
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
