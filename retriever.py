# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 两阶段混合检索引擎 (retriever.py)
核心链路：
1. L3 强力置顶捞取 (Pinned Memory Pre-fetching)：无条件置顶偏好、指令与核心约束
2. 第一阶段粗排召回 (First-Stage Recall)：稠密向量余弦相似度 Top-N + 轻量 BM25/字词重合度 Top-N，去重合并保留 candidate_k
3. 第二阶段精排重排序 (Second-Stage Rerank)：调用 BGE-Reranker Cross-Encoder 深度打分，无 API 时无缝降级至 RRF 倒数排名融合
4. 结果组装输出：生成结构化检索字典与高可读 Markdown prompt_context
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import sqlite3
import struct
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover
    logger = logging.getLogger("astrbot_plugin_agent_memory.retriever")

try:
    from .models_client import (
        EmbeddingClient,
        RerankClient,
        _compute_bm25_scores,
        _tokenize_for_similarity,
        cosine_similarity,
    )
except ImportError:  # pragma: no cover
    from models_client import (  # type: ignore
        EmbeddingClient,
        RerankClient,
        _compute_bm25_scores,
        _tokenize_for_similarity,
        cosine_similarity,
    )

try:
    from .storage import StorageManager  # type: ignore
except ImportError:
    try:
        from .storage import MemoryStorage as StorageManager
    except ImportError:  # pragma: no cover
        from storage import MemoryStorage as StorageManager  # type: ignore


def _unpack_embedding(blob: Any) -> List[float]:
    """从 SQLite BLOB (float32 bytes 或 JSON bytes/str) 安全反序列化浮点向量。"""
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
    except Exception as exc:
        logger.debug("[MemoryRetriever] 向量反序列化失败，已忽略: %s", exc)
    return []


def _compute_overlap_score(query: str, doc_text: str) -> float:
    """计算 query 与文档的字词重合度 (Overlap Ratio)，作为 BM25 在短文本上的补充特征。"""
    q_tokens = _tokenize_for_similarity(query)
    d_tokens = _tokenize_for_similarity(doc_text)
    if not q_tokens or not d_tokens:
        return 0.0
    q_counter = Counter(q_tokens)
    d_counter = Counter(d_tokens)
    overlap = sum(min(q_counter[t], d_counter.get(t, 0)) for t in q_counter)
    return float(overlap) / float(max(len(q_tokens), 1))


class MemoryRetriever:
    """
    两阶段混合记忆检索引擎：
    L3 强力置顶 + 第一阶段（向量 + BM25/词重合）粗排召回 + 第二阶段（BGE-Reranker / RRF）精排重排
    """

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
        self.config: Dict[str, Any] = config if isinstance(config, dict) else {}
        self._schema_checked: bool = False

    def _ensure_storage_columns(self, conn: sqlite3.Connection) -> None:
        """幂等检查并补齐 l2_atoms 表的扩展字段（pinned, tier, atom_type 等）。"""
        if self._schema_checked:
            return
        extended_cols: List[Tuple[str, str]] = [
            ("persona_id", "TEXT NOT NULL DEFAULT ''"),
            ("atom_type", "TEXT NOT NULL DEFAULT 'fact'"),
            ("decay_type", "TEXT NOT NULL DEFAULT 'exponential'"),
            ("ttl_days", "REAL NOT NULL DEFAULT 30.0"),
            ("reinforcement_count", "INTEGER NOT NULL DEFAULT 0"),
            ("pinned", "INTEGER NOT NULL DEFAULT 0"),
            ("tier", "INTEGER NOT NULL DEFAULT 2"),
            ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
            # 自作判定门禁：被推翻事实的软废弃标记
            ("status", "TEXT NOT NULL DEFAULT 'active'"),
            ("superseded_by", "TEXT NOT NULL DEFAULT ''"),
            ("superseded_at", "REAL NOT NULL DEFAULT 0.0"),
        ]
        try:
            cur = conn.execute("PRAGMA table_info(l2_atoms)")
            existing = {row[1] for row in cur.fetchall()}
            for col_name, col_def in extended_cols:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE l2_atoms ADD COLUMN {col_name} {col_def};")
            self._schema_checked = True
        except Exception as exc:
            logger.warning("[MemoryRetriever] 检查扩展列异常: %s", exc)

    def _row_to_atom_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将 SQLite 行记录转换为标准化的记忆原子字典。"""
        row_keys = set(row.keys())

        def _get(key: str, default: Any = None) -> Any:
            return row[key] if key in row_keys and row[key] is not None else default

        try:
            key_facts = json.loads(_get("key_facts_json", "[]") or "[]")
            if not isinstance(key_facts, list):
                key_facts = []
        except Exception:
            key_facts = []

        try:
            topics = json.loads(_get("topics_json", "[]") or "[]")
            if not isinstance(topics, list):
                topics = []
        except Exception:
            topics = []

        return {
            "atom_id": str(_get("atom_id", "")),
            "scope_key": str(_get("scope_key", "")),
            "session_id": str(_get("session_id", "")),
            "content": str(_get("content", "")).strip(),
            "atom_type": str(_get("atom_type", "fact")),
            "importance": round(float(_get("importance", 0.5)), 4),
            "pinned": int(_get("pinned", 0)),
            "tier": int(_get("tier", 2)),
            "access_count": int(_get("access_count", 0)),
            "reinforcement_count": int(_get("reinforcement_count", 0)),
            "key_facts": key_facts,
            "topics": topics,
            "created_at": float(_get("created_at", 0.0)),
            "updated_at": float(_get("updated_at", 0.0)),
            "_embedding": _unpack_embedding(_get("embedding_blob", None)),
        }

    def _fetch_pinned_l3(
        self,
        conn: sqlite3.Connection,
        session_id: Optional[str] = None,
        max_pinned: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        步骤 1：L3 强力置顶捞取。
        直接从 L2/L3 存储中捞取 pinned=1（或 tier=3）的核心记忆（偏好、指令、约束），
        若传入 session_id，则优先排列当前会话置顶项，同时兼容全局核心置顶项。
        """
        pinned_items: List[Dict[str, Any]] = []
        seen_contents: set[str] = set()

        try:
            if session_id and str(session_id).strip():
                clean_sid = str(session_id).strip()
                cur = conn.execute(
                    """
                    SELECT *
                    FROM l2_atoms
                    WHERE (pinned = 1 OR tier >= 3)
                      AND COALESCE(status, 'active') != 'superseded'
                    ORDER BY
                        CASE WHEN scope_key = ? OR session_id = ? THEN 0 ELSE 1 END ASC,
                        importance DESC,
                        updated_at DESC
                    LIMIT ?
                    """,
                    (clean_sid, clean_sid, int(max_pinned)),
                )
            else:
                cur = conn.execute(
                    """
                    SELECT *
                    FROM l2_atoms
                    WHERE (pinned = 1 OR tier >= 3)
                      AND COALESCE(status, 'active') != 'superseded'
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                    """,
                    (int(max_pinned),),
                )

            for row in cur.fetchall():
                item = self._row_to_atom_dict(row)
                content = item.get("content", "")
                if not content or content in seen_contents:
                    continue
                seen_contents.add(content)
                item.pop("_embedding", None)
                item["source_tier"] = "L3"
                pinned_items.append(item)
        except Exception as exc:
            logger.error("[MemoryRetriever] L3 置顶记忆捞取失败: %s", exc, exc_info=True)

        return pinned_items

    def _fetch_l2_pool(
        self,
        conn: sqlite3.Connection,
        session_id: Optional[str] = None,
        pool_limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """加载用于第一阶段粗排的 L2 原子记忆候选池。"""
        pool: List[Dict[str, Any]] = []
        try:
            strict_session = bool(self.config.get("strict_session_isolation", False))
            if session_id and str(session_id).strip():
                clean_sid = str(session_id).strip()
                if strict_session:
                    cur = conn.execute(
                        """
                        SELECT *
                        FROM l2_atoms
                        WHERE (scope_key = ? OR session_id = ?)
                          AND COALESCE(status, 'active') != 'superseded'
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (clean_sid, clean_sid, int(pool_limit)),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT *
                        FROM l2_atoms
                        WHERE COALESCE(status, 'active') != 'superseded'
                        ORDER BY
                            CASE WHEN scope_key = ? OR session_id = ? THEN 0 ELSE 1 END ASC,
                            updated_at DESC
                        LIMIT ?
                        """,
                        (clean_sid, clean_sid, int(pool_limit)),
                    )
            else:
                cur = conn.execute(
                    """
                    SELECT *
                    FROM l2_atoms
                    WHERE COALESCE(status, 'active') != 'superseded'
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (int(pool_limit),),
                )

            for row in cur.fetchall():
                item = self._row_to_atom_dict(row)
                if item.get("content"):
                    pool.append(item)
        except Exception as exc:
            logger.error("[MemoryRetriever] 加载 L2 候选池失败: %s", exc, exc_info=True)

        return pool

    async def _first_stage_recall(
        self,
        query: str,
        l2_pool: List[Dict[str, Any]],
        pinned_ids: set[str],
        pinned_contents: set[str],
        candidate_k: int = 20,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        步骤 2：第一阶段粗排召回（First-Stage Recall）
        - 过滤已存在于 L3 置顶集合中的记忆项；
        - 向量召回：计算 query 向量，与候选池计算余弦相似度，取 Top-N；
        - 关键词召回：计算轻量 BM25 + 字词重合度得分，取 Top-N；
        - 合并去重：保留最多 candidate_k 条进入第二阶段精排。
        """
        # 1. 过滤掉已在 L3 置顶列表中的重复项
        filtered_pool: List[Dict[str, Any]] = []
        seen_text: set[str] = set()
        for atom in l2_pool:
            aid = atom.get("atom_id", "")
            content = atom.get("content", "")
            if not content or aid in pinned_ids or content in pinned_contents or content in seen_text:
                continue
            seen_text.add(content)
            filtered_pool.append(atom)

        if not filtered_pool:
            return [], []

        # 2. 获取 Query 向量
        query_vec: List[float] = []
        if self.embedding_client is not None and query and query.strip():
            try:
                query_vec = await self.embedding_client.get_embedding(query)
            except Exception as exc:
                logger.warning("[MemoryRetriever] 获取 query 向量异常，降级为纯关键词粗排: %s", exc)
                query_vec = []

        has_valid_qvec = bool(query_vec) and any(abs(v) > 1e-12 for v in query_vec)

        # 3. 计算向量余弦相似度
        vec_scores: List[float] = []
        for atom in filtered_pool:
            doc_vec = atom.get("_embedding") or []
            if has_valid_qvec and doc_vec:
                sim = cosine_similarity(query_vec, doc_vec)
            else:
                sim = 0.0
            vec_scores.append(max(0.0, float(sim)))

        # 4. 计算关键词得分（BM25 + 字词重合度）
        docs_text = [
            f"{atom.get('content', '')} {' '.join(atom.get('key_facts', []))}".strip()
            for atom in filtered_pool
        ]
        bm25_raw = _compute_bm25_scores(query, docs_text)
        max_bm25 = max(bm25_raw) if bm25_raw else 0.0

        kw_scores: List[float] = []
        for idx, doc_str in enumerate(docs_text):
            norm_bm25 = (bm25_raw[idx] / max_bm25) if max_bm25 > 1e-9 else 0.0
            overlap = _compute_overlap_score(query, doc_str)
            kw_scores.append(0.7 * norm_bm25 + 0.3 * overlap)

        # 5. 分别取向量召回 Top-N 与关键词召回 Top-N
        top_n_each = max(1, int(candidate_k))
        n_pool = len(filtered_pool)

        vec_order = sorted(
            range(n_pool),
            key=lambda i: (vec_scores[i], float(filtered_pool[i].get("importance", 0.5)), -i),
            reverse=True,
        )[:top_n_each]

        kw_order = sorted(
            range(n_pool),
            key=lambda i: (kw_scores[i], float(filtered_pool[i].get("importance", 0.5)), -i),
            reverse=True,
        )[:top_n_each]

        # 6. 合并候选集并去重，按粗排融合分排序截断至 candidate_k
        merged_indices: Dict[int, float] = {}
        for rank, idx in enumerate(vec_order):
            merged_indices[idx] = merged_indices.get(idx, 0.0) + 0.55 * vec_scores[idx] + (0.1 / (rank + 1))
        for rank, idx in enumerate(kw_order):
            merged_indices[idx] = merged_indices.get(idx, 0.0) + 0.45 * kw_scores[idx] + (0.1 / (rank + 1))

        sorted_candidate_indices = sorted(
            merged_indices.keys(),
            key=lambda idx: (
                merged_indices[idx],
                float(filtered_pool[idx].get("importance", 0.5)),
                float(filtered_pool[idx].get("updated_at", 0.0)),
            ),
            reverse=True,
        )[: int(candidate_k)]

        candidates: List[Dict[str, Any]] = []
        for idx in sorted_candidate_indices:
            atom_copy = dict(filtered_pool[idx])
            atom_copy["coarse_vector_score"] = round(vec_scores[idx], 4)
            atom_copy["coarse_keyword_score"] = round(kw_scores[idx], 4)
            atom_copy["coarse_score"] = round(merged_indices[idx], 4)
            candidates.append(atom_copy)

        return candidates, query_vec

    async def _second_stage_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        query_vec: Sequence[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        步骤 3：第二阶段精排重排序（Second-Stage Rerank）
        调用 rerank_client.rerank(query, candidate_docs, top_n=top_k) 进行深度语义打分；
        若 RerankClient 降级走 RRF，无缝兼容其返回格式。
        """
        if not candidates or top_k <= 0:
            return []

        candidate_docs = [str(item.get("content", "")) for item in candidates]
        doc_embeddings = [item.get("_embedding") or [] for item in candidates]
        has_all_doc_vecs = bool(query_vec) and all(bool(vec) for vec in doc_embeddings)

        rerank_results: List[Dict[str, Any]] = []
        try:
            if self.rerank_client is not None:
                # 检查 rerank 方法是否支持可选的 query_embedding / doc_embeddings 参数以增强 RRF 降级精度
                sig = inspect.signature(self.rerank_client.rerank)
                if "query_embedding" in sig.parameters and has_all_doc_vecs:
                    rerank_results = await self.rerank_client.rerank(
                        query,
                        candidate_docs,
                        top_n=top_k,
                        query_embedding=query_vec,
                        doc_embeddings=doc_embeddings,
                    )
                else:
                    rerank_results = await self.rerank_client.rerank(
                        query,
                        candidate_docs,
                        top_n=top_k,
                    )
        except Exception as exc:
            logger.warning("[MemoryRetriever] Rerank 调用异常，回退至粗排顺序: %s", exc)
            rerank_results = []

        retrieved_l2: List[Dict[str, Any]] = []
        used_indices: set[int] = set()

        if isinstance(rerank_results, list) and rerank_results:
            for rank_item in rerank_results:
                if not isinstance(rank_item, dict):
                    continue
                idx = rank_item.get("index")
                doc_text = rank_item.get("document")
                score = float(
                    rank_item.get(
                        "relevance_score",
                        rank_item.get("score", 0.0),
                    )
                )

                target_idx: Optional[int] = None
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    target_idx = idx
                elif isinstance(doc_text, str):
                    for c_i, cand in enumerate(candidates):
                        if c_i not in used_indices and cand.get("content") == doc_text:
                            target_idx = c_i
                            break

                if target_idx is None or target_idx in used_indices:
                    continue

                used_indices.add(target_idx)
                selected = dict(candidates[target_idx])
                selected.pop("_embedding", None)
                selected["relevance_score"] = round(score, 6)
                selected["source_tier"] = "L2"
                retrieved_l2.append(selected)
                if len(retrieved_l2) >= int(top_k):
                    break

        # 若 rerank 返回条数不足或异常为空，使用粗排高分候选安全补齐
        if not retrieved_l2:
            for c_i, cand in enumerate(candidates[: int(top_k)]):
                if c_i in used_indices:
                    continue
                selected = dict(cand)
                selected.pop("_embedding", None)
                selected["relevance_score"] = round(float(selected.get("coarse_score", 0.0)), 6)
                selected["source_tier"] = "L2"
                retrieved_l2.append(selected)

        return retrieved_l2[: int(top_k)]

    def _touch_accessed_atoms(
        self, conn: sqlite3.Connection, atom_ids: Sequence[str]
    ) -> None:
        """轻量更新被检索命中的记忆原子的访问计数与最后访问时间。"""
        valid_ids = [str(aid) for aid in atom_ids if aid]
        if not valid_ids:
            return
        now = time.time()
        try:
            placeholders = ",".join("?" for _ in valid_ids)
            with conn:
                conn.execute(
                    f"""
                    UPDATE l2_atoms
                    SET access_count = access_count + 1,
                        last_accessed_at = ?
                    WHERE atom_id IN ({placeholders})
                    """,
                    [now, *valid_ids],
                )
        except Exception as exc:
            logger.debug("[MemoryRetriever] 更新记忆访问统计失败（忽略）: %s", exc)

    @staticmethod
    def _format_prompt_context(
        pinned_l3: Sequence[Dict[str, Any]],
        retrieved_l2: Sequence[Dict[str, Any]],
    ) -> str:
        """
        将检索出的 L3 置顶记忆与 L2 精排记忆组装为可直接注入 LLM System Prompt 的高可读 Markdown 文本。
        """
        if not pinned_l3 and not retrieved_l2:
            return ""

        sections: List[str] = ["## 🧠 长期记忆上下文 (Agent Memory Context)"]

        if pinned_l3:
            l3_lines: List[str] = ["### 📌 核心记忆与约束 (L3 Pinned)"]
            for idx, item in enumerate(pinned_l3, start=1):
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                atom_type = str(item.get("atom_type", "preference"))
                importance = float(item.get("importance", 0.9))
                l3_lines.append(
                    f"{idx}. **[核心/{atom_type}]** {content} *(重要度: {importance:.2f})*"
                )
            if len(l3_lines) > 1:
                sections.append("\n".join(l3_lines))

        if retrieved_l2:
            l2_lines: List[str] = ["### 🔍 相关情景记忆 (L2 Retrieved)"]
            for idx, item in enumerate(retrieved_l2, start=1):
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                rel_score = float(item.get("relevance_score", 0.0))
                l2_lines.append(
                    f"- {content} *(相关度: {rel_score:.4f})*"
                )
            if len(l2_lines) > 1:
                sections.append("\n".join(l2_lines))

        return "\n\n".join(sections)

    async def retrieve(
        self,
        query: str,
        session_id: str = None,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> dict:
        """
        执行两阶段混合检索：
        1. L3 强力置顶捞取 (pinned=1)
        2. 第一阶段粗排召回 (向量余弦相似度 Top-N + 轻量 BM25/词重合度 Top-N，去重保留 candidate_k)
        3. 第二阶段精排重排序 (BGE-Reranker 深度语义打分 / RRF 优雅降级，保留 top_k)
        4. 组装输出 {"pinned_l3": [...], "retrieved_l2": [...], "prompt_context": str}
        """
        clean_query = str(query or "").strip()
        effective_top_k = max(1, int(top_k or 5))
        effective_candidate_k = max(effective_top_k, int(candidate_k or 20))

        try:
            conn = self.storage.connect()
            self._ensure_storage_columns(conn)

            # 1. L3 强力置顶捞取
            max_pinned = int(self.config.get("max_pinned_l3", 15))
            pinned_l3 = self._fetch_pinned_l3(
                conn=conn,
                session_id=session_id,
                max_pinned=max_pinned,
            )
            pinned_ids = {str(item.get("atom_id", "")) for item in pinned_l3 if item.get("atom_id")}
            pinned_contents = {str(item.get("content", "")) for item in pinned_l3 if item.get("content")}

            # 若 query 为空，仅返回置顶的 L3 核心记忆与对应 Markdown 上下文
            if not clean_query:
                prompt_context = self._format_prompt_context(pinned_l3, [])
                return {
                    "pinned_l3": pinned_l3,
                    "retrieved_l2": [],
                    "prompt_context": prompt_context,
                }

            # 2. 第一阶段粗排召回
            pool_limit = int(self.config.get("recall_pool_limit", 500))
            l2_pool = self._fetch_l2_pool(
                conn=conn,
                session_id=session_id,
                pool_limit=pool_limit,
            )
            candidates, query_vec = await self._first_stage_recall(
                query=clean_query,
                l2_pool=l2_pool,
                pinned_ids=pinned_ids,
                pinned_contents=pinned_contents,
                candidate_k=effective_candidate_k,
            )

            # 3. 第二阶段精排重排序
            retrieved_l2 = await self._second_stage_rerank(
                query=clean_query,
                candidates=candidates,
                query_vec=query_vec,
                top_k=effective_top_k,
            )

            # 更新命中记忆的访问计数
            hit_ids = [item.get("atom_id", "") for item in retrieved_l2 if item.get("atom_id")]
            self._touch_accessed_atoms(conn, hit_ids)

            # 4. 组装 Markdown prompt_context
            prompt_context = self._format_prompt_context(pinned_l3, retrieved_l2)

            return {
                "pinned_l3": pinned_l3,
                "retrieved_l2": retrieved_l2,
                "prompt_context": prompt_context,
            }
        except Exception as exc:
            logger.error(
                "[MemoryRetriever] retrieve 执行异常 (query=%s, session=%s): %s",
                clean_query,
                session_id,
                exc,
                exc_info=True,
            )
            return {
                "pinned_l3": [],
                "retrieved_l2": [],
                "prompt_context": "",
            }


__all__ = ["MemoryRetriever"]
