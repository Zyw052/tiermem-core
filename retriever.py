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

import asyncio
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


# 方案 B-2：矩阵缓存命中时，候选池无需再拉取 embedding_blob（省掉逐行 struct.unpack）。
# 这里显式列出候选池所需列；缺失的扩展列会被自动过滤，保证旧库兼容。
_POOL_COLUMNS: Tuple[str, ...] = (
    "atom_id",
    "scope_key",
    "session_id",
    "content",
    "topics_json",
    "key_facts_json",
    "sentiment",
    "importance",
    "access_count",
    "source_type",
    "source_ref",
    "last_accessed_at",
    "created_at",
    "updated_at",
    "atom_type",
    "pinned",
    "tier",
    "reinforcement_count",
    "status",
    "superseded_by",
    "superseded_at",
)


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

        # ---- 方案 B：工业级引擎（FTS5 倒排索引 + 内存向量矩阵缓存）----
        # 两者均在首次检索时懒初始化，任一路径不可用都会静默回退原有 Python 实现，
        # 保证对旧库、旧配置、无 FTS5 的 SQLite 构建零破坏。
        self._engine_b_ready: bool = False
        self.fts_enabled: bool = False
        self._matrix_cache: Any = getattr(storage, "embedding_cache", None)
        self._matrix_sync_interval: float = 5.0
        try:
            self._matrix_sync_interval = max(
                0.0, float(self.config.get("matrix_cache_sync_interval", 5.0) or 0.0)
            )
        except (TypeError, ValueError):
            self._matrix_sync_interval = 5.0

        # ---- R-05：访问计数改为内存累加 + 异步定期刷盘 ----
        # 检索是高频只读链路，任何同步 UPDATE 都会抢 SQLite 写锁并拖慢首字延迟。
        self._access_counter: Dict[str, int] = {}
        self._access_last_flush: float = time.time()
        try:
            self._access_flush_interval: float = max(
                1.0, float(self.config.get("access_flush_interval", 60.0) or 60.0)
            )
        except (TypeError, ValueError):
            self._access_flush_interval = 60.0
        try:
            self._access_flush_threshold: int = max(
                1, int(self.config.get("access_flush_threshold", 200) or 200)
            )
        except (TypeError, ValueError):
            self._access_flush_threshold = 200
        self._access_flush_task: Optional[asyncio.Task] = None
        self._stat_access_buffered: int = 0
        self._stat_access_flushed: int = 0

    def _strict_isolation(self) -> bool:
        """
        R-04：是否开启严格会话隔离（默认 True）。

        默认物理隔离不同群聊/私聊；仅当明确 ``scope_key='global'`` 的记忆才允许跨会话召回。
        """
        return bool(self.config.get("strict_session_isolation", True))

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

    # ------------------------------------------------------------------
    # 方案 B：工业级引擎懒初始化（FTS5 + 向量矩阵缓存）
    # ------------------------------------------------------------------
    def _ensure_engine_b(self, conn: sqlite3.Connection) -> None:
        """
        首次检索时初始化引擎增强组件（幂等，任何异常都只会降级不会中断检索）：

        1. FTS5：确保虚拟表/触发器/预分词列就绪（旧库自动 backfill）；
        2. 向量矩阵缓存：一次性构建 ``atom_id -> index`` 映射与二维归一化矩阵。

        注意调用顺序必须在 :meth:`_ensure_storage_columns` 之后——
        软废弃触发器的创建依赖 status 扩展列已存在。
        """
        if self._engine_b_ready:
            return
        self._engine_b_ready = True

        # 1. FTS5 倒排索引
        try:
            self.fts_enabled = bool(self.storage.ensure_fts_index(conn))
        except Exception as exc:
            logger.warning(
                "[MemoryRetriever] FTS5 初始化失败，全文检索回退 Python BM25: %s", exc
            )
            self.fts_enabled = False

        # 2. 向量矩阵缓存
        try:
            if self._matrix_cache is not None:
                self._matrix_cache.ensure_built(conn)
        except Exception as exc:
            logger.warning(
                "[MemoryRetriever] 向量矩阵缓存构建失败，回退逐条余弦计算: %s", exc
            )
            self._matrix_cache = None

        logger.info(
            "[MemoryRetriever] 引擎升级状态: fts5=%s, matrix_cache=%s",
            "on" if self.fts_enabled else "off(fallback bm25)",
            (self._matrix_cache.stats() if self._matrix_cache is not None else "off"),
        )

    def _sync_matrix_cache(self, conn: sqlite3.Connection) -> None:
        """轻量一致性校验（节流），兜住绕过缓存的数据库变更。"""
        if self._matrix_cache is None:
            return
        try:
            self._matrix_cache.sync_if_stale(conn, max_age=self._matrix_sync_interval)
        except Exception as exc:
            logger.debug("[MemoryRetriever] 矩阵缓存同步失败（忽略）: %s", exc)

    def _fts_keyword_scores(
        self,
        conn: sqlite3.Connection,
        query: str,
        limit: int,
    ) -> Optional[Dict[str, float]]:
        """
        调用 FTS5 倒排索引获取关键词得分映射。

        :return: ``{atom_id: 归一化 bm25 分数}``；FTS5 不可用、无有效 token
                 或查询异常时返回 ``None``，调用方回退 Python BM25。
                 返回空字典表示「索引可用但无命中」，同样由调用方决定是否兜底。
        """
        if not self.fts_enabled:
            return None
        try:
            hits = self.storage.fts_search(query, limit=max(1, int(limit)), conn=conn)
        except Exception as exc:
            logger.debug("[MemoryRetriever] FTS5 检索异常，回退 BM25: %s", exc)
            return None
        return {aid: score for aid, score in hits}

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
            strict_session = self._strict_isolation()
            if session_id and str(session_id).strip():
                clean_sid = str(session_id).strip()
                if strict_session:
                    # R-04：严格隔离下仅召回本会话置顶 + 显式 global 全局置顶
                    cur = conn.execute(
                        """
                        SELECT *
                        FROM l2_atoms
                        WHERE (pinned = 1 OR tier >= 3)
                          AND COALESCE(status, 'active') != 'superseded'
                          AND (
                                scope_key = ?
                                OR session_id = ?
                                OR COALESCE(scope_key, '') = 'global'
                              )
                        ORDER BY importance DESC, updated_at DESC
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
                        ORDER BY
                            CASE WHEN scope_key = ? OR session_id = ? THEN 0 ELSE 1 END ASC,
                            importance DESC,
                            updated_at DESC
                        LIMIT ?
                        """,
                        (clean_sid, clean_sid, int(max_pinned)),
                    )
            else:
                if strict_session:
                    # 无会话上下文 + 严格隔离：仅召回显式 global 置顶记忆
                    cur = conn.execute(
                        """
                        SELECT *
                        FROM l2_atoms
                        WHERE (pinned = 1 OR tier >= 3)
                          AND COALESCE(status, 'active') != 'superseded'
                          AND COALESCE(scope_key, '') = 'global'
                        ORDER BY importance DESC, updated_at DESC
                        LIMIT ?
                        """,
                        (int(max_pinned),),
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

    def _pool_column_sql(self, conn: sqlite3.Connection, with_embedding: bool) -> str:
        """
        构造候选池 SELECT 列清单。

        ``with_embedding=False`` 时排除 ``embedding_blob`` 大字段，
        配合向量矩阵缓存把全表反序列化开销降到零；若表结构探测失败则安全退回 ``*``。
        """
        if with_embedding:
            return "*"
        try:
            existing = {
                str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()
            }
            if not existing:
                return "*"
            keep = [col for col in _POOL_COLUMNS if col in existing]
            return ", ".join(keep) if keep else "*"
        except Exception as exc:  # pragma: no cover - 结构探测失败极罕见
            logger.debug("[MemoryRetriever] 候选列构造失败，回退 SELECT *: %s", exc)
            return "*"

    def _fetch_l2_pool(
        self,
        conn: sqlite3.Connection,
        session_id: Optional[str] = None,
        pool_limit: int = 500,
        with_embedding: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        加载用于第一阶段粗排的 L2 原子记忆候选池。

        :param with_embedding: 是否一并拉取 embedding_blob。方案 B-2 命中矩阵缓存时
            传 False，避免每行 struct.unpack 反序列化的 CPU 开销；候选进入精排前
            会从内存矩阵按需回填向量。
        """
        pool: List[Dict[str, Any]] = []
        col_sql = self._pool_column_sql(conn, with_embedding)
        try:
            strict_session = self._strict_isolation()
            if session_id and str(session_id).strip():
                clean_sid = str(session_id).strip()
                if strict_session:
                    # R-04：严格隔离（默认）——只允许本会话 + 显式 global 全局记忆
                    cur = conn.execute(
                        f"""
                        SELECT {col_sql}
                        FROM l2_atoms
                        WHERE (
                                scope_key = ?
                                OR session_id = ?
                                OR COALESCE(scope_key, '') = 'global'
                              )
                          AND COALESCE(status, 'active') != 'superseded'
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (clean_sid, clean_sid, int(pool_limit)),
                    )
                else:
                    cur = conn.execute(
                        f"""
                        SELECT {col_sql}
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
                if strict_session:
                    # 无会话上下文 + 严格隔离：仅召回显式 global 记忆
                    cur = conn.execute(
                        f"""
                        SELECT {col_sql}
                        FROM l2_atoms
                        WHERE COALESCE(status, 'active') != 'superseded'
                          AND COALESCE(scope_key, '') = 'global'
                        ORDER BY updated_at DESC
                        LIMIT ?
                        """,
                        (int(pool_limit),),
                    )
                else:
                    cur = conn.execute(
                        f"""
                        SELECT {col_sql}
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
        fts_scores: Optional[Dict[str, float]] = None,
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

        # 3. 计算向量余弦相似度（方案 B-2：优先一次矩阵乘法，回退逐条计算）
        vec_scores: List[float] = []
        matrix_sim_map: Dict[str, float] = {}
        use_matrix = False
        if has_valid_qvec and self._matrix_cache is not None and self._matrix_cache.built:
            try:
                matrix_sim_map = self._matrix_cache.score_map(query_vec)
                use_matrix = bool(matrix_sim_map)
            except Exception as exc:
                logger.debug("[MemoryRetriever] 矩阵打分失败，回退逐条计算: %s", exc)
                use_matrix = False

        for atom in filtered_pool:
            sim = 0.0
            if has_valid_qvec:
                if use_matrix:
                    # 一次 np.dot(matrix, q) 已算出全部余弦，这里只做字典查表
                    sim = matrix_sim_map.get(str(atom.get("atom_id", "")), 0.0)
                else:
                    doc_vec = atom.get("_embedding") or []
                    if doc_vec:
                        sim = cosine_similarity(query_vec, doc_vec)
            vec_scores.append(max(0.0, float(sim)))

        # 4. 计算关键词得分
        # 方案 B-1：FTS5 命中时直接使用倒排索引 bm25 分数，跳过 Python 正则切词与双重循环；
        # FTS 不可用 / 无命中时回退原有轻量 BM25 + 字词重合度实现。
        kw_scores: List[float] = []
        if fts_scores:
            for atom in filtered_pool:
                kw_scores.append(
                    float(fts_scores.get(str(atom.get("atom_id", "")), 0.0))
                )
        else:
            docs_text = [
                f"{atom.get('content', '')} {' '.join(atom.get('key_facts', []))}".strip()
                for atom in filtered_pool
            ]
            bm25_raw = _compute_bm25_scores(query, docs_text)
            max_bm25 = max(bm25_raw) if bm25_raw else 0.0

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
            # 方案 B-2：候选池不再拉取 embedding_blob，进入精排前从内存矩阵按需回填。
            # 缓存中存放的是单位向量，余弦相似度与原始向量等价，语义零偏差。
            if not atom_copy.get("_embedding") and self._matrix_cache is not None:
                try:
                    cached_vec = self._matrix_cache.get_vector(
                        str(atom_copy.get("atom_id", ""))
                    )
                    if cached_vec:
                        atom_copy["_embedding"] = cached_vec
                except Exception as exc:
                    logger.debug("[MemoryRetriever] 回填候选向量失败（忽略）: %s", exc)
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

    def _record_access_memory(self, atom_ids: Sequence[str]) -> None:
        """
        R-05：把命中记忆的访问计数累加在内存字典里（纯内存、无 IO、无锁）。

        原因：检索是最热的只读链路，同步 UPDATE 会抢 SQLite 写锁，
        在 WAL 下仍可能与其他写入产生锁竞争，直接抬高首字延迟。
        """
        touched = 0
        for aid in atom_ids:
            atom_id = str(aid or "").strip()
            if not atom_id:
                continue
            self._access_counter[atom_id] = self._access_counter.get(atom_id, 0) + 1
            touched += 1
        if not touched:
            return
        self._stat_access_buffered += touched
        if self._should_flush_access():
            self._schedule_access_flush()

    def _should_flush_access(self) -> bool:
        """判断内存计数是否达到刷盘条件（阈值或时间窗）。"""
        if not self._access_counter:
            return False
        if sum(self._access_counter.values()) >= self._access_flush_threshold:
            return True
        return (time.time() - self._access_last_flush) >= self._access_flush_interval

    def _schedule_access_flush(self) -> None:
        """异步调度一次刷盘（同一时刻至多一个刷盘 Task，重复触发直接跳过）。"""
        if self._access_flush_task is not None and not self._access_flush_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # 无事件循环（同步上下文）时不刷盘，留给下次或 close()
            return
        self._access_flush_task = loop.create_task(
            self.flush_access_counters(), name="agent_memory_access_flush"
        )

    def _flush_access_sync(self, batch: Dict[str, int]) -> None:
        """把一批访问计数增量写回 SQLite（在后台线程执行，不阻塞事件循环）。"""
        if not batch:
            return
        conn = self.storage.connect()
        now = time.time()
        with conn:
            for atom_id, delta in batch.items():
                conn.execute(
                    """
                    UPDATE l2_atoms
                    SET access_count = COALESCE(access_count, 0) + ?,
                        last_accessed_at = ?
                    WHERE atom_id = ?
                    """,
                    (int(delta), now, atom_id),
                )

    async def flush_access_counters(self) -> int:
        """
        异步刷盘内存中的访问计数（R-05）。

        返回本次成功刷盘的原子条数；失败时把计数回填内存，保证不丢数据。
        """
        if not self._access_counter:
            self._access_last_flush = time.time()
            return 0

        batch = dict(self._access_counter)
        self._access_counter.clear()

        try:
            # SQLite 写入放到线程池，避免检索高峰时阻塞事件循环
            await asyncio.to_thread(self._flush_access_sync, batch)
            self._stat_access_flushed += sum(batch.values())
            self._access_last_flush = time.time()
            logger.debug(
                "[MemoryRetriever] 访问计数刷盘完成 atoms=%d deltas=%d",
                len(batch),
                sum(batch.values()),
            )
            return len(batch)
        except Exception as exc:
            logger.debug("[MemoryRetriever] 访问计数刷盘失败，已回填内存（忽略）: %s", exc)
            for atom_id, delta in batch.items():
                self._access_counter[atom_id] = self._access_counter.get(atom_id, 0) + delta
            return 0

    def get_access_stats(self) -> Dict[str, Any]:
        """返回访问计数缓冲状态（可观测性）。"""
        return {
            "pending_atoms": len(self._access_counter),
            "pending_deltas": sum(self._access_counter.values()),
            "flush_interval": self._access_flush_interval,
            "flush_threshold": self._access_flush_threshold,
            "stat_buffered": self._stat_access_buffered,
            "stat_flushed": self._stat_access_flushed,
            "flushing": bool(
                self._access_flush_task is not None
                and not self._access_flush_task.done()
            ),
        }

    async def close(self) -> None:
        """停机前把剩余访问计数刷盘并回收刷盘 Task（幂等）。"""
        task = self._access_flush_task
        if task is not None and not task.done():
            try:
                await task
            except Exception:  # pragma: no cover - 刷盘失败不影响停机
                pass
        self._access_flush_task = None
        try:
            await self.flush_access_counters()
        except Exception as exc:  # pragma: no cover
            logger.debug("[MemoryRetriever] 停机刷盘异常（忽略）: %s", exc)

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
            # 方案 B：懒初始化 FTS5 倒排索引与向量矩阵缓存（失败自动降级）
            self._ensure_engine_b(conn)
            self._sync_matrix_cache(conn)

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
            # 矩阵缓存可用时不再拉取 embedding_blob（省掉全表 struct.unpack）
            matrix_usable = bool(
                self._matrix_cache is not None and self._matrix_cache.built
            )
            l2_pool = self._fetch_l2_pool(
                conn=conn,
                session_id=session_id,
                pool_limit=pool_limit,
                with_embedding=not matrix_usable,
            )

            # 方案 B-1：优先走 FTS5 倒排 MATCH + bm25；无命中则回退 Python BM25
            fts_scores = self._fts_keyword_scores(
                conn=conn,
                query=clean_query,
                limit=pool_limit,
            )

            candidates, query_vec = await self._first_stage_recall(
                query=clean_query,
                l2_pool=l2_pool,
                pinned_ids=pinned_ids,
                pinned_contents=pinned_contents,
                candidate_k=effective_candidate_k,
                fts_scores=fts_scores or None,
            )

            # 3. 第二阶段精排重排序
            retrieved_l2 = await self._second_stage_rerank(
                query=clean_query,
                candidates=candidates,
                query_vec=query_vec,
                top_k=effective_top_k,
            )

            # R-05：检索保持纯读，访问计数仅在内存累加，异步定期刷盘
            hit_ids = [item.get("atom_id", "") for item in retrieved_l2 if item.get("atom_id")]
            self._record_access_memory(hit_ids)

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
