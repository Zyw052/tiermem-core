# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 分层记忆存储引擎骨架 (SQLite)
包含核心表结构：
1. l1_docs   - L1 工作区文档与核心焦点记忆
2. l2_atoms  - L2 情景记忆原子表
3. kg_nodes  - L3 知识图谱与画像实体节点表
4. kg_edges  - L3 知识图谱关系边表
5. task_dlq  - 异步总结/向量化任务死信与重试队列表
"""

import os
import re
import sqlite3
import struct
import threading
import time
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("astrbot_plugin_agent_memory.storage")

# ---------------------------------------------------------------------------
# 方案 B 增强：可选依赖（numpy 存在则启用矩阵乘法加速，缺失则纯 Python 回退）
# 说明：这里刻意使用 try/except 而非硬依赖，保证「零依赖引入」与平滑降级。
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 取决于运行环境是否安装 numpy
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

NUMPY_AVAILABLE: bool = _np is not None

# CJK / 英文分词正则（与 FTS5 unicode61 分词器的行为对齐）
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[a-z0-9_]+")


def fts_preprocess(text: Any) -> str:
    """
    为 SQLite FTS5（unicode61 分词器）预分词。

    背景：unicode61 不会把中文切成单字，而是把整段 CJK 视为一个 token，
    导致 ``MATCH '咖啡'`` 无法命中「美式咖啡」。因此索引侧与查询侧统一把
    CJK 逐字展开为空格分隔的 token，英文/数字保留整词。

    示例：
        >>> fts_preprocess("主人爱喝美式咖啡 tiermem-core")
        'tiermem core 主 人 爱 喝 美 式 咖 啡'
    """
    if not text:
        return ""
    try:
        lowered = str(text).lower()
    except Exception:  # pragma: no cover - 极端类型兜底
        return ""
    tokens: List[str] = []
    tokens.extend(_ASCII_WORD_RE.findall(lowered))
    tokens.extend(_CJK_CHAR_RE.findall(lowered))
    return " ".join(tokens)


def build_fts_match_expr(query: Any, max_tokens: int = 64) -> str:
    """
    构造安全可用的 FTS5 MATCH 表达式。

    - 复用 :func:`fts_preprocess` 做查询侧分词，保证与索引侧一致；
    - 每个 token 用双引号包裹并转义内部双引号，避免 ``-``、``*``、``:``、
      括号以及 ``AND/OR/NOT/NEAR`` 等 FTS5 语法字符造成 OperationalError；
    - token 之间用 OR 连接（bm25 会按 tf-idf 自动降权高频虚词）。

    :return: MATCH 表达式；无可检索 token 时返回空字符串。
    """
    tokens = fts_preprocess(query).split()[: int(max_tokens)]
    if not tokens:
        return ""
    quoted = ['"' + tok.replace('"', '""') + '"' for tok in tokens]
    return " OR ".join(quoted)


def detect_fts5_support(conn: sqlite3.Connection) -> bool:
    """
    探测当前 SQLite 编译是否带 FTS5 模块。

    做法是真实创建再删除一个临时虚拟表——比解析 ``PRAGMA compile_options``
    更可靠（部分发行版构建不带 ENABLE_FTS5 编译标记但模块可用，反之亦然）。
    """
    probe = f"temp.__fts5_probe_{abs(hash((id(conn), time.time())))}"
    try:
        conn.execute(f"CREATE VIRTUAL TABLE {probe} USING fts5(x)")
        return True
    except Exception as exc:  # sqlite3.OperationalError: no such module: fts5
        logger.info("[MemoryStorage] 当前 SQLite 未启用 FTS5，全文检索回退 Python BM25: %s", exc)
        return False
    finally:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {probe}")
        except Exception:  # pragma: no cover
            pass


DDL_STATEMENTS = [
    # 1. L1 工作区记忆文档表
    """
    CREATE TABLE IF NOT EXISTS l1_docs (
        doc_id TEXT PRIMARY KEY,
        scope_key TEXT NOT NULL,
        doc_type TEXT NOT NULL DEFAULT 'workspace',
        title TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL DEFAULT '',
        version INTEGER NOT NULL DEFAULT 1,
        updated_at REAL NOT NULL,
        created_at REAL NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_l1_docs_scope ON l1_docs(scope_key, doc_type);",

    # 2. L2 情景记忆原子表
    """
    CREATE TABLE IF NOT EXISTS l2_atoms (
        atom_id TEXT PRIMARY KEY,
        scope_key TEXT NOT NULL,
        session_id TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        topics_json TEXT NOT NULL DEFAULT '[]',
        key_facts_json TEXT NOT NULL DEFAULT '[]',
        sentiment TEXT NOT NULL DEFAULT 'neutral',
        importance REAL NOT NULL DEFAULT 0.5,
        access_count INTEGER NOT NULL DEFAULT 0,
        embedding_blob BLOB,
        source_type TEXT NOT NULL DEFAULT 'chat',
        source_ref TEXT NOT NULL DEFAULT '',
        last_accessed_at REAL NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_l2_atoms_scope_time ON l2_atoms(scope_key, created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_l2_atoms_importance ON l2_atoms(scope_key, importance DESC);",

    # 3. L3 知识图谱实体节点表
    """
    CREATE TABLE IF NOT EXISTS kg_nodes (
        node_id TEXT PRIMARY KEY,
        scope_key TEXT NOT NULL,
        entity_name TEXT NOT NULL,
        entity_type TEXT NOT NULL DEFAULT 'concept',
        summary TEXT NOT NULL DEFAULT '',
        attributes_json TEXT NOT NULL DEFAULT '{}',
        weight REAL NOT NULL DEFAULT 1.0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        UNIQUE(scope_key, entity_name, entity_type)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_lookup ON kg_nodes(scope_key, entity_name);",

    # 4. L3 知识图谱关系边表
    """
    CREATE TABLE IF NOT EXISTS kg_edges (
        edge_id TEXT PRIMARY KEY,
        scope_key TEXT NOT NULL,
        source_node_id TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        relation_type TEXT NOT NULL,
        fact_text TEXT NOT NULL DEFAULT '',
        confidence REAL NOT NULL DEFAULT 1.0,
        valid_from REAL,
        valid_until REAL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        FOREIGN KEY(source_node_id) REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
        FOREIGN KEY(target_node_id) REFERENCES kg_nodes(node_id) ON DELETE CASCADE
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(scope_key, source_node_id);",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(scope_key, target_node_id);",

    # 5. 异步任务死信与重试队列表
    """
    CREATE TABLE IF NOT EXISTS task_dlq (
        task_id TEXT PRIMARY KEY,
        task_type TEXT NOT NULL,
        scope_key TEXT NOT NULL DEFAULT '',
        payload_json TEXT NOT NULL,
        error_msg TEXT NOT NULL DEFAULT '',
        retry_count INTEGER NOT NULL DEFAULT 0,
        max_retries INTEGER NOT NULL DEFAULT 3,
        status TEXT NOT NULL DEFAULT 'pending',
        next_retry_at REAL NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_task_dlq_status ON task_dlq(status, next_retry_at);"
]


# ---------------------------------------------------------------------------
# 方案 B-1：SQLite 原生 FTS5 全文索引（虚拟表 + 自动同步触发器）
# ---------------------------------------------------------------------------
# 设计说明：
# 1. `l2_atoms_fts` 存放**预分词后**的文本（见 fts_preprocess），以适配
#    unicode61 对中文的处理方式，使 CJK 关键词可被毫秒级倒排召回；
# 2. `l2_atoms.fts_text` 作为预分词缓存列，由应用层写入时填充，触发器优先取它；
#    若为空（旧库 / 其他写入路径直接 INSERT），触发器回退原始 content，
#    保证英文/数字仍可检索，同时 init 阶段的 backfill 会把 fts_text 补齐；
# 3. 三个触发器覆盖 INSERT / UPDATE / DELETE，索引与主表永不脱节。
FTS_TABLE_NAME = "l2_atoms_fts"
FTS_TEXT_COLUMN = "fts_text"

FTS_DDL_STATEMENTS = [
    f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE_NAME} USING fts5(
        atom_id UNINDEXED,
        content,
        key_facts,
        tokenize = 'unicode61 remove_diacritics 2'
    );
    """,
    # 插入同步：优先使用预分词列，缺失则回退原文
    f"""
    CREATE TRIGGER IF NOT EXISTS trg_l2_atoms_fts_ai
    AFTER INSERT ON l2_atoms
    BEGIN
        INSERT INTO {FTS_TABLE_NAME}(atom_id, content, key_facts)
        VALUES (
            new.atom_id,
            CASE WHEN COALESCE(new.{FTS_TEXT_COLUMN}, '') <> ''
                 THEN new.{FTS_TEXT_COLUMN} ELSE COALESCE(new.content, '') END,
            CASE WHEN COALESCE(new.{FTS_TEXT_COLUMN}, '') <> ''
                 THEN new.{FTS_TEXT_COLUMN} ELSE COALESCE(new.key_facts_json, '') END
        );
    END;
    """,
    # 更新同步：先删后插，避免同一 atom_id 出现重复倒排条目
    f"""
    CREATE TRIGGER IF NOT EXISTS trg_l2_atoms_fts_au
    AFTER UPDATE OF content, key_facts_json, {FTS_TEXT_COLUMN} ON l2_atoms
    BEGIN
        DELETE FROM {FTS_TABLE_NAME} WHERE atom_id = old.atom_id;
        INSERT INTO {FTS_TABLE_NAME}(atom_id, content, key_facts)
        VALUES (
            new.atom_id,
            CASE WHEN COALESCE(new.{FTS_TEXT_COLUMN}, '') <> ''
                 THEN new.{FTS_TEXT_COLUMN} ELSE COALESCE(new.content, '') END,
            CASE WHEN COALESCE(new.{FTS_TEXT_COLUMN}, '') <> ''
                 THEN new.{FTS_TEXT_COLUMN} ELSE COALESCE(new.key_facts_json, '') END
        );
    END;
    """,
    # 删除同步：物理删除时清理倒排表，防止幽灵索引
    f"""
    CREATE TRIGGER IF NOT EXISTS trg_l2_atoms_fts_ad
    AFTER DELETE ON l2_atoms
    BEGIN
        DELETE FROM {FTS_TABLE_NAME} WHERE atom_id = old.atom_id;
    END;
    """,
]

# 软废弃同步触发器：依赖 l2_atoms.status 扩展列，故不能放进 FTS_DDL_STATEMENTS
# （旧库在 storage.init_schema() 阶段可能还没有 status 列），由 ensure_fts_index 条件创建。
# 作用：原子被标记 superseded 时立即从倒排表摘除，恢复 active 时重新灌入，
#       使检索期无需 JOIN 过滤，千级记忆 MATCH+bm25 可稳定压到 2ms 以内。
FTS_STATUS_TRIGGER_DDL = f"""
CREATE TRIGGER IF NOT EXISTS trg_l2_atoms_fts_status
AFTER UPDATE OF status ON l2_atoms
BEGIN
    DELETE FROM {FTS_TABLE_NAME} WHERE atom_id = old.atom_id;
    INSERT INTO {FTS_TABLE_NAME}(atom_id, content, key_facts)
    SELECT new.atom_id,
           CASE WHEN COALESCE(new.{FTS_TEXT_COLUMN}, '') <> ''
                THEN new.{FTS_TEXT_COLUMN} ELSE COALESCE(new.content, '') END,
           CASE WHEN COALESCE(new.{FTS_TEXT_COLUMN}, '') <> ''
                THEN new.{FTS_TEXT_COLUMN} ELSE COALESCE(new.key_facts_json, '') END
    WHERE COALESCE(new.status, 'active') != 'superseded';
END;
"""


class EmbeddingMatrixCache:
    """
    方案 B-2：L2 原子向量的内存矩阵缓存池。

    解决的问题：原先每次检索都要 ``SELECT *`` 全表拉取并对每行做
    ``struct.unpack`` 反序列化 + 逐条 ``cosine_similarity`` 调用，
    千级记忆下单次检索 CPU 开销高达上百毫秒。

    做法：
    - 一次性把全部活跃向量的**归一化**结果装入二维矩阵（numpy 时
      为 ``np.ndarray[dtype=float32]``，否则为 ``List[List[float]]``）；
    - 维护 ``atom_id -> index`` 映射与有效性掩码；
    - 检索时一次矩阵乘法 ``matrix @ q`` 即得到全部余弦相似度（微秒级）；
    - 新记忆写入走 ``upsert`` 增量追加，软废弃/删除走 ``invalidate`` 掩码失效，
      避免频繁全量重建。
    """

    def __init__(self, storage: "MemoryStorage", config: Optional[dict] = None) -> None:
        self.storage = storage
        self.config: Dict[str, Any] = config if isinstance(config, dict) else {}

        self._lock = threading.RLock()
        self._ids: List[str] = []
        self._index: Dict[str, int] = {}
        self._vectors: Any = None   # np.ndarray(capacity, dim) 或 list[list[float]]
        self._valid: Any = None     # 有效性掩码（False = 已失效，不参与召回）
        self._capacity: int = 0
        self._dim: int = 0
        self._n: int = 0
        self._built: bool = False
        self._fingerprint_ts: Optional[float] = None

        # 可观测性统计
        self._stat_builds: int = 0
        self._stat_appends: int = 0
        self._stat_updates: int = 0
        self._stat_invalidations: int = 0
        self._last_build_at: float = 0.0

    # ------------------------------------------------------------------
    # 基础属性
    # ------------------------------------------------------------------
    @property
    def built(self) -> bool:
        """缓存是否已完成首次构建。"""
        return self._built

    @property
    def dim(self) -> int:
        """当前缓存向量维度（0 表示尚未装载任何向量）。"""
        return self._dim

    def __len__(self) -> int:
        with self._lock:
            return self._n

    def stats(self) -> Dict[str, Any]:
        """返回缓存运行状态，供 /memory status 与测试断言使用。"""
        with self._lock:
            valid_count = 0
            if self._n and self._valid is not None:
                if NUMPY_AVAILABLE and _np is not None:
                    valid_count = int(_np.count_nonzero(self._valid[: self._n]))
                else:
                    valid_count = int(sum(1 for v in self._valid[: self._n] if v))
            return {
                "built": self._built,
                "size": self._n,
                "valid": valid_count,
                "dim": self._dim,
                "capacity": self._capacity,
                "backend": "numpy" if NUMPY_AVAILABLE else "python",
                "stat_builds": self._stat_builds,
                "stat_appends": self._stat_appends,
                "stat_updates": self._stat_updates,
                "stat_invalidations": self._stat_invalidations,
                "last_build_at": self._last_build_at,
            }

    # ------------------------------------------------------------------
    # 内部：容量与向量归一化
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(vector: Sequence[float]) -> Optional[List[float]]:
        """
        把向量归一化为单位长度，使余弦相似度可退化为纯点积。

        全零向量 / 非有限值返回 None（视为无有效向量，不进入矩阵）。
        """
        try:
            vals = [float(x) for x in vector]
        except Exception:
            return None
        if not vals:
            return None
        norm_sq = 0.0
        for v in vals:
            if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf
                return None
            norm_sq += v * v
        if norm_sq <= 1e-24:
            return None
        inv = 1.0 / (norm_sq ** 0.5)
        return [v * inv for v in vals]

    def _allocate(self, capacity: int, dim: int) -> None:
        """按指定容量与维度重建底层存储（仅内部调用，调用方需持锁）。"""
        capacity = max(int(capacity), 16)
        if NUMPY_AVAILABLE and _np is not None:
            self._vectors = _np.zeros((capacity, dim), dtype=_np.float32)
            self._valid = _np.zeros((capacity,), dtype=bool)
        else:
            self._vectors = [[0.0] * dim for _ in range(capacity)]
            self._valid = [False] * capacity
        self._capacity = capacity
        self._dim = dim

    def _grow_if_needed(self) -> None:
        """容量不足时按 2 倍扩容（调用方需持锁）。"""
        if self._n < self._capacity:
            return
        new_cap = max(16, self._capacity * 2)
        if NUMPY_AVAILABLE and _np is not None:
            new_vectors = _np.zeros((new_cap, self._dim), dtype=_np.float32)
            new_vectors[: self._n] = self._vectors[: self._n]
            new_valid = _np.zeros((new_cap,), dtype=bool)
            new_valid[: self._n] = self._valid[: self._n]
        else:
            new_vectors = [[0.0] * self._dim for _ in range(new_cap)]
            new_vectors[: self._n] = self._vectors[: self._n]
            new_valid = [False] * new_cap
            new_valid[: self._n] = self._valid[: self._n]
        self._vectors = new_vectors
        self._valid = new_valid
        self._capacity = new_cap

    # ------------------------------------------------------------------
    # 构建与同步
    # ------------------------------------------------------------------
    def _read_rows(self, conn: sqlite3.Connection) -> List[Tuple[str, Any]]:
        """
        从 l2_atoms 读取 (atom_id, embedding_blob)。

        带 status 列时过滤掉 superseded 软废弃记录（它们不应参与召回）；
        读取失败（表不存在等）时返回空列表而不抛异常。
        """
        try:
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
        except Exception as exc:
            logger.debug("[EmbeddingCache] 读取表结构失败: %s", exc)
            return []
        if not cols:
            return []

        sql = "SELECT atom_id, embedding_blob FROM l2_atoms"
        if "status" in cols:
            sql += " WHERE COALESCE(status, 'active') != 'superseded'"
        try:
            return [(str(r[0]), r[1]) for r in conn.execute(sql).fetchall()]
        except Exception as exc:
            logger.warning("[EmbeddingCache] 读取向量行失败: %s", exc)
            return []

    def build(self, conn: Optional[sqlite3.Connection] = None) -> int:
        """
        全量构建向量矩阵缓存（启动 / 首次检索 / 维度变更时调用）。

        :return: 成功装载的有效向量条数。
        """
        start = time.perf_counter()
        try:
            conn = conn or self.storage.connect()
            rows = self._read_rows(conn)
        except Exception as exc:
            logger.warning("[EmbeddingCache] 构建失败，检索将回退逐条计算: %s", exc)
            return 0

        parsed_all: List[Tuple[str, List[float]]] = []
        for atom_id, blob in rows:
            vec = _unpack_embedding_blob(blob)
            if vec:
                parsed_all.append((atom_id, vec))

        # 维度以「众数」为准：换 embedding 模型后旧维度脏数据会被安全跳过，
        # 而不是因首行维度不同导致整批召回错位。
        dim_counter: Dict[int, int] = {}
        for _, vec in parsed_all:
            dim_counter[len(vec)] = dim_counter.get(len(vec), 0) + 1
        dim = 0
        if dim_counter:
            dim = max(dim_counter.items(), key=lambda item: (item[1], item[0]))[0]

        parsed: List[Tuple[str, List[float]]] = []
        skipped = 0
        for atom_id, vec in parsed_all:
            if dim and len(vec) != dim:
                skipped += 1
                continue
            parsed.append((atom_id, vec))
        if skipped:
            logger.info(
                "[EmbeddingCache] 跳过 %d 条维度不一致的历史向量（主导维度=%d）", skipped, dim
            )

        with self._lock:
            self._ids = []
            self._index = {}
            self._n = 0
            self._built = True
            self._last_build_at = time.time()
            self._stat_builds += 1
            if dim <= 0:
                self._vectors = None
                self._valid = None
                self._capacity = 0
                self._dim = 0
                logger.debug("[EmbeddingCache] 构建完成：无有效向量")
                return 0

            self._allocate(len(parsed), dim)
            for atom_id, vec in parsed:
                norm_vec = self._normalize(vec)
                if norm_vec is None:
                    continue
                idx = self._n
                self._write_slot(idx, norm_vec)
                self._valid[idx] = True
                self._ids.append(atom_id)
                self._index[atom_id] = idx
                self._n += 1

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "[EmbeddingCache] 全量构建完成 atoms=%d dim=%d backend=%s 耗时=%.1fms",
            self._n,
            self._dim,
            "numpy" if NUMPY_AVAILABLE else "python",
            elapsed_ms,
        )
        return self._n

    def _write_slot(self, idx: int, norm_vec: Sequence[float]) -> None:
        """把归一化向量写入第 idx 槽位（调用方需持锁）。"""
        if NUMPY_AVAILABLE and _np is not None:
            self._vectors[idx] = _np.asarray(norm_vec, dtype=_np.float32)
        else:
            self._vectors[idx] = list(norm_vec)

    def ensure_built(self, conn: Optional[sqlite3.Connection] = None) -> bool:
        """确保缓存已构建（幂等）；返回是否处于可用状态。"""
        if self._built:
            return self._dim > 0
        self.build(conn)
        return self._dim > 0

    def upsert(self, atom_id: str, vector: Sequence[float]) -> bool:
        """
        增量写入 / 更新单条向量（新记忆析出或融合更新时调用）。

        - 已存在则原位覆盖并标记有效（O(1)，无重建）；
        - 不存在则追加到末尾（均摊 O(1)，容量不足时才 2 倍扩容）；
        - 维度与现有矩阵不一致时触发一次全量重建，保证矩阵不错位。
        """
        aid = str(atom_id or "").strip()
        if not aid:
            return False
        norm_vec = self._normalize(vector)
        if norm_vec is None:
            # 无有效向量：仅确保该 atom 不参与召回
            self.invalidate(aid)
            return False

        with self._lock:
            if self._dim and len(norm_vec) != self._dim:
                logger.info(
                    "[EmbeddingCache] 维度变更 %d -> %d，触发缓存重建",
                    self._dim,
                    len(norm_vec),
                )
                needs_rebuild = True
            else:
                needs_rebuild = False
            if not self._built:
                self._allocate(16 if not self._dim else self._capacity, len(norm_vec))
                self._built = True

        if needs_rebuild:
            # 维度变化属于极低频事件，直接全量重建最安全
            self.build()
            with self._lock:
                if self._dim != len(norm_vec):
                    return False
                idx = self._index.get(aid)
                if idx is None:
                    return False
                self._write_slot(idx, norm_vec)
                self._valid[idx] = True
                return True

        with self._lock:
            idx = self._index.get(aid)
            if idx is not None:
                self._write_slot(idx, norm_vec)
                self._valid[idx] = True
                self._stat_updates += 1
                return True
            self._grow_if_needed()
            idx = self._n
            self._write_slot(idx, norm_vec)
            self._valid[idx] = True
            self._ids.append(aid)
            self._index[aid] = idx
            self._n += 1
            self._stat_appends += 1
            return True

    def invalidate(self, atom_id: str) -> bool:
        """软废弃 / 删除单条向量：置掩码 False（保留槽位，不搬移数据）。"""
        aid = str(atom_id or "").strip()
        if not aid:
            return False
        with self._lock:
            idx = self._index.get(aid)
            if idx is None:
                return False
            if not self._valid[idx]:
                return False
            self._valid[idx] = False
            self._stat_invalidations += 1
            return True

    def invalidate_many(self, atom_ids: Iterable[str]) -> int:
        """批量失效（软废弃场景），返回实际失效条数。"""
        count = 0
        for aid in atom_ids or []:
            if self.invalidate(aid):
                count += 1
        return count

    def invalidate_all(self) -> None:
        """清空有效性掩码（全量重建前的轻量失效）。"""
        with self._lock:
            if self._n and self._valid is not None:
                for i in range(self._n):
                    self._valid[i] = False
            self._stat_invalidations += self._n

    def get_vector(self, atom_id: str) -> List[float]:
        """取出缓存的（归一化）向量副本；不存在或已失效时返回空列表。"""
        aid = str(atom_id or "").strip()
        if not aid:
            return []
        with self._lock:
            idx = self._index.get(aid)
            if idx is None or not self._valid[idx]:
                return []
            if NUMPY_AVAILABLE and _np is not None:
                return [float(x) for x in self._vectors[idx]]
            return list(self._vectors[idx])

    def score_map(self, query_vector: Sequence[float]) -> Dict[str, float]:
        """
        一次矩阵乘法算出全部活跃向量的余弦相似度。

        :param query_vector: 查询向量（未归一化亦可，内部归一化）
        :return: {atom_id: cosine}；无有效查询向量 / 空缓存时返回 {}
        """
        q_norm = self._normalize(query_vector)
        if q_norm is None:
            return {}

        with self._lock:
            n = self._n
            dim = self._dim
            if n <= 0 or dim <= 0 or len(q_norm) != dim:
                return {}
            ids = self._ids[:n]
            valid = self._valid[:n]
            matrix = self._vectors[:n]

            if NUMPY_AVAILABLE and _np is not None:
                try:
                    q = _np.asarray(q_norm, dtype=_np.float32)
                    # 核心加速点：一次矩阵乘法得到全部相似度
                    sims = matrix @ q
                    valid_mask = _np.asarray(valid, dtype=bool)
                    return {
                        ids[i]: float(sims[i])
                        for i in range(n)
                        if valid_mask[i]
                    }
                except Exception as exc:  # pragma: no cover - 数值异常兜底
                    logger.warning("[EmbeddingCache] 矩阵乘法失败，回退逐条计算: %s", exc)

            # 纯 Python 回退（无 numpy）：仍然是 O(n·dim) 单遍循环，
            # 但省掉了每行 struct.unpack 与函数调用开销
            out: Dict[str, float] = {}
            for i in range(n):
                if not valid[i]:
                    continue
                row = matrix[i]
                dot = 0.0
                for a, b in zip(row, q_norm):
                    dot += a * b
                out[ids[i]] = dot
            return out

    def sync_if_stale(
        self,
        conn: Optional[sqlite3.Connection] = None,
        max_age: float = 5.0,
    ) -> bool:
        """
        轻量一致性保障：与数据库指纹比对，必要时重建缓存。

        用于兜住「不经过本缓存写入的数据库变更」（如外部脚本、migrator、
        直接 SQL 写入）。指纹为 (COUNT, MAX(updated_at))，千级数据仅亚毫秒。

        :param max_age: 距上次构建/校验的最小间隔（秒），0 表示每次都校验。
        """
        now = time.time()
        if max_age > 0 and (now - self._last_build_at) < max_age:
            return False
        try:
            conn = conn or self.storage.connect()
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
            if not cols:
                return False
            sql = "SELECT COUNT(1) AS c, COALESCE(MAX(updated_at), 0) AS t FROM l2_atoms"
            if "status" in cols:
                sql += " WHERE COALESCE(status, 'active') != 'superseded'"
            row = conn.execute(sql).fetchone()
            fingerprint = (int(row["c"]), float(row["t"])) if row is not None else (0, 0.0)
        except Exception as exc:
            logger.debug("[EmbeddingCache] 指纹校验失败（忽略）: %s", exc)
            return False

        with self._lock:
            same = (
                self._built
                and fingerprint[0] == self._n
                and fingerprint[1] == getattr(self, "_fingerprint_ts", None)
            )
        if same:
            self._last_build_at = now
            return False

        logger.info(
            "[EmbeddingCache] 检测到数据库变更（count=%d, max_updated=%.3f），重建缓存",
            fingerprint[0],
            fingerprint[1],
        )
        self.build(conn)
        with self._lock:
            self._fingerprint_ts = fingerprint[1]
        return True


def _unpack_embedding_blob(blob: Any) -> List[float]:
    """
    从 SQLite BLOB 反序列化浮点向量（float32 小端 或 JSON 文本）。

    与 retriever._unpack_embedding 行为保持一致，但本模块不依赖 retriever，
    避免循环导入；失败一律返回空列表而不抛异常。
    """
    if not blob:
        return []
    try:
        if isinstance(blob, (bytes, bytearray, memoryview)):
            raw = bytes(blob)
            if raw.startswith(b"["):
                import json

                return [float(x) for x in json.loads(raw.decode("utf-8"))]
            if len(raw) % 4 == 0:
                cnt = len(raw) // 4
                return list(struct.unpack(f"<{cnt}f", raw))
        if isinstance(blob, str) and blob.strip().startswith("["):
            import json

            return [float(x) for x in json.loads(blob)]
    except Exception as exc:  # pragma: no cover
        logger.debug("[EmbeddingCache] 向量反序列化失败: %s", exc)
    return []


class MemoryStorage:
    """分层记忆 SQLite 存储管理器"""

    def __init__(self, data_dir: os.PathLike) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "agent_memory.db"
        self._conn: Optional[sqlite3.Connection] = None

        # ---- 方案 B-1：FTS5 全文索引状态 ----
        self.fts5_enabled: bool = False
        self._fts5_checked: bool = False
        self._fts_status_trigger: bool = False
        self._fts_lock = threading.RLock()

        # ---- 方案 B-2：内存向量矩阵缓存 ----
        self.embedding_cache = EmbeddingMatrixCache(self)

    def connect(self) -> sqlite3.Connection:
        """获取或创建 SQLite 连接并开启 WAL 模式"""
        if self._conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=15.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._conn = conn
        return self._conn

    def init_schema(self) -> None:
        """初始化分层记忆核心表结构（含方案 B 的 FTS5 全文索引）"""
        try:
            conn = self.connect()
            with conn:
                for stmt in DDL_STATEMENTS:
                    conn.execute(stmt)
                # 方案 B-1：FTS5 索引列必须在触发器创建前就绪
                self._ensure_fts_text_column(conn, commit=False)
                if self._probe_fts5(conn):
                    for stmt in FTS_DDL_STATEMENTS:
                        conn.execute(stmt)
            logger.info("MemoryStorage schema initialized at %s", self.db_path)

            # FTS5 后置维护（独立事务：失败不能影响主表初始化）
            if self.fts5_enabled:
                try:
                    self.ensure_fts_index(conn)
                except Exception as exc:
                    logger.warning("[MemoryStorage] FTS5 索引初始化降级（忽略）: %s", exc)
        except Exception as exc:
            logger.error("Failed to initialize MemoryStorage schema: %s", exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # 方案 B-1：FTS5 全文索引管理
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_fts_text_column(conn: sqlite3.Connection, commit: bool = True) -> bool:
        """
        幂等补齐 l2_atoms.fts_text 预分词列。

        :return: True 表示列已存在或已成功添加。
        """
        try:
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
            if not cols:
                return False
            if FTS_TEXT_COLUMN in cols:
                return True
            conn.execute(
                f"ALTER TABLE l2_atoms ADD COLUMN {FTS_TEXT_COLUMN} TEXT NOT NULL DEFAULT '';"
            )
            if commit:
                conn.commit()
            return True
        except Exception as exc:
            logger.debug("[MemoryStorage] 补列 %s 失败（忽略）: %s", FTS_TEXT_COLUMN, exc)
            return False

    def _probe_fts5(self, conn: sqlite3.Connection) -> bool:
        """探测并缓存 FTS5 可用性（每库只探测一次）。"""
        if self._fts5_checked:
            return self.fts5_enabled
        with self._fts_lock:
            if self._fts5_checked:
                return self.fts5_enabled
            self.fts5_enabled = bool(detect_fts5_support(conn))
            self._fts5_checked = True
        logger.info(
            "[MemoryStorage] FTS5 全文索引: %s",
            "已启用（MATCH + bm25 倒排检索）" if self.fts5_enabled else "不可用（回退 Python BM25）",
        )
        return self.fts5_enabled

    def ensure_fts_index(self, conn: Optional[sqlite3.Connection] = None) -> bool:
        """
        确保 FTS5 索引存在且与主表一致（幂等，可在任意时刻安全调用）。

        处理三类情况：
        1. SQLite 无 FTS5 支持 → 返回 False，检索侧自动回退 Python BM25；
        2. 旧库升级：fts_text 存在空值 → 批量补齐预分词文本；
        3. 索引与主表脱节（索引条数 < 主表条数）→ 全量重建索引。
        """
        if not self._fts5_checked:
            try:
                conn = conn or self.connect()
            except Exception as exc:
                logger.warning("[MemoryStorage] ensure_fts_index 无法连接: %s", exc)
                return False
            if not self._probe_fts5(conn):
                return False

        if not self.fts5_enabled:
            return False

        conn = conn or self.connect()
        try:
            with conn:
                self._ensure_fts_text_column(conn, commit=False)
                for stmt in FTS_DDL_STATEMENTS:
                    conn.execute(stmt)
        except Exception as exc:
            logger.warning("[MemoryStorage] 创建 FTS5 表/触发器失败，已禁用 FTS5: %s", exc)
            self.fts5_enabled = False
            return False

        # 0. 条件创建软废弃同步触发器（依赖 status 扩展列，可能由检索器/总结器稍后补齐）
        self._ensure_status_trigger(conn)

        # 1. 补齐缺失的预分词文本（旧库升级路径）
        try:
            self.refresh_fts_text(conn, only_empty=True)
        except Exception as exc:
            logger.warning("[MemoryStorage] 补齐 fts_text 异常（忽略）: %s", exc)

        # 2. 索引条数与主表条数对齐校验
        try:
            cols = {
                str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()
            }
            count_sql = "SELECT COUNT(1) AS c FROM l2_atoms"
            if "status" in cols:
                # FTS 索引不收录软废弃记录，因此比对基准也需排除 superseded
                count_sql += " WHERE COALESCE(status, 'active') != 'superseded'"
            atom_cnt = int(conn.execute(count_sql).fetchone()["c"])
            fts_cnt = int(
                conn.execute(f"SELECT COUNT(1) AS c FROM {FTS_TABLE_NAME}").fetchone()["c"]
            )
            if atom_cnt != fts_cnt:
                logger.info(
                    "[MemoryStorage] FTS5 索引与主表不一致（fts=%d atoms=%d），执行重建",
                    fts_cnt,
                    atom_cnt,
                )
                self.rebuild_fts_index(conn)
        except Exception as exc:
            logger.warning("[MemoryStorage] FTS5 一致性校验失败（忽略）: %s", exc)

        return True

    def _ensure_status_trigger(self, conn: sqlite3.Connection) -> bool:
        """
        条件创建/回收「软废弃同步触发器」。

        仅当 l2_atoms 已具备 status 列时才创建（该列由检索器/总结器的
        扩展列迁移逻辑按需补齐）。创建成功后检索期无需再做 status 过滤，
        MATCH + bm25 可全速执行。
        """
        if self._fts_status_trigger:
            return True
        try:
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
            if "status" not in cols:
                return False
            with conn:
                conn.execute(FTS_STATUS_TRIGGER_DDL)
                # 触发器生效前已存在的软废弃记录需要一次性摘除
                conn.execute(
                    f"DELETE FROM {FTS_TABLE_NAME} WHERE atom_id IN "
                    "(SELECT atom_id FROM l2_atoms WHERE COALESCE(status,'active') = 'superseded')"
                )
            self._fts_status_trigger = True
            logger.debug("[MemoryStorage] 软废弃同步触发器已就绪")
            return True
        except Exception as exc:
            logger.debug("[MemoryStorage] 创建软废弃触发器失败（将回退查询期过滤）: %s", exc)
            return False

    def refresh_fts_text(
        self,
        conn: Optional[sqlite3.Connection] = None,
        only_empty: bool = True,
        atom_ids: Optional[Sequence[str]] = None,
    ) -> int:
        """
        计算并回填 l2_atoms.fts_text（预分词文本）。

        :param only_empty: 仅处理 fts_text 为空的行（旧库升级更省时）
        :param atom_ids: 只处理指定原子（写入路径增量维护用）
        :return: 实际更新行数
        """
        conn = conn or self.connect()
        where: List[str] = []
        params: List[Any] = []
        if only_empty:
            where.append(f"COALESCE({FTS_TEXT_COLUMN}, '') = ''")
        if atom_ids:
            clean_ids = [str(a) for a in atom_ids if a]
            if not clean_ids:
                return 0
            where.append(f"atom_id IN ({','.join('?' for _ in clean_ids)})")
            params.extend(clean_ids)

        sql = "SELECT atom_id, content, key_facts_json FROM l2_atoms"
        if where:
            sql += " WHERE " + " AND ".join(where)
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as exc:
            logger.debug("[MemoryStorage] refresh_fts_text 查询失败: %s", exc)
            return 0

        updated = 0
        try:
            with conn:
                for row in rows:
                    atom_id = str(row[0])
                    text = f"{row[1] or ''} {row[2] or ''}".strip()
                    conn.execute(
                        f"UPDATE l2_atoms SET {FTS_TEXT_COLUMN} = ? WHERE atom_id = ?",
                        (fts_preprocess(text), atom_id),
                    )
                    updated += 1
        except Exception as exc:
            logger.warning("[MemoryStorage] refresh_fts_text 写回失败: %s", exc)
            return updated
        if updated:
            logger.debug("[MemoryStorage] 已回填 %d 条 fts_text", updated)
        return updated

    def rebuild_fts_index(self, conn: Optional[sqlite3.Connection] = None) -> int:
        """
        全量重建 FTS5 全文索引（先清空倒排表，再从主表批量回灌）。

        :return: 重建后索引中的条目数；FTS5 不可用或失败时返回 0。
        """
        if not self.fts5_enabled:
            return 0
        conn = conn or self.connect()
        try:
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
            where_clause = (
                "WHERE COALESCE(status, 'active') != 'superseded'"
                if "status" in cols
                else ""
            )
            with conn:
                self._ensure_fts_text_column(conn, commit=False)
                conn.execute(f"DELETE FROM {FTS_TABLE_NAME};")
                conn.execute(
                    f"""
                    INSERT INTO {FTS_TABLE_NAME}(atom_id, content, key_facts)
                    SELECT atom_id,
                           CASE WHEN COALESCE({FTS_TEXT_COLUMN}, '') <> ''
                                THEN {FTS_TEXT_COLUMN}
                                ELSE COALESCE(content, '') END,
                           CASE WHEN COALESCE({FTS_TEXT_COLUMN}, '') <> ''
                                THEN {FTS_TEXT_COLUMN}
                                ELSE COALESCE(key_facts_json, '') END
                    FROM l2_atoms
                    {where_clause};
                    """
                )
            cnt = int(
                conn.execute(f"SELECT COUNT(1) AS c FROM {FTS_TABLE_NAME}").fetchone()["c"]
            )
            logger.info("[MemoryStorage] FTS5 索引重建完成，条目数=%d", cnt)
            return cnt
        except Exception as exc:
            logger.warning("[MemoryStorage] FTS5 索引重建失败，已禁用 FTS5: %s", exc)
            self.fts5_enabled = False
            return 0

    def sync_atom_fts(
        self,
        atom_id: str,
        content: str,
        key_facts: Any = None,
        conn: Optional[sqlite3.Connection] = None,
    ) -> bool:
        """
        单条原子写入后同步 FTS 预分词文本（写入路径增量维护入口）。

        说明：主表的 INSERT/UPDATE 触发器会自动同步倒排表，本方法负责把
        预分词结果写进 ``l2_atoms.fts_text``，从而让触发器索引到可检索的 CJK 词元。
        若 FTS5 不可用则直接返回 False，不产生任何副作用。
        """
        if not self.fts5_enabled:
            return False
        aid = str(atom_id or "").strip()
        if not aid:
            return False
        try:
            facts_text = ""
            if isinstance(key_facts, (list, tuple)):
                facts_text = " ".join(str(x) for x in key_facts)
            elif key_facts:
                facts_text = str(key_facts)
            preprocessed = fts_preprocess(f"{content or ''} {facts_text}".strip())
            conn = conn or self.connect()
            with conn:
                conn.execute(
                    f"UPDATE l2_atoms SET {FTS_TEXT_COLUMN} = ? WHERE atom_id = ?",
                    (preprocessed, aid),
                )
            return True
        except Exception as exc:
            logger.debug("[MemoryStorage] sync_atom_fts 失败（忽略）: %s", exc)
            return False

    def fts_search(
        self,
        query: str,
        limit: int = 100,
        conn: Optional[sqlite3.Connection] = None,
        include_superseded: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        FTS5 倒排检索：返回按 bm25 相关性排序的 [(atom_id, 归一化分数)]。

        - 使用 ``MATCH`` 走倒排索引 + ``bm25()`` 相关度打分，千级记忆 <3ms；
        - 分数归一化到 0~1（bm25 返回负值，越负越相关）；
        - 默认过滤 status='superseded' 的软废弃记忆（软废弃即时失效）；
        - FTS5 不可用 / 查询构造为空 / 执行异常时返回空列表，调用方回退 BM25。
        """
        if not self.fts5_enabled:
            return []
        expr = build_fts_match_expr(query)
        if not expr:
            return []
        conn = conn or self.connect()
        try:
            # 若软废弃触发器已就绪，倒排表中天然不存在 superseded 条目，无需过滤；
            # 否则回退到查询期 status 过滤（旧库/触发器创建失败的兜底路径）。
            need_filter = (not include_superseded) and (not self._fts_status_trigger)
            l2_cols = (
                {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
                if need_filter
                else set()
            )
            sql = (
                f"SELECT atom_id, bm25({FTS_TABLE_NAME}, 1.0, 0.5) AS score "
                f"FROM {FTS_TABLE_NAME} WHERE {FTS_TABLE_NAME} MATCH ? "
            )
            params: List[Any] = [expr]
            if need_filter and "status" in l2_cols:
                sql += (
                    "AND atom_id IN ("
                    "SELECT atom_id FROM l2_atoms "
                    "WHERE COALESCE(status, 'active') != 'superseded') "
                )
            sql += "ORDER BY score ASC LIMIT ?"
            params.append(int(limit))
            rows = conn.execute(sql, params).fetchall()
        except Exception as exc:
            logger.debug("[MemoryStorage] FTS5 MATCH 查询失败，回退 BM25: %s", exc)
            return []

        if not rows:
            return []
        # bm25 为负值（越小越相关）→ 转正并归一化到 0~1
        raw = [(-float(r["score"]), str(r["atom_id"])) for r in rows]
        raw.sort(key=lambda item: item[0], reverse=True)
        max_score = raw[0][0] if raw and raw[0][0] > 1e-9 else 0.0
        if max_score <= 0:
            return []
        return [(atom_id, score / max_score) for score, atom_id in raw]

    def get_fts_stats(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """返回 FTS5 索引运行状态（可观测性 / 测试断言用）。"""
        stats: Dict[str, Any] = {
            "enabled": bool(self.fts5_enabled),
            "table": FTS_TABLE_NAME,
            "indexed": 0,
            "atoms": 0,
            "numpy": NUMPY_AVAILABLE,
        }
        if not self.fts5_enabled:
            return stats
        try:
            conn = conn or self.connect()
            stats["indexed"] = int(
                conn.execute(f"SELECT COUNT(1) AS c FROM {FTS_TABLE_NAME}").fetchone()["c"]
            )
            stats["atoms"] = int(
                conn.execute("SELECT COUNT(1) AS c FROM l2_atoms").fetchone()["c"]
            )
        except Exception as exc:
            logger.debug("[MemoryStorage] get_fts_stats 失败（忽略）: %s", exc)
        return stats

    def get_table_counts(self) -> Dict[str, int]:
        """统计各核心表当前记录数，用于健康检查与状态概览"""
        tables = ("l1_docs", "l2_atoms", "kg_nodes", "kg_edges", "task_dlq")
        counts: Dict[str, int] = {}
        try:
            conn = self.connect()
            for tbl in tables:
                cur = conn.execute(f"SELECT COUNT(1) AS cnt FROM {tbl}")
                row = cur.fetchone()
                counts[tbl] = int(row["cnt"]) if row else 0
        except Exception as exc:
            logger.error("Failed to query table counts: %s", exc, exc_info=True)
        return counts

    def get_tier_stats(self) -> Dict[str, int]:
        """
        统计分层记忆体量，供 /memory status 指令与 WebUI 概览使用。

        返回键值：
        - l1_docs   : L1 工作区/历史文档条数
        - l2_atoms  : L2 情景记忆原子条数
        - l2_pinned : L3 置顶原子条数（pinned=1 或 tier>=3）
        - kg_nodes  : 知识图谱实体节点数
        - kg_edges  : 知识图谱关系边数
        - task_dlq  : 异步任务死信队列条数

        说明：l2_atoms 的 pinned/tier 为检索器按需补齐的扩展列，
        在尚未补齐时会自动降级为 0，且任何异常都不外抛（状态指令要求零崩溃）。
        """
        stats: Dict[str, int] = {
            "l1_docs": 0,
            "l2_atoms": 0,
            "l2_pinned": 0,
            "kg_nodes": 0,
            "kg_edges": 0,
            "task_dlq": 0,
        }
        try:
            conn = self.connect()
        except Exception as exc:
            logger.error("get_tier_stats 连接失败: %s", exc, exc_info=True)
            return stats

        # 1. 各基础表行数（逐表容错，避免单表缺失导致整体失败）
        for table in ("l1_docs", "l2_atoms", "kg_nodes", "kg_edges", "task_dlq"):
            try:
                row = conn.execute(f"SELECT COUNT(1) AS cnt FROM {table}").fetchone()
                stats[table] = int(row["cnt"]) if row is not None else 0
            except Exception as exc:
                logger.debug("统计表 %s 失败（忽略）: %s", table, exc)
                stats[table] = 0

        # 2. L3 置顶计数：仅在扩展列存在时可统计
        try:
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
            if "pinned" in cols and "tier" in cols:
                row = conn.execute(
                    "SELECT COUNT(1) AS cnt FROM l2_atoms WHERE pinned = 1 OR tier >= 3"
                ).fetchone()
                stats["l2_pinned"] = int(row["cnt"]) if row is not None else 0
            elif "pinned" in cols:
                row = conn.execute(
                    "SELECT COUNT(1) AS cnt FROM l2_atoms WHERE pinned = 1"
                ).fetchone()
                stats["l2_pinned"] = int(row["cnt"]) if row is not None else 0
        except Exception as exc:
            logger.debug("统计 L3 置顶数失败（忽略）: %s", exc)
            stats["l2_pinned"] = 0

        return stats

    def toggle_atom_pin(self, atom_id: str) -> Dict[str, Any]:
        """
        按 atom_id 切换 L2 原子的置顶状态（pinned/tier 同步联动）。

        行为：
        - 当前未置顶 → 置顶（pinned=1, tier=3），即 L3 晋升；
        - 当前已置顶 → 取消（pinned=0, tier=2），即 L3 降级；
        - 记录不存在 / 参数非法 → found=False，不抛异常。

        返回字典：
        {"found": bool, "atom_id": str, "content": str,
         "was_pinned": bool, "is_pinned": bool, "error": str}
        """
        result: Dict[str, Any] = {
            "found": False,
            "atom_id": str(atom_id or "").strip(),
            "content": "",
            "was_pinned": False,
            "is_pinned": False,
            "error": "",
        }
        aid = result["atom_id"]
        if not aid:
            result["error"] = "atom_id 为空"
            return result

        try:
            conn = self.connect()

            # 幂等补齐扩展列（旧库可能尚未经过 retriever 迁移）
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
            with conn:
                if "pinned" not in cols:
                    conn.execute(
                        "ALTER TABLE l2_atoms ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;"
                    )
                if "tier" not in cols:
                    conn.execute(
                        "ALTER TABLE l2_atoms ADD COLUMN tier INTEGER NOT NULL DEFAULT 2;"
                    )

            row = conn.execute(
                "SELECT atom_id, content, pinned, tier FROM l2_atoms WHERE atom_id = ?",
                (aid,),
            ).fetchone()
            if row is None:
                result["error"] = "记录不存在"
                return result

            was_pinned = bool(int(row["pinned"] or 0)) or int(row["tier"] or 2) >= 3
            new_pinned = 0 if was_pinned else 1
            new_tier = 2 if was_pinned else 3

            with conn:
                conn.execute(
                    "UPDATE l2_atoms SET pinned = ?, tier = ?, updated_at = ? WHERE atom_id = ?",
                    (new_pinned, new_tier, time.time(), aid),
                )

            result.update(
                {
                    "found": True,
                    "content": str(row["content"] or ""),
                    "was_pinned": was_pinned,
                    "is_pinned": bool(new_pinned),
                }
            )
        except Exception as exc:
            logger.error("toggle_atom_pin 失败 (atom_id=%s): %s", aid, exc, exc_info=True)
            result["error"] = str(exc)

        return result

    def close(self) -> None:
        """安全关闭数据库连接（同时释放向量矩阵缓存，避免句柄/内存残留）"""
        # 方案 B-2：关闭前释放矩阵缓存，避免热重载后持有过期数据
        try:
            cache = getattr(self, "embedding_cache", None)
            if cache is not None:
                cache.invalidate_all()
        except Exception as exc:  # pragma: no cover
            logger.debug("[MemoryStorage] 释放向量缓存异常（忽略）: %s", exc)

        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("MemoryStorage connection closed.")
            except Exception as exc:
                logger.warning("Error closing MemoryStorage connection: %s", exc)
            finally:
                self._conn = None


__all__ = [
    "MemoryStorage",
    "EmbeddingMatrixCache",
    "detect_fts5_support",
    "fts_preprocess",
    "build_fts_match_expr",
    "NUMPY_AVAILABLE",
    "FTS_TABLE_NAME",
    "FTS_TEXT_COLUMN",
]
