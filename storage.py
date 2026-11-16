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

import json
import os
import re
import sqlite3
import struct
import threading
import time
import logging
import uuid
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

# ---------------------------------------------------------------------------
# 方案 C：L3 时序知识图谱扩展列（幂等补列，不破坏既有 kg_nodes/kg_edges 结构）
# ---------------------------------------------------------------------------
# 设计说明：
# 1. 基础 DDL 已建好 kg_nodes / kg_edges 两张表，但缺少运行期所需的
#    「提及次数 / 最新出现时间 / 边权重 / 关联原子 / 失效标记」字段；
# 2. 这里以 ALTER TABLE ADD COLUMN 方式幂等补齐，旧库升级零数据丢失，
#    新库直接由 DDL_STATEMENTS 建全；
# 3. 关联原子 atom_id 是「记忆原子 ↔ 图谱关系」的双向回溯锚点：
#    原子被软废弃时，可沿 atom_id 精准失效对应关系边（见 invalidate_kg_edges_for_atoms）。
KG_NODE_EXTENDED_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("mention_count", "INTEGER NOT NULL DEFAULT 1"),
    ("first_seen_at", "REAL NOT NULL DEFAULT 0.0"),
    ("last_seen_at", "REAL NOT NULL DEFAULT 0.0"),
)

KG_EDGE_EXTENDED_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("weight", "REAL NOT NULL DEFAULT 1.0"),
    ("atom_id", "TEXT NOT NULL DEFAULT ''"),
    ("status", "TEXT NOT NULL DEFAULT 'active'"),
    ("invalidated_at", "REAL NOT NULL DEFAULT 0.0"),
    ("reinforcement_count", "INTEGER NOT NULL DEFAULT 1"),
)


def ensure_kg_columns(conn: sqlite3.Connection) -> bool:
    """
    幂等补齐 kg_nodes / kg_edges 的方案 C 扩展列。

    :return: True 表示两表扩展列均已就绪；False 表示探测失败（调用方应降级处理）。
    """
    try:
        for table, cols in (
            ("kg_nodes", KG_NODE_EXTENDED_COLUMNS),
            ("kg_edges", KG_EDGE_EXTENDED_COLUMNS),
        ):
            existing = {
                str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not existing:
                # 表不存在（极端情况），交由调用方初始化后再补
                continue
            for col_name, col_def in cols:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")
        return True
    except Exception as exc:  # pragma: no cover - 结构异常极罕见
        logger.warning("[MemoryStorage] 图谱扩展列补齐失败（降级）: %s", exc)
        return False

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
        mention_count INTEGER NOT NULL DEFAULT 1,
        first_seen_at REAL NOT NULL DEFAULT 0.0,
        last_seen_at REAL NOT NULL DEFAULT 0.0,
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
        weight REAL NOT NULL DEFAULT 1.0,
        atom_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'active',
        invalidated_at REAL NOT NULL DEFAULT 0.0,
        reinforcement_count INTEGER NOT NULL DEFAULT 1,
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
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_atom ON kg_edges(scope_key, atom_id);",

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
                # 方案 C：旧库先补齐图谱扩展列，保证后续索引语句可正常创建
                ensure_kg_columns(conn)
                for stmt in DDL_STATEMENTS:
                    conn.execute(stmt)
                # 方案 B-1：FTS5 索引列必须在触发器创建前就绪
                self._ensure_fts_text_column(conn, commit=False)
                if self._probe_fts5(conn):
                    for stmt in FTS_DDL_STATEMENTS:
                        conn.execute(stmt)
                # 方案 C：幂等补齐知识图谱扩展列（旧库平滑升级）
                ensure_kg_columns(conn)
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

    # ------------------------------------------------------------------
    # 方案 C：L3 时序知识图谱读写闭环
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_kg_token(text: Any, limit: int = 64) -> str:
        """图谱实体/关系名归一化：去空白、去包裹标点、截断超长。"""
        raw = str(text or "").strip()
        raw = raw.strip(" \t\r\n,，。.;；:：!！?？\"'“”‘’()（）[]【】《》<>")
        return raw[:limit]

    def upsert_kg_triple(
        self,
        scope_key: str,
        head: str,
        relation: str,
        tail: str,
        atom_id: str = "",
        fact_text: str = "",
        now: Optional[float] = None,
        head_type: str = "concept",
        tail_type: str = "concept",
        confidence: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        写入一条 ``(head, relation, tail)`` 三元组到 kg_nodes / kg_edges。

        行为契约（全部幂等）：
        - 实体节点按 ``(scope_key, entity_name, entity_type)`` 去重，
          重复出现时 ``mention_count += 1``、``last_seen_at`` 刷新、``weight`` 轻微增强；
        - 关系边按 ``(scope_key, source, target, relation_type, atom_id)`` 去重，
          重复出现时 ``weight += 1``、``reinforcement_count += 1``；
        - 若该边此前被标记 invalidated，重新观察到即复活为 active（事实再次被提及）。

        :return: 成功时返回节点/边 ID 与是否新建的信息；失败或无有效实体返回 None。
        """
        head_name = self._normalize_kg_token(head)
        tail_name = self._normalize_kg_token(tail)
        relation_type = self._normalize_kg_token(relation, limit=32) or "related_to"
        if not head_name or not tail_name or head_name == tail_name:
            return None

        stamp = float(now if now is not None else time.time())
        scope = str(scope_key or "global")
        atom_key = str(atom_id or "")

        try:
            conn = self.connect()
            ensure_kg_columns(conn)

            def _touch_node(name: str, etype: str) -> Tuple[str, bool]:
                """获取或创建实体节点，返回 (node_id, is_new)。"""
                etype_clean = str(etype or "concept")[:24]
                row = conn.execute(
                    """
                    SELECT node_id FROM kg_nodes
                    WHERE scope_key = ? AND entity_name = ? AND entity_type = ?
                    """,
                    (scope, name, etype_clean),
                ).fetchone()
                if row is not None:
                    node_id = str(row["node_id"])
                    conn.execute(
                        """
                        UPDATE kg_nodes
                        SET mention_count = COALESCE(mention_count, 1) + 1,
                            last_seen_at = ?,
                            weight = MIN(10.0, COALESCE(weight, 1.0) + 0.1),
                            updated_at = ?
                        WHERE node_id = ?
                        """,
                        (stamp, stamp, node_id),
                    )
                    return node_id, False
                node_id = f"kgn_{uuid.uuid4().hex[:16]}"
                conn.execute(
                    """
                    INSERT INTO kg_nodes (
                        node_id, scope_key, entity_name, entity_type, summary,
                        attributes_json, weight, mention_count, first_seen_at,
                        last_seen_at, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, '', '{}', 1.0, 1, ?, ?, ?, ?)
                    """,
                    (node_id, scope, name, etype_clean, stamp, stamp, stamp, stamp),
                )
                return node_id, True

            with conn:
                head_id, head_new = _touch_node(head_name, head_type)
                tail_id, tail_new = _touch_node(tail_name, tail_type)

                edge_row = conn.execute(
                    """
                    SELECT edge_id, status FROM kg_edges
                    WHERE scope_key = ? AND source_node_id = ? AND target_node_id = ?
                      AND relation_type = ? AND atom_id = ?
                    """,
                    (scope, head_id, tail_id, relation_type, atom_key),
                ).fetchone()

                if edge_row is not None:
                    edge_id = str(edge_row["edge_id"])
                    conn.execute(
                        """
                        UPDATE kg_edges
                        SET weight = COALESCE(weight, 1.0) + 1.0,
                            reinforcement_count = COALESCE(reinforcement_count, 1) + 1,
                            fact_text = CASE WHEN ? != '' THEN ? ELSE fact_text END,
                            confidence = MAX(COALESCE(confidence, 0.0), ?),
                            status = 'active',
                            invalidated_at = 0.0,
                            updated_at = ?
                        WHERE edge_id = ?
                        """,
                        (str(fact_text or ""), str(fact_text or ""), float(confidence), stamp, edge_id),
                    )
                    edge_new = False
                else:
                    edge_id = f"kge_{uuid.uuid4().hex[:16]}"
                    conn.execute(
                        """
                        INSERT INTO kg_edges (
                            edge_id, scope_key, source_node_id, target_node_id,
                            relation_type, fact_text, confidence, weight, atom_id,
                            status, invalidated_at, reinforcement_count,
                            valid_from, valid_until, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, ?, 'active', 0.0, 1, ?, NULL, ?, ?)
                        """,
                        (
                            edge_id,
                            scope,
                            head_id,
                            tail_id,
                            relation_type,
                            str(fact_text or ""),
                            float(confidence),
                            atom_key,
                            stamp,
                            stamp,
                            stamp,
                        ),
                    )
                    edge_new = True

            return {
                "head": head_name,
                "relation": relation_type,
                "tail": tail_name,
                "head_node_id": head_id,
                "tail_node_id": tail_id,
                "edge_id": edge_id,
                "node_created": bool(head_new or tail_new),
                "edge_created": edge_new,
            }
        except Exception as exc:
            logger.warning("[MemoryStorage] 图谱三元组写入失败（已忽略）: %s", exc)
            return None

    def expand_kg_subgraph(
        self,
        scope_key: str,
        query_terms: Sequence[str],
        max_entities: int = 5,
        max_edges: int = 20,
        include_global: bool = True,
        all_scopes: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        以 query 命中的实体为种子，做 **1-hop 关联子图检索**。

        :param query_terms: 已切分的 query 关键词（实体名匹配用）
        :param all_scopes: True 表示不限作用域（仅在关闭严格会话隔离时使用）
        :return: 关联边列表，每项含 ``entity / neighbor / relation / direction /
                 atom_id / weight / fact_text``，按边权降序排列。
        """
        terms = [self._normalize_kg_token(t) for t in (query_terms or [])]
        terms = [t for t in terms if t]
        if not terms:
            return []

        scopes: List[str] = [str(scope_key or "global")]
        if include_global and "global" not in scopes:
            scopes.append("global")

        try:
            conn = self.connect()
            ensure_kg_columns(conn)

            placeholders = ",".join("?" for _ in scopes)
            scope_clause = "" if all_scopes else f"scope_key IN ({placeholders}) AND "
            scope_args: Tuple[Any, ...] = () if all_scopes else tuple(scopes)
            # 1. 实体种子：精确命中优先，其次包含命中（query 词包含实体名或反之）
            seed_nodes: List[Dict[str, Any]] = []
            seen_seed: set[str] = set()
            for term in terms:
                like_term = f"%{term}%"
                rows = conn.execute(
                    f"""
                    SELECT node_id, entity_name, entity_type, mention_count, weight
                    FROM kg_nodes
                    WHERE {scope_clause}(
                            entity_name = ?
                            OR entity_name LIKE ?
                            OR ? LIKE '%' || entity_name || '%'
                          )
                    ORDER BY
                        CASE WHEN entity_name = ? THEN 0 ELSE 1 END ASC,
                        mention_count DESC,
                        weight DESC
                    LIMIT ?
                    """,
                    (*scope_args, term, like_term, term, term, int(max_entities)),
                ).fetchall()
                for row in rows:
                    node_id = str(row["node_id"])
                    if node_id in seen_seed:
                        continue
                    seen_seed.add(node_id)
                    seed_nodes.append(
                        {
                            "node_id": node_id,
                            "entity_name": str(row["entity_name"]),
                            "entity_type": str(row["entity_type"] or "concept"),
                            "mention_count": int(row["mention_count"] or 1),
                        }
                    )
                    if len(seed_nodes) >= int(max_entities):
                        break
                if len(seed_nodes) >= int(max_entities):
                    break

            if not seed_nodes:
                return []

            # 2. 1-hop 关联边（双向：种子为 source 或 target 均纳入）
            seed_ids = [n["node_id"] for n in seed_nodes]
            id_ph = ",".join("?" for _ in seed_ids)
            edge_scope_clause = (
                "" if all_scopes else f"e.scope_key IN ({placeholders}) AND "
            )
            edge_rows = conn.execute(
                f"""
                SELECT e.edge_id, e.source_node_id, e.target_node_id,
                       e.relation_type, e.weight, e.atom_id, e.fact_text, e.confidence,
                       sn.entity_name AS source_name, tn.entity_name AS target_name
                FROM kg_edges e
                JOIN kg_nodes sn ON sn.node_id = e.source_node_id
                JOIN kg_nodes tn ON tn.node_id = e.target_node_id
                WHERE {edge_scope_clause}COALESCE(e.status, 'active') = 'active'
                  AND (e.source_node_id IN ({id_ph}) OR e.target_node_id IN ({id_ph}))
                ORDER BY COALESCE(e.weight, 1.0) DESC, e.updated_at DESC
                LIMIT ?
                """,
                (*scope_args, *seed_ids, *seed_ids, int(max_edges)),
            ).fetchall()

            seed_lookup = {n["node_id"]: n["entity_name"] for n in seed_nodes}
            results: List[Dict[str, Any]] = []
            for row in edge_rows:
                src = str(row["source_node_id"])
                tgt = str(row["target_node_id"])
                if src in seed_lookup:
                    entity, neighbor, direction = seed_lookup[src], str(row["target_name"]), "out"
                else:
                    entity, neighbor, direction = seed_lookup[tgt], str(row["source_name"]), "in"
                results.append(
                    {
                        "entity": entity,
                        "neighbor": neighbor,
                        "relation": str(row["relation_type"]),
                        "direction": direction,
                        "atom_id": str(row["atom_id"] or ""),
                        "weight": float(row["weight"] or 1.0),
                        "confidence": float(row["confidence"] or 1.0),
                        "fact_text": str(row["fact_text"] or ""),
                        "edge_id": str(row["edge_id"]),
                    }
                )
            return results
        except Exception as exc:
            logger.warning("[MemoryStorage] 图谱 1-hop 子图检索失败（降级为无图谱召回）: %s", exc)
            return []

    def invalidate_kg_edges_for_atoms(
        self, atom_ids: Sequence[str], now: Optional[float] = None
    ) -> int:
        """
        原子被软废弃时，联动失效其关联的关系边。

        - 边不物理删除，标记 ``status='invalidated'`` 并记录 ``invalidated_at``；
        - ``weight`` 归零，``valid_until`` 记录失效时刻，保留可追溯性；
        - 返回受影响边数（异常安全，失败返回 0）。
        """
        ids = [str(a) for a in (atom_ids or []) if str(a or "").strip()]
        if not ids:
            return 0
        stamp = float(now if now is not None else time.time())
        try:
            conn = self.connect()
            ensure_kg_columns(conn)
            ph = ",".join("?" for _ in ids)
            with conn:
                cur = conn.execute(
                    f"""
                    UPDATE kg_edges
                    SET status = 'invalidated',
                        invalidated_at = ?,
                        valid_until = ?,
                        weight = 0.0,
                        updated_at = ?
                    WHERE atom_id IN ({ph})
                      AND COALESCE(status, 'active') = 'active'
                    """,
                    (stamp, stamp, stamp, *ids),
                )
                return int(cur.rowcount or 0)
        except Exception as exc:
            logger.warning("[MemoryStorage] 图谱边联动失效失败（已忽略）: %s", exc)
            return 0

    def get_kg_stats(self) -> Dict[str, int]:
        """图谱健康统计：节点数 / 活跃边数 / 失效边数 / 孤立节点数（全容错）。"""
        stats = {
            "nodes": 0,
            "edges": 0,
            "active_edges": 0,
            "invalidated_edges": 0,
            "orphan_nodes": 0,
        }
        try:
            conn = self.connect()
            ensure_kg_columns(conn)
            stats["nodes"] = int(
                conn.execute("SELECT COUNT(1) AS c FROM kg_nodes").fetchone()["c"]
            )
            stats["edges"] = int(
                conn.execute("SELECT COUNT(1) AS c FROM kg_edges").fetchone()["c"]
            )
            stats["active_edges"] = int(
                conn.execute(
                    "SELECT COUNT(1) AS c FROM kg_edges WHERE COALESCE(status,'active')='active'"
                ).fetchone()["c"]
            )
            stats["invalidated_edges"] = stats["edges"] - stats["active_edges"]
            stats["orphan_nodes"] = int(
                conn.execute(
                    """
                    SELECT COUNT(1) AS c FROM kg_nodes n
                    WHERE NOT EXISTS (
                        SELECT 1 FROM kg_edges e
                        WHERE e.source_node_id = n.node_id OR e.target_node_id = n.node_id
                    )
                    """
                ).fetchone()["c"]
            )
        except Exception as exc:
            logger.debug("[MemoryStorage] 图谱统计失败（忽略）: %s", exc)
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

    # ------------------------------------------------------------------
    # P0：WebUI 只读查询接口（记忆浏览 / 搜索 / 死信排查）
    #   设计原则：全部只读、全部容错，任何异常返回空结果而非抛出。
    # ------------------------------------------------------------------
    @staticmethod
    def _atom_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """把 l2_atoms 行转换成 WebUI 友好的 dict（容错处理 JSON 字段）。"""
        def _loads(raw: Any, fallback: Any) -> Any:
            try:
                if raw is None or raw == "":
                    return fallback
                value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
                return value
            except Exception:
                return fallback

        keys = set(row.keys()) if hasattr(row, "keys") else set()
        pinned = int(row["pinned"] or 0) if "pinned" in keys else 0
        tier = int(row["tier"] or 2) if "tier" in keys else 2
        return {
            "atom_id": str(row["atom_id"]),
            "scope_key": str(row["scope_key"] or ""),
            "session_id": str(row["session_id"] or ""),
            "content": str(row["content"] or ""),
            "topics": _loads(row["topics_json"], []),
            "key_facts": _loads(row["key_facts_json"], []),
            "sentiment": str(row["sentiment"] or "neutral"),
            "importance": float(row["importance"] or 0.0),
            "access_count": int(row["access_count"] or 0),
            "pinned": bool(pinned) or tier >= 3,
            "tier": tier,
            "source_type": str(row["source_type"] or ""),
            "created_at": float(row["created_at"] or 0.0),
            "updated_at": float(row["updated_at"] or 0.0),
            "last_accessed_at": float(row["last_accessed_at"] or 0.0),
            "has_embedding": bool(row["embedding_blob"]),
            "status": str(row["status"] or "active") if "status" in keys else "active",
        }

    def _atom_columns(self, conn: sqlite3.Connection) -> set:
        try:
            return {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)")}
        except Exception:
            return set()

    def list_memories(
        self,
        limit: int = 20,
        offset: int = 0,
        query: str = "",
        pinned_only: bool = False,
        include_superseded: bool = False,
        scope_key: str = "",
    ) -> Dict[str, Any]:
        """
        分页浏览 / 搜索 L2 情景记忆（WebUI 主列表）。

        返回 ``{"items": [...], "total": int, "limit": int, "offset": int}``；
        任何异常都返回空列表，保证 WebUI 永不 500。
        """
        try:
            lim = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            lim = 20
        try:
            off = max(0, int(offset))
        except (TypeError, ValueError):
            off = 0

        result: Dict[str, Any] = {"items": [], "total": 0, "limit": lim, "offset": off}
        try:
            conn = self.connect()
            cols = self._atom_columns(conn)
            has_status = "status" in cols
            has_pinned = "pinned" in cols
            has_tier = "tier" in cols

            where: List[str] = []
            params: List[Any] = []
            if scope_key:
                where.append("scope_key = ?")
                params.append(str(scope_key))
            if pinned_only and (has_pinned or has_tier):
                conds = []
                if has_pinned:
                    conds.append("pinned = 1")
                if has_tier:
                    conds.append("tier >= 3")
                where.append("(" + " OR ".join(conds) + ")")
            if has_status and not include_superseded:
                where.append("COALESCE(status, 'active') != 'superseded'")

            term = str(query or "").strip()
            if term:
                like = f"%{term}%"
                where.append("(content LIKE ? OR key_facts_json LIKE ?)")
                params.extend([like, like])

            where_sql = (" WHERE " + " AND ".join(where)) if where else ""

            total_row = conn.execute(
                f"SELECT COUNT(1) AS cnt FROM l2_atoms{where_sql}", params
            ).fetchone()
            result["total"] = int(total_row["cnt"]) if total_row is not None else 0

            order_col = "pinned DESC, " if has_pinned else ""
            rows = conn.execute(
                f"SELECT * FROM l2_atoms{where_sql} "
                f"ORDER BY {order_col}created_at DESC LIMIT ? OFFSET ?",
                params + [lim, off],
            ).fetchall()
            result["items"] = [self._atom_row_to_dict(row) for row in rows]
        except Exception as exc:
            logger.error("list_memories 查询失败: %s", exc, exc_info=True)
        return result

    def get_memory_detail(self, atom_id: str) -> Optional[Dict[str, Any]]:
        """按 atom_id 读取单条记忆完整详情（含关联图谱三元组）。"""
        aid = str(atom_id or "").strip()
        if not aid:
            return None
        try:
            conn = self.connect()
            row = conn.execute(
                "SELECT * FROM l2_atoms WHERE atom_id = ?", (aid,)
            ).fetchone()
            if row is None:
                return None
            detail = self._atom_row_to_dict(row)
            detail["triples"] = []
            try:
                cur = conn.execute(
                    """
                    SELECT s.entity_name AS head, e.relation_type AS relation,
                           t.entity_name AS tail, e.confidence AS confidence,
                           e.status AS status
                    FROM kg_edges e
                    JOIN kg_nodes s ON s.node_id = e.source_node_id
                    JOIN kg_nodes t ON t.node_id = e.target_node_id
                    WHERE e.atom_id = ?
                    LIMIT 50
                    """,
                    (aid,),
                )
                detail["triples"] = [
                    {
                        "head": str(r["head"] or ""),
                        "relation": str(r["relation"] or ""),
                        "tail": str(r["tail"] or ""),
                        "confidence": float(r["confidence"] or 0.0),
                        "status": str(r["status"] or "active"),
                    }
                    for r in cur.fetchall()
                ]
            except Exception as exc:
                logger.debug("读取记忆关联三元组失败（忽略）: %s", exc)
            return detail
        except Exception as exc:
            logger.error("get_memory_detail 失败: %s", exc, exc_info=True)
            return None

    def set_atom_pin(self, atom_id: str, pinned: bool) -> Dict[str, Any]:
        """
        显式设置 L2 原子的置顶状态（WebUI 勾选框用，与 toggle_atom_pin 互补）。

        行为：pinned=True → pinned=1/tier=3（晋升 L3）；False → pinned=0/tier=2。
        """
        result: Dict[str, Any] = {
            "found": False,
            "atom_id": str(atom_id or "").strip(),
            "is_pinned": bool(pinned),
            "error": "",
        }
        aid = result["atom_id"]
        if not aid:
            result["error"] = "atom_id 为空"
            return result
        try:
            conn = self.connect()
            cols = self._atom_columns(conn)
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
                "SELECT atom_id FROM l2_atoms WHERE atom_id = ?", (aid,)
            ).fetchone()
            if row is None:
                result["error"] = "记录不存在"
                return result
            with conn:
                conn.execute(
                    "UPDATE l2_atoms SET pinned = ?, tier = ?, updated_at = ? WHERE atom_id = ?",
                    (1 if pinned else 0, 3 if pinned else 2, time.time(), aid),
                )
            result["found"] = True
        except Exception as exc:
            logger.error("set_atom_pin 失败 (atom_id=%s): %s", aid, exc, exc_info=True)
            result["error"] = str(exc)
        return result

    def list_dlq(self, limit: int = 20, status: str = "") -> Dict[str, Any]:
        """列出死信队列条目（WebUI 噪声/失败排查用）。"""
        try:
            lim = max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            lim = 20
        result: Dict[str, Any] = {"items": [], "counts": {}}
        try:
            conn = self.connect()
            for row in conn.execute(
                "SELECT status, COUNT(1) AS cnt FROM task_dlq GROUP BY status"
            ).fetchall():
                result["counts"][str(row["status"])] = int(row["cnt"] or 0)

            sql = (
                "SELECT task_id, task_type, scope_key, error_msg, retry_count, "
                "max_retries, status, next_retry_at, created_at, updated_at "
                "FROM task_dlq"
            )
            params: List[Any] = []
            if status:
                sql += " WHERE status = ?"
                params.append(str(status))
            sql += " ORDER BY updated_at DESC LIMIT ?"
            params.append(lim)
            result["items"] = [
                {
                    "task_id": str(r["task_id"]),
                    "task_type": str(r["task_type"] or ""),
                    "scope_key": str(r["scope_key"] or ""),
                    "error_msg": str(r["error_msg"] or "")[:400],
                    "retry_count": int(r["retry_count"] or 0),
                    "max_retries": int(r["max_retries"] or 0),
                    "status": str(r["status"] or ""),
                    "next_retry_at": float(r["next_retry_at"] or 0.0),
                    "created_at": float(r["created_at"] or 0.0),
                    "updated_at": float(r["updated_at"] or 0.0),
                }
                for r in conn.execute(sql, params).fetchall()
            ]
        except Exception as exc:
            logger.error("list_dlq 失败: %s", exc, exc_info=True)
        return result

    # ------------------------------------------------------------------
    # WebUI 写操作：记忆编辑 / 删除 / 图谱查看
    # 全部使用参数化 SQL + 事务 + 列名白名单；任何失败返回结构化错误，不抛异常。
    # ------------------------------------------------------------------
    @staticmethod
    def _public_row(row: sqlite3.Row, allowed: Optional[set] = None) -> Dict[str, Any]:
        """把 sqlite 行转为可 JSON 序列化的 dict；跳过 bytes 与未白名单列。"""
        try:
            keys = set(row.keys())
        except Exception:
            return {}
        out: Dict[str, Any] = {}
        for key in keys:
            if allowed is not None and key not in allowed:
                continue
            try:
                value = row[key]
            except Exception:
                continue
            if isinstance(value, (bytes, bytearray, memoryview)):
                continue  # 二进制向量等绝不外泄给 WebUI
            if isinstance(value, (str, int, float)) or value is None:
                out[str(key)] = value
            else:
                out[str(key)] = str(value)
        return out

    def update_memory(
        self,
        atom_id: str,
        content: Any = None,
        importance: Any = None,
        key_facts: Any = None,
        topics: Any = None,
    ) -> Dict[str, Any]:
        """
        编辑单条 L2 记忆（正文 / 重要度 / 关键事实 / 主题）。

        :return: ``{"found": bool, "atom_id": str, "updated": [...], "error": str}``
        """
        aid = str(atom_id or "").strip()
        result: Dict[str, Any] = {"found": False, "atom_id": aid, "updated": [], "error": ""}
        if not aid:
            result["error"] = "atom_id 为空"
            return result
        if len(aid) > 256:
            result["error"] = "atom_id 过长"
            return result

        # --- 输入校验（拒绝超长 / 非法类型 / 越界数值） ---
        new_content: Optional[str] = None
        if content is not None:
            if not isinstance(content, str) or not content.strip():
                result["error"] = "content 必须是非空字符串"
                return result
            new_content = content.strip()
            if len(new_content) > 20000:
                result["error"] = "content 过长（上限 20000 字符）"
                return result

        new_importance: Optional[float] = None
        if importance is not None:
            try:
                new_importance = float(importance)
            except (TypeError, ValueError):
                result["error"] = "importance 必须是数值"
                return result
            if not (0.0 <= new_importance <= 1.0):
                result["error"] = "importance 必须在 0~1 之间"
                return result

        def _normalize_list(raw: Any, label: str) -> Optional[List[str]]:
            if raw is None:
                return None
            if not isinstance(raw, list):
                raise ValueError(f"{label} 必须是数组")
            items = [str(x).strip() for x in raw if x is not None]
            if len(items) > 50 or any(len(x) > 500 for x in items):
                raise ValueError(f"{label} 条目过多或过长")
            return items

        try:
            new_facts = _normalize_list(key_facts, "key_facts")
            new_topics = _normalize_list(topics, "topics")
        except ValueError as exc:
            result["error"] = str(exc)
            return result

        try:
            conn = self.connect()
            cols = self._atom_columns(conn)
            exists = conn.execute(
                "SELECT 1 FROM l2_atoms WHERE atom_id = ?", (aid,)
            ).fetchone()
            if exists is None:
                result["error"] = "记忆不存在"
                return result

            sets: List[str] = []
            params: List[Any] = []
            if new_content is not None:
                sets.append("content = ?")
                params.append(new_content)
                result["updated"].append("content")
            if new_importance is not None and "importance" in cols:
                sets.append("importance = ?")
                params.append(new_importance)
                result["updated"].append("importance")
            if new_facts is not None and "key_facts_json" in cols:
                sets.append("key_facts_json = ?")
                params.append(json.dumps(new_facts, ensure_ascii=False))
                result["updated"].append("key_facts")
            if new_topics is not None and "topics_json" in cols:
                sets.append("topics_json = ?")
                params.append(json.dumps(new_topics, ensure_ascii=False))
                result["updated"].append("topics")
            if "updated_at" in cols:
                sets.append("updated_at = ?")
                params.append(time.time())
            if not sets:
                result["error"] = "没有可更新的字段"
                return result

            # 列名来自固定白名单，字段名不含用户输入；值全部参数化
            with conn:
                conn.execute(
                    "UPDATE l2_atoms SET " + ", ".join(sets) + " WHERE atom_id = ?",
                    params + [aid],
                )

            # 增量同步 FTS 预分词文本（失败不影响主更新）
            if new_content is not None:
                try:
                    self.sync_atom_fts(aid, new_content, new_facts)
                except Exception as exc:  # pragma: no cover
                    logger.debug("[MemoryStorage] 编辑后同步 FTS 失败（忽略）: %s", exc)
            # 正文变化会使旧向量失效，清掉该条向量缓存
            if new_content is not None:
                try:
                    cache = getattr(self, "embedding_cache", None)
                    if cache is not None and hasattr(cache, "invalidate"):
                        cache.invalidate(aid)
                except Exception as exc:  # pragma: no cover
                    logger.debug("[MemoryStorage] 失效向量缓存失败（忽略）: %s", exc)
            result["found"] = True
        except Exception as exc:
            logger.error("update_memory 失败 (atom_id=%s): %s", aid, exc, exc_info=True)
            result["error"] = str(exc)
        return result

    def delete_memory(self, atom_id: str) -> Dict[str, Any]:
        """
        删除单条 L2 记忆，并级联清理关联图谱边 / FTS 行 / 向量缓存。

        :return: ``{"found": bool, "atom_id": str, "deleted": {...}, "error": str}``
        """
        aid = str(atom_id or "").strip()
        result: Dict[str, Any] = {
            "found": False, "atom_id": aid,
            "deleted": {"atom": 0, "kg_edges": 0}, "error": "",
        }
        if not aid or len(aid) > 256:
            result["error"] = "atom_id 非法"
            return result
        try:
            conn = self.connect()
            exists = conn.execute(
                "SELECT 1 FROM l2_atoms WHERE atom_id = ?", (aid,)
            ).fetchone()
            if exists is None:
                result["error"] = "记忆不存在"
                return result
            with conn:
                # 1) 图谱关系边（按来源原子删除，表不存在时忽略）
                try:
                    cur = conn.execute(
                        "DELETE FROM kg_edges WHERE atom_id = ?", (aid,)
                    )
                    result["deleted"]["kg_edges"] = int(cur.rowcount or 0)
                except Exception as exc:  # pragma: no cover
                    logger.debug("[MemoryStorage] 清理 kg_edges 失败（忽略）: %s", exc)
                # 2) 主表
                cur = conn.execute(
                    "DELETE FROM l2_atoms WHERE atom_id = ?", (aid,)
                )
                result["deleted"]["atom"] = int(cur.rowcount or 0)
                # 3) FTS 倒排表（外部内容表可能不支持 DELETE，容错）
                try:
                    conn.execute(
                        f"DELETE FROM {FTS_TABLE_NAME} WHERE atom_id = ?", (aid,)
                    )
                except Exception as exc:  # pragma: no cover
                    logger.debug("[MemoryStorage] 清理 FTS 行失败（忽略）: %s", exc)
            try:
                cache = getattr(self, "embedding_cache", None)
                if cache is not None and hasattr(cache, "invalidate"):
                    cache.invalidate(aid)
            except Exception as exc:  # pragma: no cover
                logger.debug("[MemoryStorage] 失效向量缓存失败（忽略）: %s", exc)
            result["found"] = True
        except Exception as exc:
            logger.error("delete_memory 失败 (atom_id=%s): %s", aid, exc, exc_info=True)
            result["error"] = str(exc)
        return result

    def list_kg_graph(self, limit: int = 100, query: str = "") -> Dict[str, Any]:
        """
        知识图谱（L3）节点与关系浏览，供 WebUI「图谱」页签使用。

        使用 ``SELECT *`` + 列名白名单，兼容旧库缺列；bytes 字段一律不外泄。
        """
        try:
            lim = max(1, min(int(limit), 300))
        except (TypeError, ValueError):
            lim = 100
        term = str(query or "").strip()[:200]
        like = f"%{term}%"
        payload: Dict[str, Any] = {
            "nodes": [], "edges": [], "stats": {}, "limit": lim, "query": term,
        }
        try:
            conn = self.connect()
            node_sql = "SELECT * FROM kg_nodes"
            node_params: List[Any] = []
            if term:
                node_sql += " WHERE COALESCE(entity_name, '') LIKE ?"
                node_params.append(like)
            node_sql += " ORDER BY created_at DESC LIMIT ?"
            node_params.append(lim)
            nodes = [
                self._public_row(r)
                for r in conn.execute(node_sql, node_params).fetchall()
            ]
            payload["nodes"] = nodes

            name_of = {
                str(n.get("node_id")): str(
                    n.get("entity_name") or n.get("name") or n.get("node_id") or ""
                )
                for n in nodes
            }
            edge_sql = "SELECT * FROM kg_edges"
            edge_params: List[Any] = []
            if term:
                edge_sql += (
                    " WHERE COALESCE(relation_type, '') LIKE ?"
                    " OR COALESCE(atom_id, '') LIKE ?"
                )
                edge_params.extend([like, like])
            edge_sql += " LIMIT ?"
            edge_params.append(lim * 2)
            edges = []
            for row in conn.execute(edge_sql, edge_params).fetchall():
                item = self._public_row(row)
                src = str(item.get("source_node_id") or "")
                dst = str(item.get("target_node_id") or "")
                item["head"] = name_of.get(src, src)
                item["tail"] = name_of.get(dst, dst)
                edges.append(item)
            payload["edges"] = edges
            payload["stats"] = self.get_kg_stats()
        except Exception as exc:
            logger.error("list_kg_graph 失败: %s", exc, exc_info=True)
        return payload

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
