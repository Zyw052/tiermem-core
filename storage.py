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
import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("astrbot_plugin_agent_memory.storage")


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


class MemoryStorage:
    """分层记忆 SQLite 存储管理器"""

    def __init__(self, data_dir: os.PathLike) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "agent_memory.db"
        self._conn: Optional[sqlite3.Connection] = None

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
        """初始化分层记忆核心表结构"""
        try:
            conn = self.connect()
            with conn:
                for stmt in DDL_STATEMENTS:
                    conn.execute(stmt)
            logger.info("MemoryStorage schema initialized at %s", self.db_path)
        except Exception as exc:
            logger.error("Failed to initialize MemoryStorage schema: %s", exc, exc_info=True)
            raise

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

    def close(self) -> None:
        """安全关闭数据库连接"""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("MemoryStorage connection closed.")
            except Exception as exc:
                logger.warning("Error closing MemoryStorage connection: %s", exc)
            finally:
                self._conn = None
