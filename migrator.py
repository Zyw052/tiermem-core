# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 资产无损迁移引擎 (migrator.py)
负责从旧版 livingmemory.db 迁移文档、记忆原子与知识图谱至新版分层记忆存储。
支持：
1. 源库自动检测与自定义路径
2. 迁移前自动生成带时间戳的 .bak 安全快照
3. documents -> l1_docs, memory_atoms -> l2_atoms, graph_nodes/edges -> kg_nodes/edges 无损映射
4. --dry-run 预览统计模式与 CLI 独立运行
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from storage import MemoryStorage

logger = logging.getLogger("astrbot_plugin_agent_memory.migrator")

def _env_path(env_key: str, fallback: str) -> Path:
    """优先读取环境变量，未设置时回退到内置默认路径。

    这样既保持「零配置直接可跑」的原有默认行为（环境变量未设置时
    结果与硬编码路径完全一致），又允许非 ``/AstrBot`` 标准部署的宿主
    通过环境变量覆盖，避免把安装路径写死在开源代码里。

    支持的变量：
        - ``ASTRBOT_LIVINGMEMORY_DB``   : 旧版 livingmemory.db 绝对路径
        - ``ASTRBOT_AGENT_MEMORY_DIR``  : 新版插件数据目录绝对路径
    """
    raw = os.environ.get(env_key, "").strip()
    return Path(raw) if raw else Path(fallback)


DEFAULT_SOURCE_DB = _env_path(
    "ASTRBOT_LIVINGMEMORY_DB",
    "/AstrBot/data/plugin_data/astrbot_plugin_livingmemory/livingmemory.db",
)
DEFAULT_TARGET_DIR = _env_path(
    "ASTRBOT_AGENT_MEMORY_DIR",
    "/AstrBot/data/plugin_data/astrbot_plugin_agent_memory",
)


def _parse_timestamp(val: Any, default: Optional[float] = None) -> float:
    """将各种时间格式（float/int/ISO字符串）安全转换为 Unix 时间戳"""
    if default is None:
        default = time.time()
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str:
            return default
        try:
            return float(val_str)
        except ValueError:
            pass
        try:
            # 兼容 ISO 格式如 '2026-06-24 01:52:57.951655' 或带时区 '+00:00'
            dt = datetime.datetime.fromisoformat(val_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:
            return default
    return default


def _safe_json_loads(raw: Any, fallback: Any) -> Any:
    """安全反序列化 JSON 字符串"""
    if raw is None:
        return fallback
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        raw_str = raw.strip()
        if not raw_str:
            return fallback
        try:
            return json.loads(raw_str)
        except Exception:
            return fallback
    return fallback


def _safe_json_dumps(obj: Any, fallback: str = "{}") -> str:
    """安全序列化对象为 JSON 字符串"""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return fallback


class MemoryMigrator:
    """LivingMemory -> AgentMemory 资产无损迁移器"""

    def __init__(
        self,
        source_db_path: Optional[os.PathLike] = None,
        storage: Optional[MemoryStorage] = None,
        target_data_dir: Optional[os.PathLike] = None,
    ) -> None:
        self.source_db_path = Path(source_db_path) if source_db_path else DEFAULT_SOURCE_DB
        if storage is not None:
            self.storage = storage
            self._owns_storage = False
        else:
            data_dir = Path(target_data_dir) if target_data_dir else DEFAULT_TARGET_DIR
            self.storage = MemoryStorage(data_dir)
            self._owns_storage = True

    def check_source_exists(self) -> bool:
        """检查源数据库文件是否存在且非空"""
        return self.source_db_path.is_file() and self.source_db_path.stat().st_size > 0

    def create_backup(self) -> Path:
        """在源库同级目录生成带时间戳的 .bak 安全快照"""
        if not self.check_source_exists():
            raise FileNotFoundError(f"源数据库不存在或为空: {self.source_db_path}")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.source_db_path.with_name(f"{self.source_db_path.name}.bak.{ts}")

        # 使用 SQLite 在线备份 API 确保 WAL 模式下的数据完整性，若失败则回退到文件拷贝
        try:
            src_conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True, timeout=15.0)
            dst_conn = sqlite3.connect(str(backup_path), timeout=15.0)
            with dst_conn:
                src_conn.backup(dst_conn)
            dst_conn.close()
            src_conn.close()
        except Exception as exc:
            logger.warning("SQLite backup API 回退至文件复制: %s", exc)
            shutil.copy2(self.source_db_path, backup_path)

        logger.info("已生成源数据库安全快照: %s", backup_path)
        return backup_path

    def _ensure_extended_columns(self, conn: sqlite3.Connection) -> None:
        """确保目标表具备无损迁移所需的扩展兼容列（幂等）"""
        extended_cols: Dict[str, List[Tuple[str, str]]] = {
            "l1_docs": [
                ("session_id", "TEXT NOT NULL DEFAULT ''"),
                ("persona_id", "TEXT NOT NULL DEFAULT ''"),
                ("importance", "REAL NOT NULL DEFAULT 0.5"),
                ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
            ],
            "l2_atoms": [
                ("persona_id", "TEXT NOT NULL DEFAULT ''"),
                ("atom_type", "TEXT NOT NULL DEFAULT 'unknown'"),
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

    def _connect_source_ro(self) -> sqlite3.Connection:
        """以只读方式连接源数据库"""
        conn = sqlite3.connect(f"file:{self.source_db_path}?mode=ro", uri=True, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        )
        return cur.fetchone() is not None

    def _map_document_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将旧版 documents 行无损映射为新版 l1_docs 字段字典"""
        raw_meta = row["metadata"]
        meta_dict = _safe_json_loads(raw_meta, {})
        if not isinstance(meta_dict, dict):
            meta_dict = {"raw_metadata": raw_meta}

        doc_id = str(row["doc_id"] or f"doc_{row['id']}")
        session_id = str(meta_dict.get("session_id") or row["user_id"] or "")
        persona_id = str(meta_dict.get("persona_id") or "")
        scope_key = session_id or persona_id or "global"
        content = str(row["text"] or "")
        importance = float(meta_dict.get("importance", 0.5) or 0.5)

        created_at = _parse_timestamp(
            meta_dict.get("create_time") or row["created_at"]
        )
        updated_at = _parse_timestamp(
            meta_dict.get("last_access_time") or row["updated_at"], default=created_at
        )

        topics = meta_dict.get("topics")
        title = ""
        if isinstance(topics, list) and topics:
            title = " | ".join(str(t) for t in topics[:3])
        elif content:
            title = content[:48].replace("\n", " ")

        # 在 metadata 中完整保留原表 id、kb_doc_id 等全部上下文
        enriched_meta = dict(meta_dict)
        enriched_meta.setdefault("legacy_id", row["id"])
        if row["kb_doc_id"] is not None:
            enriched_meta.setdefault("kb_doc_id", row["kb_doc_id"])
        if row["user_id"] is not None:
            enriched_meta.setdefault("user_id", row["user_id"])

        return {
            "doc_id": doc_id,
            "scope_key": scope_key,
            "doc_type": str(meta_dict.get("interaction_type") or "episodic_doc"),
            "title": title,
            "content": content,
            "version": 1,
            "updated_at": updated_at,
            "created_at": created_at,
            "session_id": session_id,
            "persona_id": persona_id,
            "importance": importance,
            "metadata": _safe_json_dumps(enriched_meta, "{}"),
        }

    def _map_atom_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将旧版 memory_atoms 行无损映射为新版 l2_atoms 字段字典"""
        atom_id = f"atom_{row['id']}"
        session_id = str(row["session_id"] or "")
        persona_id = str(row["persona_id"] or "")
        scope_key = session_id or persona_id or "global"
        content = str(row["content"] or "")
        atom_type = str(row["atom_type"] or "unknown").strip().lower()
        importance = float(row["importance"] if row["importance"] is not None else 0.5)
        confidence = float(row["confidence"] if row["confidence"] is not None else 0.7)
        decay_type = str(row["decay_type"] or "exponential")
        ttl_days = float(row["ttl_days"] if row["ttl_days"] is not None else 30.0)
        reinforcement_count = int(row["reinforcement_count"] or 0)

        # 重要度 >= 0.85 或 preference 类型标记 pinned / tier=3
        is_pinned = importance >= 0.85 or atom_type == "preference"
        pinned = 1 if is_pinned else 0
        tier = 3 if is_pinned else 2

        entities_list = _safe_json_loads(row["entities"], [])
        if not isinstance(entities_list, list):
            entities_list = []

        raw_meta = _safe_json_loads(row["metadata"], {})
        if not isinstance(raw_meta, dict):
            raw_meta = {}

        enriched_meta = dict(raw_meta)
        enriched_meta.update(
            {
                "legacy_atom_id": row["id"],
                "parent_memory_id": row["parent_memory_id"],
                "atom_type": atom_type,
                "confidence": confidence,
                "decay_type": decay_type,
                "ttl_days": ttl_days,
                "expires_at": row["expires_at"],
                "status": row["status"],
                "reinforcement_count": reinforcement_count,
                "last_reinforced_at": row["last_reinforced_at"],
                "event_time": row["event_time"],
                "persona_id": persona_id,
                "pinned": bool(is_pinned),
                "tier": tier,
            }
        )

        created_at = _parse_timestamp(row["created_at"])
        last_accessed_at = _parse_timestamp(row["last_accessed_at"], default=created_at)
        updated_at = _parse_timestamp(row["last_reinforced_at"], default=last_accessed_at)

        return {
            "atom_id": atom_id,
            "scope_key": scope_key,
            "session_id": session_id,
            "persona_id": persona_id,
            "content": content,
            "topics_json": _safe_json_dumps(entities_list, "[]"),
            "key_facts_json": _safe_json_dumps([content] if content else [], "[]"),
            "sentiment": str(raw_meta.get("sentiment") or "neutral"),
            "importance": importance,
            "access_count": reinforcement_count,
            "embedding_blob": None,
            "source_type": atom_type,
            "source_ref": f"memory:{row['parent_memory_id']}",
            "last_accessed_at": last_accessed_at,
            "created_at": created_at,
            "updated_at": updated_at,
            "atom_type": atom_type,
            "decay_type": decay_type,
            "ttl_days": ttl_days,
            "reinforcement_count": reinforcement_count,
            "pinned": pinned,
            "tier": tier,
            "metadata": _safe_json_dumps(enriched_meta, "{}"),
        }

    def _map_graph_node_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        """将旧版 graph_nodes 行无损映射为新版 kg_nodes 字段字典"""
        node_id = f"node_{row['id']}"
        entity_type = str(row["node_type"] or "concept")
        # 使用 node_key 作为唯一实体名标识以避免 (scope_key, entity_name, entity_type) 冲突，同时在 summary/attributes 保留原始值
        entity_name = str(row["node_value"] or row["canonical_value"] or row["node_key"])
        canonical_val = str(row["canonical_value"] or "")
        node_key = str(row["node_key"] or node_id)

        meta_dict = _safe_json_loads(row["metadata"], {})
        if not isinstance(meta_dict, dict):
            meta_dict = {"raw_metadata": row["metadata"]}

        scope_key = str(meta_dict.get("session_id") or meta_dict.get("persona_id") or "global")
        attrs = dict(meta_dict)
        attrs.update(
            {
                "legacy_node_id": row["id"],
                "node_key": node_key,
                "node_value": row["node_value"],
                "canonical_value": canonical_val,
            }
        )

        created_at = _parse_timestamp(row["created_at"])
        updated_at = _parse_timestamp(row["updated_at"], default=created_at)

        return {
            "node_id": node_id,
            "scope_key": scope_key,
            "entity_name": node_key,
            "entity_type": entity_type,
            "summary": entity_name,
            "attributes_json": _safe_json_dumps(attrs, "{}"),
            "weight": float(meta_dict.get("weight", 1.0) or 1.0),
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def _map_graph_edge_row(
        self, row: sqlite3.Row, valid_node_ids: set
    ) -> Optional[Dict[str, Any]]:
        """将旧版 graph_edges 行无损映射为新版 kg_edges 字段字典"""
        src_id = f"node_{row['source_node_id']}"
        tgt_id = f"node_{row['target_node_id']}"
        if src_id not in valid_node_ids or tgt_id not in valid_node_ids:
            return None

        edge_id = f"edge_{row['id']}"
        meta_dict = _safe_json_loads(row["metadata"], {})
        if not isinstance(meta_dict, dict):
            meta_dict = {"raw_metadata": row["metadata"]}

        scope_key = str(meta_dict.get("session_id") or meta_dict.get("persona_id") or "global")
        attrs = dict(meta_dict)
        attrs.update(
            {
                "legacy_edge_id": row["id"],
                "edge_key": row["edge_key"],
                "source_memory_id": row["source_memory_id"],
                "weight": row["weight"],
                "status": row["status"],
            }
        )

        fact_text = str(
            meta_dict.get("atom_content")
            or meta_dict.get("fact_text")
            or _safe_json_dumps(attrs, "{}")
        )
        created_at = _parse_timestamp(row["created_at"])
        updated_at = _parse_timestamp(row["updated_at"], default=created_at)

        return {
            "edge_id": edge_id,
            "scope_key": scope_key,
            "source_node_id": src_id,
            "target_node_id": tgt_id,
            "relation_type": str(row["relation_type"] or "related_to"),
            "fact_text": fact_text,
            "confidence": float(row["confidence"] if row["confidence"] is not None else 0.8),
            "valid_from": created_at,
            "valid_until": None if row["status"] == "active" else updated_at,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def migrate(self, dry_run: bool = False, force: bool = False) -> Dict[str, Any]:
        """
        执行迁移或 dry-run 预览统计。
        :param dry_run: 若为 True，仅只读统计旧库各表条数与预计映射分布，不写目标库、不生成快照
        :param force: 若目标库已有数据，是否允许继续覆盖/合并写入
        :return: 迁移统计结果字典
        """
        start_ts = time.time()
        result: Dict[str, Any] = {
            "status": "pending",
            "dry_run": dry_run,
            "source_db": str(self.source_db_path),
            "target_db": str(self.storage.db_path),
            "backup_path": None,
            "source_counts": {},
            "mapped_counts": {
                "l1_docs": 0,
                "l2_atoms": 0,
                "l2_atoms_pinned_tier3": 0,
                "l2_atoms_regular_tier2": 0,
                "kg_nodes": 0,
                "kg_edges": 0,
                "kg_edges_skipped_orphan": 0,
            },
            "atom_type_distribution": {},
            "duration_sec": 0.0,
        }

        if not self.check_source_exists():
            msg = f"源数据库不存在或为空: {self.source_db_path}"
            logger.error(msg)
            result["status"] = "error"
            result["error"] = msg
            return result

        src_conn = self._connect_source_ro()
        try:
            # 1. 读取并转换 documents -> l1_docs
            mapped_docs: List[Dict[str, Any]] = []
            if self._table_exists(src_conn, "documents"):
                cur = src_conn.execute("SELECT * FROM documents ORDER BY id ASC")
                rows = cur.fetchall()
                result["source_counts"]["documents"] = len(rows)
                for r in rows:
                    mapped_docs.append(self._map_document_row(r))
            else:
                result["source_counts"]["documents"] = 0

            # 2. 读取并转换 memory_atoms -> l2_atoms
            mapped_atoms: List[Dict[str, Any]] = []
            atom_dist: Dict[str, int] = {}
            pinned_count = 0
            if self._table_exists(src_conn, "memory_atoms"):
                cur = src_conn.execute("SELECT * FROM memory_atoms ORDER BY id ASC")
                rows = cur.fetchall()
                result["source_counts"]["memory_atoms"] = len(rows)
                for r in rows:
                    item = self._map_atom_row(r)
                    mapped_atoms.append(item)
                    atype = item["atom_type"]
                    atom_dist[atype] = atom_dist.get(atype, 0) + 1
                    if item["pinned"] == 1:
                        pinned_count += 1
            else:
                result["source_counts"]["memory_atoms"] = 0

            # 3. 读取并转换 graph_nodes -> kg_nodes
            mapped_nodes: List[Dict[str, Any]] = []
            valid_node_ids = set()
            if self._table_exists(src_conn, "graph_nodes"):
                cur = src_conn.execute("SELECT * FROM graph_nodes ORDER BY id ASC")
                rows = cur.fetchall()
                result["source_counts"]["graph_nodes"] = len(rows)
                for r in rows:
                    n_item = self._map_graph_node_row(r)
                    mapped_nodes.append(n_item)
                    valid_node_ids.add(n_item["node_id"])
            else:
                result["source_counts"]["graph_nodes"] = 0

            # 4. 读取并转换 graph_edges -> kg_edges
            mapped_edges: List[Dict[str, Any]] = []
            skipped_edges = 0
            if self._table_exists(src_conn, "graph_edges"):
                cur = src_conn.execute("SELECT * FROM graph_edges ORDER BY id ASC")
                rows = cur.fetchall()
                result["source_counts"]["graph_edges"] = len(rows)
                for r in rows:
                    e_item = self._map_graph_edge_row(r, valid_node_ids)
                    if e_item is not None:
                        mapped_edges.append(e_item)
                    else:
                        skipped_edges += 1
            else:
                result["source_counts"]["graph_edges"] = 0

            result["mapped_counts"] = {
                "l1_docs": len(mapped_docs),
                "l2_atoms": len(mapped_atoms),
                "l2_atoms_pinned_tier3": pinned_count,
                "l2_atoms_regular_tier2": len(mapped_atoms) - pinned_count,
                "kg_nodes": len(mapped_nodes),
                "kg_edges": len(mapped_edges),
                "kg_edges_skipped_orphan": skipped_edges,
            }
            result["atom_type_distribution"] = atom_dist

            if dry_run:
                result["status"] = "dry_run_ok"
                result["duration_sec"] = round(time.time() - start_ts, 4)
                logger.info("Dry-run 统计完成: %s", json.dumps(result, ensure_ascii=False))
                return result

            # 正式迁移：检查目标库是否非空
            self.storage.init_schema()
            existing_counts = self.storage.get_table_counts()
            total_existing = sum(existing_counts.values())
            if total_existing > 0 and not force:
                msg = (
                    f"目标数据库已有数据 ({existing_counts})，如确认覆盖合并请使用 force=True 或 --force。"
                )
                logger.warning(msg)
                result["status"] = "aborted_non_empty_target"
                result["existing_target_counts"] = existing_counts
                result["error"] = msg
                return result

            # 生成源库安全快照
            backup_path = self.create_backup()
            result["backup_path"] = str(backup_path)

            # 写入目标数据库（单事务原子提交）
            dst_conn = self.storage.connect()
            self._ensure_extended_columns(dst_conn)

            with dst_conn:
                if mapped_docs:
                    dst_conn.executemany(
                        """
                        INSERT OR REPLACE INTO l1_docs (
                            doc_id, scope_key, doc_type, title, content,
                            version, updated_at, created_at,
                            session_id, persona_id, importance, metadata
                        ) VALUES (
                            :doc_id, :scope_key, :doc_type, :title, :content,
                            :version, :updated_at, :created_at,
                            :session_id, :persona_id, :importance, :metadata
                        )
                        """,
                        mapped_docs,
                    )

                if mapped_atoms:
                    dst_conn.executemany(
                        """
                        INSERT OR REPLACE INTO l2_atoms (
                            atom_id, scope_key, session_id, persona_id, content,
                            topics_json, key_facts_json, sentiment, importance,
                            access_count, embedding_blob, source_type, source_ref,
                            last_accessed_at, created_at, updated_at,
                            atom_type, decay_type, ttl_days, reinforcement_count,
                            pinned, tier, metadata
                        ) VALUES (
                            :atom_id, :scope_key, :session_id, :persona_id, :content,
                            :topics_json, :key_facts_json, :sentiment, :importance,
                            :access_count, :embedding_blob, :source_type, :source_ref,
                            :last_accessed_at, :created_at, :updated_at,
                            :atom_type, :decay_type, :ttl_days, :reinforcement_count,
                            :pinned, :tier, :metadata
                        )
                        """,
                        mapped_atoms,
                    )

                if mapped_nodes:
                    dst_conn.executemany(
                        """
                        INSERT OR REPLACE INTO kg_nodes (
                            node_id, scope_key, entity_name, entity_type,
                            summary, attributes_json, weight, created_at, updated_at
                        ) VALUES (
                            :node_id, :scope_key, :entity_name, :entity_type,
                            :summary, :attributes_json, :weight, :created_at, :updated_at
                        )
                        """,
                        mapped_nodes,
                    )

                if mapped_edges:
                    dst_conn.executemany(
                        """
                        INSERT OR REPLACE INTO kg_edges (
                            edge_id, scope_key, source_node_id, target_node_id,
                            relation_type, fact_text, confidence,
                            valid_from, valid_until, created_at, updated_at
                        ) VALUES (
                            :edge_id, :scope_key, :source_node_id, :target_node_id,
                            :relation_type, :fact_text, :confidence,
                            :valid_from, :valid_until, :created_at, :updated_at
                        )
                        """,
                        mapped_edges,
                    )

            result["target_counts_after"] = self.storage.get_table_counts()
            result["status"] = "migrated_ok"
            result["duration_sec"] = round(time.time() - start_ts, 4)
            logger.info("数据无损迁移完成: %s", json.dumps(result, ensure_ascii=False))
            return result

        except Exception as exc:
            logger.error("迁移过程发生异常: %s", exc, exc_info=True)
            result["status"] = "error"
            result["error"] = str(exc)
            result["duration_sec"] = round(time.time() - start_ts, 4)
            raise
        finally:
            src_conn.close()
            if self._owns_storage:
                self.storage.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LivingMemory -> AgentMemory 无损数据迁移工具"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE_DB),
        help=f"源 livingmemory.db 路径 (默认: {DEFAULT_SOURCE_DB})",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=str(DEFAULT_TARGET_DIR),
        help=f"目标存储数据目录 (默认: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅执行只读统计与映射预览，不修改目标库也不创建备份",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="当目标库非空时强制执行覆盖/合并迁移",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_arg_parser().parse_args()
    migrator = MemoryMigrator(
        source_db_path=Path(args.source),
        target_data_dir=Path(args.target_dir),
    )
    summary = migrator.migrate(dry_run=args.dry_run, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
