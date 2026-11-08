# -*- coding: utf-8 -*-
"""
Stage 8 自测：方案 C（激活真·L3 时序知识图谱 + 艾宾浩斯时间衰减曲线）专项验证。

覆盖点：
1. 【C-1】L3 知识图谱读写闭环（彻底消灭 kg_nodes / kg_edges 僵尸表）
   - 图谱扩展列幂等补齐（新库直建 / 旧库 ALTER 升级，零数据丢失）；
   - 三元组 `(head, relation, tail)` 落库：实体节点 mention_count / last_seen_at 更新；
   - 关系边按 (scope, source, target, relation, atom_id) 去重增强，weight / reinforcement_count 累加；
   - 规则三元组抽取正确性 + 疑问句/指代片段零误抽；
   - 总结器批处理真实写入链路 → 图谱落库并回指 atom_id；
   - 1-hop 实体关联子图召回（正向/反向关联、跨会话隔离、多跳解释）；
2. 【C-2】艾宾浩斯时间衰减曲线
   - exp(-λ·Δt/ttl) 数学正确性（与手算对照）；
   - pinned=1 / tier>=3 的 L3 置顶记忆衰减系数恒为 1.0（免死金牌）；
   - decay_type='linear' / 'none' 分支正确；
   - 衰减重排：陈旧琐事让位于新鲜上下文；
   - 关闭衰减开关（decay_enabled=False）→ 全量因子 1.0（向下兼容）；
3. 【C-3】软废弃图谱联动
   - 原子被 superseded → 关联边 status='invalidated'、weight=0、不物理删除；
   - 失效边不再参与 1-hop 召回；
4. 【综合】旧库平滑升级与零回归（默认配置等价性）。

运行： python3 tests/test_stage8_graph_c.py
（同时兼容 pytest 收集：所有用例均为 test_* 函数，无外部依赖）
"""

import asyncio
import math
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from retriever import MemoryRetriever  # noqa: E402
from storage import (  # noqa: E402
    KG_EDGE_EXTENDED_COLUMNS,
    KG_NODE_EXTENDED_COLUMNS,
    MemoryStorage,
    ensure_kg_columns,
)
from summarizer import MemorySummarizer  # noqa: E402

DIM = 8


# ---------------------------------------------------------------------------
# 测试桩与辅助函数
# ---------------------------------------------------------------------------


class FakeEmbeddingClient:
    """确定性零依赖 Embedding 桩：同文本必得同向量。"""

    dim = DIM

    async def get_embedding(self, text: str) -> List[float]:
        vec = [0.0] * DIM
        for idx, ch in enumerate(str(text)):
            vec[idx % DIM] += (ord(ch) % 19) / 19.0
        return [x + 0.01 for x in vec]


class FakeRerankClient:
    """直通 Rerank 桩：按原顺序返回，避免依赖外部模型。"""

    async def rerank(self, query: str, docs: Sequence[str], top_n: int = 5, **kwargs):
        return [
            {"index": i, "document": doc, "relevance_score": 1.0 - 0.01 * i}
            for i, doc in enumerate(list(docs)[:top_n])
        ]


def _new_storage(tmp_dir: Path) -> MemoryStorage:
    st = MemoryStorage(tmp_dir)
    st.init_schema()
    return st


def _insert_atom(
    conn: sqlite3.Connection,
    atom_id: str,
    content: str,
    scope: str = "s1",
    status: str = "active",
    pinned: int = 0,
    tier: int = 2,
    importance: float = 0.5,
    ttl_days: float = 30.0,
    decay_type: str = "exponential",
    updated_at: Optional[float] = None,
    vec: Optional[Sequence[float]] = None,
) -> None:
    """写入一条测试原子，自动适配扩展列是否存在。"""
    now = float(updated_at if updated_at is not None else time.time())
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(l2_atoms)").fetchall()}
    fields = [
        "atom_id", "scope_key", "session_id", "content",
        "topics_json", "key_facts_json", "sentiment", "importance",
        "access_count", "embedding_blob", "source_type", "source_ref",
        "last_accessed_at", "created_at", "updated_at",
    ]
    blob = None
    if vec:
        blob = b"".join(__import__("struct").pack("<f", float(x)) for x in vec)
    values: List[Any] = [
        atom_id, scope, scope, content,
        "[]", "[]", "neutral", float(importance),
        0, blob, "chat", "",
        now, now, now,
    ]
    optional = {
        "status": status,
        "pinned": int(pinned),
        "tier": int(tier),
        "ttl_days": float(ttl_days),
        "decay_type": decay_type,
        "atom_type": "preference" if pinned else "fact",
    }
    for key, val in optional.items():
        if key in cols:
            fields.append(key)
            values.append(val)
    placeholders = ",".join("?" for _ in fields)
    conn.execute(
        f"INSERT INTO l2_atoms({','.join(fields)}) VALUES ({placeholders})", values
    )


def _kg_node(conn: sqlite3.Connection, scope: str, name: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM kg_nodes WHERE scope_key = ? AND entity_name = ?", (scope, name)
    ).fetchone()


def _kg_edges(conn: sqlite3.Connection, scope: str) -> List[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM kg_edges WHERE scope_key = ? ORDER BY created_at", (scope,)
    ).fetchall()


def _make_retriever(st: MemoryStorage, **cfg: Any) -> MemoryRetriever:
    base: Dict[str, Any] = {"strict_session_isolation": True}
    base.update(cfg)
    return MemoryRetriever(
        storage=st,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=FakeRerankClient(),
        config=base,
    )


# ===========================================================================
# C-1：图谱扩展列与三元组写入
# ===========================================================================


async def test_kg_schema_columns_fresh_and_legacy() -> None:
    """新库直建即含扩展列；旧库（缺列）经 ensure_kg_columns 幂等补齐。"""
    with tempfile.TemporaryDirectory(prefix="stage8_kg_schema_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()

        node_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(kg_nodes)").fetchall()}
        edge_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(kg_edges)").fetchall()}
        for name, _def in KG_NODE_EXTENDED_COLUMNS:
            assert name in node_cols, f"新库 kg_nodes 缺少扩展列 {name}"
        for name, _def in KG_EDGE_EXTENDED_COLUMNS:
            assert name in edge_cols, f"新库 kg_edges 缺少扩展列 {name}"
        print(f"[OK] 新库图谱扩展列齐备: nodes={len(node_cols)} edges={len(edge_cols)}")
        st.close()

    # 旧库模拟：仅建基础列，再补齐
    with tempfile.TemporaryDirectory(prefix="stage8_kg_legacy_") as tmp:
        db_path = Path(tmp) / "agent_memory.db"
        raw = sqlite3.connect(str(db_path))
        raw.execute(
            """
            CREATE TABLE kg_nodes (
                node_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL,
                entity_name TEXT NOT NULL, entity_type TEXT NOT NULL DEFAULT 'concept',
                summary TEXT NOT NULL DEFAULT '', attributes_json TEXT NOT NULL DEFAULT '{}',
                weight REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                UNIQUE(scope_key, entity_name, entity_type)
            );
            """
        )
        raw.execute(
            """
            CREATE TABLE kg_edges (
                edge_id TEXT PRIMARY KEY, scope_key TEXT NOT NULL,
                source_node_id TEXT NOT NULL, target_node_id TEXT NOT NULL,
                relation_type TEXT NOT NULL, fact_text TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                valid_from REAL, valid_until REAL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            """
        )
        # 旧库已有数据，验证升级不丢数据
        raw.execute(
            "INSERT INTO kg_nodes VALUES ('n_old','s1','旧实体','concept','', '{}', 1.0, 1.0, 1.0)"
        )
        raw.commit()
        raw.close()

        st = MemoryStorage(Path(tmp))
        st.init_schema()
        conn = st.connect()
        node_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(kg_nodes)").fetchall()}
        edge_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(kg_edges)").fetchall()}
        assert all(n in node_cols for n, _ in KG_NODE_EXTENDED_COLUMNS)
        assert all(n in edge_cols for n, _ in KG_EDGE_EXTENDED_COLUMNS)
        old = conn.execute("SELECT * FROM kg_nodes WHERE node_id='n_old'").fetchone()
        assert old is not None and old["entity_name"] == "旧实体", "旧库数据在升级中丢失"
        assert int(old["mention_count"]) == 1, "扩展列未给旧行填充默认值"
        # 幂等：重复调用不报错
        assert ensure_kg_columns(conn) is True
        assert ensure_kg_columns(conn) is True
        print("[OK] 旧库平滑升级：扩展列补齐、历史数据保留、幂等可重入")
        st.close()


async def test_kg_triple_write_dedup_and_mentions() -> None:
    """三元组写入：节点提及次数/最新时间、边权增强、类型正确、原子回指。"""
    with tempfile.TemporaryDirectory(prefix="stage8_kg_write_") as tmp:
        st = _new_storage(Path(tmp))
        now = time.time()

        r1 = st.upsert_kg_triple(
            "s1", "主人", "偏好", "美式咖啡",
            atom_id="atom_a", fact_text="主人喜欢喝美式咖啡",
            now=now - 100, head_type="person", tail_type="object",
        )
        assert r1 and r1["node_created"] and r1["edge_created"], f"首写异常: {r1}"

        # 同一原子重复观测 → 节点 mention_count+1，边强化而非新增
        r2 = st.upsert_kg_triple(
            "s1", "主人", "偏好", "美式咖啡",
            atom_id="atom_a", fact_text="主人喜欢喝美式咖啡",
            now=now, head_type="person", tail_type="object",
        )
        assert r2 and not r2["edge_created"], "重复三元组不应新建边"

        conn = st.connect()
        head = _kg_node(conn, "s1", "主人")
        tail = _kg_node(conn, "s1", "美式咖啡")
        assert head is not None and tail is not None
        assert head["entity_type"] == "person", head["entity_type"]
        assert tail["entity_type"] == "object", tail["entity_type"]
        assert int(head["mention_count"]) == 2, f"提及次数异常: {dict(head)}"
        assert float(head["last_seen_at"]) == now, "最新出现时间未刷新"

        edges = _kg_edges(conn, "s1")
        assert len(edges) == 1, f"边数异常: {len(edges)}"
        edge = edges[0]
        assert edge["relation_type"] == "偏好"
        assert edge["atom_id"] == "atom_a", "边未回指关联 atom_id"
        assert float(edge["weight"]) == 2.0, f"边权未强化: {edge['weight']}"
        assert int(edge["reinforcement_count"]) == 2
        assert edge["status"] == "active"

        # 空实体 / 自环必须被拒绝
        assert st.upsert_kg_triple("s1", "", "偏好", "某物") is None
        assert st.upsert_kg_triple("s1", "主人", "偏好", "主人") is None
        print(
            f"[OK] 三元组写入去重: 节点 mention={head['mention_count']} "
            f"边 weight={edge['weight']} atom={edge['atom_id']}"
        )
        st.close()


async def test_triple_rule_extraction() -> None:
    """规则三元组抽取：覆盖偏好/居住/工号/厌恶，疑问句零误抽。"""
    with tempfile.TemporaryDirectory(prefix="stage8_kg_rules_") as tmp:
        sm = MemorySummarizer(
            storage=_new_storage(Path(tmp)),
            embedding_client=FakeEmbeddingClient(),
            config={"graph": {"enable": True}},
        )

        cases = {
            "我住在苏州": ("居住于", "苏州"),
            "我喜欢喝美式咖啡": ("偏好", "美式咖啡"),
            "主人讨厌香菜": ("厌恶", "香菜"),
            "我的工号是A100": ("工号为", "A100"),
            "我养了一只橘猫": ("拥有", "一只橘猫"),
        }
        for text, (relation, tail) in cases.items():
            triples = sm._rule_extract_triples(text)
            assert any(
                t["relation"] == relation and t["tail"] == tail for t in triples
            ), f"{text!r} 抽取异常: {triples}"
            assert all(t["head"] == "主人" for t in triples), f"主语未归一化: {triples}"

        # 疑问 / 聊天废话不得产出三元组
        for noise in ("你今天喜欢什么？", "今天天气不错啊", "你喜欢他吗", "这是为什么呢"):
            assert sm._rule_extract_triples(noise) == [], f"噪声误抽: {noise!r}"

        # LLM 结构化三元组归一化：dict / 字符串 / 非法结构
        llm = sm._coerce_llm_triples(
            [
                {"head": "主人", "relation": "偏好", "tail": "拿铁"},
                "主人|居住于|苏州",
                {"bad": 1},
                None,
            ]
        )
        assert len(llm) == 2, f"LLM 三元组归一化异常: {llm}"
        assert {t["tail"] for t in llm} == {"拿铁", "苏州"}
        print("[OK] 规则抽取 5 类关系正确，疑问句零误抽，LLM 三元组归一化健壮")
        sm.storage.close()


async def test_summarizer_writes_kg_end_to_end() -> None:
    """总结器批处理真实链路：事实析出同时写图谱，并回指 atom_id。"""
    with tempfile.TemporaryDirectory(prefix="stage8_kg_e2e_") as tmp:
        st = _new_storage(Path(tmp))
        sm = MemorySummarizer(
            storage=st,
            embedding_client=FakeEmbeddingClient(),
            rerank_client=None,
            config={
                "summary_turn_interval": 5,
                "evaluator": {"enable": False},
                "summary_idle_flush_seconds": 0,
                "graph": {"enable": True, "max_triples_per_fact": 6},
            },
        )
        await sm._process_batch(
            "s1",
            [
                {
                    "user_text": "我住在苏州，我喜欢喝美式咖啡",
                    "assistant_text": "记住了，苏州和美式咖啡",
                    "enqueued_at": time.time(),
                }
            ],
            reason="threshold",
        )
        conn = st.connect()
        atoms = conn.execute("SELECT atom_id, content FROM l2_atoms").fetchall()
        assert atoms, "批处理未析出 L2 原子"

        nodes = conn.execute("SELECT * FROM kg_nodes").fetchall()
        edges = conn.execute("SELECT * FROM kg_edges").fetchall()
        assert nodes, "图谱节点为空（僵尸表未激活）"
        assert edges, "图谱关系边为空（僵尸表未激活）"

        atom_ids = {str(a["atom_id"]) for a in atoms}
        edge_atoms = {str(e["atom_id"]) for e in edges if e["atom_id"]}
        assert edge_atoms & atom_ids, (
            f"边的 atom_id 未回指真实原子: edges={edge_atoms} atoms={atom_ids}"
        )
        names = {str(n["entity_name"]) for n in nodes}
        assert "主人" in names, f"未抽到主语实体: {names}"
        assert any(n in names for n in ("苏州", "美式咖啡")), f"未抽到宾语实体: {names}"
        assert all(int(n["mention_count"]) >= 1 for n in nodes)
        assert all(float(n["last_seen_at"]) > 0 for n in nodes)
        print(
            f"[OK] 端到端图谱落库: nodes={len(nodes)} edges={len(edges)} "
            f"entities={sorted(names)} atom_link={bool(edge_atoms & atom_ids)}"
        )
        st.close()


async def test_graph_disabled_zero_side_effect() -> None:
    """graph.enable=False 时写入链路不产生任何图谱数据（向下兼容）。"""
    with tempfile.TemporaryDirectory(prefix="stage8_kg_off_") as tmp:
        st = _new_storage(Path(tmp))
        sm = MemorySummarizer(
            storage=st,
            embedding_client=FakeEmbeddingClient(),
            rerank_client=None,
            config={
                "evaluator": {"enable": False},
                "summary_idle_flush_seconds": 0,
                "graph": {"enable": False},
            },
        )
        await sm._process_batch(
            "s1",
            [{"user_text": "我喜欢喝美式咖啡", "assistant_text": "好的", "enqueued_at": time.time()}],
            reason="threshold",
        )
        conn = st.connect()
        assert conn.execute("SELECT COUNT(1) c FROM kg_nodes").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(1) c FROM kg_edges").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(1) c FROM l2_atoms").fetchone()["c"] >= 1
        print("[OK] 图谱开关关闭：零图谱副作用，L2 原子照常入库")
        st.close()


# ===========================================================================
# C-1：1-hop 关联子图召回
# ===========================================================================


async def test_graph_1hop_recall_in_retriever() -> None:
    """query 命中实体时，沿 1-hop 关联边召回相连记忆并给出多跳解释。"""
    with tempfile.TemporaryDirectory(prefix="stage8_ks_recall_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()
        now = time.time()

        # 两条记忆：一条关于美式咖啡，一条关于苏州。图谱写通二者关联。
        with conn:
            _insert_atom(
                conn, "atom_coffee", "主人喜欢喝美式咖啡",
                vec=await emb.get_embedding("主人喜欢喝美式咖啡"), updated_at=now,
            )
            _insert_atom(
                conn, "atom_suzhou", "主人居住在苏州工业园区",
                vec=await emb.get_embedding("主人居住在苏州工业园区"), updated_at=now,
            )
        st.upsert_kg_triple(
            "s1", "主人", "偏好", "美式咖啡",
            atom_id="atom_coffee", fact_text="主人喜欢喝美式咖啡", now=now,
            head_type="person", tail_type="object",
        )
        st.upsert_kg_triple(
            "s1", "主人", "居住于", "苏州",
            atom_id="atom_suzhou", fact_text="主人居住在苏州工业园区", now=now,
            head_type="person", tail_type="place",
        )

        retriever = _make_retriever(st)
        # (a) query 命中叶子实体「美式咖啡」→ 1-hop 只到枢纽「主人」，不越级到「苏州」(2-hop)
        leaf_result = await retriever.retrieve(
            "美式咖啡", session_id="s1", top_k=5, candidate_k=10
        )
        leaf_pairs = {
            (e["entity"], e["neighbor"]) for e in leaf_result["graph_explanations"]
        }
        assert ("美式咖啡", "主人") in leaf_pairs or ("主人", "美式咖啡") in leaf_pairs, leaf_pairs
        assert all("苏州" not in a and "苏州" not in b for a, b in leaf_pairs), (
            f"1-hop 不应越级召回 2-hop 实体: {leaf_pairs}"
        )

        # (b) query 命中枢纽实体「主人」→ 1-hop 同时召回两条关联边
        result = await retriever.retrieve(
            "主人最近怎么样", session_id="s1", top_k=5, candidate_k=10
        )
        contents = [str(i.get("content", "")) for i in result["retrieved_l2"]]
        assert any("咖啡" in c for c in contents), f"枢纽 1-hop 未召回咖啡记忆: {contents}"
        assert any("苏州" in c for c in contents), f"枢纽 1-hop 未召回苏州记忆: {contents}"
        assert result["graph_explanations"], "未产出图谱多跳解释"
        explains = " ".join(e["explain"] for e in result["graph_explanations"])
        assert "偏好" in explains and "居住于" in explains, explains

        hop_pairs = {(e["entity"], e["neighbor"]) for e in result["graph_explanations"]}
        assert any("苏州" in a or "苏州" in b for a, b in hop_pairs), (
            f"1-hop 未召回苏州关联: {hop_pairs}"
        )
        assert any("美式咖啡" in a or "美式咖啡" in b for a, b in hop_pairs), (
            f"1-hop 未召回咖啡关联: {hop_pairs}"
        )

        # 图谱写读 API 直接校验：种子「主人」应同时命中两条关联原子
        atoms_map, explanations = retriever._graph_subgraph_recall(
            conn, "主人", session_id="s1"
        )
        assert "atom_coffee" in atoms_map, f"图谱召回未命中 atom_coffee: {atoms_map}"
        assert "atom_suzhou" in atoms_map, f"图谱召回未命中 atom_suzhou: {atoms_map}"
        assert all(0 < v["score"] <= 1.0001 for v in atoms_map.values()), atoms_map
        print(
            f"[OK] 1-hop 关联召回: atoms={sorted(atoms_map)} "
            f"explanations={len(explanations)} 例"
        )
        st.close()


async def test_graph_recall_isolation() -> None:
    """严格会话隔离：图谱召回不得跨会话串台。"""
    with tempfile.TemporaryDirectory(prefix="stage8_kg_iso_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        now = time.time()
        with conn:
            _insert_atom(conn, "atom_other", "另一会话的美式咖啡偏好", scope="s2", updated_at=now)
        st.upsert_kg_triple(
            "s2", "主人", "偏好", "美式咖啡",
            atom_id="atom_other", fact_text="另一会话的美式咖啡偏好", now=now,
        )

        retriever = _make_retriever(st)
        atoms_map, _ = retriever._graph_subgraph_recall(conn, "美式咖啡", session_id="s1")
        assert "atom_other" not in atoms_map, "严格隔离下发生了跨会话图谱召回"
        # 放宽隔离后可召回
        retriever_open = _make_retriever(st, strict_session_isolation=False)
        atoms_open, _ = retriever_open._graph_subgraph_recall(conn, "美式咖啡", session_id="s1")
        assert "atom_other" in atoms_open, "关闭隔离后仍未召回"
        print("[OK] 图谱召回遵守 R-04 严格会话隔离（默认拦截 / 放宽后可见）")
        st.close()


# ===========================================================================
# C-2：艾宾浩斯时间衰减
# ===========================================================================


async def test_decay_math_exponential_and_pinned() -> None:
    """衰减数学：exp 曲线手算对齐，pinned=1 恒为 1.0。"""
    with tempfile.TemporaryDirectory(prefix="stage8_decay_math_") as tmp:
        st = _new_storage(Path(tmp))
        retriever = _make_retriever(st, decay_lambda=1.0, decay_floor=0.0)
        now = 1_000_000_000.0

        # ttl=30 天、已过 30 天 → exp(-1) ≈ 0.3679
        atom = {
            "pinned": 0, "tier": 2, "ttl_days": 30.0,
            "decay_type": "exponential", "updated_at": now - 30 * 86400,
        }
        f = retriever._time_decay_factor(atom, now=now)
        expected = math.exp(-1.0)
        assert abs(f - expected) < 1e-9, f"指数衰减偏差: {f} vs {expected}"

        # 已过 60 天 → exp(-2)
        atom["updated_at"] = now - 60 * 86400
        assert abs(retriever._time_decay_factor(atom, now=now) - math.exp(-2.0)) < 1e-9

        # 刚更新 → 1.0
        atom["updated_at"] = now
        assert retriever._time_decay_factor(atom, now=now) == 1.0

        # pinned=1 免死金牌：无论多久都是 1.0
        pinned = dict(atom, pinned=1, updated_at=now - 3650 * 86400)
        assert retriever._time_decay_factor(pinned, now=now) == 1.0, "pinned=1 被衰减"

        # tier>=3 同样免死
        tier3 = dict(atom, pinned=0, tier=3, updated_at=now - 3650 * 86400)
        assert retriever._time_decay_factor(tier3, now=now) == 1.0, "tier=3 被衰减"

        # linear / none 分支
        lin = dict(atom, decay_type="linear", ttl_days=10.0, updated_at=now - 5 * 86400)
        assert abs(retriever._time_decay_factor(lin, now=now) - 0.5) < 1e-9
        none = dict(atom, decay_type="none", updated_at=now - 999 * 86400)
        assert retriever._time_decay_factor(none, now=now) == 1.0

        # 关闭全局衰减开关 → 全量 1.0（向下兼容）
        retriever.decay_enabled = False
        assert retriever._time_decay_factor(atom, now=now) == 1.0
        print(
            f"[OK] 衰减数学: exp(-1)={expected:.6f} / pinned 免衰 / linear=0.5 / "
            "none=1.0 / 开关关闭=1.0"
        )
        st.close()


async def test_decay_reorders_retrieval() -> None:
    """陈旧琐事降权，新鲜上下文上位；pinned 记忆不受影响。"""
    with tempfile.TemporaryDirectory(prefix="stage8_decay_rank_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()
        now = time.time()

        # 同主题两条记忆：一条一年前，一条刚刚；向量完全一致 → 基础分相同
        same_vec = await emb.get_embedding("主人喜欢喝美式咖啡")
        with conn:
            _insert_atom(
                conn, "atom_old", "主人喜欢喝美式咖啡（旧）",
                vec=same_vec, updated_at=now - 365 * 86400,
                ttl_days=30.0, importance=0.5,
            )
            _insert_atom(
                conn, "atom_new", "主人喜欢喝美式咖啡（新）",
                vec=same_vec, updated_at=now,
                ttl_days=30.0, importance=0.5,
            )

        retriever = _make_retriever(st, decay_lambda=1.0, decay_floor=0.0)
        result = await retriever.retrieve("美式咖啡", session_id="s1", top_k=5, candidate_k=10)
        by_id = {i["atom_id"]: i for i in result["retrieved_l2"]}
        assert "atom_new" in by_id and "atom_old" in by_id, by_id.keys()
        assert by_id["atom_new"]["decay_factor"] == 1.0
        assert by_id["atom_old"]["decay_factor"] < 0.1, by_id["atom_old"]["decay_factor"]
        assert (
            by_id["atom_old"]["relevance_score"] < by_id["atom_new"]["relevance_score"]
        ), "衰减未生效：旧记忆得分未被压低"
        assert by_id["atom_old"]["relevance_score"] < by_id["atom_old"]["base_score"]

        # 置顶记忆不受衰减：即使一年前也保持 base_score
        with conn:
            _insert_atom(
                conn, "atom_pin", "主人喜欢喝美式咖啡（置顶铁律）",
                vec=await emb.get_embedding("陈旧置顶偏好美式咖啡"),
                updated_at=now - 3650 * 86400, pinned=1, tier=3, importance=0.95,
            )
        result2 = await retriever.retrieve("美式咖啡", session_id="s1", top_k=5, candidate_k=10)
        pin_hits = [i for i in result2["retrieved_l2"] if i["atom_id"] == "atom_pin"]
        if pin_hits:
            assert pin_hits[0]["decay_factor"] == 1.0, "置顶记忆被衰减"
            assert pin_hits[0]["relevance_score"] == pin_hits[0]["base_score"]
        print(
            f"[OK] 衰减重排: old={by_id['atom_old']['relevance_score']} "
            f"< new={by_id['atom_new']['relevance_score']}，置顶免衰"
        )
        st.close()


# ===========================================================================
# C-3：软废弃图谱联动
# ===========================================================================


async def test_soft_deprecation_invalidates_edges() -> None:
    """原子软废弃 → 关联边标记 invalidated / 权重归零 / 不再参与召回。"""
    with tempfile.TemporaryDirectory(prefix="stage8_kg_deprecate_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        now = time.time()

        with conn:
            _insert_atom(conn, "atom_conflict", "主人喜欢喝绿茶", status="active", updated_at=now)
        st.upsert_kg_triple(
            "s1", "主人", "偏好", "绿茶",
            atom_id="atom_conflict", fact_text="主人喜欢喝绿茶", now=now,
        )
        assert st.get_kg_stats()["active_edges"] == 1

        sm = MemorySummarizer(
            storage=st,
            embedding_client=FakeEmbeddingClient(),
            config={"evaluator": {"enable": False}, "graph": {"enable": True}},
        )
        # 直接走冲突处置：判官认定 atom_conflict 与新事实冲突
        applied = sm._apply_conflict_resolution(
            [
                {
                    "atom_id": "atom_conflict",
                    "fact": "主人喜欢喝绿茶",
                    "pinned": 0,
                    "reason": "被新偏好推翻",
                }
            ],
            {"主人喜欢喝咖啡": "atom_new_fact"},
            now,
        )
        assert applied == 1, f"冲突废弃未生效: {applied}"

        atom_row = conn.execute(
            "SELECT status, superseded_by FROM l2_atoms WHERE atom_id='atom_conflict'"
        ).fetchone()
        assert atom_row["status"] == "superseded"

        edges = conn.execute("SELECT * FROM kg_edges WHERE atom_id='atom_conflict'").fetchall()
        assert edges, "关联边丢失（不应物理删除）"
        edge = edges[0]
        assert edge["status"] == "invalidated", f"边未失效: {dict(edge)}"
        assert float(edge["weight"]) == 0.0, "失效边权重未归零"
        assert float(edge["invalidated_at"]) > 0, "未记录失效时间"
        assert st.get_kg_stats()["active_edges"] == 0

        # 失效边不再参与 1-hop 召回
        expand = st.expand_kg_subgraph("s1", ["绿茶"])
        assert not [e for e in expand if e["atom_id"] == "atom_conflict"], (
            f"失效边仍被召回: {expand}"
        )

        # 置顶记忆永不废弃（R-01 免死金牌在图谱维度的延续）
        with conn:
            _insert_atom(conn, "atom_pin_keep", "主人讨厌香菜", pinned=1, tier=3, updated_at=now)
        st.upsert_kg_triple("s1", "主人", "厌恶", "香菜", atom_id="atom_pin_keep", now=now)
        applied2 = sm._apply_conflict_resolution(
            [{"atom_id": "atom_pin_keep", "fact": "主人讨厌香菜", "pinned": 1}],
            {},
            now,
        )
        assert applied2 == 0, "置顶记忆被错误废弃"
        keep_edge = conn.execute(
            "SELECT status FROM kg_edges WHERE atom_id='atom_pin_keep'"
        ).fetchone()
        assert keep_edge["status"] == "active", "置顶记忆关联边被误失效"
        print("[OK] 软废弃联动：边 invalidated/weight=0 保留可追溯，置顶记忆免死")
        st.close()


# ===========================================================================
# 综合：默认配置等价性与零回归
# ===========================================================================


async def test_default_config_equivalence() -> None:
    """默认配置下：无图谱/无衰减数据时检索结果与旧行为一致（零破坏）。"""
    with tempfile.TemporaryDirectory(prefix="stage8_default_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()
        now = time.time()
        with conn:
            for i, text in enumerate(
                ["主人喜欢喝美式咖啡", "项目使用 SQLite 存储", "天气很好适合散步"]
            ):
                _insert_atom(
                    conn, f"atom_{i}", text,
                    vec=await emb.get_embedding(text), updated_at=now,
                )

        retriever = _make_retriever(st)
        result = await retriever.retrieve("咖啡", session_id="s1", top_k=3, candidate_k=6)
        assert result["retrieved_l2"], "默认配置召回为空"
        top = result["retrieved_l2"][0]
        assert "咖啡" in top["content"], f"默认配置 Top1 不相关: {top['content']}"
        # 无图谱数据 → 不应凭空产出解释
        assert result["graph_explanations"] == []
        # 新鲜记忆衰减因子为 1.0
        assert top["decay_factor"] == 1.0
        assert set(result.keys()) >= {
            "pinned_l3", "retrieved_l2", "prompt_context", "graph_explanations"
        }
        print(
            f"[OK] 默认配置零破坏: top={top['content']!r} decay={top['decay_factor']} "
            "graph_explanations=[]"
        )
        st.close()


async def test_prompt_context_graph_explain() -> None:
    """图谱命中信息可回注 prompt_context，提供可读的多跳解释。"""
    with tempfile.TemporaryDirectory(prefix="stage8_prompt_") as tmp:
        st = _new_storage(Path(tmp))
        conn = st.connect()
        emb = FakeEmbeddingClient()
        now = time.time()
        with conn:
            _insert_atom(
                conn, "atom_coffee", "主人喜欢喝美式咖啡",
                vec=await emb.get_embedding("主人喜欢喝美式咖啡"), updated_at=now,
            )
        st.upsert_kg_triple("s1", "主人", "偏好", "美式咖啡", atom_id="atom_coffee", now=now)

        retriever = _make_retriever(st)
        result = await retriever.retrieve("美式咖啡", session_id="s1", top_k=3, candidate_k=6)
        context = result["prompt_context"]
        assert "长期记忆上下文" in context
        if any(i.get("graph_explain") for i in result["retrieved_l2"]):
            assert "图谱关联" in context, f"图谱解释未回注 prompt: {context}"
        print("[OK] prompt_context 含图谱关联解释字段（命中时回注）")
        st.close()


def main() -> int:
    cases = [
        # C-1 图谱读写
        test_kg_schema_columns_fresh_and_legacy,
        test_kg_triple_write_dedup_and_mentions,
        test_triple_rule_extraction,
        test_summarizer_writes_kg_end_to_end,
        test_graph_disabled_zero_side_effect,
        # C-1 1-hop 召回
        test_graph_1hop_recall_in_retriever,
        test_graph_recall_isolation,
        # C-2 时间衰减
        test_decay_math_exponential_and_pinned,
        test_decay_reorders_retrieval,
        # C-3 软废弃联动
        test_soft_deprecation_invalidates_edges,
        # 综合
        test_default_config_equivalence,
        test_prompt_context_graph_explain,
    ]
    for case in cases:
        asyncio.run(case())
    print(f"\n=== Stage8 方案 C 专项 用例全部通过（{len(cases)} 项） ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
