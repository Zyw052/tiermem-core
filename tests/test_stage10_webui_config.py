# -*- coding: utf-8 -*-
"""
Stage 10（主人工单）：WebUI 全量补全 / 配置保存链路 / 安全加固自测。

覆盖点：
1. 【Schema 兼容】无二级嵌套 object、type 全在宿主 DEFAULT_VALUE_MAP 白名单、
   webui.allow_config_write 开关存在；可用宿主源码时再做 AstrBotConfig + validate_config 真实校验。
2. 【配置读写】GET 视图密钥掩码（明文绝不出现在返回体）、白名单拒绝未知键、
   强类型 / options / slider 校验、db_path 路径穿越与空字节拒绝、
   掩码占位保持原密钥、原子写 + BOM 兼容、运行时热同步（不 reload）。
3. 【Web 路由】16 条路由注册、前缀/占位符合法；config / providers/test /
   memories/edit / memories/delete / graph / maintenance 处理器正常与降级行为。
4. 【存储写操作】真实 SQLite：update_memory / list_kg_graph / delete_memory 的
   正确性与级联清理。
5. 【前端契约】8 个 Tab、保存按钮、全部新 API 端点、XSS 转义函数、无外部依赖。

运行： python -m pytest tests/test_stage10_webui_config.py -q
（依赖 pytest / pytest-asyncio；本文件全部用 asyncio.run 包装，不强依赖插件）
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import config_api  # noqa: E402
import page_api  # noqa: E402
from config_api import ConfigError, ConfigManager, MASK_DISPLAY, MASK_TOKEN  # noqa: E402
from storage import MemoryStorage  # noqa: E402

HOST_SUPPORTED_TYPES = {
    "int", "float", "bool", "string", "text", "list", "file", "object",
    "template_list", "dict",
}

SECRET = "sk-SUPER-SECRET-9527"
EMB_SECRET = "emb-SECRET-7788"


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class FakeContext:
    def __init__(self) -> None:
        self.registered = []

    def register_web_api(self, route, view_handler, methods, desc):
        self.registered.append(route)


class FakeHostConfig:
    """模拟宿主 AstrBotConfig：只暴露 config_path。"""

    def __init__(self, path: Path) -> None:
        self.config_path = str(path)


class FakeStorage:
    def __init__(self) -> None:
        self.calls = []

    def list_memories(self, **kwargs):
        return {"items": [], "total": 0, "limit": 20, "offset": 0}

    def get_memory_detail(self, atom_id):
        return {"atom_id": atom_id, "content": "x", "triples": []}

    def update_memory(self, atom_id, **kwargs):
        self.calls.append(("update", atom_id, kwargs))
        return {"found": True, "atom_id": atom_id, "updated": ["content"], "error": ""}

    def delete_memory(self, atom_id):
        self.calls.append(("delete", atom_id))
        return {"found": True, "atom_id": atom_id,
                "deleted": {"atom": 1, "kg_edges": 0}, "error": ""}

    def list_kg_graph(self, limit=100, query=""):
        return {"nodes": [{"node_id": "n1", "entity_name": "A"}],
                "edges": [{"head": "A", "tail": "B", "relation_type": "likes"}],
                "stats": {"nodes": 2, "edges": 1, "active_edges": 1},
                "limit": limit, "query": query}

    def list_dlq(self, limit=20, status=""):
        return {"items": [], "counts": {}}


class FakeBackend:
    provider_id = "fake_provider"
    model = "fake-model"

    async def embed_one(self, text):
        return [0.1, 0.2, 0.3]

    async def rerank(self, query, documents, top_n=None):
        return [(0, 0.9)]

    async def chat(self, messages, model=""):
        return "OK"


class FakeBridge:
    def list_providers(self):
        return {
            "embedding": [{"id": "fake_provider"}],
            "rerank": [{"id": "fake_provider"}],
            "chat_completion": [{"id": "fake_provider"}],
        }

    def build_embedding_backend(self, provider_id="", allow_auto=False):
        return FakeBackend()

    def build_rerank_backend(self, provider_id="", allow_auto=False):
        return FakeBackend()

    def build_chat_backend(self, provider_id="", allow_auto=False):
        return FakeBackend()

    def load_bindings(self, force=False):
        return {}


class FakeSummarizer:
    async def flush_all(self):
        return {"session_1": 2}

    async def sweep_dlq_once(self, now=None):
        return {"scanned": 0, "completed": 0, "rescheduled": 0, "failed": 0, "skipped": 0}

    def get_dlq_stats(self):
        return {}


class FakePlugin:
    def __init__(self, config_path: Path) -> None:
        self.context = FakeContext()
        self.config: Dict[str, Any] = {
            "enable_memory": True,
            "webui": {"register_api": True, "allow_config_write": True},
        }
        self._host_config = FakeHostConfig(config_path)
        self.storage = FakeStorage()
        self.provider_bridge = FakeBridge()
        self.summarizer = FakeSummarizer()
        self.retriever = None
        self.embedding_client = None
        self.rerank_client = None
        self._migrator_cls = None
        self.page_api = None
        self.applied_calls = []

    def build_self_check(self):
        return {}

    def get_status(self):
        return {}

    def apply_runtime_config(self, cfg):
        self.applied_calls.append(cfg)
        return {"applied": ["enable_memory"]}


def _initial_config() -> Dict[str, Any]:
    """最小初始配置：含两处明文密钥，用于验证掩码与保持语义。"""
    return {
        "enable_memory": True,
        "memory_injection_enable": False,
        "summary_turn_interval": 5,
        "db_path": "",
        "summarizer": {"api_key": SECRET, "api_base": "", "model": ""},
        "models": {
            "embedding": {"api_key": EMB_SECRET, "model": "BAAI/bge-m3", "dim": 1024},
            "rerank": {"api_key": "", "model": "BAAI/bge-reranker-v2-m3"},
        },
        "retrieval": {"top_k": 6, "inject_position": "system_head"},
        "webui": {"register_api": True, "allow_config_write": True},
    }


def _write_config(path: Path, data: Dict[str, Any], bom: bool = False) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(("\ufeff" if bom else "") + text, encoding="utf-8")


def _read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _manager(config_path: Path, plugin=None) -> ConfigManager:
    return ConfigManager(
        plugin,
        schema_path=str(PLUGIN_ROOT / "_conf_schema.json"),
        config_path=str(config_path),
    )


def _patch_body(monkeypatch, body: Dict[str, Any]) -> Dict[str, Any]:
    """替换 page_api._json_body，模拟 WebUI POST 请求体。"""
    holder = {"body": body}

    async def fake_json_body():
        return holder["body"]

    monkeypatch.setattr(page_api, "_json_body", fake_json_body)
    return holder


# ---------------------------------------------------------------------------
# 一、Schema 兼容性（根因加固）
# ---------------------------------------------------------------------------
def test_schema_has_no_nested_object_and_is_host_compatible():
    schema = json.loads(
        (PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8-sig")
    )
    compat = _manager(Path("/tmp/unused.json")).validate_schema_compatibility()
    assert compat["ok"], compat["issues"]
    assert compat["nested_objects"] == [], "不得存在二级嵌套 object"

    # 所有 type 必须落在宿主 DEFAULT_VALUE_MAP 白名单
    def walk(node):
        for _, meta in (node or {}).items():
            if not isinstance(meta, dict):
                continue
            assert meta["type"] in HOST_SUPPORTED_TYPES, meta["type"]
            if meta["type"] == "object":
                walk(meta.get("items") or {})

    walk(schema)
    # models 的二级分组已改为宿主原生支持的 dict 自由映射，且默认配置完整
    assert schema["models"]["items"]["embedding"]["type"] == "dict"
    assert schema["models"]["items"]["embedding"]["default"]["model"] == "BAAI/bge-m3"
    # 面板写开关存在且默认开启
    assert schema["webui"]["items"]["allow_config_write"]["default"] is True
    print("[OK] schema 无二级嵌套、类型全合法、面板写开关存在")


def test_astrbot_native_config_roundtrip(tmp_path):
    """可用宿主源码时，用真实 AstrBotConfig + validate_config 校验 schema。"""
    try:
        from astrbot.core.config.astrbot_config import AstrBotConfig  # type: ignore
        from astrbot.dashboard.services.config_service import validate_config  # type: ignore
    except Exception as exc:  # pragma: no cover - 脱离宿主源码时跳过
        pytest.skip(f"宿主 AstrBot 源码不可用，跳过原生兼容性校验: {exc}")

    schema = json.loads(
        (PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8-sig")
    )
    cfg_path = tmp_path / "native_config.json"
    cfg = AstrBotConfig(config_path=str(cfg_path), schema=schema)
    assert len(cfg) == len(schema), "AstrBotConfig 未按 schema 生成全部顶层键"
    assert cfg["models"]["embedding"]["model"] == "BAAI/bge-m3"
    errors, _ = validate_config(json.loads(json.dumps(dict(cfg))), schema, is_core=False)
    assert not errors, f"宿主 validate_config 拒绝 schema: {errors}"
    print("[OK] 宿主 AstrBotConfig + validate_config 双向校验通过")


# ---------------------------------------------------------------------------
# 二、配置视图：密钥掩码
# ---------------------------------------------------------------------------
def test_config_view_masks_all_secrets(tmp_path):
    cfg_path = tmp_path / "plugin_config.json"
    _write_config(cfg_path, _initial_config(), bom=True)
    view = _manager(cfg_path).view()

    assert view["values"]["summarizer"]["api_key"] == MASK_DISPLAY
    assert view["values"]["models"]["embedding"]["api_key"] == MASK_DISPLAY
    assert "summarizer.api_key" in view["sensitive_paths"]
    assert "models.embedding.api_key" in view["sensitive_paths"]
    # 序列化后的完整返回体不得出现任何明文密钥
    blob = json.dumps(view, ensure_ascii=False)
    assert SECRET not in blob and EMB_SECRET not in blob, "GET /config 泄漏了明文密钥"
    # 只暴露文件名，不泄漏绝对路径
    assert "/" not in view["config_file"]
    assert view["mask_token"] == MASK_TOKEN
    print("[OK] GET /config 全量掩码，无明文、无绝对路径泄漏")


# ---------------------------------------------------------------------------
# 三、配置保存：白名单 / 类型 / 枚举 / 范围 / 路径穿越
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload, reason",
    [
        ({"evil_unknown_key": 1}, "未知键"),
        ({"summary_turn_interval": "not-a-number"}, "非法 int"),
        ({"retrieval": {"top_k": 999}}, "slider 越界"),
        ({"retrieval": {"inject_position": "not_allowed"}}, "options 枚举"),
        ({"db_path": "../../etc/passwd"}, "路径穿越"),
        ({"db_path": "http://evil.example/x.db"}, "协议头"),
        ({"db_path": "a\x00b.db"}, "空字节"),
        ({"models": {"embedding": {"injected_key": "x"}}}, "dict 子键白名单"),
    ],
)
def test_config_save_rejects_invalid_payload(tmp_path, payload, reason):
    cfg_path = tmp_path / "plugin_config.json"
    _write_config(cfg_path, _initial_config())
    before = cfg_path.read_bytes()
    with pytest.raises(ConfigError):
        _manager(cfg_path).save(payload)
    assert cfg_path.read_bytes() == before, f"{reason} 校验失败时不得写文件"
    print(f"[OK] 拒绝非法载荷：{reason}")


def test_config_save_roundtrip_mask_preserve_and_runtime_sync(tmp_path):
    cfg_path = tmp_path / "plugin_config.json"
    _write_config(cfg_path, _initial_config(), bom=True)
    plugin = FakePlugin(cfg_path)
    manager = _manager(cfg_path, plugin)

    result = manager.save({
        "enable_memory": False,
        "retrieval": {"top_k": 8},
        "noise_filter": {"enabled": "yes"},          # 字符串 bool 归一化
        "summarizer": {"api_key": MASK_DISPLAY},     # 掩码占位 → 保持
    })
    assert result["changed"], result
    assert "enable_memory" in result["changed"]
    assert "retrieval.top_k" in result["changed"]
    assert not any("api_key" in c for c in result["changed"]), "掩码占位不应产生变更"
    assert result["requires_reload"] is False

    saved = _read_config(cfg_path)
    assert saved["enable_memory"] is False
    assert saved["retrieval"]["top_k"] == 8
    assert saved["noise_filter"]["enabled"] is True
    assert saved["summarizer"]["api_key"] == SECRET, "掩码占位误清空了线上密钥"
    assert saved["models"]["embedding"]["model"] == "BAAI/bge-m3", "未变更的键被丢失"
    # 运行时热同步（不 reload）
    assert plugin.config["enable_memory"] is False
    assert plugin.applied_calls, "未调用 apply_runtime_config"
    # 原子写不留临时文件
    assert not list(tmp_path.glob(".*.tmp"))

    # 显式提交新密钥时应真正更新
    manager.save({"summarizer": {"api_key": "NEW-KEY-001"}})
    assert _read_config(cfg_path)["summarizer"]["api_key"] == "NEW-KEY-001"
    assert _manager(cfg_path).view()["values"]["summarizer"]["api_key"] == MASK_DISPLAY
    print("[OK] 保存往返 / 掩码保持 / 运行时热同步 / 原子写全部通过")


def test_config_save_requires_reload_flag(tmp_path):
    cfg_path = tmp_path / "plugin_config.json"
    _write_config(cfg_path, _initial_config())
    result = _manager(cfg_path).save({"db_path": str(tmp_path / "memory.db")})
    assert result["requires_reload"] is True, "db_path 变更应提示需重载"
    assert _read_config(cfg_path)["db_path"].endswith("memory.db")
    print("[OK] db_path 合法绝对路径可保存，并正确标注 requires_reload")


# ---------------------------------------------------------------------------
# 四、Web 路由与处理器
# ---------------------------------------------------------------------------
def test_page_api_route_table_is_complete():
    routes = page_api.PluginPageApi.ROUTES
    suffixes = {r[0] for r in routes}
    for needed in (
        "config", "config/save", "providers/test", "memories/edit",
        "memories/delete", "graph", "maintenance", "recall", "memory/<atom_id>",
        "status", "memories", "pin", "noise", "dlq", "providers", "bindings",
    ):
        assert needed in suffixes, f"缺少路由 {needed}"
    assert len(routes) == 16

    plugin = FakePlugin(Path("/tmp/unused.json"))
    api = page_api.PluginPageApi(plugin)
    assert api.register_routes() == len(routes)
    assert all(
        r.startswith("/astrbot_plugin_agent_memory/page") for r in plugin.context.registered
    )
    assert not any("{" in r for r in plugin.context.registered)
    print(f"[OK] 注册 {len(routes)} 条路由，前缀与占位符合法")


def test_page_api_config_handlers(tmp_path, monkeypatch):
    cfg_path = tmp_path / "plugin_config.json"
    _write_config(cfg_path, _initial_config())
    plugin = FakePlugin(cfg_path)
    api = page_api.PluginPageApi(plugin)

    async def scenario():
        # GET /config
        holder = _patch_body(monkeypatch, {})
        got = await api.get_config()
        assert got["status"] == "ok"
        assert SECRET not in json.dumps(got, ensure_ascii=False)

        # POST /config/save 正常
        holder["body"] = {"config": {"retrieval": {"top_k": 7}}}
        saved = await api.save_config()
        assert saved["status"] == "ok", saved
        assert "retrieval.top_k" in saved["data"]["changed"]
        assert _read_config(cfg_path)["retrieval"]["top_k"] == 7

        # POST /config/save 非法载荷 → error 不抛
        holder["body"] = {"config": {"unknown": 1}}
        bad = await api.save_config()
        assert bad["status"] == "error" and "未知配置项" in bad["message"]

        # 只读开关
        plugin.config["webui"]["allow_config_write"] = False
        holder["body"] = {"config": {"enable_memory": False}}
        denied = await api.save_config()
        assert denied["status"] == "error" and "禁用" in denied["message"]
        plugin.config["webui"]["allow_config_write"] = True

    asyncio.run(scenario())
    print("[OK] config / config-save 处理器：掩码、保存、拒绝、只读开关全部正确")


def test_page_api_write_and_maintenance_handlers(tmp_path, monkeypatch):
    plugin = FakePlugin(tmp_path / "plugin_config.json")
    api = page_api.PluginPageApi(plugin)
    holder = _patch_body(monkeypatch, {})

    async def scenario():
        # 编辑记忆
        holder["body"] = {}
        assert (await api.update_memory())["status"] == "error"
        holder["body"] = {"atom_id": "atom_1", "content": "新正文"}
        assert (await api.update_memory())["status"] == "ok"

        # 删除记忆（必须显式确认）
        holder["body"] = {"atom_id": "atom_1"}
        assert (await api.delete_memory())["status"] == "error"
        holder["body"] = {"atom_id": "atom_1", "confirm": True}
        assert (await api.delete_memory())["status"] == "ok"

        # 图谱
        graph = await api.get_graph()
        assert graph["status"] == "ok" and graph["data"]["stats"]["nodes"] == 2

        # 维护
        holder["body"] = {"op": "not_allowed"}
        assert (await api.run_maintenance())["status"] == "error"
        holder["body"] = {"op": "flush"}
        flush = await api.run_maintenance()
        assert flush["status"] == "ok" and flush["data"]["flushed"] == {"session_1": 2}

        # 连通性测试
        holder["body"] = {"kind": "bad"}
        assert (await api.test_provider())["status"] == "error"
        holder["body"] = {"kind": "embedding", "provider_id": "fake_provider"}
        probe = await api.test_provider()
        assert probe["status"] == "ok", probe
        assert probe["data"]["dim"] == 3
        assert "api_key" not in json.dumps(probe, ensure_ascii=False)

    asyncio.run(scenario())
    print("[OK] 记忆编辑/删除、图谱、维护、连通性测试处理器全部通过")


def test_page_api_handlers_degrade_without_components(tmp_path, monkeypatch):
    plugin = FakePlugin(tmp_path / "plugin_config.json")
    plugin.storage = None
    plugin.provider_bridge = None
    plugin.summarizer = None
    plugin._migrator_cls = None
    api = page_api.PluginPageApi(plugin)
    holder = _patch_body(monkeypatch, {"atom_id": "x", "confirm": True,
                                      "op": "flush", "kind": "embedding"})

    async def scenario():
        for coro in (
            api.update_memory(), api.delete_memory(), api.get_graph(),
            api.run_maintenance(), api.test_provider(),
            api.list_memories(), api.get_dlq(),
        ):
            result = await coro
            assert result["status"] == "error", result  # 不抛异常、可读错误

    asyncio.run(scenario())
    print("[OK] 组件缺失时全部新处理器安全降级为 error")


# ---------------------------------------------------------------------------
# 五、真实 SQLite：编辑 / 删除 / 图谱
# ---------------------------------------------------------------------------
def test_storage_edit_delete_and_graph(tmp_path):
    storage = MemoryStorage(tmp_path)
    storage.init_schema()
    conn = storage.connect()
    now = 1000.0
    with conn:
        conn.execute(
            "INSERT INTO l2_atoms (atom_id, scope_key, content, last_accessed_at, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("atom_1", "scope_1", "原始内容", now, now, now),
        )
        conn.execute(
            "INSERT INTO kg_nodes (node_id, scope_key, entity_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)", ("n1", "scope_1", "薇薇安", now, now),
        )
        conn.execute(
            "INSERT INTO kg_nodes (node_id, scope_key, entity_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)", ("n2", "scope_1", "主人", now, now),
        )
        conn.execute(
            "INSERT INTO kg_edges (edge_id, scope_key, source_node_id, target_node_id, "
            "relation_type, atom_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("e1", "scope_1", "n1", "n2", "serves", "atom_1", now, now),
        )

    # 编辑
    edited = storage.update_memory(
        "atom_1", content="编辑后的内容", importance=0.8,
        key_facts=["事实A"], topics=["主题A"],
    )
    assert edited["found"] and not edited["error"], edited
    row = conn.execute(
        "SELECT content, importance, key_facts_json, topics_json FROM l2_atoms "
        "WHERE atom_id = ?", ("atom_1",)
    ).fetchone()
    assert row["content"] == "编辑后的内容"
    assert abs(row["importance"] - 0.8) < 1e-6
    assert json.loads(row["key_facts_json"]) == ["事实A"]

    # 非法输入被拒绝
    assert storage.update_memory("atom_1", content="")["error"]
    assert storage.update_memory("atom_1", importance=5.0)["error"]
    assert not storage.update_memory("missing_id", content="x")["found"]

    # 图谱浏览
    graph = storage.list_kg_graph(limit=50)
    assert graph["stats"]["nodes"] == 2 and graph["stats"]["edges"] == 1
    assert graph["edges"][0]["head"] == "薇薇安" and graph["edges"][0]["tail"] == "主人"
    assert not any(isinstance(v, (bytes, bytearray)) for e in graph["edges"] for v in e.values())

    # 删除 + 级联
    deleted = storage.delete_memory("atom_1")
    assert deleted["found"] and deleted["deleted"]["atom"] == 1, deleted
    assert deleted["deleted"]["kg_edges"] == 1
    assert conn.execute(
        "SELECT COUNT(1) c FROM l2_atoms WHERE atom_id = ?", ("atom_1",)
    ).fetchone()["c"] == 0
    assert storage.delete_memory("atom_1")["error"]  # 重复删除给出可读错误
    storage.close()
    print("[OK] 真实 SQLite 编辑 / 图谱 / 级联删除全部通过")


# ---------------------------------------------------------------------------
# 六、前端契约与安全
# ---------------------------------------------------------------------------
def test_webui_page_full_contract():
    html = (PLUGIN_ROOT / "views" / "dashboard" / "index.html").read_text(encoding="utf-8")
    # 8 个 Tab
    for tab in ("overview", "config", "providers", "memories", "recall",
                "graph", "noise", "maintenance"):
        assert f'data-tab="{tab}"' in html, f"缺少 Tab: {tab}"
    # 配置保存按钮（主人痛点①）
    assert 'id="btn-save-config"' in html
    assert "page/config/save" in html
    assert "page/config" in html
    # 全部新增 API 端点已接线
    for endpoint in (
        "page/memory/", "page/recall", "page/graph", "page/maintenance",
        "page/providers/test", "page/memories/edit", "page/memories/delete",
        "page/status", "page/memories", "page/noise", "page/dlq",
    ):
        assert endpoint in html, f"前端缺少接口调用: {endpoint}"
    # 删除二次确认 + 掩码输入 + XSS 转义
    assert "confirm" in html and "window.confirm" in html
    assert "type=\"password\"" in html or "'password'" in html
    assert "function esc(" in html and "esc(" in html
    assert "eval(" not in html and "document.write(" not in html
    # 无外部 CDN / 无内联明文密钥
    for bad in ("http://", "https://cdn", "node_modules"):
        assert bad not in html, f"页面不应依赖外部资源: {bad}"
    assert SECRET not in html and EMB_SECRET not in html
    # 桥接 SDK 仍保留
    assert "/api/plugin/page/bridge-sdk.js" in html
    print("[OK] 前端 8 Tab / 保存按钮 / 全量接口 / XSS 防护 / 无外链")


def test_config_api_source_has_no_hardcoded_paths_or_secrets():
    src = (PLUGIN_ROOT / "config_api.py").read_text(encoding="utf-8")
    assert "/data/apps/" not in src and "/AstrBot/" not in src, "源码不得写死绝对路径"
    assert SECRET not in src and EMB_SECRET not in src
    assert "os.replace" in src and "mkstemp" in src, "必须原子写"
    assert "utf-8-sig" in src, "必须兼容线上 BOM 配置文件"
    assert "0o600" in src, "配置文件权限必须收紧"
    print("[OK] config_api.py 无硬编码路径/密钥，原子写 + BOM + 权限齐备")


if __name__ == "__main__":  # 便于脱离 pytest 直接自检
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
