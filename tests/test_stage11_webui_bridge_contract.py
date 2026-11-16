# -*- coding: utf-8 -*-
"""Stage 11：按 astrbot-plugin-dev 规范校验 WebUI 前后端链路闭环。

依据规范章节（产出注明依据）：
  - §2 目录与 metadata.yaml（官方原文 plugin-new.md「克隆项目到本地」）
  - §3 main.py（plugin.md「提要 / 最小实例」：主类继承 Star，宿主自动识别）
  - §6 配置（guides/plugin-config.md「配置定义」「敏感配置项」「dict 类型的 schema」）
  - §9 存储（guides/storage.md「存储大文件规范」）
  - §10 工程规范（plugin-new.md「开发原则」：注释 / 测试 / 不写死路径）
  - WebUI 专章 guides/plugin-pages.md：「目录结构」「后端 Web API / 路由注册」
    「请求对象」「响应对象」「Bridge API / 请求和返回值规则」
    「视图元数据与国际化」「安全约束」「调试建议」
  - 官方原文 guides/plugin-i18n.md「插件可视化视图」/「约束」

本文件不依赖运行中的 AstrBot 主进程，全部为静态契约 + 纯 Python 校验。
运行： python -m pytest tests/test_stage11_webui_bridge_contract.py -q
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import config_api  # noqa: E402
import page_api  # noqa: E402


def _html() -> str:
    return (PLUGIN_ROOT / "views" / "dashboard" / "index.html").read_text(encoding="utf-8")


def _schema() -> Dict[str, Any]:
    return json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8-sig"))


# ---------------------------------------------------------------------------
# 1) Bridge 返回值契约（本次「配置没有保存按钮」根因回归测试）
#    依据：guides/plugin-pages.md「Bridge API / 请求和返回值规则」
# ---------------------------------------------------------------------------

def test_frontend_does_not_double_unwrap_bridge_response():
    html = _html()
    # 旧错误写法：在 bridge 已解包的业务数据上再检查 status / 再取 data
    assert "res.status" not in html, "禁止对 bridge 解包后的结果二次检查 status"
    assert "return res.data" not in html, "禁止对 bridge 解包后的结果再次取 data"
    assert "return await VIEW.apiGet(endpoint" in html
    assert "return await VIEW.apiPost(endpoint" in html
    assert "请求和返回值规则" in html, "前端须注明依据的规范章节"


def test_backend_envelope_matches_host_bridge_contract():
    """宿主 Dashboard 只识别 {status:ok,data} / {status:error,message} 两种外壳。"""
    ok = page_api.ok({"changed": ["enable_memory"]})
    assert ok["status"] == "ok" and ok["data"]["changed"] == ["enable_memory"]
    err = page_api.err("保存配置失败")
    assert err["status"] == "error" and err["message"]
    # 模拟父页面解包（PluginViewPage.vue: sendBridgeResponse(..., response.data?.data)）
    unwrapped = ok.get("data", ok)
    assert unwrapped == {"changed": ["enable_memory"]}
    assert unwrapped.get("status") is None, "解包后不应残留 status 字段"


# ---------------------------------------------------------------------------
# 2) 视图元数据与国际化
#    依据：guides/plugin-pages.md「视图元数据与国际化」/ plugin-i18n.md
# ---------------------------------------------------------------------------

def test_i18n_view_metadata_files():
    for locale in ("zh-CN", "en-US"):
        path = PLUGIN_ROOT / ".astrbot-plugin" / "i18n" / f"{locale}.json"
        assert path.is_file(), f"缺少视图元数据 {locale}.json"
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        assert isinstance(data, dict) and isinstance(data.get("views"), dict)
        view = data["views"].get("dashboard")
        assert isinstance(view, dict) and view.get("title"), "views.dashboard.title 必填"
        assert view.get("description"), "views.dashboard.description 必填"
        assert isinstance(data.get("metadata"), dict)
        assert data["metadata"].get("display_name")


def test_frontend_uses_bridge_i18n_api():
    html = _html()
    assert "applyI18n" in html
    assert "VIEW.ready()" in html, "页面初始化须先等待 bridge.ready()"
    assert "VIEW.onContext(applyI18n)" in html, "语言/主题切换须通过 onContext 响应"
    assert 'VIEW.t("views.dashboard.title"' in html or "views.dashboard.title" in html


# ---------------------------------------------------------------------------
# 3) 配置 schema 合规（§6）
# ---------------------------------------------------------------------------

def test_conf_schema_compliance():
    schema = _schema()
    assert isinstance(schema, dict) and schema
    for key, meta in schema.items():
        assert isinstance(meta, dict) and meta.get("type"), f"{key} 缺少必填 type"
        if meta["type"] == "object":
            assert isinstance(meta.get("items"), dict), f"{key} 的 object 缺少 items"
        if meta["type"] == "dict":
            assert isinstance(meta.get("items"), dict), f"{key} 的 dict 缺少 items"


def test_secret_fields_are_marked():
    schema = _schema()
    for section in ("summarizer", "evaluator"):
        key = schema[section]["items"]["api_key"]
        assert key.get("secret") is True, f"{section}.api_key 须标注 secret:true"
    assert config_api.ConfigManager().validate_schema_compatibility()["ok"]


# ---------------------------------------------------------------------------
# 4) 插件主类 / 元数据单一来源（§2 / §3）
# ---------------------------------------------------------------------------

def test_main_class_uses_star_without_deprecated_register():
    src = (PLUGIN_ROOT / "main.py").read_text(encoding="utf-8")
    assert "class AgentMemoryPlugin(Star)" in src
    assert "@register(" not in src, "旧式 @register 装饰器已废弃（§3）"
    assert "Star, register" not in src
    meta = (PLUGIN_ROOT / "metadata.yaml").read_text(encoding="utf-8")
    for field in ("name:", "desc:", "version:", "author:", "repo:"):
        assert field in meta, f"metadata.yaml 缺少必填字段 {field}"


# ---------------------------------------------------------------------------
# 5) 前后端路由闭环 / 页面模块 / 错误提示（WebUI 专章）
# ---------------------------------------------------------------------------

def test_route_table_and_frontend_endpoints_are_closed():
    html = _html()
    suffixes = {suffix for suffix, *_ in page_api.PluginPageApi.ROUTES}
    must = {
        "status", "memories", "memory/<atom_id>", "pin", "noise", "dlq",
        "providers", "bindings", "recall", "config", "config/save",
        "providers/test", "memories/edit", "memories/delete", "graph", "maintenance",
    }
    assert must <= suffixes, f"后端路由缺失: {must - suffixes}"
    assert page_api.PAGE_API_PREFIX.startswith("/astrbot_plugin_agent_memory/"), \
        "路由必须含插件名前缀（guides/plugin-pages.md「路由注册」）"
    for endpoint in (
        "page/status", "page/config", "page/config/save", "page/memories",
        "page/memory/", "page/pin", "page/noise", "page/dlq", "page/providers",
        "page/bindings", "page/recall", "page/graph", "page/maintenance",
        "page/providers/test", "page/memories/edit", "page/memories/delete",
    ):
        assert endpoint in html, f"前端未接线接口 {endpoint}"


def test_page_modules_buttons_and_error_feedback():
    html = _html()
    for tab in ("overview", "config", "providers", "memories", "recall",
                "graph", "noise", "maintenance"):
        assert f'data-tab="{tab}"' in html, f"缺少页面模块 {tab}"
    for btn in ("btn-save-config", "btn-reload-config", "btn-save-bindings",
                "btn-search", "btn-recall", "btn-graph-search", "btn-test-provider",
                "btn-mem-save", "btn-mem-delete", "btn-maint-flush",
                "btn-maint-fts", "btn-maint-dlq", "btn-maint-migrate"):
        assert f'id="{btn}"' in html, f"缺少操作按钮 {btn}"
    assert "toast(" in html and '"err"' in html, "缺少统一错误提示"
    assert "window.confirm" in html, "删除等破坏性操作须二次确认"
    assert "type=\"password\"" in html, "敏感配置项须密码遮罩（§6）"
    assert "function esc(" in html and "eval(" not in html
    for bad in ("http://", "https://cdn", "node_modules"):
        assert bad not in html, f"视图不应依赖外部资源 {bad}"
    assert "/api/plugin/page/bridge-sdk.js" in html


# ---------------------------------------------------------------------------
# 6) 前端语法 / 不写死路径（§10）
# ---------------------------------------------------------------------------

def test_frontend_javascript_syntax():
    if not shutil.which("node"):
        import pytest
        pytest.skip("node 不可用，跳过 JS 语法校验")
    html = _html()
    scripts = re.findall(
        r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S | re.I
    )
    assert scripts, "未找到内联脚本"
    with tempfile.TemporaryDirectory(prefix="tiermem_jscheck_") as tmp:
        for i, script in enumerate(scripts):
            f = Path(tmp) / f"inline_{i}.js"
            f.write_text(script, encoding="utf-8")
            proc = subprocess.run(
                ["node", "--check", str(f)], capture_output=True, text=True
            )
            assert proc.returncode == 0, f"JS 语法错误: {proc.stderr[:800]}"


def test_no_hardcoded_absolute_paths_in_runtime_sources():
    for rel in ("config_api.py", "page_api.py", "main.py"):
        src = (PLUGIN_ROOT / rel).read_text(encoding="utf-8")
        assert "/data/apps/" not in src, f"{rel} 写死绝对路径"
