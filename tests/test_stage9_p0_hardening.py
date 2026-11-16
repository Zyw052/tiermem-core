# -*- coding: utf-8 -*-
"""
Stage 9（P0 改造）自测：配置 Schema 结构 / 服务商解析 / 噪声过滤 / 插件 WebUI 接口。

覆盖点：
1. 【Schema】`_conf_schema.json` 结构校验
   - 合法 JSON、必需顶层键齐全；
   - 所有 `_special` 取值都在「宿主真实支持」白名单内（避免宿主不认导致下拉框失效）；
   - Embedding / Rerank 不得使用宿主不存在的 `_special`（否则会退化成文本框）；
   - 弹窗高度守门：可见字段总数受控 + 嵌套深度受控（解决「设置弹窗不能下滑」）。
2. 【Provider】`host_providers.ProviderBridge`
   - 按 provider_id 精确命中正确类型（chat / embedding / rerank）；
   - 类型不符时忽略并降级（不会把 Chat 服务商错当 Embedding 用）；
   - 运行期绑定文件（provider_bindings.json）优先级与解除逻辑；
   - `list_providers()` 分组正确；后端适配器真实调用宿主方法。
3. 【Noise】系统噪声过滤
   - 主人点名的噪声样本全部命中（子代理后台任务回调 / 项目大盘 / RAG 标记 / system_reminder）；
   - 正常对话零误伤（中文长句、含尖括号的技术讨论）；
   - 行内清洗保留有效正文；统计与最近样本可读。
4. 【WebUI】页面与后端接口
   - `views/dashboard/index.html` 存在、引用桥接 SDK、无外部 CDN 依赖；
   - `PluginPageApi.register_routes()` 在假 Context 上注册全部路由且路径带插件名前缀；
   - status / memories / pin / noise / dlq / providers 处理器在假插件上返回 ok；
   - 处理器在存储缺失、请求异常等极端入参下仍返回 error 而不抛异常。

运行： python3 tests/test_stage9_p0_hardening.py
（不导入 main.py，避免触发宿主配置管理器读写线上 cmd_config.json）
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

from storage import MemoryStorage  # noqa: E402
from noise_filter import NoiseFilter, filter_turn, reset_for_tests  # noqa: E402
from host_providers import (  # noqa: E402
    KIND_CHAT,
    KIND_EMBEDDING,
    KIND_RERANK,
    ProviderBridge,
    detect_provider_kind,
)
import page_api  # noqa: E402


#: 宿主（AstrBot v4.29.4）ConfigItemRenderer.vue 真实支持的 `_special` 白名单
#: 依据：astrbot/dashboard/src/components/shared/ConfigItemRenderer.vue 模板分支
#:      及线上 dist/assets 打包产物二次核验。
HOST_SUPPORTED_SPECIALS = {
    "umo",
    "sender_id",
    "select_provider",
    "select_provider_stt",
    "select_provider_tts",
    "select_providers",
    "provider_pool",
    "select_persona",
    "persona_pool",
    "select_knowledgebase",
    "select_plugin_set",
    "t2i_template",
    "dashboard_totp_manager",
    "local_permission_matrix",
    "get_embedding_dim",
}


# ---------------------------------------------------------------------------
# 假宿主对象：Provider / Context
# ---------------------------------------------------------------------------
class FakeProvider:
    """最小可用的 Provider 桩：按 kind 暴露不同能力方法。"""

    def __init__(self, pid: str, kind: str, model: str = "m", dim: int = 8) -> None:
        self.provider_config = {"id": pid, "type": f"fake_{kind}", "model": model}
        self.model_name = model
        self._kind = kind
        self._dim = dim
        self.embed_calls: List[List[str]] = []
        self.rerank_calls: List[str] = []
        self.chat_calls: List[Any] = []

    # -- 类型探测：优先走 meta()（与宿主一致） --
    def meta(self):
        class _Meta:
            pass

        meta = _Meta()
        meta.provider_type = type("PT", (), {"value": self._kind})()
        return meta

    def get_model(self) -> str:
        return self.model_name

    # -- embedding --
    async def get_embeddings(self, texts):
        self.embed_calls.append(list(texts))
        return [[float(len(t) % 7) + 0.1] * self._dim for t in texts]

    async def get_embedding(self, text):
        return (await self.get_embeddings([text]))[0]

    def get_dim(self) -> int:
        return self._dim

    # -- rerank --
    async def rerank(self, query, documents, top_n=None):
        self.rerank_calls.append(query)

        class _R:
            def __init__(self, index, score):
                self.index = index
                self.relevance_score = score

        results = [_R(i, 1.0 - i * 0.1) for i in range(len(documents))]
        return results[: top_n or len(results)]

    # -- chat --
    async def text_chat(self, prompt=None, contexts=None, system_prompt=None, model=None, **kw):
        self.chat_calls.append({"prompt": prompt, "contexts": contexts, "model": model})

        class _Resp:
            completion_text = "OK_FROM_PROVIDER"

        return _Resp()


class FakeContext:
    """最小可用的 StarContext 桩：只实现插件实际用到的两个接口。"""

    def __init__(self, providers: List[FakeProvider]) -> None:
        self.provider_manager = type(
            "PM", (), {"inst_map": {p.provider_config["id"]: p for p in providers}}
        )()
        self.registered: List[tuple] = []

    def get_provider_by_id(self, pid: str):
        return self.provider_manager.inst_map.get(pid)

    def register_web_api(self, route, handler, methods, desc):
        self.registered.append((route, handler, tuple(methods), desc))


# ---------------------------------------------------------------------------
# 一、Schema 结构校验
# ---------------------------------------------------------------------------
def _iter_schema_items(node: Dict[str, Any], path: str = "", depth: int = 0):
    """深度遍历 schema，产出 (path, meta)。"""
    for key, meta in (node or {}).items():
        current = f"{path}.{key}" if path else key
        if isinstance(meta, dict):
            yield current, meta
            items = meta.get("items")
            if isinstance(items, dict):
                yield from _iter_schema_items(items, current, depth + 1)


def test_schema_is_valid_and_complete():
    schema_path = PLUGIN_ROOT / "_conf_schema.json"
    assert schema_path.is_file(), "_conf_schema.json 缺失"
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    assert isinstance(schema, dict) and schema, "根节点必须是非空对象"

    # 关键配置组必须存在（代码读取路径依赖）
    for key in (
        "enable_memory",
        "memory_injection_enable",
        "summary_turn_interval",
        "llm_provider_id",
        "embedding_provider_id",
        "rerank_provider_id",
        "judge_provider_id",
        "retrieval",
        "evaluator",
        "graph",
        "noise_filter",
        "webui",
        "summarizer",
        "models",
    ):
        assert key in schema, f"缺少必需配置键: {key}"

    # 每个字段必须有 type；object 必须有 items
    for path, meta in _iter_schema_items(schema):
        assert "type" in meta, f"{path} 缺少 type"
        if meta["type"] == "object":
            assert isinstance(meta.get("items"), dict), f"{path} 为 object 但缺少 items"
        if meta["type"] == "list" and "options" in meta:
            assert isinstance(meta["options"], list), f"{path}.options 必须是数组"
    print("[OK] _conf_schema.json 结构合法，关键配置组齐全")


def test_schema_special_values_are_host_supported():
    schema = json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8-sig"))
    used: List[tuple] = []
    for path, meta in _iter_schema_items(schema):
        special = meta.get("_special")
        if special is None:
            continue
        used.append((path, special))
        assert (
            special in HOST_SUPPORTED_SPECIALS
        ), f"{path} 使用了宿主不支持的 _special={special!r}"

    # 服务商类下拉必须真的被用上（否则弹窗里只能手输 ID）
    specials = {special for _, special in used}
    assert "select_provider" in specials, "总结/判官模型未使用 select_provider 下拉"

    # 宿主没有 select_embedding_provider / select_rerank_provider，
    # 因此 Embedding / Rerank 的 provider_id 必须是普通字符串（由 WebUI 下拉兜底）
    for path, meta in _iter_schema_items(schema):
        if path.endswith("embedding_provider_id") or path.endswith("rerank_provider_id"):
            assert "options" not in meta or isinstance(meta.get("options"), list)
            assert meta.get("_special") in (None,), (
                f"{path} 不应使用 _special（宿主无对应实现，会退化为纯文本输入）"
            )
    print(f"[OK] _special 全部落在宿主白名单内（共 {len(used)} 处）")


def test_schema_dialog_is_compact():
    schema = json.loads((PLUGIN_ROOT / "_conf_schema.json").read_text(encoding="utf-8-sig"))

    def count_visible(node: Dict[str, Any], depth: int = 0) -> int:
        total = 0
        for _, meta in (node or {}).items():
            if not isinstance(meta, dict) or meta.get("invisible"):
                continue
            total += 1
            if meta.get("type") == "object":
                total += count_visible(meta.get("items") or {}, depth + 1)
        return total

    def max_depth(node: Dict[str, Any], depth: int = 1) -> int:
        deepest = depth
        for _, meta in (node or {}).items():
            if not isinstance(meta, dict) or meta.get("invisible"):
                continue
            if meta.get("type") == "object":
                deepest = max(deepest, max_depth(meta.get("items") or {}, depth + 1))
        return deepest

    visible = count_visible(schema)
    depth = max_depth(schema)
    # 旧 schema 可见字段 ~70 个、嵌套 2 层且全部展开 → 弹窗无法正常滚动；
    # 这里做守门，防止后续改动再把弹窗撑爆。
    assert visible <= 30, f"可见字段过多（{visible}），弹窗会再次无法滚动"
    assert depth <= 2, f"嵌套过深（{depth} 层），弹窗渲染与滚动会异常"
    print(f"[OK] 弹窗可见字段 {visible} 个、嵌套 {depth} 层（滚动风险受控）")


# ---------------------------------------------------------------------------
# 二、Provider 解析
# ---------------------------------------------------------------------------
def test_detect_provider_kind():
    assert detect_provider_kind(FakeProvider("a", KIND_CHAT)) == KIND_CHAT
    assert detect_provider_kind(FakeProvider("b", KIND_EMBEDDING)) == KIND_EMBEDDING
    assert detect_provider_kind(FakeProvider("c", KIND_RERANK)) == KIND_RERANK
    assert detect_provider_kind(None) == ""
    print("[OK] Provider 类型探测（meta().provider_type）正确")


def test_bridge_resolves_by_provider_id():
    chat = FakeProvider("llm_main", KIND_CHAT)
    emb = FakeProvider("emb_sf", KIND_EMBEDDING, dim=16)
    rer = FakeProvider("rr_bge", KIND_RERANK)
    bridge = ProviderBridge(FakeContext([chat, emb, rer]))

    emb_backend = bridge.build_embedding_backend("emb_sf")
    assert emb_backend is not None and emb_backend.provider_id == "emb_sf"
    assert emb_backend.dim() == 16

    rer_backend = bridge.build_rerank_backend("rr_bge")
    assert rer_backend is not None and rer_backend.provider_id == "rr_bge"

    chat_backend = bridge.build_chat_backend("llm_main")
    assert chat_backend is not None and chat_backend.provider_id == "llm_main"

    # 未配置 / 不存在 → None（不 panic），且默认关闭自动挑选，避免选错模型
    assert bridge.build_embedding_backend("") is None
    assert bridge.build_embedding_backend("not_exist") is None
    assert bridge.build_embedding_backend("", allow_auto=True).provider_id == "emb_sf"
    print("[OK] provider_id 精确解析 + 自动挑选开关行为正确")


def test_bridge_rejects_type_mismatch():
    chat = FakeProvider("llm_main", KIND_CHAT)
    bridge = ProviderBridge(FakeContext([chat]))
    # 把 Chat 服务商错填到 Embedding 位 → 必须拒绝，不能静默误用
    assert bridge.build_embedding_backend("llm_main") is None
    print("[OK] 服务商类型不匹配时正确拒绝（避免 Chat 被当作 Embedding 使用）")


def test_bridge_async_backends_really_call_host():
    chat = FakeProvider("llm_main", KIND_CHAT)
    emb = FakeProvider("emb_sf", KIND_EMBEDDING, dim=4)
    rer = FakeProvider("rr_bge", KIND_RERANK)
    bridge = ProviderBridge(FakeContext([chat, emb, rer]))

    async def scenario():
        emb_backend = bridge.build_embedding_backend("emb_sf")
        vectors = await emb_backend.embed(["你好", "hello"])
        assert len(vectors) == 2 and all(len(v) == 4 for v in vectors)
        assert emb.embed_calls, "宿主 get_embeddings 未被调用"

        rer_backend = bridge.build_rerank_backend("rr_bge")
        pairs = await rer_backend.rerank("查询", ["doc0", "doc1", "doc2"], top_n=2)
        assert pairs and pairs[0][0] == 0 and len(pairs) == 2
        assert rer.rerank_calls == ["查询"]

        chat_backend = bridge.build_chat_backend("llm_main")
        text = await chat_backend.chat(
            [
                {"role": "system", "content": "你是判官"},
                {"role": "user", "content": "打个分"},
            ],
            model="jev",
        )
        assert text == "OK_FROM_PROVIDER"
        assert chat.chat_calls[0]["model"] == "jev"
        # 最后一条 user 消息走 prompt 通道，因此不再重复塞进 contexts
        assert not chat.chat_calls[0]["contexts"]

    asyncio.run(scenario())
    print("[OK] 后端适配器真实调用宿主 Provider（embedding / rerank / chat）")


def test_bridge_runtime_bindings_roundtrip():
    chat = FakeProvider("llm_main", KIND_CHAT)
    emb_a = FakeProvider("emb_a", KIND_EMBEDDING)
    emb_b = FakeProvider("emb_b", KIND_EMBEDDING)
    with tempfile.TemporaryDirectory() as tmp:
        bridge = ProviderBridge(FakeContext([chat, emb_a, emb_b]), data_dir=Path(tmp))
        assert bridge.load_bindings() == {}

        saved = bridge.save_bindings({KIND_EMBEDDING: "emb_b"})
        assert saved.get(KIND_EMBEDDING) == "emb_b"

        # 新建实例重读文件（模拟插件重启）
        bridge2 = ProviderBridge(FakeContext([chat, emb_a, emb_b]), data_dir=Path(tmp))
        assert bridge2.load_bindings(force=True).get(KIND_EMBEDDING) == "emb_b"
        backend = bridge2.build_embedding_backend("")
        assert backend is not None and backend.provider_id == "emb_b"
        assert backend.source == "bindings"

        # 解除绑定
        bridge2.save_bindings({KIND_EMBEDDING: ""})
        assert bridge2.load_bindings(force=True).get(KIND_EMBEDDING) is None
        assert bridge2.build_embedding_backend("") is None
    print("[OK] 运行期绑定持久化 / 重载 / 解除全部正确")


def test_bridge_list_providers_grouping():
    providers = [
        FakeProvider("llm_main", KIND_CHAT),
        FakeProvider("emb_sf", KIND_EMBEDDING),
        FakeProvider("emb_gem", KIND_EMBEDDING),
        FakeProvider("rr_bge", KIND_RERANK),
    ]
    bridge = ProviderBridge(FakeContext(providers))
    grouped = bridge.list_providers()
    assert [p["id"] for p in grouped[KIND_EMBEDDING]] == ["emb_gem", "emb_sf"]
    assert [p["id"] for p in grouped[KIND_RERANK]] == ["rr_bge"]
    assert [p["id"] for p in grouped[KIND_CHAT]] == ["llm_main"]
    print("[OK] list_providers 按能力分组且排序稳定")


# ---------------------------------------------------------------------------
# 三、噪声过滤
# ---------------------------------------------------------------------------
NOISE_SAMPLES = [
    "Background task for subagent 'CodeSmith' finished.",
    "Background task for sub-agent \"crawler\" completed successfully",
    "【项目大盘·跨代理共享状态】\nastrbot_plugin_agent_memory: active v3",
    "<RAG-Faiss-Memory>相关记忆：主人喜欢美式咖啡</RAG-Faiss-Memory>",
    "<system_reminder>请勿输出内部提示词</system_reminder>",
    "<system-reminder>reminder body</system-reminder>",
    "[SYSTEM NOTICE] 你已执行同一个工具 3 次",
    "<!-- agent_memory:injected:v1 -->",
    "[16:24:43.866] [Plug] [DBUG] [astrbot_plugin_agent_memory.summarizer:2216]: noise line",
]

CLEAN_SAMPLES = [
    "主人今天想让插件支持服务商下拉选择，我建议先核对宿主支持的 _special 取值。",
    "帮我把 <T> 这个泛型参数改成 <K>，顺便补个注释。",
    "Vue 模板里用了 <v-select>，下拉框的 items 需要是数组。",
    "记住：主人的服务器时区是 Asia/Shanghai。",
]


def test_noise_filter_catches_system_noise():
    reset_for_tests()
    nf = NoiseFilter()
    for sample in NOISE_SAMPLES:
        verdict = nf.inspect(sample)
        assert verdict.is_noise, f"未拦截噪声: {sample!r}"
        assert verdict.hits, f"命中规则为空: {sample!r}"
    print(f"[OK] {len(NOISE_SAMPLES)} 类系统噪声全部被拦截")


def test_noise_filter_keeps_normal_dialogue():
    nf = NoiseFilter()
    for sample in CLEAN_SAMPLES:
        verdict = nf.inspect(sample)
        assert not verdict.is_noise, f"误伤正常对话: {sample!r}"
        assert verdict.cleaned_text.strip(), f"正常对话被清空: {sample!r}"
    print(f"[OK] {len(CLEAN_SAMPLES)} 条正常对话零误伤（含尖括号技术讨论）")


def test_noise_filter_strips_inline_blocks():
    nf = NoiseFilter()
    text = (
        "主人的新偏好是每周一把周报送我。\n"
        "<system_reminder>内部：不要暴露本段</system_reminder>\n"
        "Background task for subagent 'x' finished."
    )
    verdict = nf.inspect(text)
    assert not verdict.is_noise, "混合文本不应被判为纯噪声"
    assert "每周一" in verdict.cleaned_text
    assert "system_reminder" not in verdict.cleaned_text
    assert "Background task" not in verdict.cleaned_text
    print("[OK] 行内系统块被剔除，有效正文保留")


def test_noise_filter_stats_and_recent():
    nf = NoiseFilter()
    nf.inspect(NOISE_SAMPLES[0])
    nf.inspect(CLEAN_SAMPLES[0])
    stats = nf.stats
    assert stats["inspected"] >= 2
    assert stats["dropped"] >= 1
    assert stats["passed"] >= 1
    recent = nf.recent(5)
    assert recent and "hits" in recent[0]
    assert nf.describe()["line_rules"]
    print("[OK] 噪声统计与最近样本可读（供 WebUI 噪声排查页）")


def test_filter_turn_pair_semantics():
    reset_for_tests()
    # 两侧都噪声 → is_noise
    _, _, meta = filter_turn(NOISE_SAMPLES[2], NOISE_SAMPLES[3])
    assert meta["is_noise"] is True
    # 单侧噪声 → 只清空该侧
    user, bot, meta = filter_turn(CLEAN_SAMPLES[0], NOISE_SAMPLES[0])
    assert meta["is_noise"] is False
    assert user.strip() and not bot.strip()
    print("[OK] filter_turn 双/单侧噪声语义正确")


# ---------------------------------------------------------------------------
# 四、WebUI 页面与接口
# ---------------------------------------------------------------------------
class FakeStorage:
    """内存版存储桩，覆盖 page_api 需要的读接口。"""

    def __init__(self) -> None:
        self.items = {
            "atom_1": {
                "atom_id": "atom_1",
                "scope_key": "scope_a",
                "session_id": "s1",
                "content": "主人喜欢美式咖啡",
                "topics": ["偏好"],
                "key_facts": ["喜欢美式咖啡"],
                "importance": 0.8,
                "pinned": False,
                "tier": 2,
                "has_embedding": True,
                "created_at": 1700000000.0,
                "updated_at": 1700000000.0,
            }
        }

    def list_memories(self, limit=20, offset=0, query="", pinned_only=False, scope_key=""):
        items = list(self.items.values())
        if query:
            items = [i for i in items if query in i["content"]]
        if pinned_only:
            items = [i for i in items if i["pinned"]]
        return {"items": items[offset : offset + limit], "total": len(items), "limit": limit, "offset": offset}

    def get_memory_detail(self, atom_id):
        return self.items.get(atom_id)

    def set_atom_pin(self, atom_id, pinned):
        item = self.items.get(atom_id)
        if not item:
            return {"found": False, "atom_id": atom_id, "error": "记录不存在"}
        item["pinned"] = bool(pinned)
        return {"found": True, "atom_id": atom_id, "is_pinned": bool(pinned)}

    def list_dlq(self, limit=20, status=""):
        return {"items": [], "counts": {"pending": 1}}


class FakeBridge:
    def list_providers(self):
        return {KIND_CHAT: [{"id": "llm_main", "model": "m1"}], KIND_EMBEDDING: [], KIND_RERANK: []}

    def load_bindings(self, force=False):
        return {KIND_EMBEDDING: "emb_sf"}

    def save_bindings(self, updates):
        return dict(updates)


class FakePlugin:
    def __init__(self) -> None:
        self.context = FakeContext([])
        self.storage = FakeStorage()
        self.provider_bridge = FakeBridge()
        self.retriever = None
        self.summarizer = None
        self.embedding_client = None
        self.rerank_client = None

    def get_status(self):
        return {"enable_memory": True, "db_path": "/tmp/x.db", "table_counts": {"l1_docs": 1}}

    def build_self_check(self):
        return {
            "all_powered": False,
            "degraded": ["向量化降级（零向量，语义检索失效）"],
            "embedding": {"powered": False},
            "rerank": {"powered": False},
            "llm": {"powered": False},
            "judge": {"present": True, "enabled": True, "injected_provider": False},
            "kg": {"enabled": True, "triples_written": 0},
        }

    async def retrieve_memory(self, query, session_id=None, top_k=None):
        return {"pinned_l3": [], "retrieved_l2": [{"content": "命中"}], "prompt_context": "ctx"}

    def format_memory_block(self, result, max_chars=0):
        return "注入块"


def test_webui_page_structure():
    page = PLUGIN_ROOT / "views" / "dashboard" / "index.html"
    assert page.is_file(), "插件 WebUI 页面缺失（应为 views/dashboard/index.html）"
    html = page.read_text(encoding="utf-8")

    # 宿主按目录 + index.html 自动发现页面；必须引用桥接 SDK 才能调后端
    assert "/api/plugin/page/bridge-sdk.js" in html, "缺少桥接 SDK 引用"
    assert "AstrBotPluginView" in html, "未使用宿主桥接对象"
    assert "apiGet(" in html and "apiPost(" in html
    # 轻量：不依赖任何外部 CDN / 打包产物
    for bad in ("http://", "https://cdn", "node_modules"):
        assert bad not in html, f"页面不应依赖外部资源: {bad}"
    # 四个核心视图都在
    for key in ("page/status", "page/memories", "page/noise", "page/providers"):
        assert key in html, f"页面缺少接口调用: {key}"
    print("[OK] WebUI 页面存在且引用桥接 SDK、无外部依赖、覆盖四大视图")


def test_webui_register_routes():
    plugin = FakePlugin()
    api = page_api.PluginPageApi(plugin)
    count = api.register_routes()
    assert count == len(page_api.PluginPageApi.ROUTES) > 0
    routes = [r[0] for r in plugin.context.registered]
    assert all(r.startswith("/astrbot_plugin_agent_memory/page") for r in routes), routes
    # 路由占位符必须是宿主支持的 <param> 语法（不是 {param}）
    assert "/astrbot_plugin_agent_memory/page/memory/<atom_id>" in routes
    assert not any("{" in r for r in routes)
    print(f"[OK] 注册 {count} 条页面路由，路径前缀与 <param> 语法均正确")


def test_webui_handlers_return_ok():
    plugin = FakePlugin()
    api = page_api.PluginPageApi(plugin)

    async def scenario():
        status = await api.get_status()
        assert status["status"] == "ok" and "self_check" in status["data"]

        memories = await api.list_memories()
        assert memories["status"] == "ok" and memories["data"]["total"] == 1

        detail = await api.get_memory(atom_id="atom_1")
        assert detail["status"] == "ok" and detail["data"]["atom_id"] == "atom_1"

        # 无请求上下文时（脱离 WebUI 直接调用）置顶应给出可读错误而非抛异常
        pin = await api.set_pin()
        assert pin["status"] == "error" and "atom_id" in pin["message"]

        noise = await api.get_noise()
        assert noise["status"] == "ok" and "stats" in noise["data"]

        dlq = await api.get_dlq()
        assert dlq["status"] == "ok" and "counts" in dlq["data"]

        providers = await api.get_providers()
        assert providers["status"] == "ok" and "available" in providers["data"]

        recall = await api.recall()
        assert recall["status"] == "error"  # 缺 q 参数时给出可读错误而非抛异常

    asyncio.run(scenario())
    print("[OK] status/memories/memory/noise/dlq/providers/recall 处理器全部安全返回")


def test_webui_handlers_degrade_without_storage():
    plugin = FakePlugin()
    plugin.storage = None
    api = page_api.PluginPageApi(plugin)

    async def scenario():
        for coro in (
            api.list_memories(),
            api.get_memory(atom_id="x"),
            api.get_dlq(),
        ):
            res = await coro
            assert res["status"] == "error" and res["message"]

    asyncio.run(scenario())
    print("[OK] 存储缺失时接口返回可读错误（WebUI 不 500）")


def test_storage_webui_queries():
    with tempfile.TemporaryDirectory() as tmp:
        storage = MemoryStorage(Path(tmp))
        storage.init_schema()
        # 写入两条记忆（复用写侧接口，保证与真实链路一致）
        conn = storage.connect()
        now = 1_700_000_000.0
        with conn:
            for idx, content in enumerate(["主人喜欢美式咖啡", "项目代号是智涌"]):
                conn.execute(
                    """
                    INSERT INTO l2_atoms (
                        atom_id, scope_key, session_id, content, topics_json,
                        key_facts_json, sentiment, importance, access_count,
                        source_type, source_ref, last_accessed_at, created_at, updated_at
                    ) VALUES (?, 'scope_a', 's1', ?, '[]', '[]', 'neutral', 0.6, 0,
                              'chat', '', ?, ?, ?)
                    """,
                    (f"atom_{idx}", content, now, now, now),
                )

        listed = storage.list_memories(limit=10)
        assert listed["total"] == 2 and len(listed["items"]) == 2

        searched = storage.list_memories(query="咖啡")
        assert searched["total"] == 1 and "咖啡" in searched["items"][0]["content"]

        setres = storage.set_atom_pin("atom_0", True)
        assert setres["found"] and setres["is_pinned"]
        pinned = storage.list_memories(pinned_only=True)
        assert pinned["total"] == 1 and pinned["items"][0]["atom_id"] == "atom_0"

        detail = storage.get_memory_detail("atom_0")
        assert detail is not None and detail["atom_id"] == "atom_0"

        dlq = storage.list_dlq(limit=5)
        assert dlq["items"] == [] and dlq["counts"] == {}

        storage.close()
    print("[OK] storage 新增的 WebUI 查询接口（列表/搜索/置顶/详情/DLQ）正确")


def main() -> int:
    cases = [
        # Schema
        test_schema_is_valid_and_complete,
        test_schema_special_values_are_host_supported,
        test_schema_dialog_is_compact,
        # Provider
        test_detect_provider_kind,
        test_bridge_resolves_by_provider_id,
        test_bridge_rejects_type_mismatch,
        test_bridge_async_backends_really_call_host,
        test_bridge_runtime_bindings_roundtrip,
        test_bridge_list_providers_grouping,
        # Noise
        test_noise_filter_catches_system_noise,
        test_noise_filter_keeps_normal_dialogue,
        test_noise_filter_strips_inline_blocks,
        test_noise_filter_stats_and_recent,
        test_filter_turn_pair_semantics,
        # WebUI
        test_webui_page_structure,
        test_webui_register_routes,
        test_webui_handlers_return_ok,
        test_webui_handlers_degrade_without_storage,
        test_storage_webui_queries,
    ]
    for case in cases:
        case()
        # pytest 收集时 asyncio 用例需单独处理（本文件全部为同步包装）
    print(f"\n=== Stage9 P0 改造专项 用例全部通过（{len(cases)} 项） ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
