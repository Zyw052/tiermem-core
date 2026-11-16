# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 插件 WebUI 后端 API (page_api.py)

规范依据（astrbot-plugin-dev，产出注明依据章节）：
  §3 main.py —— 插件初始化阶段用 context.register_web_api 注册页面 API
  §6 配置 —— _conf_schema.json / AstrBotConfig.save_config
  §9 存储 —— 持久化数据落 data/plugin_data/<plugin>/
  §10 工程规范 —— from astrbot.api import logger、异步网络库、ruff、异常处理
  WebUI 专章 guides/plugin-pages.md —— 「后端 Web API / 路由注册」「请求对象」
    「响应对象 / Quart 兼容」「Bridge API / 请求和返回值规则」「安全约束」

宿主源码实证（v4.29.4，非猜测）：
    - ``Context.register_web_api(route, view_handler, methods, desc)``
      → astrbot/core/star/context.py:774
    - 路由匹配语法为 Quart 风格 ``<param>`` / ``<path:param>``
      → astrbot/dashboard/api/plugins.py:198 ``_plugin_api_route_pattern``
    - 前端经 ``/api/v1/plugins/extensions/{plugin}/{endpoint}`` 调用，
      页面内使用 ``window.AstrBotPluginView.apiGet/apiPost``
      → astrbot/dashboard/plugin_page_bridge.js

因此本模块注册的每个路由都必须带 ``/{PLUGIN_NAME}/`` 前缀，
否则后端 ``_match_registered_web_api`` 无法命中（插件名是路径第一段）。

设计原则：
    1. 所有处理器 **不抛异常**，失败统一返回 ``{"status": "error", ...}``；
    2. 只读为主，唯一写操作是「置顶」与「运行期服务商绑定」（写插件数据目录）；
    3. 返回体保持 ``{"status": "ok", "data": {...}}`` 结构，与宿主插件页约定一致。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

try:  # 宿主日志
    from astrbot.api import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("astrbot_plugin_agent_memory.page_api")

PLUGIN_NAME = "astrbot_plugin_agent_memory"
PAGE_API_PREFIX = f"/{PLUGIN_NAME}/page"


# ---------------------------------------------------------------------------
# 请求/响应辅助（兼容宿主 PluginRequestProxy 与 Quart 两种取参方式）
# ---------------------------------------------------------------------------
def _get_request():
    """获取当前插件 Web 请求对象；不可用时返回 None。"""
    try:
        from astrbot.api.web import request  # type: ignore

        return request
    except Exception:
        pass
    try:  # 兼容性兜底（宿主内部走 Quart 兼容层）
        from quart import request as quart_request  # type: ignore

        return quart_request
    except Exception:
        return None


def _query_one(name: str, default: str = "") -> str:
    """安全读取单个 query 参数。"""
    req = _get_request()
    if req is None:
        return default
    try:
        query = getattr(req, "query", None)
        if query is None:
            args = getattr(req, "args", None)
            value = args.get(name) if args is not None else None
        else:
            value = query.get(name)
        if value is None:
            return default
        return str(value)
    except Exception:
        return default


def _query_int(name: str, default: int, minimum: int = 0, maximum: int = 10_000) -> int:
    """安全读取整型 query 参数并夹逼到合法区间。"""
    raw = _query_one(name, "")
    if not raw.strip():
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(value, maximum))


def _query_bool(name: str, default: bool = False) -> bool:
    raw = _query_one(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on", "y", "t")


async def _json_body() -> Dict[str, Any]:
    """安全读取 JSON 请求体，失败返回空 dict。"""
    req = _get_request()
    if req is None:
        return {}
    try:
        payload = await req.json(default={})
    except TypeError:
        try:
            payload = await req.json()
        except Exception:
            payload = {}
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def ok(data: Any = None) -> Dict[str, Any]:
    return {"status": "ok", "data": data if data is not None else {}}


def err(message: str, data: Any = None) -> Dict[str, Any]:
    return {"status": "error", "message": str(message), "data": data or {}}


# ---------------------------------------------------------------------------
# API 门面
# ---------------------------------------------------------------------------
class PluginPageApi:
    """把插件运行期组件暴露为 WebUI 接口（全部幂等、只读优先）。"""

    #: (路径后缀, 处理方法, HTTP 方法列表, 描述)
    ROUTES = (
        ("status", "get_status", ["GET"], "记忆引擎通电自检与分层统计"),
        ("memories", "list_memories", ["GET"], "分页浏览 / 关键词搜索记忆"),
        ("memory/<atom_id>", "get_memory", ["GET"], "单条记忆详情与关联三元组"),
        ("pin", "set_pin", ["POST"], "设置 / 取消记忆置顶（L3 晋升）"),
        ("noise", "get_noise", ["GET"], "系统噪声过滤统计与最近拦截样本"),
        ("dlq", "get_dlq", ["GET"], "死信队列条目（失败任务排查）"),
        ("providers", "get_providers", ["GET"], "宿主服务商清单与当前绑定"),
        ("bindings", "save_bindings", ["POST"], "保存运行期服务商绑定并热重建"),
        ("recall", "recall", ["GET"], "手动召回测试（观察检索与注入结果）"),
        # ---- 本次补全：配置在线查看/保存、连通性测试、记忆写操作、图谱、维护 ----
        ("config", "get_config", ["GET"], "在线查看插件配置（敏感字段掩码）"),
        ("config/save", "save_config", ["POST"], "在线保存配置并热更新（不重载主进程）"),
        ("providers/test", "test_provider", ["POST"], "服务商连通性测试（不返回密钥）"),
        ("memories/edit", "update_memory", ["POST"], "编辑记忆正文/重要度/关键事实/主题"),
        ("memories/delete", "delete_memory", ["POST"], "删除记忆及关联图谱边"),
        ("graph", "get_graph", ["GET"], "知识图谱节点与关系浏览"),
        ("maintenance", "run_maintenance", ["POST"], "维护：flush/rebuild_fts/retry_dlq/migrate_preview"),
    )

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin

    # -- 注册 --------------------------------------------------------------
    def register_routes(self) -> int:
        """向宿主注册全部页面 API，返回成功注册的路由数。"""
        registered = 0
        context = getattr(self.plugin, "context", None)
        register = getattr(context, "register_web_api", None)
        if not callable(register):
            logger.warning(
                "[agent_memory] 宿主 Context 缺少 register_web_api，插件 WebUI API 未注册。"
            )
            return 0

        for suffix, handler_name, methods, description in self.ROUTES:
            handler = getattr(self, handler_name, None)
            if not callable(handler):
                logger.warning("[agent_memory] 页面 API 处理器缺失: %s", handler_name)
                continue
            try:
                register(f"{PAGE_API_PREFIX}/{suffix}", handler, methods, description)
                registered += 1
            except Exception as exc:
                logger.warning(
                    "[agent_memory] 注册页面 API 失败 %s: %s", suffix, exc
                )
        logger.info("[agent_memory] 插件 WebUI API 已注册 %d 条路由。", registered)
        return registered

    # -- GET /status -------------------------------------------------------
    async def get_status(self, **_: Any) -> Dict[str, Any]:
        """通电自检 + 分层统计 + 引擎可观测性快照（WebUI 首屏）。"""
        try:
            self_check = {}
            builder = getattr(self.plugin, "build_self_check", None)
            if callable(builder):
                self_check = builder()
            status_fn = getattr(self.plugin, "get_status", None)
            status = status_fn() if callable(status_fn) else {}
            return ok({"self_check": self_check, "status": status})
        except Exception as exc:
            logger.error("[agent_memory] page.status 失败: %s", exc, exc_info=True)
            return err(f"获取状态失败: {exc}")

    # -- GET /memories -----------------------------------------------------
    async def list_memories(self, **_: Any) -> Dict[str, Any]:
        """分页浏览/搜索记忆；支持 q / limit / offset / pinned / scope。"""
        try:
            storage = getattr(self.plugin, "storage", None)
            if storage is None:
                return err("存储未就绪")
            lister = getattr(storage, "list_memories", None)
            if not callable(lister):
                return err("存储缺少 list_memories 接口")

            payload = lister(
                limit=_query_int("limit", 20, 1, 200),
                offset=_query_int("offset", 0, 0, 1_000_000),
                query=_query_one("q", ""),
                pinned_only=_query_bool("pinned", False),
                scope_key=_query_one("scope", ""),
            )
            return ok(payload)
        except Exception as exc:
            logger.error("[agent_memory] page.memories 失败: %s", exc, exc_info=True)
            return err(f"查询记忆失败: {exc}")

    # -- GET /memory/<atom_id> --------------------------------------------
    async def get_memory(self, atom_id: str = "", **_: Any) -> Dict[str, Any]:
        """单条记忆详情（含图谱三元组）。"""
        try:
            storage = getattr(self.plugin, "storage", None)
            if storage is None:
                return err("存储未就绪")
            getter = getattr(storage, "get_memory_detail", None)
            if not callable(getter):
                return err("存储缺少 get_memory_detail 接口")
            detail = getter(atom_id)
            if not detail:
                return err("记忆不存在", {"atom_id": atom_id})
            return ok(detail)
        except Exception as exc:
            logger.error("[agent_memory] page.memory 失败: %s", exc, exc_info=True)
            return err(f"读取记忆详情失败: {exc}")

    # -- POST /pin ---------------------------------------------------------
    async def set_pin(self, **_: Any) -> Dict[str, Any]:
        """设置 / 取消置顶；请求体 {"atom_id": "...", "pinned": true}。"""
        try:
            body = await _json_body()
            atom_id = str(body.get("atom_id") or "").strip()
            if not atom_id:
                return err("缺少 atom_id")
            pinned = bool(body.get("pinned", True))

            storage = getattr(self.plugin, "storage", None)
            if storage is None:
                return err("存储未就绪")
            setter = getattr(storage, "set_atom_pin", None)
            if callable(setter):
                result = setter(atom_id, pinned)
            else:  # 兼容回退：老接口为切换语义
                toggler = getattr(storage, "toggle_atom_pin", None)
                result = toggler(atom_id) if callable(toggler) else {"error": "无可用接口"}

            if result.get("error") and not result.get("found"):
                return err(str(result.get("error")), result)

            # 置顶变化会影响检索加分，清掉检索侧缓存（失败忽略）
            retriever = getattr(self.plugin, "retriever", None)
            invalidator = getattr(retriever, "invalidate_cache", None)
            if callable(invalidator):
                try:
                    invalidator()
                except Exception:
                    pass
            return ok(result)
        except Exception as exc:
            logger.error("[agent_memory] page.pin 失败: %s", exc, exc_info=True)
            return err(f"置顶操作失败: {exc}")

    # -- GET /noise --------------------------------------------------------
    async def get_noise(self, **_: Any) -> Dict[str, Any]:
        """噪声过滤统计 + 最近拦截样本（噪声排查）。"""
        try:
            from .noise_filter import get_noise_filter
        except ImportError:  # pragma: no cover
            try:
                from noise_filter import get_noise_filter  # type: ignore
            except ImportError:
                return err("噪声过滤器不可用")
        try:
            nf = get_noise_filter()
            limit = _query_int("limit", 20, 1, 50)
            return ok(
                {
                    "stats": dict(getattr(nf, "stats", {}) or {}),
                    "recent": nf.recent(limit),
                    "line_rules": [
                        name for name, _ in getattr(nf, "_line_patterns", []) or []
                    ],
                }
            )
        except Exception as exc:
            logger.error("[agent_memory] page.noise 失败: %s", exc, exc_info=True)
            return err(f"读取噪声统计失败: {exc}")

    # -- GET /dlq ----------------------------------------------------------
    async def get_dlq(self, **_: Any) -> Dict[str, Any]:
        """死信队列条目与各状态计数。"""
        try:
            storage = getattr(self.plugin, "storage", None)
            if storage is None:
                return err("存储未就绪")
            lister = getattr(storage, "list_dlq", None)
            if not callable(lister):
                return err("存储缺少 list_dlq 接口")
            payload = lister(
                limit=_query_int("limit", 20, 1, 200),
                status=_query_one("status", ""),
            )
            # 附带 summarizer 内部 DLQ 统计（重试闭环计数）
            summarizer = getattr(self.plugin, "summarizer", None)
            getter = getattr(summarizer, "get_dlq_stats", None)
            if callable(getter):
                try:
                    payload["runtime"] = getter()
                except Exception:
                    pass
            return ok(payload)
        except Exception as exc:
            logger.error("[agent_memory] page.dlq 失败: %s", exc, exc_info=True)
            return err(f"读取死信队列失败: {exc}")

    # -- GET /providers ----------------------------------------------------
    async def get_providers(self, **_: Any) -> Dict[str, Any]:
        """宿主服务商清单（按能力分组）+ 当前生效配置与绑定。"""
        try:
            bridge = getattr(self.plugin, "provider_bridge", None)
            if bridge is None:
                return err("服务商桥接未初始化")
            return ok(
                {
                    "available": bridge.list_providers(),
                    "bindings": bridge.load_bindings(force=True),
                    "effective": _effective_provider_snapshot(self.plugin),
                }
            )
        except Exception as exc:
            logger.error("[agent_memory] page.providers 失败: %s", exc, exc_info=True)
            return err(f"读取服务商失败: {exc}")

    # -- POST /bindings ----------------------------------------------------
    async def save_bindings(self, **_: Any) -> Dict[str, Any]:
        """
        保存运行期服务商绑定并热重建客户端。

        请求体：{"embedding": "provider_id", "rerank": "provider_id", "chat": "provider_id"}
        空字符串表示解除绑定（回退到插件配置里的 provider_id / 手动凭证）。
        """
        try:
            body = await _json_body()
            bridge = getattr(self.plugin, "provider_bridge", None)
            if bridge is None:
                return err("服务商桥接未初始化")

            updates: Dict[str, str] = {}
            for key in ("embedding", "rerank", "chat_completion"):
                if key in body:
                    updates[key] = str(body.get(key) or "").strip()

            if not updates:
                return err("请求体为空（支持 embedding / rerank / chat_completion）")

            saved = bridge.save_bindings(updates)

            # 热重建：让新绑定立即生效（避免重载整个插件）
            refresher = getattr(self.plugin, "refresh_provider_bindings", None)
            rebuilt: Dict[str, Any] = {}
            if callable(refresher):
                try:
                    rebuilt = await refresher() or {}
                except Exception as exc:
                    logger.warning("[agent_memory] 热重建失败（绑定已保存）: %s", exc)
                    rebuilt = {"error": str(exc)}

            return ok({"bindings": saved, "reload": rebuilt})
        except Exception as exc:
            logger.error("[agent_memory] page.bindings 失败: %s", exc, exc_info=True)
            return err(f"保存绑定失败: {exc}")

    # -- GET /recall -------------------------------------------------------
    async def recall(self, **_: Any) -> Dict[str, Any]:
        """手动召回测试：返回置顶/精排结果与最终注入块（便于观察降级）。"""
        try:
            query = _query_one("q", "").strip()
            if not query:
                return err("缺少查询参数 q")
            retrieve = getattr(self.plugin, "retrieve_memory", None)
            if not callable(retrieve):
                return err("检索器未就绪")
            started = time.time()
            result = await retrieve(
                query=query,
                session_id=_query_one("session", "") or None,
                top_k=_query_int("top_k", 6, 1, 50),
            )
            block = ""
            formatter = getattr(self.plugin, "format_memory_block", None)
            if callable(formatter):
                try:
                    block = formatter(result, 2000)
                except Exception:
                    block = ""
            return ok(
                {
                    "query": query,
                    "elapsed_ms": int((time.time() - started) * 1000),
                    "pinned_l3": result.get("pinned_l3", []) if isinstance(result, dict) else [],
                    "retrieved_l2": result.get("retrieved_l2", []) if isinstance(result, dict) else [],
                    "prompt_block": block,
                }
            )
        except Exception as exc:
            logger.error("[agent_memory] page.recall 失败: %s", exc, exc_info=True)
            return err(f"召回测试失败: {exc}")

    # ==================================================================
    # 本次补全：配置 / 连通性 / 记忆写操作 / 图谱 / 维护
    # 约定：全部处理器不抛异常，失败统一返回 {"status": "error", ...}
    # ==================================================================
    def _invalidate_cache(self) -> None:
        """写操作后清理检索侧缓存（失败忽略，不影响主链路）。"""
        retriever = getattr(self.plugin, "retriever", None)
        fn = getattr(retriever, "invalidate_cache", None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # pragma: no cover
                logger.debug("[agent_memory] 清理检索缓存失败（忽略）: %s", exc)

    @staticmethod
    def _load_config_manager():
        """兼容包内相对导入与扁平导入两种加载方式。"""
        try:
            from .config_api import ConfigError, ConfigManager  # type: ignore

            return ConfigManager, ConfigError
        except ImportError:  # pragma: no cover
            from config_api import ConfigError, ConfigManager  # type: ignore

            return ConfigManager, ConfigError

    # -- GET /config -------------------------------------------------------
    async def get_config(self, **_: Any) -> Dict[str, Any]:
        """在线查看插件配置；api_key / token / secret 一律掩码，绝不返回明文。"""
        try:
            manager_cls, _ = self._load_config_manager()
            return ok(manager_cls(self.plugin).view())
        except ImportError:
            return err("配置模块不可用")
        except Exception as exc:
            logger.error("[agent_memory] page.config 失败: %s", exc, exc_info=True)
            return err(f"读取配置失败: {exc}")

    # -- POST /config/save -------------------------------------------------
    async def save_config(self, **_: Any) -> Dict[str, Any]:
        """在线保存配置：白名单校验 → 原子写 → 运行时热更新（不重载/不重启）。"""
        try:
            manager_cls, config_error = self._load_config_manager()
        except ImportError:
            return err("配置模块不可用")
        try:
            body = await _json_body()
            # 安全开关：webui.allow_config_write=false 时面板只读，拒绝任何写入
            runtime_cfg = getattr(self.plugin, "config", None)
            if isinstance(runtime_cfg, dict):
                webui_cfg = runtime_cfg.get("webui")
                if isinstance(webui_cfg, dict) and not bool(
                    webui_cfg.get("allow_config_write", True)
                ):
                    return err("面板配置写入已被 webui.allow_config_write=false 禁用")
            # 兼容 {"config": {...}} 与直接提交 {...} 两种前端载荷
            payload = body.get("config") if isinstance(body.get("config"), dict) else body
            result = manager_cls(self.plugin).save(payload)
            return ok(result)
        except config_error as exc:
            return err(str(exc))
        except Exception as exc:
            logger.error("[agent_memory] page.config-save 失败: %s", exc, exc_info=True)
            return err(f"保存配置失败: {exc}")

    # -- POST /providers/test ---------------------------------------------
    async def test_provider(self, **_: Any) -> Dict[str, Any]:
        """服务商连通性测试：轻量真实调用 + 超时保护，绝不返回任何密钥。"""
        import asyncio

        try:
            body = await _json_body()
            kind = str(body.get("kind") or "").strip().lower()
            kind = {"chat": "chat_completion", "llm": "chat_completion"}.get(kind, kind)
            if kind not in ("embedding", "rerank", "chat_completion"):
                return err("kind 必须是 embedding / rerank / chat_completion")
            provider_id = str(body.get("provider_id") or "").strip()[:200]
            bridge = getattr(self.plugin, "provider_bridge", None)
            if bridge is None:
                return err("服务商桥接未初始化")

            # 指定 provider_id 时先核对能力清单（只比对 id，不外泄凭证）
            if provider_id:
                try:
                    bucket = (bridge.list_providers() or {}).get(kind) or []
                    if not any(str(p.get("id")) == provider_id for p in bucket):
                        return err(f"服务商 {provider_id} 不在 {kind} 清单中")
                except Exception as exc:
                    return err(f"读取服务商清单失败: {exc}")
            auto = not bool(provider_id)

            async def _probe() -> Dict[str, Any]:
                if kind == "embedding":
                    backend = bridge.build_embedding_backend(provider_id, allow_auto=auto)
                    if backend is None:
                        raise RuntimeError("未找到可用 Embedding 服务商")
                    vec = await backend.embed_one("TierMem connectivity probe")
                    return {
                        "dim": len(vec),
                        "model": str(getattr(backend, "model", "") or ""),
                        "effective_provider_id": str(getattr(backend, "provider_id", "") or ""),
                    }
                if kind == "rerank":
                    backend = bridge.build_rerank_backend(provider_id, allow_auto=auto)
                    if backend is None:
                        raise RuntimeError("未找到可用 Rerank 服务商")
                    ranked = await backend.rerank(
                        "分层记忆连通性测试", ["记忆可用", "无关文本"]
                    )
                    return {
                        "results": len(ranked),
                        "model": str(getattr(backend, "model", "") or ""),
                        "effective_provider_id": str(getattr(backend, "provider_id", "") or ""),
                    }
                backend = bridge.build_chat_backend(provider_id, allow_auto=auto)
                if backend is None:
                    raise RuntimeError("未找到可用对话服务商")
                text = await backend.chat(
                    [{"role": "user", "content": "请只回复 OK 两个字符。"}]
                )
                return {
                    "reply_preview": str(text or "")[:60],
                    "model": str(getattr(backend, "model", "") or ""),
                    "effective_provider_id": str(getattr(backend, "provider_id", "") or ""),
                }

            started = time.time()
            try:
                detail = await asyncio.wait_for(_probe(), timeout=10.0)
            except asyncio.TimeoutError:
                return err(
                    "连通性测试超时（10 秒）",
                    {"kind": kind, "provider_id": provider_id},
                )
            return ok(
                {
                    "kind": kind,
                    "provider_id": provider_id,
                    "elapsed_ms": int((time.time() - started) * 1000),
                    **detail,
                }
            )
        except Exception as exc:
            logger.error("[agent_memory] page.provider-test 失败: %s", exc, exc_info=True)
            return err(f"连通性测试失败: {exc}")

    # -- POST /memories/edit ----------------------------------------------
    async def update_memory(self, **_: Any) -> Dict[str, Any]:
        """编辑记忆正文 / 重要度 / 关键事实 / 主题（参数化 SQL，输入强校验）。"""
        try:
            body = await _json_body()
            atom_id = str(body.get("atom_id") or "").strip()
            if not atom_id:
                return err("缺少 atom_id")
            storage = getattr(self.plugin, "storage", None)
            if storage is None:
                return err("存储未就绪")
            fn = getattr(storage, "update_memory", None)
            if not callable(fn):
                return err("存储不支持编辑操作")
            result = fn(
                atom_id,
                content=body.get("content"),
                importance=body.get("importance"),
                key_facts=body.get("key_facts"),
                topics=body.get("topics"),
            )
            if not isinstance(result, dict) or result.get("error"):
                return err(str((result or {}).get("error") or "编辑失败"), result)
            self._invalidate_cache()
            return ok(result)
        except Exception as exc:
            logger.error("[agent_memory] page.memory-edit 失败: %s", exc, exc_info=True)
            return err(f"编辑记忆失败: {exc}")

    # -- POST /memories/delete --------------------------------------------
    async def delete_memory(self, **_: Any) -> Dict[str, Any]:
        """删除记忆及关联图谱边；必须显式 confirm=true，防止误删。"""
        try:
            body = await _json_body()
            atom_id = str(body.get("atom_id") or "").strip()
            if not atom_id:
                return err("缺少 atom_id")
            if body.get("confirm") is not True:
                return err("删除操作需要显式 confirm=true")
            storage = getattr(self.plugin, "storage", None)
            if storage is None:
                return err("存储未就绪")
            fn = getattr(storage, "delete_memory", None)
            if not callable(fn):
                return err("存储不支持删除操作")
            result = fn(atom_id)
            if not isinstance(result, dict) or result.get("error"):
                return err(str((result or {}).get("error") or "删除失败"), result)
            self._invalidate_cache()
            return ok(result)
        except Exception as exc:
            logger.error("[agent_memory] page.memory-delete 失败: %s", exc, exc_info=True)
            return err(f"删除记忆失败: {exc}")

    # -- GET /graph --------------------------------------------------------
    async def get_graph(self, **_: Any) -> Dict[str, Any]:
        """知识图谱（L3）节点与关系浏览，支持 q 关键词与 limit 上限。"""
        try:
            storage = getattr(self.plugin, "storage", None)
            if storage is None:
                return err("存储未就绪")
            fn = getattr(storage, "list_kg_graph", None)
            if not callable(fn):
                return err("存储不支持图谱浏览")
            return ok(
                fn(
                    limit=_query_int("limit", 100, 1, 300),
                    query=_query_one("q", ""),
                )
            )
        except Exception as exc:
            logger.error("[agent_memory] page.graph 失败: %s", exc, exc_info=True)
            return err(f"读取图谱失败: {exc}")

    # -- POST /maintenance -------------------------------------------------
    async def run_maintenance(self, **_: Any) -> Dict[str, Any]:
        """维护操作白名单：flush / rebuild_fts / retry_dlq / migrate_preview。"""
        allowed = ("flush", "rebuild_fts", "retry_dlq", "migrate_preview")
        try:
            body = await _json_body()
            op = str(body.get("op") or "").strip()
            if op not in allowed:
                return err("op 必须是 " + " / ".join(allowed))
            started = time.time()
            data: Dict[str, Any] = {"op": op}

            if op == "flush":
                summarizer = getattr(self.plugin, "summarizer", None)
                flush = getattr(summarizer, "flush_all", None)
                if not callable(flush):
                    return err("总结器未就绪")
                data["flushed"] = await flush()
            elif op == "rebuild_fts":
                storage = getattr(self.plugin, "storage", None)
                rebuild = getattr(storage, "rebuild_fts_index", None)
                if not callable(rebuild):
                    return err("存储未就绪")
                data["indexed"] = rebuild()
            elif op == "retry_dlq":
                summarizer = getattr(self.plugin, "summarizer", None)
                sweep = getattr(summarizer, "sweep_dlq_once", None)
                if not callable(sweep):
                    return err("总结器未就绪")
                data["sweep"] = await sweep()
            elif op == "migrate_preview":
                cls = getattr(self.plugin, "_migrator_cls", None)
                storage = getattr(self.plugin, "storage", None)
                if cls is None or storage is None:
                    return err("迁移器/存储未就绪")
                migrator = cls(storage=storage)
                if not migrator.check_source_exists():
                    data["status"] = "no_source"
                    data["source_db"] = str(getattr(migrator, "source_db_path", ""))
                else:
                    # dry_run=True 只读统计，不写目标库、不生成快照
                    data["preview"] = migrator.migrate(dry_run=True)

            data["elapsed_ms"] = int((time.time() - started) * 1000)
            return ok(data)
        except Exception as exc:
            logger.error("[agent_memory] page.maintenance 失败: %s", exc, exc_info=True)
            return err(f"维护操作失败: {exc}")


def _effective_provider_snapshot(plugin: Any) -> Dict[str, Any]:
    """汇总当前实际生效的 Embedding / Rerank / LLM 通电信息。"""
    snapshot: Dict[str, Any] = {}
    for key, attr in (("embedding", "embedding_client"), ("rerank", "rerank_client")):
        client = getattr(plugin, attr, None)
        describe = getattr(client, "describe", None)
        if callable(describe):
            try:
                snapshot[key] = describe()
                continue
            except Exception:
                pass
        snapshot[key] = {"present": client is not None}
    builder = getattr(plugin, "build_self_check", None)
    if callable(builder):
        try:
            snapshot["llm"] = builder().get("llm", {})
        except Exception:
            pass
    return snapshot


__all__ = ["PAGE_API_PREFIX", "PLUGIN_NAME", "PluginPageApi", "err", "ok"]
