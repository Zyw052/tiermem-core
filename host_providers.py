# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 宿主服务商桥接 (host_providers.py)

背景（P0 改造第 1 项）：
    历史实现只从插件配置里读 ``api_base`` / ``api_key`` / ``model`` 三个字段，
    ``provider_id`` 仅作为「顺带解析凭证」的可选项；一旦三者留空，Embedding /
    Rerank / 判官 LLM 就全部 fail-open 退化为零向量 + 本地规则，表现为
    「配了服务商也不通电」。

本模块的职责：
    1. 通过 ``context.get_provider_by_id`` 拿到 AstrBot 宿主真实的 Provider 实例；
    2. 依据 Provider 类型（chat_completion / embedding / rerank）产出「后端适配器」；
    3. 适配器直接调用宿主 Provider 的官方方法（``text_chat`` / ``get_embedding`` /
       ``rerank``），从而自动继承宿主已配置的 api_base、密钥、代理、重试与计费；
    4. 全链路零异常外抛：任何失败都返回 None，由上层回退到「手动凭证」或「本地降级」；
    5. 提供 ``status()`` 通电自检快照，供 WebUI 与 /memory status 展示。

类型探测口径（依据宿主 astrbot/core/provider/entities.py:ProviderType）：
    - ``provider.meta().provider_type`` 为权威值；
    - 若 meta() 不可用，退化为鸭子类型（有 get_embedding → embedding；
      有 rerank → rerank；有 text_chat → chat）。
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # 宿主日志；脱离宿主时退化为标准 logging
    from astrbot.api import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("astrbot_plugin_agent_memory.host_providers")


#: 期望的能力类型标识（与宿主 ProviderType.value 对齐）
KIND_CHAT = "chat_completion"
KIND_EMBEDDING = "embedding"
KIND_RERANK = "rerank"

#: WebUI 运行期绑定文件名（存放于插件数据目录，不改动宿主 cmd_config.json）
BINDINGS_FILENAME = "provider_bindings.json"


# ---------------------------------------------------------------------------
# 1. 能力探测
# ---------------------------------------------------------------------------
def detect_provider_kind(provider: Any) -> str:
    """探测 Provider 实例的能力类型，返回 KIND_* 之一；无法判定时返回空串。"""
    if provider is None:
        return ""

    # 1) 权威来源：ProviderMeta.provider_type（enum，value 为 chat_completion 等）
    try:
        meta = provider.meta()
        provider_type = getattr(meta, "provider_type", None)
        value = getattr(provider_type, "value", None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass

    # 2) provider_config["type"] 为适配器类名（如 openai_embedding），需再映射
    try:
        cfg = getattr(provider, "provider_config", None)
        if isinstance(cfg, dict):
            type_name = str(cfg.get("type") or "").strip()
            if type_name:
                try:
                    from astrbot.core.provider.register import provider_cls_map

                    meta_data = provider_cls_map.get(type_name)
                    pt = getattr(meta_data, "provider_type", None)
                    value = getattr(pt, "value", None)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                except Exception:
                    pass
            # 3) 名称启发式（最后兜底，命中率足够高）
            lowered = type_name.lower()
            if "embedding" in lowered:
                return KIND_EMBEDDING
            if "rerank" in lowered:
                return KIND_RERANK
    except Exception:
        pass

    # 4) 鸭子类型
    if callable(getattr(provider, "get_embedding", None)):
        return KIND_EMBEDDING
    if callable(getattr(provider, "rerank", None)):
        return KIND_RERANK
    if callable(getattr(provider, "text_chat", None)):
        return KIND_CHAT
    return ""


def _provider_id_of(provider: Any) -> str:
    """尽量取出 Provider 的配置 ID。"""
    for attr in ("provider_config",):
        cfg = getattr(provider, attr, None)
        if isinstance(cfg, dict):
            pid = cfg.get("id")
            if isinstance(pid, str) and pid.strip():
                return pid.strip()
    for attr in ("id", "provider_id"):
        val = getattr(provider, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _provider_model_of(provider: Any) -> str:
    """尽量取出 Provider 当前模型名。"""
    try:
        get_model = getattr(provider, "get_model", None)
        if callable(get_model):
            name = get_model()
            if isinstance(name, str) and name.strip():
                return name.strip()
    except Exception:
        pass
    name = getattr(provider, "model_name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    cfg = getattr(provider, "provider_config", None)
    if isinstance(cfg, dict):
        for key in ("model", "model_name", "embedding_model", "rerank_model"):
            val = cfg.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


# ---------------------------------------------------------------------------
# 2. 三类后端适配器
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingBackend:
    """Embedding 能力适配器：直接走宿主 EmbeddingProvider。"""

    provider: Any
    provider_id: str = ""
    model: str = ""
    source: str = "provider"
    """provider=宿主服务商；bindings=WebUI 运行期绑定（同样走宿主服务商）。"""

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """批量向量化（优先 get_embeddings，缺失时逐条 get_embedding）。"""
        items = [str(t or "") for t in texts]
        if not items:
            return []
        batch_fn = getattr(self.provider, "get_embeddings", None)
        if callable(batch_fn):
            vectors = batch_fn(items)
            if inspect.isawaitable(vectors):
                vectors = await vectors
            return [list(map(float, v or [])) for v in (vectors or [])]
        one_fn = getattr(self.provider, "get_embedding", None)
        if not callable(one_fn):
            raise RuntimeError("Provider 缺少 get_embedding/get_embeddings 方法")
        results: List[List[float]] = []
        for text in items:
            vec = one_fn(text)
            if inspect.isawaitable(vec):
                vec = await vec
            results.append(list(map(float, vec or [])))
        return results

    async def embed_one(self, text: str) -> List[float]:
        vecs = await self.embed([text])
        return vecs[0] if vecs else []

    def dim(self) -> int:
        """Provider 声明的向量维度（拿不到时返回 0，由客户端动态对齐）。"""
        try:
            get_dim = getattr(self.provider, "get_dim", None)
            if callable(get_dim):
                value = get_dim()
                if inspect.isawaitable(value):
                    return 0
                return int(value or 0)
        except Exception:
            pass
        return 0

    def describe(self) -> Dict[str, Any]:
        return {
            "bound": True,
            "kind": KIND_EMBEDDING,
            "source": self.source,
            "provider_id": self.provider_id,
            "model": self.model,
            "dim": self.dim(),
        }


@dataclass
class RerankBackend:
    """Rerank 能力适配器：直接走宿主 RerankProvider。"""

    provider: Any
    provider_id: str = ""
    model: str = ""
    source: str = "provider"

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """返回 [(文档索引, 相关性分数)]，按分数降序。"""
        docs = [str(d or "") for d in documents]
        if not docs:
            return []
        fn = getattr(self.provider, "rerank", None)
        if not callable(fn):
            raise RuntimeError("Provider 缺少 rerank 方法")
        try:
            raw = fn(str(query or ""), docs, top_n)
        except TypeError:
            # 部分适配器不支持 top_n 位置参数
            raw = fn(str(query or ""), docs)
        if inspect.isawaitable(raw):
            raw = await raw

        results: List[Tuple[int, float]] = []
        for item in raw or []:
            idx = getattr(item, "index", None)
            score = getattr(item, "relevance_score", None)
            if idx is None and isinstance(item, dict):
                idx = item.get("index")
                score = item.get("relevance_score", item.get("score"))
            if idx is None:
                continue
            try:
                results.append((int(idx), float(score or 0.0)))
            except (TypeError, ValueError):
                continue
        results.sort(key=lambda pair: pair[1], reverse=True)
        if top_n is not None and top_n > 0:
            results = results[: int(top_n)]
        return results

    def describe(self) -> Dict[str, Any]:
        return {
            "bound": True,
            "kind": KIND_RERANK,
            "source": self.source,
            "provider_id": self.provider_id,
            "model": self.model,
        }


@dataclass
class ChatBackend:
    """Chat 能力适配器：把 messages 列表翻译成宿主 ``text_chat`` 调用。"""

    provider: Any
    provider_id: str = ""
    model: str = ""
    source: str = "provider"

    async def chat(
        self,
        messages: Sequence[Dict[str, str]],
        model: str = "",
    ) -> str:
        """
        调用宿主 chat provider 生成回复文本。

        ``messages`` 为 OpenAI 风格 [{"role": "system|user|assistant", "content": "..."}]；
        system 消息会被抽成 ``system_prompt``，其余作为 ``contexts`` 传入。
        """
        system_parts: List[str] = []
        contexts: List[Dict[str, str]] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip().lower()
            content = msg.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
                continue
            if role not in ("user", "assistant"):
                role = "user"
            contexts.append({"role": role, "content": text})

        text_chat = getattr(self.provider, "text_chat", None)
        if not callable(text_chat):
            raise RuntimeError("Provider 缺少 text_chat 方法")
        if not system_parts and not contexts:
            raise ValueError("messages 为空，无法调用 chat provider")

        kwargs: Dict[str, Any] = {}
        if system_parts:
            kwargs["system_prompt"] = "\n\n".join(system_parts)
        # 最后一条若是 user，则用 prompt 传入，保持与宿主调用习惯一致
        prompt = None
        if contexts and contexts[-1]["role"] == "user":
            prompt = contexts.pop()["content"]
        if contexts:
            kwargs["contexts"] = contexts
        target_model = model or self.model
        if target_model:
            kwargs["model"] = target_model

        if prompt is not None:
            resp = text_chat(prompt=prompt, **kwargs)
        else:
            resp = text_chat(**kwargs)
        if inspect.isawaitable(resp):
            resp = await resp
        return _extract_response_text(resp)

    def describe(self) -> Dict[str, Any]:
        return {
            "bound": True,
            "kind": KIND_CHAT,
            "source": self.source,
            "provider_id": self.provider_id,
            "model": self.model,
        }


def _extract_response_text(resp: Any) -> str:
    """从 LLMResponse 中尽量取出纯文本（兼容旧版 completion_text 与 result_chain）。"""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    text = getattr(resp, "completion_text", None)
    if isinstance(text, str) and text.strip():
        return text
    chain = getattr(resp, "result_chain", None)
    if chain is not None:
        try:
            plain = getattr(chain, "message_str", None)
            if isinstance(plain, str) and plain.strip():
                return plain
            parts: List[str] = []
            for comp in getattr(chain, "chain", []) or []:
                piece = getattr(comp, "text", None)
                if isinstance(piece, str) and piece:
                    parts.append(piece)
            if parts:
                return "".join(parts)
        except Exception:
            pass
    raw = getattr(resp, "raw_completion", None)
    try:
        choice = (getattr(raw, "choices", None) or [])[0]
        content = getattr(getattr(choice, "message", None), "content", "")
        if isinstance(content, str):
            return content
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# 3. 桥接器
# ---------------------------------------------------------------------------
@dataclass
class ProviderBridge:
    """
    宿主 Provider 桥接器。

    参数：
        context: AstrBot ``Context``（Star 基类自身即持有 ``self.context``）；
        data_dir: 插件数据目录（用于读写 WebUI 运行期绑定文件）。
    """

    context: Any
    data_dir: Optional[Path] = None
    #: 运行期绑定缓存（避免每次装配都读盘）
    _bindings: Dict[str, str] = field(default_factory=dict)
    _bindings_mtime: float = 0.0
    #: 最近一次自检结果
    _status: Dict[str, Any] = field(default_factory=dict)

    # -- 内部：访问宿主 Provider 清单 -------------------------------------
    def _inst_map(self) -> Dict[str, Any]:
        try:
            manager = getattr(self.context, "provider_manager", None)
            inst_map = getattr(manager, "inst_map", None)
            if isinstance(inst_map, dict):
                return inst_map
        except Exception:
            pass
        return {}

    def _get_by_id(self, provider_id: str) -> Any:
        """按 ID 取 Provider 实例（兼容 sync / async 两种宿主实现）。"""
        pid = str(provider_id or "").strip()
        if not pid:
            return None

        # 1) 优先用 Context 的同步方法（AstrBot v4 现状）
        getter = getattr(self.context, "get_provider_by_id", None)
        if callable(getter):
            try:
                provider = getter(pid)
                if not inspect.isawaitable(provider):
                    if provider is not None:
                        return provider
            except Exception as exc:
                logger.debug("[ProviderBridge] get_provider_by_id(%s) 异常: %s", pid, exc)

        # 2) 直接读 inst_map（零副作用，且不会触发宿主的 warning 日志）
        return self._inst_map().get(pid)

    def iter_providers(self) -> List[Any]:
        """枚举全部宿主 Provider 实例（去重，按 ID 排序保证稳定）。"""
        seen: Dict[int, Any] = {}
        for provider in self._inst_map().values():
            if provider is None:
                continue
            seen.setdefault(id(provider), provider)
        return list(seen.values())

    def list_providers(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        按能力类型列出宿主 Provider（供 WebUI 下拉框使用）。

        返回：{"chat_completion": [{...}], "embedding": [...], "rerank": [...]}
        """
        result: Dict[str, List[Dict[str, Any]]] = {
            KIND_CHAT: [],
            KIND_EMBEDDING: [],
            KIND_RERANK: [],
        }
        for provider in self.iter_providers():
            kind = detect_provider_kind(provider)
            if kind not in result:
                continue
            item = {
                "id": _provider_id_of(provider),
                "model": _provider_model_of(provider),
                "kind": kind,
                "adapter": str(
                    (getattr(provider, "provider_config", {}) or {}).get("type") or ""
                ),
            }
            if not item["id"]:
                continue
            result[kind].append(item)
        for items in result.values():
            items.sort(key=lambda entry: entry["id"])
        return result

    # -- 内部：WebUI 运行期绑定 -------------------------------------------
    def _bindings_path(self) -> Optional[Path]:
        if self.data_dir is None:
            return None
        try:
            return Path(self.data_dir) / BINDINGS_FILENAME
        except Exception:
            return None

    def load_bindings(self, force: bool = False) -> Dict[str, str]:
        """读取 WebUI 运行期绑定（按 mtime 缓存，文件不存在时返回空表）。"""
        path = self._bindings_path()
        if path is None:
            return dict(self._bindings)
        try:
            if not path.is_file():
                if force:
                    self._bindings = {}
                return dict(self._bindings)
            mtime = path.stat().st_mtime
            if not force and mtime == self._bindings_mtime:
                return dict(self._bindings)
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cleaned: Dict[str, str] = {}
                for key in (KIND_CHAT, KIND_EMBEDDING, KIND_RERANK):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        cleaned[key] = val.strip()
                self._bindings = cleaned
            self._bindings_mtime = mtime
        except Exception as exc:
            logger.debug("[ProviderBridge] 读取绑定文件失败(忽略): %s", exc)
        return dict(self._bindings)

    def save_bindings(self, updates: Dict[str, str]) -> Dict[str, str]:
        """
        写入 WebUI 运行期绑定（只写插件数据目录，不触碰宿主配置文件）。

        ``updates`` 形如 {"embedding": "gemini_embedding"}；空字符串表示解除绑定。
        """
        current = self.load_bindings(force=True)
        for key, value in (updates or {}).items():
            if key not in (KIND_CHAT, KIND_EMBEDDING, KIND_RERANK):
                continue
            val = str(value or "").strip()
            if val:
                current[key] = val
            else:
                current.pop(key, None)

        path = self._bindings_path()
        if path is None:
            return current
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._bindings = dict(current)
            self._bindings_mtime = path.stat().st_mtime
            logger.info("[ProviderBridge] 已更新运行期绑定: %s", current)
        except Exception as exc:
            logger.warning("[ProviderBridge] 写入绑定文件失败: %s", exc)
        return current

    # -- 对外：构建后端 ---------------------------------------------------
    def _pick(
        self,
        provider_id: str,
        kind: str,
        allow_auto: bool,
    ) -> Tuple[Any, str, str]:
        """
        按 ID / 运行期绑定 / 首个可用 Provider 的顺序解析，返回 (provider, 实际ID, 来源)。
        """
        # 1) 显式 ID
        pid = str(provider_id or "").strip()
        if pid:
            provider = self._get_by_id(pid)
            if provider is not None and detect_provider_kind(provider) == kind:
                return provider, pid, "provider"
            if provider is not None:
                logger.warning(
                    "[ProviderBridge] provider_id=%s 类型不符（期望 %s，实际 %s），已忽略。",
                    pid,
                    kind,
                    detect_provider_kind(provider) or "unknown",
                )

        # 2) WebUI 运行期绑定
        bound = self.load_bindings().get(kind, "")
        if bound:
            provider = self._get_by_id(bound)
            if provider is not None and detect_provider_kind(provider) == kind:
                return provider, bound, "bindings"

        # 3) 同类型首个可用 Provider（可选，默认关闭以避免「选错模型」）
        if allow_auto:
            for provider in self.iter_providers():
                if detect_provider_kind(provider) == kind:
                    return provider, _provider_id_of(provider), "auto"

        return None, "", "none"

    def build_embedding_backend(
        self, provider_id: str = "", allow_auto: bool = False
    ) -> Optional[EmbeddingBackend]:
        provider, pid, source = self._pick(provider_id, KIND_EMBEDDING, allow_auto)
        if provider is None:
            return None
        return EmbeddingBackend(
            provider=provider,
            provider_id=pid,
            model=_provider_model_of(provider),
            source=source,
        )

    def build_rerank_backend(
        self, provider_id: str = "", allow_auto: bool = False
    ) -> Optional[RerankBackend]:
        provider, pid, source = self._pick(provider_id, KIND_RERANK, allow_auto)
        if provider is None:
            return None
        return RerankBackend(
            provider=provider,
            provider_id=pid,
            model=_provider_model_of(provider),
            source=source,
        )

    def build_chat_backend(
        self, provider_id: str = "", allow_auto: bool = False
    ) -> Optional[ChatBackend]:
        provider, pid, source = self._pick(provider_id, KIND_CHAT, allow_auto)
        if provider is None:
            return None
        return ChatBackend(
            provider=provider,
            provider_id=pid,
            model=_provider_model_of(provider),
            source=source,
        )

    # -- 对外：自检快照 ---------------------------------------------------
    def remember_status(self, **kwargs: Any) -> None:
        """记录一次组件装配结果（供 WebUI 通电自检读取）。"""
        self._status.update(kwargs)
        self._status["checked_at"] = time.time()

    def status(self) -> Dict[str, Any]:
        """返回最近一次装配的通电自检快照（只读副本）。"""
        return json.loads(json.dumps(self._status, ensure_ascii=False, default=str))


__all__ = [
    "BINDINGS_FILENAME",
    "ChatBackend",
    "EmbeddingBackend",
    "KIND_CHAT",
    "KIND_EMBEDDING",
    "KIND_RERANK",
    "ProviderBridge",
    "RerankBackend",
    "detect_provider_kind",
]
