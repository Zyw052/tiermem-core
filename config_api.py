# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory · 配置在线读写（记忆面板 WebUI 后端核心）

规范依据（astrbot-plugin-dev，产出注明依据章节）：
  §6 配置 / guides/plugin-config.md「配置定义」「敏感配置项」「dict 类型的 schema」
  §9 存储 / guides/storage.md「存储大文件规范」
  §10 工程规范 / plugin-new.md「开发原则」（注释、测试、不写死路径、异常处理）
  WebUI 专章 guides/plugin-pages.md「安全约束」（后端 handler 必须校验输入）

为什么需要这个模块
------------------
AstrBot 原生插件配置弹窗（``ExtensionPage.vue``）把「保存并关闭」按钮与配置表单
共同绑定在 ``v-if="extension_config.metadata"`` 上；而后端
``ConfigDisplayService.get_plugin_config()`` 只有在 ``star_registry`` 中
``plugin_md.config``（由 ``_conf_schema.json`` 生成的 ``AstrBotConfig``）为真值时
才返回 metadata。保存成功后宿主会自动 ``plugin_manager.reload()``，一旦重载链路
让插件落入 failed 列表（或 schema 注册异常），metadata 变为 null，前端只显示
「无配置」，**连保存按钮一起消失** —— 这正是主人反馈的「重载后配置没有保存按钮」。

本模块提供一条**不依赖宿主配置弹窗**的独立保存通道：
  * 直接原子写插件配置文件（与 ``AstrBotConfig`` 同一路径、同一 JSON 结构）；
  * 同步更新插件运行时 ``config``（可热更新的字段立即生效）；
  * 明确返回 ``requires_reload`` 提示，由用户决定何时重载；
  * **不调用 reload、不重启 AstrBot 主进程**。

安全约束（对应主人要求）
------------------------
1. 读取一律对敏感字段（api_key / token / secret / password…）做掩码，绝不返回明文；
2. 写入只允许 schema 白名单内的键，未知键一律拒绝（防配置注入）；
3. 强类型 + options 枚举 + slider 范围 + 字符串长度 + 递归深度校验；
4. ``db_path`` 做路径穿越 / 空字节 / 协议头 / 长度校验；
5. 原子写（同目录临时文件 + ``os.replace``）+ ``0o600`` 权限；
6. 掩码占位符表示「保持原值」，避免误清空线上密钥；
7. 全程异常处理，失败抛 ``ConfigError``，由 ``page_api`` 转成结构化 error；
8. 路径不写死：schema 相对本文件解析，配置文件优先取宿主 ``config_path``。
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from astrbot.api import logger

PLUGIN_NAME = "astrbot_plugin_agent_memory"
SCHEMA_FILENAME = "_conf_schema.json"

#: 前端展示的掩码；回传本 token 表示「保持原密钥不变」
MASK_DISPLAY = "******"
MASK_TOKEN = "__TIERMEM_KEEP_UNCHANGED__"

SENSITIVE_RE = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|credential|access[_-]?key|authorization)",
    re.IGNORECASE,
)
SUPPORTED_TYPES = {"bool", "int", "float", "string", "text", "object", "list", "dict"}
MAX_STRING_LEN = 4096
MAX_TEXT_LEN = 20000
MAX_LIST_LEN = 200
MAX_DEPTH = 4
MAX_PATH_LEN = 1024

DEFAULT_VALUE_MAP: Dict[str, Any] = {
    "int": 0,
    "float": 0.0,
    "bool": False,
    "string": "",
    "text": "",
    "list": [],
    "object": {},
    "dict": {},
}

#: 修改这些键后必须重载插件才能完全生效（面板会给出提示，但绝不自动重载）
RELOAD_REQUIRED_PREFIXES = (
    "db_path",
    "summarizer",
    "models",
    "migration",
    "webui",
    "evaluator.api_",
    "graph.",
)


class ConfigError(Exception):
    """配置读写的可预期错误（会被 page_api 转成 ``{"status": "error"}``）。"""


def _is_sensitive(key: str) -> bool:
    """按字段名判定是否为敏感凭证（不依赖值，避免漏判）。"""
    return bool(SENSITIVE_RE.search(str(key or "")))


def _default_of(meta: Dict[str, Any]) -> Any:
    """取 schema 节点的默认值，兼容缺省 default。"""
    if "default" in meta:
        return copy.deepcopy(meta["default"])
    return copy.deepcopy(DEFAULT_VALUE_MAP.get(str(meta.get("type", "string")), ""))


def _walk_schema(
    node: Dict[str, Any], prefix: str = "", depth: int = 0
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """深度遍历 schema，产出 (点分路径, meta)；带最大深度保护。"""
    if not isinstance(node, dict) or depth > MAX_DEPTH:
        return
    for key, meta in node.items():
        if not isinstance(meta, dict):
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        yield path, meta
        if meta.get("type") == "object" and isinstance(meta.get("items"), dict):
            yield from _walk_schema(meta["items"], path, depth + 1)


def _validate_db_path(value: Any) -> Tuple[bool, str]:
    """
    校验 db_path：允许空（使用默认）、普通目录或 .db/.sqlite 文件；
    拒绝路径穿越、空字节、URL 协议头与超长路径。
    """
    raw = str(value or "").strip()
    if not raw:
        return True, ""
    if "\x00" in raw:
        return False, "路径包含空字节"
    if len(raw) > MAX_PATH_LEN:
        return False, "路径过长"
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
        return False, "不接受 URL / 协议形式路径"
    try:
        parts = Path(os.path.normpath(raw)).parts
    except (ValueError, OSError) as exc:  # pragma: no cover - 极端输入
        return False, f"路径无法解析: {exc}"
    if ".." in parts:
        return False, "禁止路径穿越（..）"
    return True, ""


def _normalize_scalar(
    path: str, key: str, meta: Dict[str, Any], value: Any
) -> Tuple[Any, str]:
    """按 schema 归一化并校验单个标量值，返回 (值, 错误文本)。"""
    type_ = str(meta.get("type", "string"))

    # 敏感字段：掩码占位符直接透传（_apply 会据此保持原值）
    if _is_sensitive(key) and type_ in ("string", "text") and value == MASK_TOKEN:
        return MASK_TOKEN, ""

    if type_ == "bool":
        if isinstance(value, bool):
            return value, ""
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value), ""
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("1", "true", "yes", "on"):
                return True, ""
            if low in ("0", "false", "no", "off"):
                return False, ""
        return None, f"{path} 期望 bool"

    if type_ == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None, f"{path} 期望 int"
        try:
            casted = int(float(value))
        except (TypeError, ValueError):
            return None, f"{path} 期望 int"
    elif type_ == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None, f"{path} 期望 float"
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return None, f"{path} 期望 float"
    else:
        casted = value

    # slider 数值范围
    slider = meta.get("slider")
    if isinstance(slider, dict) and isinstance(casted, (int, float)):
        low = slider.get("min")
        high = slider.get("max")
        try:
            if low is not None and casted < float(low):
                return None, f"{path} 不得小于 {low}"
            if high is not None and casted > float(high):
                return None, f"{path} 不得大于 {high}"
        except (TypeError, ValueError):  # pragma: no cover
            pass

    if type_ in ("string", "text"):
        if not isinstance(casted, str):
            return None, f"{path} 期望 string"
        limit = MAX_TEXT_LEN if type_ == "text" else MAX_STRING_LEN
        if len(casted) > limit:
            return None, f"{path} 长度超过 {limit}"
        options = meta.get("options")
        if isinstance(options, list) and options and casted not in options:
            return None, f"{path} 取值必须是 {options} 之一"

    if type_ == "list":
        if not isinstance(casted, list):
            return None, f"{path} 期望 list"
        if len(casted) > MAX_LIST_LEN:
            return None, f"{path} 列表过长"

    return casted, ""


class ConfigManager:
    """插件配置读写门面：GET 掩码视图 / POST 白名单校验 + 原子写 + 运行时同步。"""

    def __init__(
        self,
        plugin: Any = None,
        *,
        schema_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.plugin = plugin
        self._schema_path = (
            Path(schema_path)
            if schema_path
            else Path(__file__).resolve().parent / SCHEMA_FILENAME
        )
        self._config_path_override = Path(config_path) if config_path else None
        self._schema_cache: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()

    # -- 路径与 schema ----------------------------------------------------
    def load_schema(self) -> Dict[str, Any]:
        """读取并缓存 ``_conf_schema.json``（utf-8-sig 兼容 BOM）。"""
        with self._lock:
            if self._schema_cache is not None:
                return self._schema_cache
            try:
                with open(self._schema_path, encoding="utf-8-sig") as fh:
                    schema = json.load(fh)
            except FileNotFoundError as exc:
                raise ConfigError(f"配置 schema 不存在: {self._schema_path.name}") from exc
            except (OSError, ValueError) as exc:
                raise ConfigError(f"配置 schema 无法解析: {exc}") from exc
            if not isinstance(schema, dict) or not schema:
                raise ConfigError("配置 schema 必须是非空对象")
            self._schema_cache = schema
            return schema

    def config_path(self) -> Path:
        """定位插件配置文件：宿主 AstrBotConfig 优先，其次宿主工具函数。"""
        if self._config_path_override is not None:
            return self._config_path_override
        host = getattr(self.plugin, "_host_config", None)
        candidate = getattr(host, "config_path", None)
        if candidate:
            return Path(candidate)
        try:  # 不写死路径：向宿主工具函数询问配置目录
            from astrbot.core.utils.astrbot_path import (  # type: ignore
                get_astrbot_config_path,
            )

            return Path(get_astrbot_config_path()) / f"{PLUGIN_NAME}_config.json"
        except Exception as exc:  # pragma: no cover - 仅在脱离宿主时触发
            raise ConfigError(
                "无法定位插件配置文件（宿主未提供 config_path，且 astrbot 工具函数不可用）"
            ) from exc

    def read_raw(self) -> Dict[str, Any]:
        """读取线上配置原文（仅供内部使用，绝不直接返回给前端）。"""
        path = self.config_path()
        try:
            with open(path, encoding="utf-8-sig") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise ConfigError(f"读取插件配置失败: {exc}") from exc

    # -- 默认值 / 掩码 ----------------------------------------------------
    @staticmethod
    def _fill_defaults(schema: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """按 schema 补齐缺省键（不写文件，仅用于展示/落盘完整性）。"""
        out: Dict[str, Any] = copy.deepcopy(data) if isinstance(data, dict) else {}
        for key, meta in schema.items():
            if not isinstance(meta, dict):
                continue
            type_ = str(meta.get("type", "string"))
            if type_ == "object":
                child = out.get(key)
                out[key] = ConfigManager._fill_defaults(
                    meta.get("items") or {}, child if isinstance(child, dict) else {}
                )
            elif key not in out:
                out[key] = _default_of(meta)
        return out

    def _mask_tree(
        self,
        schema: Dict[str, Any],
        values: Dict[str, Any],
        prefix: str,
        sensitive: List[str],
    ) -> Dict[str, Any]:
        """递归生成掩码视图，并收集敏感字段点分路径。"""
        out: Dict[str, Any] = {}
        src = values if isinstance(values, dict) else {}
        for key, meta in schema.items():
            if not isinstance(meta, dict):
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            type_ = str(meta.get("type", "string"))
            value = src.get(key, _default_of(meta))
            if type_ == "object":
                out[key] = self._mask_tree(
                    meta.get("items") or {},
                    value if isinstance(value, dict) else {},
                    path,
                    sensitive,
                )
            elif type_ == "dict":
                child = copy.deepcopy(value) if isinstance(value, dict) else {}
                for sub_key in list(child):
                    if _is_sensitive(sub_key):
                        sensitive.append(f"{path}.{sub_key}")
                        child[sub_key] = MASK_DISPLAY if child[sub_key] not in (None, "") else ""
                out[key] = child
            else:
                if _is_sensitive(key) and type_ in ("string", "text"):
                    sensitive.append(path)
                    out[key] = MASK_DISPLAY if value not in (None, "") else ""
                else:
                    out[key] = value
        return out

    def view(self) -> Dict[str, Any]:
        """GET /config 的返回体：schema + 掩码后的值 + 敏感字段清单。"""
        schema = self.load_schema()
        raw = self.read_raw()
        values = self._fill_defaults(schema, raw)
        sensitive: List[str] = []
        masked = self._mask_tree(schema, values, "", sensitive)
        path = self.config_path()
        return {
            "schema": schema,
            "values": masked,
            "sensitive_paths": sensitive,
            "mask_display": MASK_DISPLAY,
            "mask_token": MASK_TOKEN,
            "config_file": path.name,  # 只给文件名，不泄漏绝对路径
            "updated_at": self._mtime(path),
            "reload_required_prefixes": list(RELOAD_REQUIRED_PREFIXES),
            "note": "保存后写入插件配置文件并热更新运行时；标注「需重载」的项在下次插件重载后完全生效（面板不会自动重载）。",
        }

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    # -- 写入 --------------------------------------------------------------
    def _apply(
        self,
        schema: Dict[str, Any],
        payload: Dict[str, Any],
        target: Dict[str, Any],
        current: Dict[str, Any],
        prefix: str,
        changes: List[str],
        errors: List[str],
        depth: int = 0,
    ) -> None:
        """按 schema 白名单递归套用 payload；未知键/非法值写入 errors。"""
        if depth > MAX_DEPTH:
            errors.append(f"{prefix or 'root'} 配置层级过深")
            return
        if not isinstance(payload, dict):
            errors.append(f"{prefix or 'root'} 必须是对象")
            return
        cur = current if isinstance(current, dict) else {}
        for key, value in payload.items():
            meta = schema.get(key)
            path = f"{prefix}.{key}" if prefix else str(key)
            if not isinstance(meta, dict):
                errors.append(f"未知配置项: {path}")
                continue
            type_ = str(meta.get("type", "string"))
            if type_ not in SUPPORTED_TYPES:
                errors.append(f"不支持的配置类型: {path} ({type_})")
                continue

            if type_ == "object":
                if not isinstance(value, dict):
                    errors.append(f"{path} 必须是对象")
                    continue
                child_cur = cur.get(key) if isinstance(cur.get(key), dict) else {}
                child_tgt = target.get(key) if isinstance(target.get(key), dict) else {}
                target[key] = child_tgt
                self._apply(
                    meta.get("items") or {}, value, child_tgt, child_cur,
                    path, changes, errors, depth + 1,
                )
                continue

            if type_ == "dict":
                # 自由映射（models.embedding / models.rerank）：只允许 default 中的键
                if not isinstance(value, dict):
                    errors.append(f"{path} 必须是对象")
                    continue
                allowed = meta.get("default") if isinstance(meta.get("default"), dict) else {}
                child_cur = cur.get(key) if isinstance(cur.get(key), dict) else {}
                child_tgt = copy.deepcopy(child_cur)
                for sub_key, sub_value in value.items():
                    sub_path = f"{path}.{sub_key}"
                    if allowed and sub_key not in allowed:
                        errors.append(f"未知配置项: {sub_path}")
                        continue
                    if sub_value is None or not isinstance(sub_value, (str, int, float, bool)):
                        errors.append(f"{sub_path} 只接受标量值")
                        continue
                    if isinstance(sub_value, str) and len(sub_value) > MAX_STRING_LEN:
                        errors.append(f"{sub_path} 字符串过长")
                        continue
                    if _is_sensitive(sub_key) and sub_value in (MASK_TOKEN, MASK_DISPLAY):
                        continue  # 掩码占位 → 保持原密钥
                    if child_cur.get(sub_key) != sub_value:
                        changes.append(sub_path)
                    child_tgt[sub_key] = sub_value
                target[key] = child_tgt
                continue

            normalized, error = _normalize_scalar(path, key, meta, value)
            if error:
                errors.append(error)
                continue
            if _is_sensitive(key) and type_ in ("string", "text") and normalized == MASK_TOKEN:
                continue  # 掩码占位 → 保持原密钥
            if normalized == MASK_DISPLAY and _is_sensitive(key):
                continue  # 前端直接回显掩码也算「保持」
            if key == "db_path":
                valid, reason = _validate_db_path(normalized)
                if not valid:
                    errors.append(f"{path}: {reason}")
                    continue
            if cur.get(key, _default_of(meta)) != normalized:
                changes.append(path)
            target[key] = normalized

    @staticmethod
    def _atomic_write(path: Path, data: Dict[str, Any]) -> None:
        """同目录临时文件 + os.replace 原子落盘，失败不留半截文件。"""
        parent = path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(f"配置目录不可写: {exc}") from exc
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, path)
            tmp_path = None
        except OSError as exc:
            raise ConfigError(f"写入插件配置失败: {exc}") from exc
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _sync_runtime(self, merged: Dict[str, Any]) -> List[str]:
        """把新配置同步到插件运行时（不 reload、不重启主进程）。"""
        applied: List[str] = []
        plugin = self.plugin
        if plugin is None:
            return applied
        try:
            runtime_cfg = getattr(plugin, "config", None)
            if isinstance(runtime_cfg, dict):
                runtime_cfg.clear()
                runtime_cfg.update(copy.deepcopy(merged))
                applied.append("runtime.config")
        except Exception as exc:  # pragma: no cover
            logger.warning("[agent_memory] 同步运行时 config 失败（已落盘）: %s", exc)
        applier = getattr(plugin, "apply_runtime_config", None)
        if callable(applier):
            try:
                result = applier(merged)
                if isinstance(result, dict):
                    applied.extend(str(x) for x in (result.get("applied") or []))
            except Exception as exc:  # pragma: no cover
                logger.warning("[agent_memory] 运行时热更新失败（已落盘）: %s", exc)
        return applied

    def save(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        POST /config/save：校验 → 合并 → 原子写 → 运行时同步。

        :raises ConfigError: 请求体非法或校验失败（不写任何文件）。
        """
        if not isinstance(payload, dict) or not payload:
            raise ConfigError("请求体为空（应提交待修改的配置项）")
        with self._lock:
            schema = self.load_schema()
            current = self.read_raw()
            merged = self._fill_defaults(schema, copy.deepcopy(current))
            changes: List[str] = []
            errors: List[str] = []
            self._apply(schema, payload, merged, current, "", changes, errors)
            if errors:
                raise ConfigError("配置校验未通过: " + "；".join(errors[:8]))

            self._atomic_write(self.config_path(), merged)
            applied = self._sync_runtime(merged)
            requires_reload = any(
                any(str(c).startswith(p.rstrip(".")) or str(c).startswith(p) for p in RELOAD_REQUIRED_PREFIXES)
                for c in changes
            )
            result = {
                "changed": sorted(set(changes)),
                "applied": applied,
                "requires_reload": bool(requires_reload),
                "config_file": self.config_path().name,
                "saved_at": time.time(),
                "note": (
                    "已写入配置文件并热更新运行时；标注「需重载」的项请稍后在插件页重载后完全生效。"
                    if requires_reload
                    else "已写入配置文件并即时生效。"
                ),
            }
            logger.info(
                "[agent_memory] 面板保存配置成功: %d 项变更, requires_reload=%s",
                len(result["changed"]), result["requires_reload"],
            )
            return result

    def validate_schema_compatibility(self) -> Dict[str, Any]:
        """
        供单测/自检使用：校验 schema 是否落在 AstrBot 渲染器安全子集内。

        检查项：type 白名单、object 必带 items、无二级嵌套 object、字段齐全。
        """
        schema = self.load_schema()
        issues: List[str] = []
        nested: List[str] = []
        for path, meta in _walk_schema(schema):
            type_ = str(meta.get("type", ""))
            if type_ not in SUPPORTED_TYPES:
                issues.append(f"{path}: 不支持的 type={type_!r}")
            if "default" not in meta and type_ != "object":
                issues.append(f"{path}: 缺少 default")
            if type_ == "object":
                if not isinstance(meta.get("items"), dict):
                    issues.append(f"{path}: object 缺少 items")
                for sub_path, sub_meta in _walk_schema(meta.get("items") or {}, path, 1):
                    if sub_meta.get("type") == "object":
                        nested.append(sub_path)
        if nested:
            issues.append("存在二级嵌套 object: " + ", ".join(nested))
        return {"ok": not issues, "issues": issues, "nested_objects": nested}


__all__ = [
    "ConfigError",
    "ConfigManager",
    "MASK_DISPLAY",
    "MASK_TOKEN",
    "PLUGIN_NAME",
    "_validate_db_path",
]
