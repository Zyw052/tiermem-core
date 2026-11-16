# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 系统噪声过滤 (noise_filter.py)

背景（P0 改造第 3 项）：
    分层记忆的写入/抽取入口会把「宿主机/其它插件注入到上下文的系统块」误当成
    人类对话内容，例如：

        Background task for subagent 'xxx' finished.
        【项目大盘·跨代理共享状态】...
        <RAG-Faiss-Memory>...</RAG-Faiss-Memory>
        <system_reminder>...</system_reminder>
        [SYSTEM NOTICE] ...

    这些噪声一旦进入 L1/L2，会污染长期记忆（并被反复召回、反复总结）。

设计目标：
    1. 纯函数、零依赖、零 IO：可被写入入口、抽取入口、单测随意调用；
    2. 只做「减法」：能识别的系统块整体剔除，行级噪声逐行剔除，其余原样保留；
    3. 永不抛异常：任何正则/编码异常都退化为「原文放行」，绝不阻断记忆主链路；
    4. 可观测：记录命中次数与最近被拦截样本，供 WebUI「噪声排查」页展示。

使用方式（推荐在写入入口统一调用）::

    from noise_filter import get_noise_filter

    nf = get_noise_filter()
    verdict = nf.inspect(text)         # -> NoiseVerdict
    if verdict.is_noise:               # 整段都是噪声 → 直接丢弃，不入库
        return False
    text = verdict.cleaned_text        # 清洗后的正文（可能等于原文）
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Pattern, Tuple

try:  # 宿主日志；脱离宿主时退化为标准 logging
    from astrbot.api import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("astrbot_plugin_agent_memory.noise_filter")


# ---------------------------------------------------------------------------
# 1. 整块噪声：成对标签 / 分隔符包裹的系统注入块（含内容整体剔除）
# ---------------------------------------------------------------------------
# 说明：
# - 标签名统一 ASCII 化并做宽松匹配（下划线/连字符/空格/大小写均可）；
# - 使用 DOTALL 让内容跨行；使用非贪婪避免吞掉后续正常文本；
# - 未闭合的起始标签单独兜底（见 _UNCLOSED_BLOCK_PATTERNS）。
_BLOCK_TAGS: Tuple[str, ...] = (
    "RAG-Faiss-Memory",
    "RAG-Faiss-Memories",
    "long_term_memory",
    "system_reminder",
    "system-reminder",
    "system_prompt",
    "system",
    "tools",
    "tool_result",
    "tool-call",
    "function_results",
    "function_result",
    "agent_memory",
    "memory_context",
    "recall_context",
    "knowledge_context",
    "project_board",
    "subagent_result",
    "background_task",
    "thinking",
    "scratchpad",
)


def _tag_pattern(tag: str) -> str:
    """把标签名转换成宽松匹配的正则片段（下划线/连字符可互换）。"""
    parts: List[str] = []
    for ch in tag:
        if ch in "-_ ":
            parts.append(r"[-_\s]?")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def _build_block_patterns() -> List[Pattern[str]]:
    patterns: List[Pattern[str]] = []
    for tag in _BLOCK_TAGS:
        relaxed = _tag_pattern(tag)
        patterns.append(
            re.compile(
                rf"<{relaxed}\b[^>]*>.*?</{relaxed}\s*>",
                re.IGNORECASE | re.DOTALL,
            )
        )
    # 中文方头括号包裹的系统块：【项目大盘·跨代理共享状态】……【/项目大盘…】
    patterns.append(
        re.compile(
            r"【[^】]{0,40}(项目大盘|跨代理共享状态|系统提示|系统注入)[^】]{0,40}】"
            r".*?(?:【\s*/\s*[^】]{0,40}】|$)",
            re.IGNORECASE | re.DOTALL,
        )
    )
    # Markdown 注释块：宿主的记忆注入占位标记形如 <!-- agent_memory:injected:v1 -->
    patterns.append(re.compile(r"<!--.*?-->", re.IGNORECASE | re.DOTALL))
    # 围栏代码块中的 system 说明（极少数插件会这样注入）
    patterns.append(
        re.compile(r"```(?:system|sys)[^\n]*\n.*?```", re.IGNORECASE | re.DOTALL)
    )
    return patterns


#: 未闭合的起始标签：从标签起到文本结束全部剔除（宁可少记，不可记噪声）
_UNCLOSED_BLOCK_PATTERNS: List[Pattern[str]] = [
    re.compile(
        rf"<{_tag_pattern(tag)}\b[^>]*>.*\Z",
        re.IGNORECASE | re.DOTALL,
    )
    for tag in _BLOCK_TAGS
]


# ---------------------------------------------------------------------------
# 2. 行级噪声：单行即可判定的系统回调 / 后台任务通知 / 调试残留
# ---------------------------------------------------------------------------
_LINE_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    # 子代理后台任务完成回调（主人明确点名的高频噪声）
    (
        "subagent_background_task",
        re.compile(
            r"background\s+task\s+for\s+(subagent|sub[-_ ]?agent|agent|task)"
            r"\s*['\"“”]?[^'\"“”\n]{0,120}['\"“”]?\s*"
            r"(finished|completed|done|failed|cancelled|canceled|started)\b",
            re.IGNORECASE,
        ),
    ),
    # [SYSTEM NOTICE] / [system] / [INFO] 之类的系统提示行
    (
        "system_notice",
        re.compile(
            r"^\s*[\[【(（]\s*(system|sys|notice|系统|系统提示|system notice|"
            r"system notice|reminder)\s*[\]】)）]",
            re.IGNORECASE,
        ),
    ),
    # 跨代理共享状态（项目大盘）
    (
        "project_board",
        re.compile(
            r"^\s*[\[【]?\s*(项目大盘|跨代理共享状态|共享状态|project\s+board)",
            re.IGNORECASE,
        ),
    ),
    # 记忆/RAG 检索结果回灌标记
    (
        "rag_marker",
        re.compile(
            r"^\s*[\[【<]?\s*(RAG[-_ ]?Faiss[-_ ]?Memor(y|ies)|"
            r"long[-_ ]term[-_ ]memory|recall[-_ ]context|"
            r"agent[-_ ]memory)\b",
            re.IGNORECASE,
        ),
    ),
    # 后台任务/异步任务流水账
    (
        "background_task_line",
        re.compile(
            r"^\s*[\[【]?\s*(background\s+task|async\s+task|task\s+queue|"
            r"worker\s+task)\b.*\b(finished|completed|done|failed|retry)",
            re.IGNORECASE,
        ),
    ),
    # 工具调用回显（function/tool 调用结果块）
    (
        "tool_echo",
        re.compile(
            r"^\s*[\[【]?\s*(tool[_ ]?call|function[_ ]?call|tool[_ ]?result)\b",
            re.IGNORECASE,
        ),
    ),
    # 宿主调试/框架日志误入上下文
    (
        "framework_log",
        re.compile(
            r"^\s*\[(?:\d{2}:\d{2}:\d{2}|\d{4}-\d{2}-\d{2})[^\]]*\]\s*"
            r"\[(?:Core|Plug|DBUG|INFO|WARN|ERROR|DEBUG)\]",
            re.IGNORECASE,
        ),
    ),
]

#: 全角/半角空白与零宽字符（清洗时统一清理）
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

#: 判定「清洗后仍为噪声」的最小有效字符数（中文按字符，英文按字符）
_MIN_MEANINGFUL_CHARS = 2


@dataclass
class NoiseVerdict:
    """一次噪声判定结果（不可变语义，便于日志与单测断言）。"""

    original: str = ""
    cleaned_text: str = ""
    is_noise: bool = False
    """整段是否判定为纯噪声（清洗后无有效内容）。"""
    hits: List[str] = field(default_factory=list)
    """命中的规则名列表，便于定位是哪类噪声。"""
    removed_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_noise": self.is_noise,
            "hits": list(self.hits),
            "removed_chars": self.removed_chars,
            "original_len": len(self.original),
            "cleaned_len": len(self.cleaned_text),
            "preview": self.original[:120],
            "cleaned_preview": self.cleaned_text[:120],
        }


class NoiseFilter:
    """系统噪声过滤器（进程内单例，带轻量统计与最近样本环形缓冲）。"""

    #: 最近被拦截样本的保留条数（供 WebUI 噪声排查页）
    RECENT_LIMIT = 50

    def __init__(self) -> None:
        self._block_patterns: List[Pattern[str]] = _build_block_patterns()
        self._line_patterns: List[Tuple[str, Pattern[str]]] = list(_LINE_PATTERNS)
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=self.RECENT_LIMIT)
        #: 规则名 -> 命中次数
        self.stats: Dict[str, int] = {
            "inspected": 0,
            "dropped": 0,
            "cleaned": 0,
            "passed": 0,
        }

    # -- 核心 API ---------------------------------------------------------
    def inspect(self, text: Any) -> NoiseVerdict:
        """判定并清洗一段文本。任何异常都退化为「原文放行」。"""
        raw = "" if text is None else str(text)
        try:
            self.stats["inspected"] += 1
            cleaned, hits = self._clean(raw)
            meaningful = self._meaningful_length(cleaned)
            is_noise = meaningful < _MIN_MEANINGFUL_CHARS
            verdict = NoiseVerdict(
                original=raw,
                cleaned_text="" if is_noise else cleaned,
                is_noise=is_noise,
                hits=hits,
                removed_chars=max(0, len(raw) - len(cleaned)),
            )
            self._record(verdict)
            return verdict
        except Exception as exc:  # pragma: no cover - 防御式兜底
            logger.debug("[NoiseFilter] 判定异常，原文放行: %s", exc)
            return NoiseVerdict(original=raw, cleaned_text=raw)

    def is_noise(self, text: Any) -> bool:
        """快捷判定：整段是否纯噪声。"""
        return self.inspect(text).is_noise

    def clean(self, text: Any) -> str:
        """快捷清洗：返回清洗后的正文（纯噪声时返回空串）。"""
        return self.inspect(text).cleaned_text

    def describe(self) -> Dict[str, Any]:
        """返回统计快照（供 /memory status 与 WebUI 使用）。"""
        return {
            "stats": dict(self.stats),
            "block_patterns": [
                getattr(p, "pattern", "")[:80] for p in self._block_patterns
            ],
            "line_rules": [name for name, _ in self._line_patterns],
            "recent": list(self._recent),
        }

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """返回最近被拦截/清洗的样本（最多 limit 条，最新在前）。"""
        try:
            n = max(1, int(limit))
        except (TypeError, ValueError):
            n = 20
        return list(self._recent)[:n]

    # -- 内部实现 ---------------------------------------------------------
    def _clean(self, text: str) -> Tuple[str, List[str]]:
        """执行逐层清洗，返回 (清洗后文本, 命中规则名列表)。"""
        if not text:
            return "", []

        hits: List[str] = []
        working = _INVISIBLE_RE.sub("", text)

        # 1) 成对标签整体剔除
        for pattern in self._block_patterns:
            new_working, count = pattern.subn(" ", working)
            if count:
                hits.append(f"block:{getattr(pattern, 'pattern', '')[:40]}")
                working = new_working

        # 2) 未闭合标签兜底剔除
        for pattern in _UNCLOSED_BLOCK_PATTERNS:
            new_working, count = pattern.subn(" ", working)
            if count:
                hits.append("block:unclosed")
                working = new_working

        # 3) 行级剔除
        kept_lines: List[str] = []
        for line in working.splitlines():
            stripped = line.strip()
            if not stripped:
                kept_lines.append("")
                continue
            hit_name = ""
            for rule_name, pattern in self._line_patterns:
                if pattern.search(stripped):
                    hit_name = rule_name
                    break
            if hit_name:
                hits.append(f"line:{hit_name}")
                continue
            kept_lines.append(line)

        working = "\n".join(kept_lines)

        # 4) 归一化空白：连续空行折叠为单个换行，首尾空白裁剪
        working = re.sub(r"[ \t]+", " ", working)
        working = re.sub(r"\n{3,}", "\n\n", working).strip()

        # 5) 去重 hits，保持顺序
        deduped: List[str] = []
        for item in hits:
            if item not in deduped:
                deduped.append(item)
        return working, deduped

    @staticmethod
    def _meaningful_length(text: str) -> int:
        """统计有效字符数（忽略空白与常见标点）。"""
        if not text:
            return 0
        meaningful = re.sub(r"[\s\u3000.,;:!?，。；：！？、·\-—_=*#`'\"<>/\\|()\[\]{}]+", "", text)
        return len(meaningful)

    def _record(self, verdict: NoiseVerdict) -> None:
        """更新统计并保留最近样本。"""
        if verdict.is_noise:
            self.stats["dropped"] += 1
        elif verdict.removed_chars > 0:
            self.stats["cleaned"] += 1
        else:
            self.stats["passed"] += 1

        if not verdict.hits:
            return
        self._recent.appendleft(
            {
                "ts": time.time(),
                "is_noise": verdict.is_noise,
                "hits": list(verdict.hits),
                "removed_chars": verdict.removed_chars,
                "preview": verdict.original[:200],
                "cleaned_preview": verdict.cleaned_text[:200],
            }
        )


# ---------------------------------------------------------------------------
# 3. 进程级单例（写入入口/抽取入口共用同一份统计）
# ---------------------------------------------------------------------------
_INSTANCE: NoiseFilter | None = None


def get_noise_filter() -> NoiseFilter:
    """获取进程内共享的 NoiseFilter 实例（首次调用时惰性创建）。"""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = NoiseFilter()
    return _INSTANCE


def is_noise(text: Any) -> bool:
    """模块级快捷函数：整段是否纯噪声。"""
    return get_noise_filter().is_noise(text)


def clean_text(text: Any) -> str:
    """模块级快捷函数：清洗正文。"""
    return get_noise_filter().clean(text)


def inspect_text(text: Any) -> NoiseVerdict:
    """模块级快捷函数：返回完整判定结果。"""
    return get_noise_filter().inspect(text)


def filter_turn(
    user_text: Any,
    assistant_text: Any,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    对一轮对话做统一清洗（写入入口推荐入口函数）。

    返回 ``(clean_user, clean_assistant, meta)``：
    - 两侧都被判定为纯噪声时，``meta["is_noise"]`` 为 True，调用方应丢弃该轮；
    - 任一侧被部分清洗时，``meta["cleaned"]`` 为 True，并给出命中规则名。
    """
    nf = get_noise_filter()
    u_verdict = nf.inspect(user_text)
    a_verdict = nf.inspect(assistant_text)

    meta: Dict[str, Any] = {
        "is_noise": u_verdict.is_noise and a_verdict.is_noise,
        "cleaned": bool(u_verdict.removed_chars or a_verdict.removed_chars),
        "hits": sorted(set(u_verdict.hits) | set(a_verdict.hits)),
        "removed_chars": u_verdict.removed_chars + a_verdict.removed_chars,
    }
    return u_verdict.cleaned_text, a_verdict.cleaned_text, meta


def reset_for_tests() -> None:
    """仅供单测：重置全局单例。"""
    global _INSTANCE
    _INSTANCE = None


__all__ = [
    "NoiseFilter",
    "NoiseVerdict",
    "clean_text",
    "filter_turn",
    "get_noise_filter",
    "inspect_text",
    "is_noise",
    "reset_for_tests",
]
