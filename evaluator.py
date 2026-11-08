# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 自作判定门禁 Evaluator Gate (evaluator.py)

职责（在 L2 原子事实入库前做一轮“廉价模型自作判定”）：
1. **信息价值打分 (value_score)**：过滤寒暄、语气词与低价值废话，避免污染 L2；
2. **事实冲突与推翻检测 (conflict_ids)**：把“我不喜欢喝绿茶”这类与旧事实矛盾的新事实
   识别出来，并给出需要更新/废弃的旧 atom_id；
3. **异常降级（容错回退）**：判官模型调用失败、超时、返回非法 JSON 时，
   自动平滑回退到本地规则判定（fail-open 保留事实），绝不阻断主记忆流程。

设计原则：
- 判官模型（默认 ``jev``）只是“廉价加速器”，规则判定永远是可用底座；
- 本模块不直接写库，只产出判定报告，由 summarizer 负责落库与废弃标记；
- 所有对外方法均为异常安全，不向上抛异常。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover
    logger = logging.getLogger("astrbot_plugin_agent_memory.evaluator")


# ---------------------------------------------------------------------------
# 常量与词表
# ---------------------------------------------------------------------------

DEFAULT_EVALUATOR_MODEL: str = "jev"

#: 明显无长期价值的寒暄/应答/语气填充（整句命中即判为低价值）
CHATTER_PHRASES: Tuple[str, ...] = (
    "你好",
    "您好",
    "在吗",
    "在不在",
    "谢谢",
    "多谢",
    "感谢",
    "好的",
    "好吧",
    "行吧",
    "收到",
    "明白了",
    "知道了",
    "了解",
    "嗯",
    "嗯嗯",
    "哦",
    "哦哦",
    "啊",
    "哈哈",
    "哈哈哈",
    "嘿嘿",
    "晚安",
    "早安",
    "午安",
    "再见",
    "拜拜",
    "没事",
    "没关系",
    "不客气",
    "稍等",
    "我来了",
    "辛苦了",
    "你说得对",
    "确实",
)

#: 具备长期记忆价值的高信号词（出现即显著加分）
MEMORY_SIGNAL_KEYWORDS: Tuple[str, ...] = (
    "我叫",
    "我是",
    "我的",
    "我喜欢",
    "我不喜欢",
    "我讨厌",
    "我偏好",
    "我习惯",
    "我住",
    "我在",
    "我养",
    "我的生日",
    "生日",
    "年龄",
    "手机",
    "电话",
    "邮箱",
    "地址",
    "账号",
    "密码",
    "工作",
    "职业",
    "公司",
    "学校",
    "专业",
    "家人",
    "女友",
    "男友",
    "老婆",
    "老公",
    "孩子",
    "宠物",
    "过敏",
    "禁忌",
    "必须",
    "严禁",
    "记住",
    "以后",
    "从此",
    "改成",
    "改为",
    "不再是",
    "换成了",
)

#: 否定/反义极性词
NEGATION_TOKENS: Tuple[str, ...] = (
    "不",
    "没",
    "无",
    "别",
    "勿",
    "非",
    "不再",
    "不会",
    "不想",
    "取消",
)

#: 明确的“推翻/更新”语义标记
REPLACEMENT_MARKERS: Tuple[str, ...] = (
    "改成",
    "改为",
    "换成",
    "换成了",
    "不再是",
    "现在已经",
    "现在是",
    "以后不",
    "我不再",
    "已经不用",
    "已经戒了",
    "搬家到",
    "调到",
)

#: 只用于检索冲突候选的属性槽关键词（同槽不同值 = 潜在冲突）
ATTRIBUTE_SLOTS: Tuple[str, ...] = (
    "住",
    "住在",
    "叫",
    "名字",
    "生日",
    "年龄",
    "手机",
    "电话",
    "邮箱",
    "地址",
    "工作",
    "公司",
    "学校",
    "专业",
    "喜欢",
    "讨厌",
    "偏好",
    "习惯",
    "养",
)

_SENT_SPLIT_RE = re.compile(r"[。！？!?\n；;]+")
_PUNCT_RE = re.compile(r"[\s，。！？、；：,.!?;:~～\-—_\"'“”‘’()（）\[\]【】]+")
#: 寒暄句尾常见语气助词（剥离后若为空即视为纯寒暄）
_CHATTER_PARTICLES_RE = re.compile(r"[呀啊呢吧哦啦哈嘛喔噢咯哇诶欸的了]+")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class GateReport:
    """一次自作判定的完整结果，供调用方落库与可观测性使用。"""

    kept: List[Dict[str, Any]] = field(default_factory=list)
    dropped: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "rule"          # rule | llm | fallback | disabled
    degraded: bool = False        # True 表示判官模型失败已降级
    error: str = ""               # 降级原因（便于排障）
    latency_ms: float = 0.0

    def summary(self) -> Dict[str, Any]:
        """返回精简的可观测性快照（不含完整事实文本）。"""
        return {
            "kept": len(self.kept),
            "dropped": len(self.dropped),
            "conflicts": len(self.conflicts),
            "source": self.source,
            "degraded": self.degraded,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """去除空白与标点，用于相似度与极性比较。"""
    return _PUNCT_RE.sub("", str(text or "")).lower()


def _char_bigram_jaccard(left: str, right: str) -> float:
    """字符二元组 Jaccard 相似度：无需外部分词，中文场景足够稳健。"""
    a = _normalize_text(left)
    b = _normalize_text(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) < 2 or len(b) < 2:
        return 0.0
    grams_a = {a[i : i + 2] for i in range(len(a) - 1)}
    grams_b = {b[i : i + 2] for i in range(len(b) - 1)}
    if not grams_a or not grams_b:
        return 0.0
    inter = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return inter / union if union else 0.0


def _has_negation(text: str) -> bool:
    """判断文本是否含否定极性（仅在否定词位于句首或关键词前时生效）。"""
    normalized = _normalize_text(text)
    for token in NEGATION_TOKENS:
        if token in normalized:
            # 排除“不错/不少/非常/特别”等非否定用法
            if token == "不" and (
                "不错" in normalized
                or "不少" in normalized
                or "不久" in normalized
                or "不好意思" in normalized
            ):
                continue
            return True
    return False


#: 共享二元组中的高频虚词/代词，不构成“同一实体”证据
_STOP_BIGRAMS = frozenset(
    {
        "我们", "你们", "他们", "她们", "我的", "你的", "他的", "我是", "你是",
        "现在", "已经", "一个", "什么", "这个", "那个", "就是", "还是", "可以",
        "因为", "所以", "但是", "然后", "如果", "不要", "没有", "不是", "知道",
        "觉得", "时候", "一点", "有点", "真的", "非常", "特别",
    }
)


def _strip_negation(normalized_text: str) -> str:
    """移除否定词，得到“核心谓词”，用于判断极性翻转是否针对同一命题。"""
    core = normalized_text
    for token in sorted(NEGATION_TOKENS, key=len, reverse=True):
        core = core.replace(_normalize_text(token), "")
    return core


def _shared_entities(left: str, right: str) -> List[str]:
    """提取两段文本共享的、非虚词的中文二元组（视为同一实体证据）。"""
    a = _normalize_text(left)
    b = _normalize_text(right)
    if len(a) < 2 or len(b) < 2:
        return []
    grams_a = {a[i : i + 2] for i in range(len(a) - 1)}
    grams_b = {b[i : i + 2] for i in range(len(b) - 1)}
    shared = [g for g in (grams_a & grams_b) if g not in _STOP_BIGRAMS]
    return shared


def _shares_meaningful_entity(left: str, right: str) -> bool:
    """两段文本是否共享至少一个非虚词实体（用于撤销/搬家等推翻语义判定）。"""
    return bool(_shared_entities(left, right))


def _is_chatter(text: str) -> bool:
    """识别寒暄/应答/纯语气填充。"""
    normalized = _normalize_text(text)
    if not normalized:
        return True
    # 纯重复字符（如 哈哈哈哈哈 / 嗯嗯嗯）
    if len(set(normalized)) <= 2 and len(normalized) <= 8:
        return True
    for phrase in CHATTER_PHRASES:
        if normalized == _normalize_text(phrase):
            return True
    # 短句且仅由寒暄词 + 语气尾缀拼成（例如“好的谢谢”“你好呀”“嗯嗯呢”）
    if len(normalized) <= 10:
        residual = normalized
        for phrase in sorted(CHATTER_PHRASES, key=len, reverse=True):
            residual = residual.replace(_normalize_text(phrase), "")
        residual = _CHATTER_PARTICLES_RE.sub("", residual)
        if not residual:
            return True
    return False


# ---------------------------------------------------------------------------
# 判官门禁
# ---------------------------------------------------------------------------


class EvaluatorGate:
    """
    廉价模型自作判定门禁。

    参数：
        config:     插件运行时配置（读取 ``config["evaluator"]``，并回退 summarizer 凭证）
        llm_client: 可选的异步调用桩，签名 ``async (messages, model) -> str``；
                    默认使用 aiohttp 直连 OpenAI 兼容端点。便于单测注入与替换。
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Callable[[List[Dict[str, str]], str], Awaitable[str]]] = None,
    ) -> None:
        self.config: Dict[str, Any] = config if isinstance(config, dict) else {}
        cfg = self.config.get("evaluator")
        cfg = cfg if isinstance(cfg, dict) else {}
        summarizer_cfg = self.config.get("summarizer")
        summarizer_cfg = summarizer_cfg if isinstance(summarizer_cfg, dict) else {}

        # ---- 开关 ----
        self.enabled: bool = _safe_bool(cfg.get("enable"), True)
        self.conflict_check: bool = _safe_bool(cfg.get("conflict_check"), True)
        #: 判官失败时是否保留事实（fail-open）。默认 True：宁多记，不丢记忆。
        self.fail_open: bool = _safe_bool(cfg.get("fail_open"), True)

        # ---- 模型凭证（支持留空回退主总结模型）----
        self.api_base: str = str(
            cfg.get("api_base")
            or self.config.get("llm_api_base")
            or summarizer_cfg.get("api_base")
            or ""
        ).strip().rstrip("/")
        self.api_key: str = str(
            cfg.get("api_key")
            or self.config.get("llm_api_key")
            or summarizer_cfg.get("api_key")
            or ""
        ).strip()
        #: 判官模型名：优先 evaluator.model_name，其次顶层，最后默认 jev
        self.model_name: str = str(
            cfg.get("model_name")
            or cfg.get("model")
            or self.config.get("evaluator_model")
            or DEFAULT_EVALUATOR_MODEL
        ).strip() or DEFAULT_EVALUATOR_MODEL

        # ---- 参数 ----
        self.timeout_seconds: float = _safe_float(cfg.get("timeout_seconds"), 8.0, 0.5)
        self.min_value_score: float = _clamp(
            _safe_float(cfg.get("min_value_score"), 0.35), 0.0, 1.0
        )
        #: 降级时阈值放宽，避免误伤（回退路径更保守地保留记忆）
        self.fallback_min_value_score: float = _clamp(
            _safe_float(cfg.get("fallback_min_value_score"), self.min_value_score * 0.5),
            0.0,
            1.0,
        )
        self.max_candidates: int = max(1, _safe_int(cfg.get("max_candidates"), 8))
        self.conflict_similarity: float = _clamp(
            _safe_float(cfg.get("conflict_similarity"), 0.5), 0.1, 1.0
        )

        self._llm_client = llm_client
        # R-06：判官默认走持久连接池，避免每次裁决重建 Session
        self._http: Optional[Any] = None
        try:
            from .models_client import _HttpSessionOwner  # 复用同一套连接池实现
        except ImportError:  # pragma: no cover - 兼容脚本式导入
            try:
                from models_client import _HttpSessionOwner  # type: ignore
            except ImportError:
                _HttpSessionOwner = None  # type: ignore
        if _HttpSessionOwner is not None:
            self._http = _HttpSessionOwner(self.timeout_seconds)

        # ---- 运行期统计（仅可观测性）----
        self.stats: Dict[str, Any] = {
            "gate_calls": 0,
            "llm_ok": 0,
            "llm_failed": 0,
            "rule_only": 0,
            "facts_dropped": 0,
            "conflicts_detected": 0,
        }

    # -- 配置快照（便于日志/测试断言）------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "has_api_base": bool(self.api_base),
            "has_api_key": bool(self.api_key),
            "timeout_seconds": self.timeout_seconds,
            "min_value_score": self.min_value_score,
            "conflict_check": self.conflict_check,
        }

    # -- 规则判定：信息价值 ------------------------------------------------
    def score_value(self, text: str) -> Tuple[float, str]:
        """
        规则信息价值打分（0.0~1.0）。

        返回 ``(score, reason)``，score < min_value_score 视为低价值废话。
        """
        raw = str(text or "").strip()
        if not raw:
            return 0.0, "empty"

        normalized = _normalize_text(raw)
        if _is_chatter(raw):
            return 0.1, "chatter"

        score = 0.5
        reasons: List[str] = ["base"]

        # 长度加成（过短的信息通常不构成稳定事实）
        if len(raw) >= 12:
            score += 0.10
            reasons.append("len>=12")
        if len(raw) >= 30:
            score += 0.10
            reasons.append("len>=30")
        if len(raw) < 6:
            score -= 0.15
            reasons.append("too_short")

        # 高信号记忆词
        hits = [kw for kw in MEMORY_SIGNAL_KEYWORDS if kw in raw]
        if hits:
            score += 0.30
            reasons.append("signal:" + ",".join(hits[:3]))

        # 纯提问、且无第一人称与记忆信号 → 低价值
        is_question = raw.endswith("？") or raw.endswith("?")
        if is_question and not hits and "我" not in raw:
            score = min(score, 0.25)
            reasons.append("pure_question")

        # 纯符号/纯数字
        if not re.search(r"[\u4e00-\u9fffA-Za-z]", normalized):
            score = min(score, 0.1)
            reasons.append("no_word")

        return round(_clamp(score, 0.0, 1.0), 4), "|".join(reasons)

    # -- 规则判定：冲突检测 ------------------------------------------------
    def detect_conflicts(
        self,
        fact_text: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        与候选旧事实逐条比对，识别“相互矛盾/被推翻”的旧原子。

        判定组合（任一成立即冲突）：
        1. 文本高相似（>= conflict_similarity）且否定极性相反；
        2. 文本高相似且包含明确推翻标记（改成/不再是/搬家到…）；
        3. 文本极高相似（>= 0.85）但内容不完全一致，且命中同一属性槽。
        """
        results: List[Dict[str, Any]] = []
        if not self.conflict_check:
            return results

        new_text = str(fact_text or "").strip()
        if not new_text:
            return results
        new_neg = _has_negation(new_text)
        new_has_replace = any(m in new_text for m in REPLACEMENT_MARKERS)
        new_slots = {s for s in ATTRIBUTE_SLOTS if s in new_text}

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            atom_id = str(cand.get("atom_id") or cand.get("id") or "").strip()
            old_text = str(cand.get("content") or cand.get("fact") or "").strip()
            if not atom_id or not old_text:
                continue
            # R-01 修复：L3 置顶记忆拥有“免死金牌”，规则层不得将其判为冲突/推翻目标。
            # 置顶代表用户明确要求长期记住的核心偏好或硬约束，任何自动判决都无权推翻。
            if int(cand.get("pinned") or 0) == 1:
                continue
            if _normalize_text(old_text) == _normalize_text(new_text):
                continue  # 完全相同不算冲突，交给余弦去重融合

            sim = _char_bigram_jaccard(new_text, old_text)
            old_neg = _has_negation(old_text)
            old_slots = {s for s in ATTRIBUTE_SLOTS if s in old_text}

            # 子串包含 + 极性相反：例如旧“我喜欢喝绿茶”，新“我不喜欢喝绿茶，改喝乌龙茶”
            # （分句未切开时相似度会被稀释，这里用包含关系兜住）
            old_norm = _normalize_text(old_text)
            new_norm = _normalize_text(new_text)
            # 剥离否定词后比较核心谓词：旧“我喜欢喝绿茶” vs 新“我不喜欢喝绿茶，改喝乌龙茶”
            core_old = _strip_negation(old_norm)
            core_new = _strip_negation(new_norm)
            core_match = bool(core_old) and (
                core_old in core_new or core_new in core_old
            )
            if (new_neg != old_neg) and (
                (old_norm and old_norm in new_norm) or core_match
            ):
                results.append(
                    {
                        "atom_id": atom_id,
                        "fact": new_text,
                        "old_content": old_text,
                        "reason": f"polarity_flip(core_match={core_match},sim={sim:.2f})",
                        "similarity": round(sim, 4),
                    }
                )
                continue

            # 推翻语义优先：即使相似度不高，只要带明确推翻标记且共享同一实体，也算冲突
            entity_shared = new_has_replace and _shares_meaningful_entity(new_text, old_text)
            if sim < self.conflict_similarity and not entity_shared:
                continue

            reason = ""
            if sim >= self.conflict_similarity and (new_neg != old_neg):
                reason = f"polarity_flip(sim={sim:.2f})"
            elif new_has_replace and _shares_meaningful_entity(new_text, old_text):
                reason = f"replacement_marker(shared_entity,sim={sim:.2f})"
            elif new_has_replace and sim >= self.conflict_similarity * 0.8:
                reason = f"replacement_marker(sim={sim:.2f})"
            elif sim >= 0.85 and new_slots and (new_slots & old_slots):
                reason = f"same_slot_value_changed(sim={sim:.2f})"

            if reason:
                results.append(
                    {
                        "atom_id": atom_id,
                        "fact": new_text,
                        "old_content": old_text,
                        "reason": reason,
                        "similarity": round(sim, 4),
                    }
                )

        # 同一旧原子只保留一次，且限制冲突数量避免雪崩
        dedup: Dict[str, Dict[str, Any]] = {}
        for item in results:
            dedup.setdefault(item["atom_id"], item)
        return list(dedup.values())[: self.max_candidates]

    # -- 规则回退路径 ------------------------------------------------------
    def _rule_gate(
        self,
        facts: List[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> GateReport:
        """纯规则判定（无外部依赖，永远是可用底座）。"""
        limits = self.min_value_score if threshold is None else threshold
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []

        for item in facts:
            fact_text = str(item.get("fact", "")).strip()
            if not fact_text:
                continue
            score, reason = self.score_value(fact_text)
            enriched = dict(item)
            enriched["value_score"] = round(score, 4)
            enriched["value_reason"] = reason
            if score < limits:
                dropped.append(enriched)
                continue
            kept.append(enriched)
            if self.conflict_check:
                conflicts.extend(self.detect_conflicts(fact_text, candidates))

        return GateReport(kept=kept, dropped=dropped, conflicts=conflicts, source="rule")

    # -- 判官模型调用 ------------------------------------------------------
    def _build_endpoint(self) -> str:
        base = (self.api_base or "").strip().rstrip("/")
        if not base:
            return ""
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    async def _default_llm_call(
        self, messages: List[Dict[str, str]], model: str
    ) -> str:
        """默认判官调用：OpenAI 兼容 /chat/completions。"""
        if aiohttp is None:
            raise RuntimeError("aiohttp 不可用，无法调用判官模型")
        endpoint = self._build_endpoint()
        if not endpoint:
            raise RuntimeError("未配置 evaluator api_base")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        # R-06：优先复用持久连接池；极少数场景（未成功装配）才回退临时 Session
        if self._http is not None:
            session = await self._http.get()
            async with session.post(
                endpoint, headers=headers, json=payload, timeout=timeout
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"judge http {resp.status}: {body[:200]}")
                data = await resp.json()
                return (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                )

        async with aiohttp.ClientSession(timeout=timeout) as session:  # pragma: no cover
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"judge http {resp.status}: {body[:200]}")
                data = await resp.json()
                return (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    or ""
                )

    async def close(self) -> None:
        """释放判官 HTTP 连接池（插件停用 / 测试清理时调用）。"""
        if self._http is not None:
            await self._http.close()

    def _build_prompt(
        self,
        facts: List[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
    ) -> str:
        fact_lines = "\n".join(
            f"{idx}. {str(item.get('fact', '')).strip()}"
            for idx, item in enumerate(facts, 1)
            if str(item.get("fact", "")).strip()
        )
        cand_lines = "\n".join(
            f"- id={c.get('atom_id')} | 内容={str(c.get('content', ''))[:120]}"
            for c in candidates[: self.max_candidates]
            if isinstance(c, dict)
        ) or "（无历史事实）"

        return (
            "你是长期记忆入库前的判定官，只输出 JSON，不要解释。\n"
            "任务：\n"
            "1) 为每条【待判定事实】打 0~1 的信息价值分 value_score"
            "（寒暄、寒暄式应答、无信息量废话应 <=0.2；稳定的个人信息/偏好/约束应 >=0.7）；\n"
            "2) 与【历史事实】逐条比对，若新事实与某条历史事实矛盾或推翻它，"
            "则输出该历史事实的 id 到 conflict_id；\n"
            "3) 不得修改事实原文。\n\n"
            "输出格式：\n"
            '{"keep":[{"fact":"原文","value_score":0.8,"reason":"简短理由"}],'
            '"drop":[{"fact":"原文","value_score":0.1,"reason":"简短理由"}],'
            '"conflicts":[{"fact":"原文","conflict_id":"atom_xxx","reason":"矛盾点"}]}\n\n'
            f"【待判定事实】\n{fact_lines}\n\n"
            f"【历史事实】\n{cand_lines}\n"
        )

    @staticmethod
    def _extract_json_object(content: str) -> Optional[Dict[str, Any]]:
        """从模型输出中稳健提取第一个 JSON 对象（容忍 markdown 代码块与前后废话）。"""
        text = str(content or "").strip()
        if not text:
            return None
        # 去掉 ```json ... ``` 包裹
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        # 退化：截取最外层花括号
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    def _merge_llm_verdict(
        self,
        rule_report: GateReport,
        verdict: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
    ) -> GateReport:
        """
        把判官模型结论合并进规则结果：
        - value_score 以模型结果为准（但规则判定为寒暄的仍强制丢弃）；
        - conflicts 仅接受能在候选集中找到的 atom_id，防止模型臆造。
        """
        valid_ids = {
            str(c.get("atom_id")).strip()
            for c in candidates
            if isinstance(c, dict) and c.get("atom_id")
        }

        llm_scores: Dict[str, float] = {}
        llm_reasons: Dict[str, str] = {}
        for bucket in ("keep", "drop"):
            items = verdict.get(bucket)
            if not isinstance(items, list):
                continue
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                fact_text = _normalize_text(raw.get("fact", ""))
                if not fact_text:
                    continue
                score = raw.get("value_score")
                if score is None:
                    score = raw.get("score")
                if score is None:
                    continue
                try:
                    llm_scores[fact_text] = _clamp(float(score), 0.0, 1.0)
                except (TypeError, ValueError):
                    continue
                if raw.get("reason"):
                    llm_reasons[fact_text] = str(raw["reason"])[:120]

        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = list(rule_report.dropped)

        for item in rule_report.kept:
            fact_text = str(item.get("fact", "")).strip()
            key = _normalize_text(fact_text)
            enriched = dict(item)
            final_score = enriched.get("value_score", 0.5)
            if key in llm_scores:
                final_score = llm_scores[key]
                enriched["value_reason"] = llm_reasons.get(key, "llm")
            # 规则判定的寒暄仍然强制丢弃（廉价模型的兜底保险）
            forced_chatter = enriched.get("value_reason") == "chatter"
            if final_score < self.min_value_score or forced_chatter:
                enriched["value_score"] = round(float(final_score), 4)
                enriched["drop_source"] = "llm" if key in llm_scores else "rule"
                dropped.append(enriched)
                continue
            enriched["value_score"] = round(float(final_score), 4)
            kept.append(enriched)

        # 处理模型额外要求丢弃、但规则未产出的低分事实
        for item in list(kept):
            key = _normalize_text(item.get("fact", ""))
            if key in llm_scores and llm_scores[key] < self.min_value_score:
                kept.remove(item)
                dropped.append(item)

        conflicts: List[Dict[str, Any]] = list(rule_report.conflicts)
        seen_keys = {c.get("atom_id") for c in conflicts if isinstance(c, dict)}
        raw_conflicts = verdict.get("conflicts")
        if isinstance(raw_conflicts, list):
            for raw in raw_conflicts:
                if not isinstance(raw, dict):
                    continue
                cid = str(raw.get("conflict_id") or raw.get("atom_id") or "").strip()
                if not cid or cid not in valid_ids or cid in seen_keys:
                    continue
                seen_keys.add(cid)
                conflicts.append(
                    {
                        "atom_id": cid,
                        "fact": str(raw.get("fact", "")).strip(),
                        "reason": f"llm:{str(raw.get('reason', 'conflict'))[:80]}",
                        "similarity": None,
                    }
                )

        return GateReport(
            kept=kept,
            dropped=dropped,
            conflicts=conflicts[: self.max_candidates],
            source="llm",
            degraded=False,
        )

    # -- 主入口 ------------------------------------------------------------
    async def gate(
        self,
        session_id: str,
        facts: Sequence[Dict[str, Any]],
        candidates: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> GateReport:
        """
        对一批待入库事实执行判定门禁。

        异常安全：任何错误都会降级为规则判定（或 fail-open 全量保留），
        调用方可以无条件信任本方法的返回值。
        """
        started = time.perf_counter()
        self.stats["gate_calls"] += 1
        candidates = list(candidates or [])
        fact_list = [dict(f) for f in (facts or []) if isinstance(f, dict)]

        # 开关关闭：原样放行，不做任何过滤
        if not self.enabled:
            report = GateReport(
                kept=[{**f, "value_score": 1.0} for f in fact_list],
                source="disabled",
            )
            self._finish(report, started)
            return report

        if not fact_list:
            report = GateReport(source="rule")
            self._finish(report, started)
            return report

        # 1) 规则底座
        try:
            rule_report = self._rule_gate(fact_list, candidates)
        except Exception as exc:  # 规则层理论不会失败，仍做兜底
            logger.warning("[EvaluatorGate] 规则判定异常，fail-open 放行: %s", exc)
            rule_report = GateReport(
                kept=[{**f, "value_score": 0.5} for f in fact_list],
                source="fallback",
                degraded=True,
                error=f"rule_error:{exc}",
            )
            self._finish(rule_report, started)
            return rule_report

        # 2) 无凭证：仅规则判定
        if not self.api_base or not self.api_key:
            self.stats["rule_only"] += 1
            self._finish(rule_report, started)
            return rule_report

        # 3) 判官模型判定（带硬超时）
        caller = self._llm_client or self._default_llm_call
        messages = [
            {"role": "system", "content": "你是严格的记忆入库判定官，只返回 JSON。"},
            {"role": "user", "content": self._build_prompt(fact_list, candidates)},
        ]
        try:
            content = await asyncio.wait_for(
                caller(messages, self.model_name), timeout=self.timeout_seconds
            )
            verdict = self._extract_json_object(content)
            if not isinstance(verdict, dict):
                raise ValueError("判官返回非法 JSON")
            merged = self._merge_llm_verdict(rule_report, verdict, candidates)
            self.stats["llm_ok"] += 1
            self._finish(merged, started)
            return merged
        except asyncio.TimeoutError:
            # 超时：平滑回退到规则判定，阈值放宽，保证不丢有效记忆
            self.stats["llm_failed"] += 1
            return await self._degrade(
                fact_list, candidates, started, "judge_timeout"
            )
        except Exception as exc:
            self.stats["llm_failed"] += 1
            return await self._degrade(
                fact_list, candidates, started, f"judge_error:{type(exc).__name__}:{exc}"
            )

    async def _degrade(
        self,
        fact_list: List[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        started: float,
        error: str,
    ) -> GateReport:
        """降级路径：规则重判（放宽阈值）+ fail-open 兜底，绝不阻断主流程。"""
        logger.warning(
            "[EvaluatorGate] 判官模型不可用（%s），已平滑降级为规则判定，model=%s",
            error,
            self.model_name,
        )
        try:
            # fail-open（默认）：放宽阈值，尽量保留事实；
            # fail-close：仍按常规阈值严格过滤，避免降级期噪声入库。
            threshold = (
                self.fallback_min_value_score if self.fail_open else self.min_value_score
            )
            report = self._rule_gate(fact_list, candidates, threshold=threshold)
            report.source = "fallback"
            report.degraded = True
            report.error = error
        except Exception as exc:  # 二次兜底：全量放行
            report = GateReport(
                kept=[{**f, "value_score": 0.5} for f in fact_list],
                source="fallback",
                degraded=True,
                error=f"{error};rule_error:{exc}",
            )
        self._finish(report, started)
        return report

    def _finish(self, report: GateReport, started: float) -> None:
        """补齐耗时统计并累加全局指标。"""
        report.latency_ms = (time.perf_counter() - started) * 1000.0
        self.stats["facts_dropped"] += len(report.dropped)
        self.stats["conflicts_detected"] += len(report.conflicts)
        logger.debug("[EvaluatorGate] report=%s", report.summary())


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(raw: Any, default: float, minimum: Optional[float] = None) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


def _safe_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _safe_bool(raw: Any, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on", "y"):
        return True
    if text in ("0", "false", "no", "off", "n"):
        return False
    return default
