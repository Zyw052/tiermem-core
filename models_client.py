# -*- coding: utf-8 -*-
"""
astrbot_plugin_agent_memory - 向量与重排序模型客户端 (models_client.py)
提供轻量、高内聚的 OpenAI 兼容 Embedding 客户端 (EmbeddingClient)
与 Cross-Encoder 重排序客户端 (RerankClient)，支持动态维度校验、
本地余弦相似度计算以及基于 BM25 + 余弦相似度的 RRF 优雅降级。
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from astrbot.api import logger
except Exception:  # pragma: no cover
    logger = logging.getLogger("astrbot_plugin_agent_memory.models_client")


class _HttpSessionOwner:
    """
    惰性持有的 aiohttp.ClientSession 连接池（R-06 修复）。

    背景：原实现每次请求都以上下文管理器方式新建 ClientSession，
    高频 Embedding / Rerank 下产生大量 TCP/TLS 握手开销（N+1 账单与延迟）。

    行为：
    - 首次使用时创建 Session，后续所有请求复用同一连接池；
    - 允许外部注入 Session（测试桩 / 宿主统一管理），此时不负责关闭；
    - ``close()`` 幂等释放，重复调用与未创建场景均安全。
    """

    def __init__(
        self,
        timeout: float,
        session: Optional["aiohttp.ClientSession"] = None,
    ) -> None:
        self._timeout = float(timeout)
        self._session = session
        self._owns_session = session is None

    async def get(self) -> "aiohttp.ClientSession":
        """获取（必要时惰性创建）共享 Session。"""
        # 外部注入的 Session 优先直接复用，不依赖 aiohttp 是否可导入
        if self._session is not None and not self._session.closed:
            return self._session
        if aiohttp is None:  # pragma: no cover - 依赖缺失时的显式报错
            raise RuntimeError("aiohttp 不可用，无法创建 HTTP 连接池")
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        )
        self._owns_session = True
        return self._session

    @property
    def session(self) -> Optional["aiohttp.ClientSession"]:
        """当前持有的 Session（可能为 None / 已关闭）。"""
        return self._session

    async def close(self) -> None:
        """释放自建 Session；外部注入的 Session 不在此处关闭。"""
        session = self._session
        self._session = None
        if session is None or not self._owns_session:
            return
        try:
            if not session.closed:
                await session.close()
        except Exception as exc:  # pragma: no cover - 关闭失败不应影响停机
            logger.warning("[models_client] 关闭 HTTP 连接池异常（已忽略）: %s", exc)
        finally:
            self._owns_session = False


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """
    计算两个向量的余弦相似度（支持 numpy 加速与纯 Python 回退）。
    当输入为空、维度不匹配、全零向量或出现数值异常时，安全降级返回 0.0。
    """
    try:
        if not vec1 or not vec2:
            return 0.0
        if len(vec1) != len(vec2):
            min_len = min(len(vec1), len(vec2))
            if min_len == 0:
                return 0.0
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]

        if np is not None:
            a = np.asarray(vec1, dtype=np.float64)
            b = np.asarray(vec2, dtype=np.float64)
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            if norm_a <= 1e-12 or norm_b <= 1e-12:
                return 0.0
            sim = float(np.dot(a, b) / (norm_a * norm_b))
            if math.isnan(sim) or math.isinf(sim):
                return 0.0
            return max(-1.0, min(1.0, sim))

        dot_prod = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        for x, y in zip(vec1, vec2):
            fx = float(x)
            fy = float(y)
            dot_prod += fx * fy
            norm_a_sq += fx * fx
            norm_b_sq += fy * fy

        if norm_a_sq <= 1e-24 or norm_b_sq <= 1e-24:
            return 0.0
        sim = dot_prod / (math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq))
        if math.isnan(sim) or math.isinf(sim):
            return 0.0
        return max(-1.0, min(1.0, sim))
    except Exception as exc:
        logger.warning("[models_client] 余弦相似度计算异常，已降级返回 0.0: %s", exc)
        return 0.0


def _tokenize_for_similarity(text: str) -> List[str]:
    """
    中英文混合轻量分词：提取英文单词/数字串、单字与双字 (bigram) 特征，
    供本地 BM25 与词袋余弦相似度降级计算使用。
    """
    if not text:
        return []
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = [cjk_chars[i] + cjk_chars[i + 1] for i in range(len(cjk_chars) - 1)]
    return words + bigrams


def _compute_bm25_scores(
    query: str,
    documents: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """计算 query 对一组 documents 的轻量级 BM25 相关性得分。"""
    n_docs = len(documents)
    if n_docs == 0:
        return []

    query_tokens = _tokenize_for_similarity(query)
    if not query_tokens:
        return [0.0] * n_docs

    doc_tokens_list = [_tokenize_for_similarity(doc) for doc in documents]
    doc_lens = [len(tokens) for tokens in doc_tokens_list]
    avgdl = (sum(doc_lens) / n_docs) if n_docs > 0 else 1.0
    if avgdl <= 0:
        avgdl = 1.0

    # 统计文档频率 df
    df_counter: Dict[str, int] = {}
    for tokens in doc_tokens_list:
        for term in set(tokens):
            df_counter[term] = df_counter.get(term, 0) + 1

    scores: List[float] = []
    query_term_counts = Counter(query_tokens)

    for tokens, doc_len in zip(doc_tokens_list, doc_lens):
        if not tokens:
            scores.append(0.0)
            continue
        tf_counter = Counter(tokens)
        score = 0.0
        for term, q_freq in query_term_counts.items():
            tf = tf_counter.get(term, 0)
            if tf <= 0:
                continue
            df = df_counter.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * (doc_len / avgdl))
            score += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-9)) * q_freq
        scores.append(score)

    return scores


def _compute_lexical_cosine_scores(
    query: str,
    documents: Sequence[str],
) -> List[float]:
    """当未提供预计算稠密向量时，基于特征词频向量计算余弦相似度。"""
    if not documents:
        return []
    q_tokens = _tokenize_for_similarity(query)
    if not q_tokens:
        return [0.0] * len(documents)

    q_counter = Counter(q_tokens)
    doc_counters = [Counter(_tokenize_for_similarity(d)) for d in documents]

    vocab = list(
        set(q_counter.keys()).union(*(c.keys() for c in doc_counters))
    )
    if not vocab:
        return [0.0] * len(documents)

    q_vec = [float(q_counter.get(t, 0)) for t in vocab]
    scores: List[float] = []
    for d_counter in doc_counters:
        d_vec = [float(d_counter.get(t, 0)) for t in vocab]
        scores.append(cosine_similarity(q_vec, d_vec))
    return scores


def rrf_fusion(
    query: str,
    documents: Sequence[str],
    top_n: int = 5,
    rrf_k: int = 60,
    query_embedding: Optional[Sequence[float]] = None,
    doc_embeddings: Optional[Sequence[Sequence[float]]] = None,
    weight_bm25: float = 0.5,
    weight_cosine: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    基于 BM25 与余弦相似度的倒数排名融合 (Reciprocal Rank Fusion, RRF) 算法。
    返回格式与 RerankClient.rerank 保持严格一致：
    [{"index": int, "document": str, "relevance_score": float}, ...]
    """
    n_docs = len(documents)
    if n_docs == 0:
        return []

    bm25_scores = _compute_bm25_scores(query, documents)

    if (
        query_embedding is not None
        and doc_embeddings is not None
        and len(doc_embeddings) == n_docs
    ):
        cosine_scores = [
            cosine_similarity(query_embedding, doc_vec)
            for doc_vec in doc_embeddings
        ]
    else:
        cosine_scores = _compute_lexical_cosine_scores(query, documents)

    # 按分数降序排列得到名次 (1-based rank)
    bm25_order = sorted(
        range(n_docs),
        key=lambda idx: (bm25_scores[idx], -idx),
        reverse=True,
    )
    cosine_order = sorted(
        range(n_docs),
        key=lambda idx: (cosine_scores[idx], -idx),
        reverse=True,
    )

    bm25_rank = {doc_idx: rank + 1 for rank, doc_idx in enumerate(bm25_order)}
    cosine_rank = {doc_idx: rank + 1 for rank, doc_idx in enumerate(cosine_order)}

    results: List[Dict[str, Any]] = []
    for idx, doc_text in enumerate(documents):
        r_bm25 = bm25_rank.get(idx, n_docs)
        r_cos = cosine_rank.get(idx, n_docs)
        rrf_score = (
            weight_bm25 / float(rrf_k + r_bm25)
            + weight_cosine / float(rrf_k + r_cos)
        )
        results.append(
            {
                "index": idx,
                "document": doc_text,
                "relevance_score": round(float(rrf_score), 6),
            }
        )

    results.sort(
        key=lambda item: (item["relevance_score"], -item["index"]),
        reverse=True,
    )
    limit = max(1, int(top_n)) if top_n is not None else n_docs
    return results[:limit]


class EmbeddingClient:
    """
    OpenAI 兼容 API 向量客户端 (EmbeddingClient)
    - 默认模型: BAAI/bge-m3
    - 默认维度: 1024 (支持传入任意正整数 dim 并进行维度校验/对齐)
    - 异常安全降级: 网络或解析异常时返回全零向量 [0.0] * dim

    P0 改造（通电）：
    - 新增 ``backend`` 注入位，可传入 ``host_providers.EmbeddingBackend``，
      此时优先走 AstrBot 宿主已配置的 Embedding 服务商（无需在插件里重复填密钥）；
    - 宿主后端失败时按「宿主后端 → 手动 api_key HTTP → 零向量」三级降级，
      任何一级失败都不会抛出异常；
    - ``sync_provider_dim`` 为 True 时，会用宿主 Provider 声明的维度覆盖插件配置
      的 dim，避免「配置 1024 / 实际 1536」导致向量截断补零、相似度失真。
    """

    def __init__(
        self,
        api_base: str = "https://api.siliconflow.cn/v1",
        api_key: str = "",
        model: str = "BAAI/bge-m3",
        dim: int = 1024,
        timeout: float = 30.0,
        send_dimensions_param: bool = False,
        session: Optional["aiohttp.ClientSession"] = None,
        backend: Optional[Any] = None,
        sync_provider_dim: bool = True,
    ) -> None:
        if int(dim) <= 0:
            raise ValueError(f"Embedding 维度 dim 必须为正整数，当前收到: {dim}")
        self.api_base = (api_base or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = (model or "BAAI/bge-m3").strip()
        self.dim = int(dim)
        self.timeout = float(timeout)
        self.send_dimensions_param = bool(send_dimensions_param)
        # R-06：持久连接池，避免每次请求重建 Session
        self._http = _HttpSessionOwner(self.timeout, session)

        # ---- P0：宿主服务商后端（可选，存在即优先使用）----
        self.backend: Optional[Any] = backend
        self.backend_source: str = str(
            getattr(backend, "source", "") if backend is not None else "none"
        ) or "none"
        self.provider_id: str = str(
            getattr(backend, "provider_id", "") if backend is not None else ""
        )
        if backend is not None:
            backend_model = str(getattr(backend, "model", "") or "")
            if backend_model:
                self.model = backend_model
            if sync_provider_dim:
                try:
                    provider_dim = int(getattr(backend, "dim", lambda: 0)() or 0)
                except Exception:
                    provider_dim = 0
                if provider_dim > 0 and provider_dim != self.dim:
                    logger.info(
                        "[EmbeddingClient] 采用宿主 Provider 维度 %d（原配置 %d）。",
                        provider_dim,
                        self.dim,
                    )
                    self.dim = provider_dim

    @property
    def powered(self) -> bool:
        """是否具备真实向量化能力（宿主 Provider 或手动 API Key 二者其一）。"""
        return self.backend is not None or bool(self.api_key)

    def describe(self) -> Dict[str, Any]:
        """返回向量化通电状态（供自检/WebUI 使用）。"""
        return {
            "powered": self.powered,
            "mode": "provider" if self.backend is not None else ("manual" if self.api_key else "none"),
            "source": self.backend_source,
            "provider_id": self.provider_id,
            "model": self.model,
            "dim": self.dim,
        }

    async def close(self) -> None:
        """释放 HTTP 连接池（插件停用 / 测试清理时调用）。"""
        await self._http.close()

    def zero_vector(self) -> List[float]:
        """生成当前配置维度的安全零向量。"""
        return [0.0] * self.dim

    def _build_endpoint(self) -> str:
        """构建 OpenAI 兼容的 /embeddings 完整请求地址。"""
        if not self.api_base:
            return ""
        if self.api_base.endswith("/embeddings"):
            return self.api_base
        return f"{self.api_base}/embeddings"

    def _validate_and_align_vector(self, raw_vec: Any) -> List[float]:
        """
        校验返回向量的维度与数值合法性：
        - 若维度恰好等于 self.dim，直接转为 float 列表返回；
        - 若维度大于 self.dim（如支持 Matryoshka 的模型返回更高维），截断并重新 L2 归一化；
        - 若维度小于 self.dim，右侧补 0.0 并告警；
        - 若非序列或包含非法数值，降级返回零向量。
        """
        try:
            if not isinstance(raw_vec, (list, tuple)):
                logger.warning("[EmbeddingClient] 向量格式非法，已降级为零向量。")
                return self.zero_vector()

            vec = [float(x) for x in raw_vec]
            if any(math.isnan(x) or math.isinf(x) for x in vec):
                logger.warning("[EmbeddingClient] 向量包含 NaN/Inf，已降级为零向量。")
                return self.zero_vector()

            actual_dim = len(vec)
            if actual_dim == 0:
                return self.zero_vector()

            if actual_dim == self.dim:
                return vec

            if actual_dim > self.dim:
                logger.warning(
                    "[EmbeddingClient] 返回向量维度 (%d) 大于预期 (%d)，自动执行截断与归一化。",
                    actual_dim,
                    self.dim,
                )
                sliced = vec[: self.dim]
                norm = math.sqrt(sum(v * v for v in sliced))
                if norm > 1e-12:
                    return [v / norm for v in sliced]
                return sliced

            logger.warning(
                "[EmbeddingClient] 返回向量维度 (%d) 小于预期 (%d)，自动补零对齐。",
                actual_dim,
                self.dim,
            )
            return vec + [0.0] * (self.dim - actual_dim)
        except Exception as exc:
            logger.warning("[EmbeddingClient] 向量校验异常，降级为零向量: %s", exc)
            return self.zero_vector()

    async def get_embedding(self, text: str) -> List[float]:
        """获取单条文本的向量表示，异常时安全降级返回零向量。"""
        if not text or not text.strip():
            return self.zero_vector()
        results = await self.get_embeddings([text])
        if results and len(results) > 0:
            return results[0]
        return self.zero_vector()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取多条文本的向量表示，异常时安全降级返回对应数量的零向量。"""
        if not texts:
            return []

        cleaned_texts = [t.strip() if isinstance(t, str) and t.strip() else " " for t in texts]

        # ---- P0 一级：宿主服务商后端（真实通电路径）----
        if self.backend is not None:
            try:
                raw_vectors = await self.backend.embed(cleaned_texts)
                if raw_vectors:
                    vectors: List[List[float]] = []
                    for vec in raw_vectors:
                        aligned = self._validate_and_align_vector(list(vec or []))
                        vectors.append(aligned)
                    # 数量对齐：宿主返回不足时补零向量，保证与输入严格一一对应
                    while len(vectors) < len(cleaned_texts):
                        vectors.append(self.zero_vector())
                    return vectors[: len(cleaned_texts)]
                logger.warning(
                    "[EmbeddingClient] 宿主 Provider(%s) 返回空向量，降级到下一级。",
                    self.provider_id or "unknown",
                )
            except Exception as exc:
                logger.warning(
                    "[EmbeddingClient] 宿主 Provider(%s) 调用失败，降级到下一级: %s",
                    self.provider_id or "unknown",
                    exc,
                )

        endpoint = self._build_endpoint()

        if not endpoint or not self.api_key or aiohttp is None:
            logger.debug(
                "[EmbeddingClient] 未配置有效 API 端点/密钥或缺少 aiohttp，返回安全零向量 (count=%d, dim=%d)。",
                len(texts),
                self.dim,
            )
            return [self.zero_vector() for _ in texts]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": cleaned_texts,
        }
        if self.send_dimensions_param:
            payload["dimensions"] = self.dim

        try:
            session = await self._http.get()
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logger.warning(
                        "[EmbeddingClient] API 请求失败 (HTTP %d): %s",
                        resp.status,
                        err_text[:300],
                    )
                    return [self.zero_vector() for _ in texts]

                data = await resp.json()
                items = data.get("data", [])
                if not isinstance(items, list) or len(items) == 0:
                    logger.warning("[EmbeddingClient] 响应缺少有效 data 列表，降级返回零向量。")
                    return [self.zero_vector() for _ in texts]

                # 按 index 排序保证与输入顺序严格一致
                items_sorted = sorted(
                    items,
                    key=lambda item: item.get("index", 0) if isinstance(item, dict) else 0,
                )
                vectors: List[List[float]] = []
                for i in range(len(texts)):
                    if i < len(items_sorted) and isinstance(items_sorted[i], dict):
                        raw_emb = items_sorted[i].get("embedding")
                        vectors.append(self._validate_and_align_vector(raw_emb))
                    else:
                        vectors.append(self.zero_vector())
                return vectors
        except Exception as exc:
            logger.warning("[EmbeddingClient] 调用异常，已安全降级返回零向量: %s", exc)
            return [self.zero_vector() for _ in texts]

    @staticmethod
    def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        """便捷静态方法：调用模块级余弦相似度计算。"""
        return cosine_similarity(vec1, vec2)


class RerankClient:
    """
    Cross-Encoder API 重排序客户端 (RerankClient)
    - 默认模型: BAAI/bge-reranker-v2-m3
    - 支持传入 query, documents, top_n
    - 优雅降级: 当未配置 Rerank API 或请求异常时，自动回退为基于 BM25 与余弦相似度的 RRF 算法
    """

    def __init__(
        self,
        api_base: str = "https://api.siliconflow.cn/v1",
        api_key: str = "",
        model: str = "BAAI/bge-reranker-v2-m3",
        timeout: float = 20.0,
        rrf_k: int = 60,
        embedding_client: Optional[EmbeddingClient] = None,
        session: Optional["aiohttp.ClientSession"] = None,
        backend: Optional[Any] = None,
    ) -> None:
        self.api_base = (api_base or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = (model or "BAAI/bge-reranker-v2-m3").strip()
        self.timeout = float(timeout)
        self.rrf_k = int(rrf_k)
        self.embedding_client = embedding_client
        # R-06：持久连接池，避免每次请求重建 Session
        self._http = _HttpSessionOwner(self.timeout, session)

        # ---- P0：宿主 Rerank 服务商后端（可选，存在即优先使用）----
        self.backend: Optional[Any] = backend
        self.backend_source: str = str(
            getattr(backend, "source", "") if backend is not None else "none"
        ) or "none"
        self.provider_id: str = str(
            getattr(backend, "provider_id", "") if backend is not None else ""
        )
        if backend is not None:
            backend_model = str(getattr(backend, "model", "") or "")
            if backend_model:
                self.model = backend_model

    @property
    def powered(self) -> bool:
        """是否具备真实交叉编码器重排能力（宿主 Provider 或手动 API Key）。"""
        return self.backend is not None or bool(self.api_key)

    def describe(self) -> Dict[str, Any]:
        """返回重排通电状态（供自检/WebUI 使用）。"""
        return {
            "powered": self.powered,
            "mode": "provider" if self.backend is not None else ("manual" if self.api_key else "none"),
            "source": self.backend_source,
            "provider_id": self.provider_id,
            "model": self.model,
        }

    async def close(self) -> None:
        """释放 HTTP 连接池（插件停用 / 测试清理时调用）。"""
        await self._http.close()

    def _build_endpoint(self) -> str:
        """构建 Rerank 完整请求地址（兼容 /rerank 端点）。"""
        if not self.api_base:
            return ""
        if self.api_base.endswith("/rerank"):
            return self.api_base
        return f"{self.api_base}/rerank"

    def _fallback_rrf(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
        query_embedding: Optional[Sequence[float]] = None,
        doc_embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[Dict[str, Any]]:
        """执行本地 BM25 + 余弦相似度的倒数排名融合 (RRF) 降级重排。"""
        return rrf_fusion(
            query=query,
            documents=documents,
            top_n=top_n,
            rrf_k=self.rrf_k,
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
        )

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 5,
        query_embedding: Optional[Sequence[float]] = None,
        doc_embeddings: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        对候选文档列表进行重排序。
        返回格式: [{"index": int, "document": str, "relevance_score": float}, ...]
        若未配置 API 密钥/端点或请求异常，自动降级为基于 BM25 / 余弦相似度的 RRF 算法。
        """
        if not documents:
            return []

        effective_top_n = max(1, min(int(top_n), len(documents))) if top_n else len(documents)

        # ---- P0 一级：宿主 Rerank 服务商后端（真实通电路径）----
        if self.backend is not None:
            try:
                pairs = await self.backend.rerank(query or "", documents, effective_top_n)
                if pairs:
                    return [
                        {
                            "index": idx,
                            "document": documents[idx],
                            "relevance_score": float(score),
                        }
                        for idx, score in pairs
                        if 0 <= idx < len(documents)
                    ]
                logger.debug(
                    "[RerankClient] 宿主 Provider(%s) 返回空结果，降级到下一级。",
                    self.provider_id or "unknown",
                )
            except Exception as exc:
                logger.warning(
                    "[RerankClient] 宿主 Provider(%s) 调用失败，降级到下一级: %s",
                    self.provider_id or "unknown",
                    exc,
                )

        endpoint = self._build_endpoint()

        if not endpoint or not self.api_key or aiohttp is None:
            logger.debug(
                "[RerankClient] 未配置 Rerank API 或无密钥，自动回退至本地 RRF 融合重排 (docs=%d, top_n=%d)。",
                len(documents),
                effective_top_n,
            )
            return self._fallback_rrf(
                query=query,
                documents=documents,
                top_n=effective_top_n,
                query_embedding=query_embedding,
                doc_embeddings=doc_embeddings,
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query or "",
            "documents": documents,
            "top_n": effective_top_n,
            "return_documents": False,
        }

        try:
            session = await self._http.get()
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logger.warning(
                        "[RerankClient] Rerank API 返回非 200 (HTTP %d)，回退至 RRF: %s",
                        resp.status,
                        err_text[:300],
                    )
                    return self._fallback_rrf(
                        query=query,
                        documents=documents,
                        top_n=effective_top_n,
                        query_embedding=query_embedding,
                        doc_embeddings=doc_embeddings,
                    )

                data = await resp.json()
                raw_results = data.get("results") or data.get("data") or []
                if not isinstance(raw_results, list) or len(raw_results) == 0:
                    logger.warning("[RerankClient] Rerank 响应无有效 results，回退至 RRF。")
                    return self._fallback_rrf(
                        query=query,
                        documents=documents,
                        top_n=effective_top_n,
                        query_embedding=query_embedding,
                        doc_embeddings=doc_embeddings,
                    )

                formatted: List[Dict[str, Any]] = []
                for item in raw_results:
                    if not isinstance(item, dict):
                        continue
                    idx = int(item.get("index", -1))
                    if idx < 0 or idx >= len(documents):
                        continue
                    score = float(
                        item.get(
                            "relevance_score",
                            item.get("score", 0.0),
                        )
                    )
                    formatted.append(
                        {
                            "index": idx,
                            "document": documents[idx],
                            "relevance_score": score,
                        }
                    )

                if not formatted:
                    return self._fallback_rrf(
                        query=query,
                        documents=documents,
                        top_n=effective_top_n,
                        query_embedding=query_embedding,
                        doc_embeddings=doc_embeddings,
                    )

                formatted.sort(
                    key=lambda x: (x["relevance_score"], -x["index"]),
                    reverse=True,
                )
                return formatted[:effective_top_n]
        except Exception as exc:
            logger.warning("[RerankClient] 请求异常，已优雅降级至 RRF 算法: %s", exc)
            return self._fallback_rrf(
                query=query,
                documents=documents,
                top_n=effective_top_n,
                query_embedding=query_embedding,
                doc_embeddings=doc_embeddings,
            )


__all__ = [
    "EmbeddingClient",
    "RerankClient",
    "cosine_similarity",
    "rrf_fusion",
]
