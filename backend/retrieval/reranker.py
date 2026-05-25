from __future__ import annotations

import re
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Sequence

from sentence_transformers import CrossEncoder
from backend.retrieval.schemas import RetrievedChunk

@dataclass(frozen=True)
class CrossEncoderRerankerConfig:
    """
    Configuration for the second-stage cross-encoder reranker.

    The reranker is designed to work after first-stage retrieval, such as:
    dense retrieval, keyword retrieval, or hybrid retrieval.

    The final score combines:
    - cross-encoder relevance score
    - optional original retrieval score
    - optional legal-reference boost

    For legal RAG, the legal-reference boost is useful when the user explicitly
    asks about a specific article, such as "Art. 337".
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str | None = None

    cross_encoder_weight: float = 0.85
    original_score_weight: float = 0.10
    legal_boost_weight: float = 0.05

    batch_size: int = 16
    max_length: int = 512

    normalize_cross_encoder_scores: bool = True
    normalize_original_scores: bool = True

    enable_legal_boost: bool = True


class CrossEncoderReranker:
    """
    Cross-encoder reranker for legal RAG.

    Usage:
        reranker = CrossEncoderReranker()
        reranked_results = reranker.rerank(
            query=query,
            results=candidate_results,
            top_k=5,
        )

    Expected result object fields:
        - text: str
        - score: float
        - source_label: str, optional
        - metadata: dict, optional

    The reranker is intentionally schema-tolerant. It works with dataclasses,
    normal Python objects, or simple objects that expose the expected attributes.
    """

    ARTICLE_PATTERN = re.compile(
        r"\b(?:art\.?|article)\s*(\d+[a-z]?)\b",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        config: CrossEncoderRerankerConfig | None = None,
    ) -> None:
        self.config = config or CrossEncoderRerankerConfig()

        self.model = CrossEncoder(
            self.config.model_name,
            device=self.config.device,
            max_length=self.config.max_length,
        )

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank candidate retrieval results with a cross-encoder.

        Parameters
        ----------
        query:
            User question.
        results:
            Candidate chunks returned by the first-stage retriever.
        top_k:
            Optional number of results to keep after reranking.

        Returns
        -------
        list[Any]
            Reranked results. The result score is replaced by the final rerank
            score when possible. Metadata stores:
            - original_score
            - cross_encoder_score
            - cross_encoder_score_normalized
            - legal_boost_score
            - rerank_score
        """

        if not query.strip():
            return list(results[:top_k]) if top_k is not None else list(results)

        if not results:
            return []

        candidate_results = list(results)

        pairs = [
            [query, self._get_text(result)]
            for result in candidate_results
        ]

        raw_cross_scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        cross_scores = [float(score) for score in raw_cross_scores]
        original_scores = [self._get_score(result) for result in candidate_results]

        normalized_cross_scores = (
            self._min_max_normalize(cross_scores)
            if self.config.normalize_cross_encoder_scores
            else cross_scores
        )

        normalized_original_scores = (
            self._min_max_normalize(original_scores)
            if self.config.normalize_original_scores
            else original_scores
        )

        scored_results: list[tuple[float, Any]] = []

        for result, raw_cross_score, cross_score, original_score, original_score_norm in zip(
            candidate_results,
            cross_scores,
            normalized_cross_scores,
            original_scores,
            normalized_original_scores,
        ):
            legal_boost = (
                self._legal_reference_boost(query=query, result=result)
                if self.config.enable_legal_boost
                else 0.0
            )

            final_score = self._combine_scores(
                cross_score=cross_score,
                original_score=original_score_norm,
                legal_boost=legal_boost,
            )

            updated_result = self._with_updated_score_and_metadata(
                result=result,
                final_score=final_score,
                original_score=original_score,
                cross_encoder_score=raw_cross_score,
                cross_encoder_score_normalized=cross_score,
                legal_boost_score=legal_boost,
            )

            scored_results.append((final_score, updated_result))

        scored_results.sort(key=lambda item: item[0], reverse=True)

        reranked_results = [result for _, result in scored_results]

        if top_k is not None:
            return reranked_results[:top_k]

        return reranked_results

    def _combine_scores(
        self,
        cross_score: float,
        original_score: float,
        legal_boost: float,
    ) -> float:
        cfg = self.config

        return (
            cfg.cross_encoder_weight * cross_score
            + cfg.original_score_weight * original_score
            + cfg.legal_boost_weight * legal_boost
        )

    def _legal_reference_boost(self, query: str, result: Any) -> float:
        """
        Give a small boost when the query explicitly mentions an article number
        and the retrieved chunk appears to come from that article.

        Example:
            query: "What are the rules for immediate termination under Art. 337?"
            source_label: "Art. 337"
            boost: 1.0
        """

        query_articles = {
            match.group(1).lower()
            for match in self.ARTICLE_PATTERN.finditer(query)
        }

        if not query_articles:
            return 0.0

        source_label = self._get_source_label(result)
        metadata = self._get_metadata(result)

        article_candidates = [
            source_label,
            metadata.get("source_label", ""),
            metadata.get("article_id", ""),
            metadata.get("article_number", ""),
            metadata.get("chunk_id", ""),
        ]

        candidate_text = " ".join(str(item) for item in article_candidates).lower()

        for article in query_articles:
            if article in candidate_text:
                return 1.0

        return 0.0

    def _min_max_normalize(self, scores: Sequence[float]) -> list[float]:
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0 for _ in scores]

        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]

    def _get_text(self, result: Any) -> str:
        text = getattr(result, "text", "")

        if text is None:
            return ""

        return str(text)

    def _get_score(self, result: Any) -> float:
        score = getattr(result, "score", 0.0)

        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    def _get_source_label(self, result: Any) -> str:
        source_label = getattr(result, "source_label", None)

        if source_label:
            return str(source_label)

        metadata = self._get_metadata(result)
        return str(metadata.get("source_label", ""))

    def _get_metadata(self, result: Any) -> dict[str, Any]:
        metadata = getattr(result, "metadata", None)

        if isinstance(metadata, dict):
            return dict(metadata)

        return {}

    def _with_updated_score_and_metadata(
        self,
        result: RetrievedChunk,
        final_score: float,
        original_score: float,
        cross_encoder_score: float,
        cross_encoder_score_normalized: float,
        legal_boost_score: float,
    ) -> RetrievedChunk:
        metadata = dict(result.metadata or {})

        metadata["original_score"] = original_score
        metadata["cross_encoder_score"] = cross_encoder_score
        metadata["cross_encoder_score_normalized"] = cross_encoder_score_normalized
        metadata["legal_boost_score"] = legal_boost_score
        metadata["rerank_score"] = final_score
        metadata["reranker"] = self.config.model_name

        return replace(
            result,
            score=final_score,
            metadata=metadata,
        )