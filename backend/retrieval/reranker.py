from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence

from sentence_transformers import CrossEncoder

from backend.retrieval.schemas import RetrievedChunk


@dataclass(frozen=True)
class CrossEncoderRerankerConfig:
    """
    Configuration for the second-stage cross-encoder reranker.

    The final score combines:
    - cross-encoder relevance score
    - original retrieval score
    - legal article-reference boost
    - small legal role-aware adjustment

    The legal role-aware adjustment is intentionally conservative. It does not
    replace the cross-encoder; it only corrects common legal actor mismatches,
    such as employer dismissal vs. employee departure.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str | None = None

    cross_encoder_weight: float = 0.85
    original_score_weight: float = 0.10
    legal_boost_weight: float = 0.05
    legal_role_boost_weight: float = 0.08
    legal_role_penalty_weight: float = 0.10

    batch_size: int = 16
    max_length: int = 512

    normalize_cross_encoder_scores: bool = True
    normalize_original_scores: bool = True

    enable_legal_boost: bool = True
    enable_legal_role_adjustment: bool = True


class CrossEncoderReranker:
    """
    Cross-encoder reranker for legal RAG.

    This reranker is used after first-stage retrieval, for example:
    dense retrieval, keyword retrieval, or hybrid retrieval.

    Input:
        query + candidate RetrievedChunk objects

    Output:
        reranked RetrievedChunk objects with updated scores and debug metadata
    """

    ARTICLE_PATTERN = re.compile(
        r"\b(?:art\.?|article)\s*(\d+[a-z]?(?:\s+bis)?)\b",
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
        """
        if not query.strip():
            reranked = list(results)
            return reranked[:top_k] if top_k is not None else reranked

        if not results:
            return []

        candidate_results = list(results)

        pairs = [
            [query, result.text]
            for result in candidate_results
        ]

        raw_cross_scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        cross_scores = [float(score) for score in raw_cross_scores]
        original_scores = [float(result.score) for result in candidate_results]

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

        scored_results: list[tuple[float, RetrievedChunk]] = []

        for (
            result,
            raw_cross_score,
            normalized_cross_score,
            original_score,
            normalized_original_score,
        ) in zip(
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

            legal_role_boost, legal_role_penalty = (
                self._legal_role_adjustment(query=query, result=result)
                if self.config.enable_legal_role_adjustment
                else (0.0, 0.0)
            )

            final_score = self._combine_scores(
                cross_score=normalized_cross_score,
                original_score=normalized_original_score,
                legal_boost=legal_boost,
                legal_role_boost=legal_role_boost,
                legal_role_penalty=legal_role_penalty,
            )

            updated_result = self._with_updated_score_and_metadata(
                result=result,
                final_score=final_score,
                original_score=original_score,
                cross_encoder_score=raw_cross_score,
                cross_encoder_score_normalized=normalized_cross_score,
                legal_boost_score=legal_boost,
                legal_role_boost_score=legal_role_boost,
                legal_role_penalty_score=legal_role_penalty,
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
        legal_role_boost: float,
        legal_role_penalty: float,
    ) -> float:
        cfg = self.config

        return (
            cfg.cross_encoder_weight * cross_score
            + cfg.original_score_weight * original_score
            + cfg.legal_boost_weight * legal_boost
            + cfg.legal_role_boost_weight * legal_role_boost
            - cfg.legal_role_penalty_weight * legal_role_penalty
        )

    def _legal_reference_boost(
        self,
        query: str,
        result: RetrievedChunk,
    ) -> float:
        """
        Boost a chunk when the user explicitly mentions an article number.

        Example:
            Query: "What are the rules under Art. 337?"
            Chunk: Art. 337
            Boost: 1.0
        """
        query_articles = {
            self._normalize_article_number(match.group(1))
            for match in self.ARTICLE_PATTERN.finditer(query)
        }

        if not query_articles:
            return 0.0

        article_candidates = {
            self._normalize_article_number(result.article_number),
            self._normalize_article_number(result.source_label),
            self._normalize_article_number(result.chunk_id),
        }

        metadata = result.metadata or {}
        article_candidates.update(
            {
                self._normalize_article_number(str(metadata.get("article_number", ""))),
                self._normalize_article_number(str(metadata.get("source_label", ""))),
                self._normalize_article_number(str(metadata.get("chunk_id", ""))),
            }
        )

        if query_articles.intersection(article_candidates):
            return 1.0

        return 0.0

    def _legal_role_adjustment(
        self,
        query: str,
        result: RetrievedChunk,
    ) -> tuple[float, float]:
        """
        Apply a small legal role-aware adjustment after cross-encoder scoring.

        This helps distinguish legally different actor directions.

        Example:
            Query: "Can my employer dismiss me immediately without notice?"
            More relevant:
                Art. 337  - requirements for immediate termination
                Art. 337c - employer dismisses employee without good cause
            Less relevant:
                Art. 337d - employee leaves or fails to take up post
        """
        query_text = query.lower()
        article_number = self._normalize_article_number(result.article_number)
        title_path = str(result.title_path or "").lower()
        text = str(result.text or "").lower()

        result_text = f"{article_number} {title_path} {text}"

        employer_dismissal_query = self._is_employer_dismissal_query(query_text)
        employee_departure_query = self._is_employee_departure_query(query_text)

        boost = 0.0
        penalty = 0.0

        if employer_dismissal_query:
            if article_number in {"337", "337c"}:
                boost = 1.0

            if article_number == "337d":
                penalty = 1.0

            if (
                "failure to take up post" in result_text
                or "departure without just cause" in result_text
                or "employee fails to take up" in result_text
                or "leaves it without notice" in result_text
            ):
                penalty = max(penalty, 1.0)

        if employee_departure_query:
            if article_number == "337d":
                boost = 1.0

        return boost, penalty

    @staticmethod
    def _is_employer_dismissal_query(query_text: str) -> bool:
        employer_terms = [
            "employer",
            "boss",
            "company",
            "workplace",
        ]

        dismissal_terms = [
            "dismiss me",
            "fire me",
            "terminate me",
            "dismissed me",
            "fired me",
            "terminate my employment",
            "dismiss employee",
            "dismiss the employee",
            "dismisses the employee",
            "termination by employer",
        ]

        immediate_terms = [
            "immediately",
            "immediate",
            "without notice",
            "with immediate effect",
        ]

        has_employer = any(term in query_text for term in employer_terms)
        has_dismissal = any(term in query_text for term in dismissal_terms)
        has_immediate = any(term in query_text for term in immediate_terms)

        return has_employer and has_dismissal and has_immediate

    @staticmethod
    def _is_employee_departure_query(query_text: str) -> bool:
        employee_departure_terms = [
            "can i leave",
            "can i quit",
            "i want to leave",
            "i left",
            "leave without notice",
            "quit without notice",
            "employee leaves",
            "employee fails to take up",
            "fail to take up",
            "departure without just cause",
        ]

        return any(term in query_text for term in employee_departure_terms)

    @staticmethod
    def _normalize_article_number(value: str) -> str:
        normalized = str(value).lower().strip()

        normalized = normalized.replace("article", "")
        normalized = normalized.replace("art.", "")
        normalized = normalized.replace("art_", "")
        normalized = normalized.replace("art ", "")
        normalized = normalized.replace("_", " ")
        normalized = " ".join(normalized.split())

        if normalized.startswith("art"):
            normalized = normalized.removeprefix("art").strip()

        return normalized

    @staticmethod
    def _min_max_normalize(scores: Sequence[float]) -> list[float]:
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

    def _with_updated_score_and_metadata(
        self,
        result: RetrievedChunk,
        final_score: float,
        original_score: float,
        cross_encoder_score: float,
        cross_encoder_score_normalized: float,
        legal_boost_score: float,
        legal_role_boost_score: float,
        legal_role_penalty_score: float,
    ) -> RetrievedChunk:
        metadata = dict(result.metadata or {})

        metadata["original_score"] = original_score
        metadata["cross_encoder_score"] = cross_encoder_score
        metadata["cross_encoder_score_normalized"] = cross_encoder_score_normalized
        metadata["legal_boost_score"] = legal_boost_score
        metadata["legal_role_boost_score"] = legal_role_boost_score
        metadata["legal_role_penalty_score"] = legal_role_penalty_score
        metadata["rerank_score"] = final_score
        metadata["reranker"] = self.config.model_name

        return replace(
            result,
            score=final_score,
            metadata=metadata,
        )