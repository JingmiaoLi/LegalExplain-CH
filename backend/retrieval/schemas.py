from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RetrievalMethod = Literal[
    "dense",
    "keyword",
    "hybrid",
    "reranked",
    "multi_hop",
]


@dataclass(slots=True)
class RetrievedChunk:
    """
    A normalized retrieval result used across all retrievers.

    Dense, keyword, hybrid, reranker, and multi-hop retrieval should all return
    this structure so the rest of the system does not need to know where the
    result came from.
    """

    chunk_id: str
    text: str

    source_label: str = ""
    article_number: str = ""
    title_path: str = ""

    score: float = 0.0
    distance: float | None = None
    rank: int | None = None

    retrieval_method: RetrievalMethod = "dense"
    metadata: dict[str, Any] = field(default_factory=dict)

    def short_label(self) -> str:
        """
        Return a compact legal source label for display.
        """
        if self.source_label:
            return self.source_label

        if self.article_number:
            return f"Art. {self.article_number}"

        return self.chunk_id

    def preview(self, max_chars: int = 300) -> str:
        """
        Return a compact text preview for debugging and terminal output.
        """
        compact_text = " ".join(self.text.split())

        if len(compact_text) <= max_chars:
            return compact_text

        return compact_text[:max_chars].rstrip() + "..."

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the retrieved chunk into a serializable dictionary.

        Useful later for API responses, Streamlit display, logging, or tests.
        """
        return {
            "chunk_id": self.chunk_id,
            "source_label": self.source_label,
            "article_number": self.article_number,
            "title_path": self.title_path,
            "text": self.text,
            "score": self.score,
            "distance": self.distance,
            "rank": self.rank,
            "retrieval_method": self.retrieval_method,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RetrievalConfig:
    """
    Shared retrieval configuration.

    Multi-hop is intentionally disabled by default because most queries should
    first use normal single-hop retrieval. It can be enabled explicitly for
    complex legal questions.
    """

    top_k: int = 5
    candidate_k: int = 10

    dense_weight: float = 0.7
    keyword_weight: float = 0.3

    enable_reranker: bool = False
    enable_multi_hop: bool = False

    max_hops: int = 2

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0.")

        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k.")

        if self.max_hops <= 0:
            raise ValueError("max_hops must be greater than 0.")

        if self.dense_weight < 0 or self.keyword_weight < 0:
            raise ValueError("retrieval weights must be non-negative.")

        if self.dense_weight == 0 and self.keyword_weight == 0:
            raise ValueError("at least one retrieval weight must be greater than 0.")


@dataclass(slots=True)
class RetrievalResponse:
    """
    A structured response from a retriever.

    This is useful because later we may want to return not only chunks, but also
    the original query, rewritten queries, hop count, or debugging information.
    """

    query: str
    chunks: list[RetrievedChunk]

    retrieval_method: RetrievalMethod = "dense"
    rewritten_queries: list[str] = field(default_factory=list)
    hop_count: int = 1
    debug_info: dict[str, Any] = field(default_factory=dict)

    def top_sources(self) -> list[str]:
        """
        Return unique source labels in ranked order.
        """
        sources: list[str] = []
        seen: set[str] = set()

        for chunk in self.chunks:
            label = chunk.short_label()
            if label not in seen:
                sources.append(label)
                seen.add(label)

        return sources

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieval_method": self.retrieval_method,
            "rewritten_queries": self.rewritten_queries,
            "hop_count": self.hop_count,
            "top_sources": self.top_sources(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "debug_info": self.debug_info,
        }