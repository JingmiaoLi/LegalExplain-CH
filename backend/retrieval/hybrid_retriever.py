from __future__ import annotations

from collections import defaultdict
from typing import Any

from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.keyword_retriever import KeywordRetriever
from backend.retrieval.reranker import CrossEncoderReranker
from backend.retrieval.schemas import RetrievedChunk, RetrievalConfig, RetrievalResponse


class HybridRetriever:
    """
    Hybrid retriever that combines dense retrieval and keyword retrieval.

    It uses Reciprocal Rank Fusion (RRF) to merge ranked results from:
    - DenseRetriever: semantic similarity via embeddings
    - KeywordRetriever: lexical matching via BM25

    Important design:
    - retrieve_candidates() returns a larger fused candidate pool.
    - retrieve() returns the final top_k results, with or without reranking
      depending on config.enable_reranker.
    - retrieve_without_reranker() is kept for explicit debugging/comparison.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        keyword_retriever: KeywordRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.keyword_retriever = keyword_retriever or KeywordRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k

    def retrieve_candidates(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResponse:
        """
        Return a fused candidate pool using RRF.

        This is the method used before reranking. It keeps config.candidate_k
        fused candidates, so potentially useful chunks are not discarded too early.
        """
        if config is None:
            config = RetrievalConfig()

        config.validate()

        candidate_config = RetrievalConfig(
            top_k=config.candidate_k,
            candidate_k=config.candidate_k,
            dense_weight=config.dense_weight,
            keyword_weight=config.keyword_weight,
            enable_reranker=False,
            enable_multi_hop=False,
            max_hops=config.max_hops,
        )

        dense_response = self.dense_retriever.retrieve(
            query=query,
            config=candidate_config,
        )

        keyword_response = self.keyword_retriever.retrieve(
            query=query,
            config=candidate_config,
        )

        fused_chunks = self._fuse_with_rrf(
            dense_chunks=dense_response.chunks,
            keyword_chunks=keyword_response.chunks,
            top_k=config.candidate_k,
        )

        return RetrievalResponse(
            query=query,
            chunks=fused_chunks,
            retrieval_method="hybrid",
            hop_count=1,
            debug_info={
                "mode": "candidate_pool",
                "fusion_method": "rrf",
                "rrf_k": self.rrf_k,
                "candidate_k": config.candidate_k,
                "dense_candidate_count": len(dense_response.chunks),
                "keyword_candidate_count": len(keyword_response.chunks),
                "dense_top_sources": dense_response.top_sources(),
                "keyword_top_sources": keyword_response.top_sources(),
            },
        )

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResponse:
        """
        Return final top_k hybrid results.

        If config.enable_reranker is True, this method applies the cross-encoder
        reranker to the larger candidate pool. Otherwise, it directly returns
        the top_k fused hybrid results.
        """
        if config is None:
            config = RetrievalConfig()

        config.validate()

        candidate_response = self.retrieve_candidates(
            query=query,
            config=config,
        )

        if config.enable_reranker:
            final_chunks = self.reranker.rerank(
                query=query,
                results=candidate_response.chunks,
                top_k=config.top_k,
            )

            debug_info = dict(candidate_response.debug_info)
            debug_info["mode"] = "final_with_cross_encoder_reranker"
            debug_info["top_k"] = config.top_k
            debug_info["candidate_count_before_reranking"] = len(
                candidate_response.chunks
            )
            debug_info["final_count_after_reranking"] = len(final_chunks)
            debug_info["reranker"] = self.reranker.config.model_name

            return RetrievalResponse(
                query=query,
                chunks=final_chunks,
                retrieval_method="hybrid",
                hop_count=1,
                debug_info=debug_info,
            )

        final_chunks = self._take_top_k(
            chunks=candidate_response.chunks,
            top_k=config.top_k,
        )

        debug_info = dict(candidate_response.debug_info)
        debug_info["mode"] = "final_without_reranker"
        debug_info["top_k"] = config.top_k

        return RetrievalResponse(
            query=query,
            chunks=final_chunks,
            retrieval_method="hybrid",
            hop_count=1,
            debug_info=debug_info,
        )

    def retrieve_without_reranker(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResponse:
        """
        Return final top_k hybrid results without the model reranker.

        This is useful for debugging and comparing:
        - hybrid retrieval only
        - hybrid retrieval + cross-encoder reranking
        """
        if config is None:
            config = RetrievalConfig()

        config.validate()

        candidate_response = self.retrieve_candidates(
            query=query,
            config=config,
        )

        final_chunks = self._take_top_k(
            chunks=candidate_response.chunks,
            top_k=config.top_k,
        )

        debug_info = dict(candidate_response.debug_info)
        debug_info["mode"] = "final_without_reranker"
        debug_info["top_k"] = config.top_k

        return RetrievalResponse(
            query=query,
            chunks=final_chunks,
            retrieval_method="hybrid",
            hop_count=1,
            debug_info=debug_info,
        )

    def _fuse_with_rrf(
        self,
        dense_chunks: list[RetrievedChunk],
        keyword_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = defaultdict(float)
        chunk_store: dict[str, RetrievedChunk] = {}
        source_methods: dict[str, set[str]] = defaultdict(set)
        original_ranks: dict[str, dict[str, int | None]] = defaultdict(dict)

        for chunk in dense_chunks:
            chunk_id = chunk.chunk_id
            rank = chunk.rank or 9999

            scores[chunk_id] += 1.0 / (self.rrf_k + rank)
            chunk_store[chunk_id] = chunk
            source_methods[chunk_id].add("dense")
            original_ranks[chunk_id]["dense_rank"] = chunk.rank

        for chunk in keyword_chunks:
            chunk_id = chunk.chunk_id
            rank = chunk.rank or 9999

            scores[chunk_id] += 1.0 / (self.rrf_k + rank)

            if chunk_id not in chunk_store:
                chunk_store[chunk_id] = chunk

            source_methods[chunk_id].add("keyword")
            original_ranks[chunk_id]["keyword_rank"] = chunk.rank

        ranked_chunk_ids = sorted(
            scores.keys(),
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        fused_chunks: list[RetrievedChunk] = []

        for final_rank, chunk_id in enumerate(ranked_chunk_ids[:top_k], start=1):
            original_chunk = chunk_store[chunk_id]
            methods = sorted(source_methods[chunk_id])

            metadata: dict[str, Any] = dict(original_chunk.metadata)
            metadata["rrf_score"] = scores[chunk_id]
            metadata["retrieval_sources"] = methods
            metadata.update(original_ranks[chunk_id])

            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=original_chunk.chunk_id,
                    text=original_chunk.text,
                    source_label=original_chunk.source_label,
                    article_number=original_chunk.article_number,
                    title_path=original_chunk.title_path,
                    score=scores[chunk_id],
                    distance=original_chunk.distance,
                    rank=final_rank,
                    retrieval_method="hybrid",
                    metadata=metadata,
                )
            )

        return fused_chunks

    @staticmethod
    def _take_top_k(
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        final_chunks: list[RetrievedChunk] = []

        for rank, chunk in enumerate(chunks[:top_k], start=1):
            metadata = dict(chunk.metadata)

            final_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_label=chunk.source_label,
                    article_number=chunk.article_number,
                    title_path=chunk.title_path,
                    score=chunk.score,
                    distance=chunk.distance,
                    rank=rank,
                    retrieval_method="hybrid",
                    metadata=metadata,
                )
            )

        return final_chunks
    