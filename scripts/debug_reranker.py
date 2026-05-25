from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.schemas import RetrievalConfig, RetrievalResponse

def print_response(title: str, response: RetrievalResponse) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    print(f"query: {response.query}")
    print(f"retrieval_method: {response.retrieval_method}")
    print(f"hop_count: {response.hop_count}")
    print(f"debug_info: {response.debug_info}")

    for idx, chunk in enumerate(response.chunks, start=1):
        metadata = chunk.metadata or {}

        print("\n" + "-" * 120)
        print(f"[{idx}] {chunk.source_label}")
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"article_number: {chunk.article_number}")
        print(f"rank: {chunk.rank}")
        print(f"score: {chunk.score:.6f}")
        print(f"retrieval_method: {chunk.retrieval_method}")

        if "rrf_score" in metadata:
            print(f"rrf_score: {metadata['rrf_score']:.6f}")

        if "original_score" in metadata:
            print(f"original_score: {metadata['original_score']:.6f}")

        if "cross_encoder_score" in metadata:
            print(f"cross_encoder_score: {metadata['cross_encoder_score']:.6f}")

        if "cross_encoder_score_normalized" in metadata:
            print(
                "cross_encoder_score_normalized: "
                f"{metadata['cross_encoder_score_normalized']:.6f}"
            )

        if "legal_boost_score" in metadata:
            print(f"legal_boost_score: {metadata['legal_boost_score']:.6f}")

        if "rerank_score" in metadata:
            print(f"rerank_score: {metadata['rerank_score']:.6f}")

        if "retrieval_sources" in metadata:
            print(f"retrieval_sources: {metadata['retrieval_sources']}")

        if "dense_rank" in metadata:
            print(f"dense_rank: {metadata['dense_rank']}")

        if "keyword_rank" in metadata:
            print(f"keyword_rank: {metadata['keyword_rank']}")

        if isinstance(chunk.title_path, list):
            title_path = " > ".join(str(part) for part in chunk.title_path)
        else:
            title_path = str(chunk.title_path or "")

        if title_path:
            print(f"title_path: {title_path}")

        preview = chunk.text[:700].replace("\n", " ")
        print(f"text: {preview}")


def compare_query(
    retriever: HybridRetriever,
    query: str,
    config: RetrievalConfig,
) -> None:
    print("\n\n" + "#" * 120)
    print(f"QUERY: {query}")
    print("#" * 120)

    hybrid_only_response = retriever.retrieve_without_reranker(
        query=query,
        config=config,
    )

    reranked_response = retriever.retrieve(
        query=query,
        config=config,
    )

    print_response(
        title="HYBRID ONLY",
        response=hybrid_only_response,
    )

    print_response(
        title="HYBRID + CROSS-ENCODER RERANKER",
        response=reranked_response,
    )


def main() -> None:
    retriever = HybridRetriever()

    config = RetrievalConfig(
        top_k=5,
        candidate_k=20,
        enable_reranker=True,
    )

    test_queries = [
        "What are the rules for immediate termination under Art. 337?",
        "Can an employer dismiss an employee without notice?",
        "What does Swiss employment law say about salary payment?",
        "What happens if the employee is prevented from working through no fault of their own?",
        "What are the employee's duties of loyalty and care?",
    ]

    for query in test_queries:
        compare_query(
            retriever=retriever,
            query=query,
            config=config,
        )


if __name__ == "__main__":
    main()