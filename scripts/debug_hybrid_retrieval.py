from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.schemas import RetrievalConfig, RetrievalResponse


TEST_QUERIES = [
    "Can my employer fire me immediately?",
    "What is an employment contract?",
    "Does my employer have to pay my salary?",
    "Can I take holidays during employment?",
    "What happens if the employee is sick and cannot work?",
    "Can an employee compete with the employer?",
]


def print_retrieval_response(response: RetrievalResponse) -> None:
    print("=" * 100)
    print(f"Query: {response.query}")
    print(f"Method: {response.retrieval_method}")
    print(f"Top sources: {', '.join(response.top_sources())}")
    print("=" * 100)

    debug_info = response.debug_info

    print("Debug info:")
    print(f"  Mode: {debug_info.get('mode')}")
    print(f"  Fusion method: {debug_info.get('fusion_method')}")
    print(f"  RRF k: {debug_info.get('rrf_k')}")
    print(f"  Dense candidates: {debug_info.get('dense_candidate_count')}")
    print(f"  Keyword candidates: {debug_info.get('keyword_candidate_count')}")
    print(f"  Dense top sources: {debug_info.get('dense_top_sources')}")
    print(f"  Keyword top sources: {debug_info.get('keyword_top_sources')}")
    print("-" * 100)

    for chunk in response.chunks:
        print(f"\nResult {chunk.rank}")
        print("-" * 100)
        print(f"Source: {chunk.short_label()}")
        print(f"Hybrid RRF score: {chunk.score:.6f}")

        retrieval_sources = chunk.metadata.get("retrieval_sources", [])
        dense_rank = chunk.metadata.get("dense_rank")
        keyword_rank = chunk.metadata.get("keyword_rank")

        print(f"Retrieved by: {retrieval_sources}")

        if dense_rank is not None:
            print(f"Dense rank: {dense_rank}")

        if keyword_rank is not None:
            print(f"Keyword rank: {keyword_rank}")

        if chunk.title_path:
            print(f"Title path: {chunk.title_path}")

        print(f"Text: {chunk.preview(max_chars=500)}")

    print()


def main() -> None:
    config = RetrievalConfig(
        top_k=5,
        candidate_k=10,
        enable_reranker=False,
        enable_multi_hop=False,
    )

    retriever = HybridRetriever()

    for query in TEST_QUERIES:
        response = retriever.retrieve(
            query=query,
            config=config,
        )
        print_retrieval_response(response)


if __name__ == "__main__":
    main()