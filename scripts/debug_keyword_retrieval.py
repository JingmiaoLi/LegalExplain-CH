from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval.keyword_retriever import KeywordRetriever
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

    query_tokens = response.debug_info.get("query_tokens", [])
    if query_tokens:
        print(f"Query tokens: {query_tokens}")
        print("-" * 100)

    for chunk in response.chunks:
        print(f"\nResult {chunk.rank}")
        print("-" * 100)
        print(f"Source: {chunk.short_label()}")
        print(f"Score: {chunk.score:.4f}")

        if chunk.title_path:
            print(f"Title path: {chunk.title_path}")

        print(f"Text: {chunk.preview(max_chars=500)}")

    print()


def main() -> None:
    config = RetrievalConfig(
        top_k=5,
        candidate_k=20,
        enable_reranker=False,
        enable_multi_hop=False,
    )

    retriever = KeywordRetriever()

    for query in TEST_QUERIES:
        response = retriever.retrieve(
            query=query,
            config=config,
        )
        print_retrieval_response(response)


if __name__ == "__main__":
    main()