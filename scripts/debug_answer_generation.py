from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.generation.answer_generator import (  # noqa: E402
    AnswerGenerationConfig,
    AnswerGenerator,
)
from backend.retrieval.schemas import RetrievalConfig  # noqa: E402


def print_sources(generated_answer) -> None:
    print("\n" + "=" * 120)
    print("RETRIEVED SOURCES")
    print("=" * 120)

    for index, chunk in enumerate(generated_answer.retrieval_response.chunks, start=1):
        print("\n" + "-" * 120)
        print(f"[{index}] {chunk.source_label}")
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"article_number: {chunk.article_number}")
        print(f"rank: {chunk.rank}")
        print(f"score: {chunk.score:.6f}")
        print(f"title_path: {chunk.title_path}")

        metadata = chunk.metadata or {}

        if "cross_encoder_score" in metadata:
            print(f"cross_encoder_score: {metadata['cross_encoder_score']:.6f}")

        if "rerank_score" in metadata:
            print(f"rerank_score: {metadata['rerank_score']:.6f}")

        preview = chunk.text[:500].replace("\n", " ")
        print(f"text: {preview}")


def main() -> None:
    query = "Can my employer dismiss me immediately without notice?"

    retrieval_config = RetrievalConfig(
        top_k=5,
        candidate_k=20,
        enable_reranker=True,
    )

    generation_config = AnswerGenerationConfig(
        llm_mode="prompt_only",
        model_name="gpt-4o-mini",
        temperature=0.0,
        max_tokens=800,
    )

    generator = AnswerGenerator(config=generation_config)

    generated_answer = generator.generate(
        query=query,
        retrieval_config=retrieval_config,
    )

    print("\n" + "#" * 120)
    print("QUERY")
    print("#" * 120)
    print(generated_answer.query)

    print_sources(generated_answer)

    print("\n" + "=" * 120)
    print("ANSWER / PROMPT")
    print("=" * 120)
    print(generated_answer.answer)


if __name__ == "__main__":
    main()