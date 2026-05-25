from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.generation.answer_generator import (  # noqa: E402
    AnswerGenerationConfig,
    AnswerGenerator,
)
from backend.retrieval.schemas import RetrievalConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a source-grounded Swiss employment-law RAG assistant.",
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="User question, for example: 'Can my employer dismiss me immediately?'",
    )

    parser.add_argument(
        "--llm-mode",
        choices=["prompt_only", "openai_compatible"],
        default="prompt_only",
        help=(
            "Generation mode. Use 'prompt_only' for debugging without an LLM call, "
            "or 'openai_compatible' to call an OpenAI-compatible chat endpoint."
        ),
    )

    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name used when --llm-mode openai_compatible is selected.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final sources to pass to the answer generator.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of hybrid retrieval candidates before reranking.",
    )

    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Disable cross-encoder reranking and use hybrid retrieval only.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature when using openai_compatible mode.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum output tokens when using openai_compatible mode.",
    )

    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved sources before the answer.",
    )

    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the grounded prompt after the answer.",
    )

    return parser


def read_query(args: argparse.Namespace) -> str:
    if args.query:
        return str(args.query).strip()

    print("Enter your legal question. Press Enter when done:")
    query = input("> ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    return query


def print_sources(generated_answer) -> None:
    print("\n" + "=" * 100)
    print("SOURCES")
    print("=" * 100)

    for index, chunk in enumerate(generated_answer.retrieval_response.chunks, start=1):
        metadata = chunk.metadata or {}

        print("\n" + "-" * 100)
        print(f"[{index}] {chunk.source_label}")
        print(f"article_number: {chunk.article_number}")
        print(f"chunk_id: {chunk.chunk_id}")
        print(f"rank: {chunk.rank}")
        print(f"score: {chunk.score:.6f}")

        if "cross_encoder_score" in metadata:
            print(f"cross_encoder_score: {metadata['cross_encoder_score']:.6f}")

        if "rerank_score" in metadata:
            print(f"rerank_score: {metadata['rerank_score']:.6f}")

        if "legal_role_boost_score" in metadata:
            print(f"legal_role_boost_score: {metadata['legal_role_boost_score']:.6f}")

        if "legal_role_penalty_score" in metadata:
            print(f"legal_role_penalty_score: {metadata['legal_role_penalty_score']:.6f}")

        print(f"title_path: {chunk.title_path}")

        preview = chunk.text[:500].replace("\n", " ")
        print(f"text: {preview}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        query = read_query(args)

        retrieval_config = RetrievalConfig(
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            enable_reranker=not args.no_reranker,
        )

        generation_config = AnswerGenerationConfig(
            llm_mode=args.llm_mode,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        generator = AnswerGenerator(config=generation_config)

        generated_answer = generator.generate(
            query=query,
            retrieval_config=retrieval_config,
        )

        if args.show_sources:
            print_sources(generated_answer)

        print("\n" + "=" * 100)
        print("ANSWER")
        print("=" * 100)
        print(generated_answer.answer)

        if args.show_prompt and generated_answer.answer != generated_answer.prompt:
            print("\n" + "=" * 100)
            print("GROUNDED PROMPT")
            print("=" * 100)
            print(generated_answer.prompt)

        print("\n" + "=" * 100)
        print("DEBUG")
        print("=" * 100)
        print(f"llm_mode: {generated_answer.llm_mode}")
        print(f"model_name: {generated_answer.model_name}")
        print(f"retrieval_method: {generated_answer.retrieval_response.retrieval_method}")
        print(f"sources_used: {len(generated_answer.retrieval_response.chunks)}")
        print(f"retrieval_debug_info: {generated_answer.retrieval_response.debug_info}")

    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()