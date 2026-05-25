from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHROMA_DIR = PROJECT_ROOT / "data" / "vectorstore" / "chroma_articles"

COLLECTION_NAME = "swiss_employment_law_articles"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """
    Embed a user query using the BGE query instruction prefix.

    BGE models recommend adding an instruction to queries so that the model
    represents the text as a retrieval query rather than as a normal passage.
    """
    query_text = QUERY_PREFIX + query
    embedding = model.encode(
        query_text,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def format_metadata(metadata: dict[str, Any]) -> str:
    source_label = metadata.get("source_label", "")
    article_number = metadata.get("article_number", "")
    chunk_type = metadata.get("chunk_type", "")
    word_count = metadata.get("word_count", "")

    parts = []

    if source_label:
        parts.append(f"source_label={source_label}")
    if article_number:
        parts.append(f"article_number={article_number}")
    if chunk_type:
        parts.append(f"chunk_type={chunk_type}")
    if word_count:
        parts.append(f"word_count={word_count}")

    return " | ".join(parts)


def preview_text(text: str, max_chars: int = 500) -> str:
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def run_query(
    collection: Any,
    model: SentenceTransformer,
    query: str,
    top_k: int = 5,
) -> None:
    print("=" * 100)
    print(f"Query: {query}")
    print("=" * 100)

    query_embedding = embed_query(model, query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No results found.")
        return

    for rank, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1,
    ):
        print(f"\nResult {rank}")
        print("-" * 100)
        print(f"Distance: {distance:.4f}")
        print(f"Metadata: {format_metadata(metadata)}")
        print(f"Text: {preview_text(document)}")

    print()


def main() -> None:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma directory not found: {CHROMA_DIR}\n"
            "Please run scripts/build_vector_index.py first."
        )

    print(f"Loading Chroma index from: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing_collections = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME not in existing_collections:
        raise ValueError(
            f"Collection not found: {COLLECTION_NAME}\n"
            f"Existing collections: {existing_collections}"
        )

    collection = client.get_collection(COLLECTION_NAME)
    print(f"Loaded collection: {COLLECTION_NAME}")
    print(f"Indexed documents: {collection.count()}")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    test_queries = [
        "Can my employer fire me immediately?",
        "What is an employment contract?",
        "Does my employer have to pay my salary?",
        "Can I take holidays during employment?",
        "What happens if the employee is sick and cannot work?",
        "Can an employee compete with the employer?",
    ]

    for query in test_queries:
        run_query(
            collection=collection,
            model=model,
            query=query,
            top_k=5,
        )


if __name__ == "__main__":
    main()