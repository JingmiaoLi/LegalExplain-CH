from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "article_chunks.json"
CHROMA_DIR = PROJECT_ROOT / "data" / "vectorstore" / "chroma_articles"

COLLECTION_NAME = "swiss_employment_law_articles"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        return data["chunks"]

    raise ValueError(
        "Expected the input JSON to contain either a list of chunks "
        "or a dictionary with a 'chunks' list."
    )


def chunk_to_document(chunk: dict[str, Any]) -> str:
    source_label = chunk.get("source_label", "")
    title_path = chunk.get("title_path", [])
    text = chunk.get("text", "")

    title_context = " > ".join(title_path) if isinstance(title_path, list) else str(title_path)

    parts = []

    if source_label:
        parts.append(source_label)

    if title_context:
        parts.append(title_context)

    if text:
        parts.append(text)

    return "\n".join(parts).strip()


def build_metadata(article: dict[str, Any]) -> dict[str, Any]:
    """
    Chroma metadata values must be simple scalar types.
    Lists/dicts should be converted to strings.
    """
    title_path = article.get("title_path", [])

    return {
        "chunk_id": article.get("chunk_id", ""),
        "source_label": article.get("source_label", ""),
        "word_count": int(article.get("word_count", 0)),
        "title_path": json.dumps(title_path, ensure_ascii=False),
        "num_paragraphs": len(article.get("paragraphs", [])),
        "num_footnotes": len(article.get("footnotes", [])),
    }


def main() -> None:
    print(f"Loading chunks from: {INPUT_PATH}")
    chunks = load_chunks(INPUT_PATH)

    print(f"Loaded chunks: {len(chunks)}")

    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict[str, str | int | float | bool]] = []

    for idx, chunk in enumerate(chunks):
        document = chunk_to_document(chunk)

        if not document:
            print(f"Skipping empty chunk at index {idx}")
            continue

        chunk_id = chunk.get("chunk_id") or f"chunk_{idx}"

        documents.append(document)
        ids.append(str(chunk_id))
        metadatas.append(build_metadata(chunk))

    print(f"Documents prepared for indexing: {len(documents)}")

    if not documents:
        raise ValueError("No valid documents to index.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Computing embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Creating Chroma index at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing_collections = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Swiss employment-law article chunks",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "hnsw:space": "cosine",
        },
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas, # type: ignore
    )

    print("Vector index built successfully.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Indexed documents: {collection.count()}")
    print(f"Persisted at: {CHROMA_DIR}")

if __name__ == "__main__":
    main()