from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from backend.retrieval.schemas import RetrievedChunk, RetrievalConfig, RetrievalResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DIR = PROJECT_ROOT / "data" / "vectorstore" / "chroma_articles"

COLLECTION_NAME = "swiss_employment_law_articles"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """
    Dense vector retriever based on Chroma and BGE embeddings.

    This retriever performs single-hop semantic search. It does not do BM25,
    reranking, or multi-hop expansion. Those should be added in separate layers.
    """

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        self._client: Any | None = None
        self._collection: Any | None = None
        self._model: SentenceTransformer | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.chroma_dir.exists():
                raise FileNotFoundError(
                    f"Chroma directory not found: {self.chroma_dir}. "
                    "Please run scripts/build_vector_index.py first."
                )

            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))

        return self._client
   
    @property
    def collection(self) -> Any:
        if self._collection is None:
            existing_collections = [
                collection.name for collection in self.client.list_collections()
            ]

            if self.collection_name not in existing_collections:
                raise ValueError(
                    f"Collection not found: {self.collection_name}. "
                    f"Existing collections: {existing_collections}"
                )

            self._collection = self.client.get_collection(self.collection_name)

        return self._collection

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)

        return self._model

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query using the BGE query instruction prefix.
        """
        query_text = QUERY_PREFIX + query
        embedding = self.model.encode(
            query_text,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResponse:
        """
        Retrieve relevant chunks from the Chroma vector index.
        """
        if config is None:
            config = RetrievalConfig()

        config.validate()

        query_embedding = self.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=config.top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = self._parse_chroma_results(results)

        return RetrievalResponse(
            query=query,
            chunks=chunks,
            retrieval_method="dense",
            hop_count=1,
            debug_info={
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "top_k": config.top_k,
            },
        )

    def _parse_chroma_results(self, results: dict[str, Any]) -> list[RetrievedChunk]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved_chunks: list[RetrievedChunk] = []

        for rank, (chunk_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1,
        ):
            metadata = metadata or {}

            source_label = str(metadata.get("source_label", ""))
            article_number = str(metadata.get("article_number", ""))
            title_path = self._decode_title_path(metadata.get("title_path", ""))

            # Chroma cosine distance: smaller is better.
            # This score is only a simple convenience score for display/fusion.
            score = 1.0 - float(distance)

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(document),
                    source_label=source_label,
                    article_number=article_number,
                    title_path=title_path,
                    score=score,
                    distance=float(distance),
                    rank=rank,
                    retrieval_method="dense",
                    metadata=dict(metadata),
                )
            )

        return retrieved_chunks

    @staticmethod
    def _decode_title_path(raw_title_path: Any) -> str:
        """
        Chroma metadata stores title_path as a JSON string.
        Convert it into a readable string for display.
        """
        if not raw_title_path:
            return ""

        if isinstance(raw_title_path, str):
            try:
                decoded = json.loads(raw_title_path)
                if isinstance(decoded, list):
                    return " > ".join(str(item) for item in decoded)
            except json.JSONDecodeError:
                return raw_title_path

            return raw_title_path

        return str(raw_title_path)