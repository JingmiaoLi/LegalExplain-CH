from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from nltk.stem import SnowballStemmer
from rank_bm25 import BM25Okapi

from backend.retrieval.schemas import RetrievedChunk, RetrievalConfig, RetrievalResponse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "article_chunks.json"


GENERAL_STOPWORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "can", "could", "should", "would", "may", "might",
    "i", "me", "my", "you", "your", "he", "she", "his", "her",
    "we", "our", "they", "their", "them",
    "what", "when", "where", "why", "how", "if",
    "to", "of", "in", "on", "for", "from", "by", "with", "at", "as",
    "into", "over", "after", "before", "between", "through",
    "this", "that", "these", "those",
    "have", "has", "had",
    "it", "its",
    "not",
    "under",
    "during",
    "there",
    "here",
    "any",
    "all",
    "such",
    "than",
    "then",
    "so",
    "also",
}


class KeywordRetriever:
    """
    Keyword retriever based on BM25.

    This retriever performs lexical retrieval over article chunks. It uses:
    - lowercasing
    - regex-based tokenization
    - general English stopword removal
    - lightweight stemming

    Dense retrieval should still use the full natural-language text. These
    preprocessing steps are only for BM25 keyword matching.
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        use_stemming: bool = True,
        stopwords: set[str] | None = None,
    ) -> None:
        self.chunks_path = chunks_path
        self.use_stemming = use_stemming
        self.stopwords = stopwords if stopwords is not None else GENERAL_STOPWORDS

        self.stemmer = SnowballStemmer("english")

        self._chunks: list[dict[str, Any]] | None = None
        self._documents: list[str] | None = None
        self._tokenized_documents: list[list[str]] | None = None
        self._bm25: BM25Okapi | None = None

    @property
    def chunks(self) -> list[dict[str, Any]]:
        if self._chunks is None:
            self._chunks = self._load_chunks(self.chunks_path)

        return self._chunks

    @property
    def documents(self) -> list[str]:
        if self._documents is None:
            self._documents = [self._chunk_to_document(chunk) for chunk in self.chunks]

        return self._documents

    @property
    def tokenized_documents(self) -> list[list[str]]:
        if self._tokenized_documents is None:
            self._tokenized_documents = [
                self._tokenize(document) for document in self.documents
            ]

        return self._tokenized_documents

    @property
    def bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            self._bm25 = BM25Okapi(self.tokenized_documents)

        return self._bm25

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResponse:
        """
        Retrieve chunks using BM25 keyword matching.
        """
        if config is None:
            config = RetrievalConfig()

        config.validate()

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return RetrievalResponse(
                query=query,
                chunks=[],
                retrieval_method="keyword",
                hop_count=1,
                debug_info={
                    "chunks_path": str(self.chunks_path),
                    "reason": "empty_query_after_tokenization",
                    "use_stemming": self.use_stemming,
                },
            )

        raw_scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(raw_scores)),
            key=lambda index: raw_scores[index],
            reverse=True,
        )

        top_indices = ranked_indices[: config.top_k]

        max_score = max(raw_scores) if len(raw_scores) > 0 else 0.0

        retrieved_chunks: list[RetrievedChunk] = []

        for rank, index in enumerate(top_indices, start=1):
            chunk = self.chunks[index]
            document = self.documents[index]
            raw_score = float(raw_scores[index])
            normalized_score = self._normalize_score(raw_score, max_score)

            metadata = self._build_metadata(chunk)
            metadata["bm25_raw_score"] = raw_score

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk.get("chunk_id", f"chunk_{index}")),
                    text=document,
                    source_label=str(chunk.get("source_label", "")),
                    article_number=str(chunk.get("article_number", "")),
                    title_path=self._format_title_path(chunk.get("title_path", "")),
                    score=normalized_score,
                    distance=None,
                    rank=rank,
                    retrieval_method="keyword",
                    metadata=metadata,
                )
            )

        return RetrievalResponse(
            query=query,
            chunks=retrieved_chunks,
            retrieval_method="keyword",
            hop_count=1,
            debug_info={
                "chunks_path": str(self.chunks_path),
                "top_k": config.top_k,
                "query_tokens": query_tokens,
                "use_stemming": self.use_stemming,
                "stopword_count": len(self.stopwords),
            },
        )

    @staticmethod
    def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {chunks_path}. "
                "Please run scripts/build_article_chunks.py first."
            )

        with chunks_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            return data["chunks"]

        raise ValueError(
            "Expected article_chunks.json to contain either a list of chunks "
            "or a dictionary with a 'chunks' list."
        )

    @staticmethod
    def _chunk_to_document(chunk: dict[str, Any]) -> str:
        """
        Build the searchable BM25 document.

        We include source label and title path because article numbers and legal
        section headings are useful lexical signals in legal retrieval.
        """
        source_label = str(chunk.get("source_label", ""))
        title_path = chunk.get("title_path", [])
        text = str(chunk.get("text", ""))

        title_context = KeywordRetriever._format_title_path(title_path)

        parts = []

        if source_label:
            parts.append(source_label)

        if title_context:
            parts.append(title_context)

        if text:
            parts.append(text)

        return "\n".join(parts).strip()

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Steps:
        1. Lowercase text.
        2. Extract alphanumeric tokens.
        3. Remove general English stopwords.
        4. Apply lightweight stemming if enabled.

        This preprocessing is intentionally used only for keyword/BM25 retrieval,
        not for dense embedding retrieval.
        """
        text = text.lower()
        raw_tokens = re.findall(r"[a-z0-9]+", text)

        tokens: list[str] = []

        for token in raw_tokens:
            if len(token) <= 1:
                continue

            if token in self.stopwords:
                continue

            if self.use_stemming:
                token = self.stemmer.stem(token)

            tokens.append(token)

        return tokens

    @staticmethod
    def _normalize_score(raw_score: float, max_score: float) -> float:
        """
        Normalize BM25 scores into a rough 0-1 range.

        BM25 scores are not probabilities. This normalization is mainly for
        easier display and later hybrid score fusion.
        """
        if max_score <= 0:
            return 0.0

        score = raw_score / max_score

        if math.isnan(score) or math.isinf(score):
            return 0.0

        return max(0.0, min(1.0, score))

    @staticmethod
    def _format_title_path(raw_title_path: Any) -> str:
        if not raw_title_path:
            return ""

        if isinstance(raw_title_path, list):
            return " > ".join(str(item) for item in raw_title_path)

        if isinstance(raw_title_path, str):
            try:
                decoded = json.loads(raw_title_path)
                if isinstance(decoded, list):
                    return " > ".join(str(item) for item in decoded)
            except json.JSONDecodeError:
                return raw_title_path

            return raw_title_path

        return str(raw_title_path)

    @staticmethod
    def _build_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "chunk_id": str(chunk.get("chunk_id", "")),
            "chunk_type": str(chunk.get("chunk_type", "")),
            "article_number": str(chunk.get("article_number", "")),
            "source_label": str(chunk.get("source_label", "")),
            "title_path": KeywordRetriever._format_title_path(
                chunk.get("title_path", "")
            ),
            "paragraph_count": int(chunk.get("paragraph_count", 0)),
            "word_count": int(chunk.get("word_count", 0)),
            "char_count": int(chunk.get("char_count", 0)),
            "source_url": str(chunk.get("source_url", "")),
            "source_type": str(chunk.get("source_type", "")),
            "status": str(chunk.get("status", "")),
        }