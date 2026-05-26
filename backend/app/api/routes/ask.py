from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.generation.answer_generator import (
    AnswerGenerationConfig,
    AnswerGenerator,
)
from backend.retrieval.schemas import RetrievalConfig


router = APIRouter(prefix="/ask", tags=["ask"])


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    llm_mode: str = "prompt_only"
    top_k: int = 5
    candidate_k: int = 20
    enable_reranker: bool = True


class SourceResponse(BaseModel):
    source_label: str
    article_number: str
    title_path: str
    text: str
    score: float
    rank: int | None = None
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    debug_info: dict[str, Any] = Field(default_factory=dict)


@router.post("", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    retrieval_config = RetrievalConfig(
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        enable_reranker=request.enable_reranker,
    )

    generation_config = AnswerGenerationConfig(
        llm_mode=request.llm_mode,
    )

    generator = AnswerGenerator(config=generation_config)

    generated_answer = generator.generate(
        query=request.query,
        retrieval_config=retrieval_config,
    )

    sources = [
        SourceResponse(
            source_label=chunk.source_label,
            article_number=chunk.article_number,
            title_path=str(chunk.title_path or ""),
            text=chunk.text,
            score=float(chunk.score),
            rank=chunk.rank,
            chunk_id=chunk.chunk_id,
            metadata=dict(chunk.metadata or {}),
        )
        for chunk in generated_answer.retrieval_response.chunks
    ]

    return AskResponse(
        answer=generated_answer.answer,
        sources=sources,
        debug_info=generated_answer.debug_info,
    )