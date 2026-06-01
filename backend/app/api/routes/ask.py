from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.generation.answer_generator import (
    AnswerGenerationConfig,
    AnswerGenerator,
)
from backend.retrieval.schemas import RetrievalConfig


router = APIRouter(prefix="/ask", tags=["ask"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    response_mode: Literal["text_map", "text_only", "map_only"] = "text_map"
    
    llm_mode: str = "prompt_only"

    top_k: int = 5
    candidate_k: int = 20
    enable_reranker: bool = True

    enable_query_rewrite: bool = True
    enable_reasoning_map: bool = True

class SourceResponse(BaseModel):
    source_label: str
    article_number: str
    title_path: str
    text: str
    score: float
    rank: int | None = None
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasoningNode(BaseModel):
    id: str
    node_type: str
    label: str
    description: str = ""
    article_label: str | None = None
    article_number: str | None = None
    source_url: str | None = None


class ReasoningEdge(BaseModel):
    source: str
    target: str
    label: str | None = None


class ReasoningMap(BaseModel):
    title: str = "Legal reasoning map"
    map_type: str = "mixed"
    summary: str = ""
    nodes: list[ReasoningNode] = Field(default_factory=list)
    edges: list[ReasoningEdge] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    reasoning_map: ReasoningMap | None = None
    sources: list[SourceResponse]
    debug_info: dict[str, Any] = Field(default_factory=dict)


def build_source_items(chunks: list[Any]) -> list[SourceResponse]:
    """
    Convert retrieved chunks into API source responses.

    This keeps the /ask route independent from the exact retrieved chunk class,
    while preserving source metadata needed by the frontend.
    """
    sources: list[SourceResponse] = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = getattr(chunk, "metadata", {}) or {}

        if not isinstance(metadata, dict):
            metadata = {}

        source_label = getattr(chunk, "source_label", "") or ""
        article_number = getattr(chunk, "article_number", "") or ""

        sources.append(
            SourceResponse(
                source_label=str(source_label),
                article_number=str(article_number),
                title_path=str(getattr(chunk, "title_path", "") or ""),
                text=str(getattr(chunk, "text", "") or ""),
                score=float(getattr(chunk, "score", 0.0) or 0.0),
                rank=getattr(chunk, "rank", index),
                chunk_id=str(
                    getattr(chunk, "chunk_id", f"chunk_{index}")
                    or f"chunk_{index}"
                ),
                metadata=metadata,
            )
        )

    return sources


def dump_chat_message(message: ChatMessage) -> dict[str, str]:
    """
    Support both Pydantic v2 and v1 style serialization.
    """
    if hasattr(message, "model_dump"):
        return message.model_dump()

    return message.dict()


@router.post("", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    retrieval_config = RetrievalConfig(
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        enable_reranker=request.enable_reranker,
    )

    generation_config = AnswerGenerationConfig(
        llm_mode=request.llm_mode,
        enable_query_rewrite=request.enable_query_rewrite,
        enable_reasoning_map=request.enable_reasoning_map,
        response_mode=request.response_mode,
    )

    generator = AnswerGenerator(config=generation_config)

    generated_answer = generator.generate(
        query=request.query,
        retrieval_config=retrieval_config,
        conversation_history=[
            dump_chat_message(message)
            for message in request.conversation_history
        ],
    )

    sources = build_source_items(
        generated_answer.retrieval_response.chunks
    )

    debug_info = {
        **generated_answer.debug_info,
    }

    reasoning_map: ReasoningMap | None = None

    if generated_answer.reasoning_map is not None:
        try:
            reasoning_map = ReasoningMap.model_validate(
                generated_answer.reasoning_map
            )
        except AttributeError:
            reasoning_map = ReasoningMap.parse_obj(
                generated_answer.reasoning_map
            )
        except Exception as error:
            reasoning_map = None
            debug_info["reasoning_map_validation_error"] = str(error)

    debug_info["reasoning_map_generated"] = reasoning_map is not None
    debug_info["reasoning_map_mode"] = (
        "llm_structured" if reasoning_map is not None else "none"
    )

    return AskResponse(
        answer=generated_answer.answer,
        sources=sources,
        reasoning_map=reasoning_map,
        debug_info=debug_info,
    )