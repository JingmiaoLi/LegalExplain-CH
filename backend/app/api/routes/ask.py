from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
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


ALLOWED_NODE_TYPES = {
    "issue",
    "rule",
    "condition",
    "outcome",
    "consequence",
    "exception",
    "limitation",
    "duty",
    "right",
    "remedy",
    "deadline",
    "procedure",
    "additional_check",
    "related",
}



def build_source_items(chunks: list[Any]) -> list[SourceResponse]:
    """
    Convert retrieved chunks into API source responses.

    This keeps the /ask route independent from the exact retrieved chunk class,
    while still preserving all source metadata needed by the frontend.
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
                chunk_id=str(getattr(chunk, "chunk_id", f"chunk_{index}") or f"chunk_{index}"),
                metadata=metadata,
            )
        )

    return sources

def _get_source_url(source: SourceResponse) -> str | None:
    source_url = source.metadata.get("source_url")

    if isinstance(source_url, str) and source_url:
        return source_url

    return None


def _normalize_article_number(article_number: str | None) -> str:
    if not article_number:
        return ""

    return article_number.lower().replace("art.", "").replace("art", "").strip()


def _find_source_by_article(
    sources: list[SourceResponse],
    article_number: str | None,
) -> SourceResponse | None:
    target = _normalize_article_number(article_number)

    if not target:
        return None

    for source in sources:
        if _normalize_article_number(source.article_number) == target:
            return source

    return None


def _format_sources_for_reasoning_prompt(sources: list[SourceResponse]) -> str:
    blocks: list[str] = []

    for index, source in enumerate(sources, start=1):
        source_text = source.text.strip()

        if len(source_text) > 1200:
            source_text = f"{source_text[:1200]}..."

        blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"source_label: {source.source_label}",
                    f"article_number: {source.article_number}",
                    f"title_path: {source.title_path}",
                    "text:",
                    source_text,
                ]
            )
        )

    return "\n\n".join(blocks)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract the first JSON object from an LLM response.

    The model should return pure JSON, but local models sometimes wrap it in
    markdown fences or add short explanations. This keeps the parser tolerant.
    """
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return None

    candidate = cleaned[first_brace : last_brace + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed

    return None


def _call_openai_compatible_json(
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any] | None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "llama3.1:8b")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }

    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            response_body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        return None

    try:
        parsed_response = json.loads(response_body)
    except json.JSONDecodeError:
        return None

    choices = parsed_response.get("choices")

    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]

    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")

    if not isinstance(message, dict):
        return None

    content = message.get("content")

    if not isinstance(content, str):
        return None

    return _extract_json_object(content)


def build_reasoning_map_prompt(
    query: str,
    answer: str,
    sources: list[SourceResponse],
) -> tuple[str, str]:
    source_context = _format_sources_for_reasoning_prompt(sources)
    allowed_articles = ", ".join(
        source.article_number for source in sources if source.article_number
    )

    system_prompt = """
You create compact visual legal reasoning maps from source-grounded legal answers.

Return only valid JSON.
Do not include markdown fences.
Do not include explanations outside JSON.
""".strip()

    user_prompt = f"""
User question:
{query}

Generated answer:
{answer}

Retrieved legal sources:
{source_context}

Allowed article numbers:
{allowed_articles}

Task:
Create a compact reasoning_map JSON object that helps a user quickly understand the legal reasoning path.

The map should be useful for users who do not want to read a long text answer.

Rules:
- Use only the retrieved legal sources.
- Do not invent legal rules, article numbers, remedies, deadlines, exceptions, or procedures.
- Every legal rule node should cite an article from the allowed article numbers.
- Do not include long statutory wording.
- Keep node labels short and user-friendly.
- Keep descriptions brief, ideally one short sentence.
- Prefer a decision structure when the issue depends on a legal condition.
- Use branches such as "Yes" and "No" only when the sources support a condition-based reasoning path.
- Do not add a "facts needed" node. If facts are needed, that belongs in the answer, not the graph.
- The graph should usually have 2 to 6 nodes.
- The graph should not repeat the full text answer.

Allowed node_type values:
issue, rule, condition, outcome, consequence, exception, limitation, duty, right, remedy, deadline, procedure, additional_check, related

Required JSON shape:
{{
  "title": "short title",
  "summary": "one-sentence explanation of what the map shows",
  "nodes": [
    {{
      "id": "short_unique_id",
      "node_type": "condition",
      "label": "short node label",
      "description": "brief explanation",
      "article_number": "337"
    }}
  ],
  "edges": [
    {{
      "source": "source_node_id",
      "target": "target_node_id",
      "label": "Yes"
    }}
  ]
}}

Important:
- article_number must be null or one of the allowed article numbers.
- Omit article_label and source_url. The backend will add them.
- Return only JSON.
""".strip()

    return system_prompt, user_prompt


def _sanitize_node_id(raw_id: Any, fallback: str) -> str:
    if not isinstance(raw_id, str):
        return fallback

    cleaned = re.sub(r"[^a-zA-Z0-9_\\-]", "_", raw_id.strip())

    if not cleaned:
        return fallback

    return cleaned[:60]


def validate_reasoning_map(
    raw_map: dict[str, Any],
    sources: list[SourceResponse],
) -> ReasoningMap | None:
    title = raw_map.get("title")
    summary = raw_map.get("summary")
    raw_nodes = raw_map.get("nodes")
    raw_edges = raw_map.get("edges")

    if not isinstance(title, str) or not title.strip():
        title = "Legal reasoning map"

    if not isinstance(summary, str) or not summary.strip():
        summary = "A visual path showing the main legal reasoning."

    if not isinstance(raw_nodes, list) or not raw_nodes:
        return None

    source_by_article = {
        _normalize_article_number(source.article_number): source for source in sources
    }

    nodes: list[ReasoningNode] = []
    used_ids: set[str] = set()

    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            continue

        node_id = _sanitize_node_id(raw_node.get("id"), fallback=f"node_{index + 1}")

        while node_id in used_ids:
            node_id = f"{node_id}_{index + 1}"

        node_type = raw_node.get("node_type")

        if not isinstance(node_type, str):
            node_type = "related"

        node_type = node_type.strip().lower()

        if node_type not in ALLOWED_NODE_TYPES:
            node_type = "related"

        label = raw_node.get("label")
        description = raw_node.get("description")
        article_number = raw_node.get("article_number")

        if not isinstance(label, str) or not label.strip():
            continue

        if not isinstance(description, str):
            description = ""

        if article_number is not None and not isinstance(article_number, str):
            article_number = None

        source = _find_source_by_article(sources, article_number)

        article_label: str | None = None
        source_url: str | None = None
        normalized_article = ""

        if source is not None:
            normalized_article = source.article_number
            article_label = source.source_label
            source_url = _get_source_url(source)
        else:
            article_number = None

        nodes.append(
            ReasoningNode(
                id=node_id,
                node_type=node_type,
                label=label.strip()[:120],
                description=description.strip()[:260],
                article_label=article_label,
                article_number=normalized_article or None,
                source_url=source_url,
            )
        )
        used_ids.add(node_id)

    if not nodes:
        return None

    node_ids = {node.id for node in nodes}
    edges: list[ReasoningEdge] = []

    if isinstance(raw_edges, list):
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue

            source_id = raw_edge.get("source")
            target_id = raw_edge.get("target")
            label = raw_edge.get("label")

            if not isinstance(source_id, str) or not isinstance(target_id, str):
                continue

            if source_id not in node_ids or target_id not in node_ids:
                continue

            if label is not None and not isinstance(label, str):
                label = None

            edges.append(
                ReasoningEdge(
                    source=source_id,
                    target=target_id,
                    label=label.strip()[:40] if isinstance(label, str) else None,
                )
            )

    if not edges and len(nodes) >= 2:
        for current, following in zip(nodes, nodes[1:]):
            edges.append(
                ReasoningEdge(
                    source=current.id,
                    target=following.id,
                )
            )

    return ReasoningMap(
        title=title.strip()[:120],
        summary=summary.strip()[:220],
        nodes=nodes,
        edges=edges,
    )


def build_fallback_reasoning_map(
    query: str,
    sources: list[SourceResponse],
) -> ReasoningMap | None:
    if not sources:
        return None

    top_sources = sources[:3]
    nodes: list[ReasoningNode] = [
        ReasoningNode(
            id="issue",
            node_type="issue",
            label="Legal question",
            description=query,
        )
    ]

    edges: list[ReasoningEdge] = []

    for index, source in enumerate(top_sources, start=1):
        node_id = f"source_{index}"

        nodes.append(
            ReasoningNode(
                id=node_id,
                node_type="rule" if index == 1 else "related",
                label=f"Relevant legal basis: {source.source_label}",
                description=source.title_path,
                article_label=source.source_label,
                article_number=source.article_number,
                source_url=_get_source_url(source),
            )
        )
        edges.append(
            ReasoningEdge(
                source="issue",
                target=node_id,
            )
        )

    return ReasoningMap(
        title="Main legal basis",
        summary="A source-grounded map of the main legal articles retrieved for this question.",
        nodes=nodes,
        edges=edges,
    )


def build_reasoning_map(
    query: str,
    answer: str,
    sources: list[SourceResponse],
    llm_mode: str,
) -> ReasoningMap | None:
    """
    Build a general reasoning map for any legal question.

    The preferred path uses the configured OpenAI-compatible local/remote LLM
    to generate a compact JSON graph. The backend validates article references
    and graph structure before returning it. If generation or validation fails,
    the function falls back to a simple source-based map.
    """
    if not sources:
        return None

    if llm_mode == "openai_compatible":
        system_prompt, user_prompt = build_reasoning_map_prompt(
            query=query,
            answer=answer,
            sources=sources,
        )
        raw_map = _call_openai_compatible_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if raw_map is not None:
            validated_map = validate_reasoning_map(
                raw_map=raw_map,
                sources=sources,
            )

            if validated_map is not None:
                return validated_map

    return build_fallback_reasoning_map(
        query=query,
        sources=sources,
    )


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
