from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from backend.generation.prompts import (
    SYSTEM_PROMPT,
    build_answer_prompt,
    build_reasoning_map_prompt,
)
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.schemas import RetrievalConfig, RetrievalResponse

KNOWN_NODE_TYPES = {
    "issue",
    "rule",
    "condition",
    "test",
    "outcome",
    "consequence",
    "remedy",
    "exception",
    "limitation",
    "duty",
    "related",
}

NODE_TYPE_ALIASES = {
    "main_rule": "rule",
    "legal_rule": "rule",
    "legal_basis": "rule",
    "standard": "test",
    "legal_standard": "test",
    "how_to_decide": "test",
    "risk": "consequence",
    "liability": "consequence",
    "claim": "remedy",
    "compensation": "remedy",
    "damages": "remedy",
    "right": "outcome",
    "entitlement": "outcome",
    "obligation": "duty",
    "responsibility": "duty",
}


@dataclass(frozen=True)
class AnswerGenerationConfig:
    """
    Configuration for source-grounded answer generation.

    llm_mode:
        - "prompt_only": do not call an LLM; return the built prompt for debugging.
        - "openai_compatible": call an OpenAI-compatible chat completions endpoint.

    Environment variables used in openai_compatible mode:
        - LLM_API_KEY
        - LLM_BASE_URL
        - LLM_MODEL

    Fallbacks:
        - OPENAI_API_KEY can be used if LLM_API_KEY is not set.
        - LLM_BASE_URL defaults to https://api.openai.com/v1
    """

    llm_mode: str = "prompt_only"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 800
    request_timeout_seconds: int = 60


@dataclass(frozen=True)
class GeneratedAnswer:
    query: str
    answer: str
    prompt: str
    retrieval_response: RetrievalResponse
    llm_mode: str
    model_name: str
    debug_info: dict[str, Any]
    reasoning_map: dict[str, Any] | None = None
    reasoning_map_prompt: str | None = None
    reasoning_map_raw: str | None = None


class AnswerGenerator:
    """
    Source-grounded answer generator for the law_rag project.

    This class deliberately separates retrieval from generation:
    - HybridRetriever retrieves and reranks legal chunks.
    - AnswerGenerator first asks the LLM for a compact reasoning_map JSON.
    - AnswerGenerator then uses that reasoning_map to guide the final answer.

    In prompt_only mode, no external API call is made. This is useful for
    debugging and for portfolio demos where retrieval quality is the focus.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        config: AnswerGenerationConfig | None = None,
    ) -> None:
        load_dotenv()

        self.retriever = retriever or HybridRetriever()
        self.config = config or AnswerGenerationConfig()

    def generate(
        self,
        query: str,
        retrieval_config: RetrievalConfig | None = None,
    ) -> GeneratedAnswer:
        if retrieval_config is None:
            retrieval_config = RetrievalConfig(
                top_k=5,
                candidate_k=20,
                enable_reranker=True,
            )

        retrieval_response = self.retriever.retrieve(
            query=query,
            config=retrieval_config,
        )

        reasoning_map_prompt = build_reasoning_map_prompt(
            query=query,
            chunks=retrieval_response.chunks,
        )

        reasoning_map_raw: str | None = None
        reasoning_map: dict[str, Any] | None = None
        reasoning_map_json_for_answer: str | None = None

        if self.config.llm_mode == "prompt_only":
            prompt = build_answer_prompt(
                query=query,
                chunks=retrieval_response.chunks,
                reasoning_map_json=None,
            )

            answer = self._build_prompt_only_answer(
                reasoning_map_prompt=reasoning_map_prompt,
                answer_prompt=prompt,
            )

        elif self.config.llm_mode == "openai_compatible":
            reasoning_map_raw = self._call_openai_compatible_chat_completion(
                reasoning_map_prompt,
                max_tokens=700,
            )

            reasoning_map = self._parse_reasoning_map(reasoning_map_raw)

            if reasoning_map is not None:
                reasoning_map_json_for_answer = json.dumps(
                    reasoning_map,
                    ensure_ascii=False,
                    indent=2,
                )

            prompt = build_answer_prompt(
                query=query,
                chunks=retrieval_response.chunks,
                reasoning_map_json=reasoning_map_json_for_answer,
            )

            answer = self._call_openai_compatible_chat_completion(
                prompt,
                max_tokens=self.config.max_tokens,
            )

        else:
            raise ValueError(
                "Unsupported llm_mode. Expected 'prompt_only' or "
                f"'openai_compatible', got: {self.config.llm_mode}"
            )

        return GeneratedAnswer(
            query=query,
            answer=answer,
            prompt=prompt,
            retrieval_response=retrieval_response,
            llm_mode=self.config.llm_mode,
            model_name=self.config.model_name,
            debug_info={
                "retrieval_method": retrieval_response.retrieval_method,
                "retrieval_debug_info": retrieval_response.debug_info,
                "source_count": len(retrieval_response.chunks),
                "reasoning_map_parse_success": reasoning_map is not None,
            },
            reasoning_map=reasoning_map,
            reasoning_map_prompt=reasoning_map_prompt,
            reasoning_map_raw=reasoning_map_raw,
        )

    def _build_prompt_only_answer(
        self,
        reasoning_map_prompt: str,
        answer_prompt: str,
    ) -> str:
        return (
            "PROMPT_ONLY_MODE\n\n"
            "No LLM call was made. The grounded prompts are shown below.\n\n"
            "=== REASONING MAP PROMPT ===\n\n"
            f"{reasoning_map_prompt}\n\n"
            "=== ANSWER PROMPT ===\n\n"
            f"{answer_prompt}"
        )

    def _call_openai_compatible_chat_completion(
        self,
        prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model_name = os.getenv("LLM_MODEL", self.config.model_name)

        if not api_key:
            raise ValueError(
                "Missing API key. Set LLM_API_KEY or OPENAI_API_KEY, "
                "or use llm_mode='prompt_only'."
            )

        url = f"{base_url}/chat/completions"

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")

        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8")
            raise RuntimeError(
                f"LLM request failed with status {error.code}: {error_body}"
            ) from error

        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error}") from error

        data = json.loads(response_body)

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM response did not contain choices: {data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if not content:
            raise RuntimeError(f"LLM response did not contain content: {data}")

        return str(content).strip()

    def _parse_reasoning_map(self, raw_content: str) -> dict[str, Any] | None:
        """
        Parse and lightly validate the reasoning_map JSON returned by the LLM.

        If parsing or validation fails, return None. The frontend already handles
        a missing reasoning_map by not rendering the graph.
        """
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            extracted_json = self._extract_json_object(raw_content)

            if extracted_json is None:
                return None

            try:
                parsed = json.loads(extracted_json)
            except json.JSONDecodeError:
                return None

        if not isinstance(parsed, dict):
            return None

        nodes = parsed.get("nodes")
        edges = parsed.get("edges")

        if not isinstance(nodes, list) or not isinstance(edges, list):
            return None

        normalized_nodes: list[dict[str, Any]] = []
        node_ids: set[str] = set()

        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue

            raw_id = node.get("id")
            raw_node_type = node.get("node_type")
            raw_label = node.get("label")

            if not isinstance(raw_label, str) or not raw_label.strip():
                continue

            node_id = (
                self._to_snake_case(raw_id)
                if isinstance(raw_id, str) and raw_id.strip()
                else f"node_{index + 1}"
            )

            node_type = (
                raw_node_type.strip().lower()
                if isinstance(raw_node_type, str)
                else "related"
            )

            node_type = NODE_TYPE_ALIASES.get(node_type, node_type)

            if node_type not in KNOWN_NODE_TYPES:
                node_type = "related"
                
            if node_id in node_ids:
                node_id = f"{node_id}_{index + 1}"

            node_ids.add(node_id)

            normalized_nodes.append(
                {
                    "id": node_id,
                    "node_type": node_type,
                    "label": raw_label.strip(),
                    "description": self._safe_optional_string(
                        node.get("description"),
                        default="",
                    ),
                    "article_label": self._safe_optional_string(
                        node.get("article_label"),
                        default=None,
                    ),
                    "source_url": self._safe_optional_string(
                        node.get("source_url"),
                        default=None,
                    ),
                }
            )

        if len(normalized_nodes) < 2:
            return None

        normalized_node_ids = {node["id"] for node in normalized_nodes}
        normalized_edges: list[dict[str, Any]] = []

        for edge in edges:
            if not isinstance(edge, dict):
                continue

            source = edge.get("source")
            target = edge.get("target")
            label = edge.get("label")

            if not isinstance(source, str) or not isinstance(target, str):
                continue

            source = self._to_snake_case(source)
            target = self._to_snake_case(target)

            if source not in normalized_node_ids or target not in normalized_node_ids:
                continue

            normalized_label: str | None = None
            if isinstance(label, str) and label.strip():
                clean_label = label.strip()

                if clean_label.lower() == "yes":
                    normalized_label = "Yes"
                elif clean_label.lower() == "no":
                    normalized_label = "No"
                else:
                    normalized_label = clean_label

            normalized_edges.append(
                {
                    "source": source,
                    "target": target,
                    "label": normalized_label,
                }
            )

        if not normalized_edges:
            for index in range(len(normalized_nodes) - 1):
                normalized_edges.append(
                    {
                        "source": normalized_nodes[index]["id"],
                        "target": normalized_nodes[index + 1]["id"],
                        "label": None,
                    }
                )

        summary = parsed.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "Legal reasoning map."

        return {
            "summary": summary.strip(),
            "nodes": normalized_nodes,
            "edges": normalized_edges,
        }

    def _extract_json_object(self, text: str) -> str | None:
        """
        Extract the first likely JSON object from an LLM response.

        This is a fallback in case the model accidentally wraps JSON in text or
        markdown despite the prompt asking for JSON only.
        """
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)

        if fenced_match:
            return fenced_match.group(1)

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return None

        return text[start : end + 1]

    def _to_snake_case(self, value: str) -> str:
        cleaned = value.strip()
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned)
        cleaned = cleaned.strip("_").lower()

        return cleaned or "node"

    def _safe_optional_string(
        self,
        value: Any,
        default: str | None,
    ) -> str | None:
        if value is None:
            return default

        if not isinstance(value, str):
            return default

        cleaned = value.strip()

        if not cleaned:
            return default

        return cleaned