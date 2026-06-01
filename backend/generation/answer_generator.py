from __future__ import annotations

import time
import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from backend.generation.prompts import (
    SYSTEM_PROMPT,
    build_answer_prompt,
    build_query_rewrite_prompt,
    build_combined_answer_map_prompt,
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

KNOWN_MAP_TYPES = {
    "decision",
    "overview",
    "consequence",
    "duty",
    "mixed",
}


@dataclass(frozen=True)
class AnswerGenerationConfig:
    """
    Configuration for source-grounded answer generation.

    llm_mode:
        - "prompt_only": do not call an LLM; return the built prompt for debugging.
        - "openai_compatible": call an OpenAI-compatible chat completions endpoint.

    Cost / latency controls:
        - enable_query_rewrite: allow LLM-based follow-up query rewriting.
        - enable_reasoning_map: allow optional visual reasoning map generation.
        - query_rewrite_mode:
            - "auto": rewrite only when the query looks like a follow-up.
            - "always": rewrite whenever conversation history exists.
            - "disabled": never rewrite.
        - reasoning_map_max_tokens: cap output size for map generation.
        - query_rewrite_timeout_seconds: short timeout for rewrite calls.
        - reasoning_map_timeout_seconds: short timeout for optional map calls.
        - request_timeout_seconds: default timeout for the main answer call.

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
    max_tokens: int = 1600

    enable_query_rewrite: bool = True
    enable_reasoning_map: bool = True
    query_rewrite_mode: str = "auto"
    response_mode: str = "text_map"
    text_map_generation_mode: str = "separate"

    query_rewrite_max_tokens: int = 160
    reasoning_map_max_tokens: int = 900

    query_rewrite_timeout_seconds: int = 20
    reasoning_map_timeout_seconds: int = 30
    request_timeout_seconds: int = 120




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
    original_query: str | None = None
    standalone_query: str | None = None
    query_rewrite_prompt: str | None = None


class AnswerGenerator:
    """
    Source-grounded answer generator for the law_rag project.

    This class deliberately separates retrieval from generation:
    - HybridRetriever retrieves and reranks legal chunks.
    - AnswerGenerator rewrites follow-up queries only when needed.
    - AnswerGenerator always prioritizes the text answer.
    - AnswerGenerator optionally generates a compact reasoning_map.
    - Reasoning map failure never blocks the main answer.

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
        conversation_history: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        started_at = time.perf_counter()
        timings: dict[str, float] = {}

        if retrieval_config is None:
            retrieval_config = RetrievalConfig(
                top_k=5,
                candidate_k=20,
                enable_reranker=True,
            )

        step_started_at = time.perf_counter()

        (
            standalone_query,
            query_rewrite_prompt,
            query_rewrite_error,
        ) = self._rewrite_query_for_retrieval(
            query=query,
            conversation_history=conversation_history,
        )

        timings["query_rewrite_seconds"] = round(
            time.perf_counter() - step_started_at,
            3,
        )

        step_started_at = time.perf_counter()

        retrieval_response = self.retriever.retrieve(
            query=standalone_query,
            config=retrieval_config,
        )

        timings["retrieval_seconds"] = round(
            time.perf_counter() - step_started_at,
            3,
        )

        prompt_query = self._build_prompt_query(
            original_query=query,
            standalone_query=standalone_query,
        )

        should_generate_answer = self.config.response_mode in {
            "text_map",
            "text_only",
        }

        should_generate_reasoning_map = (
            self.config.enable_reasoning_map
            and self.config.response_mode in {"text_map", "map_only"}
        )

        use_combined_text_map = (
            self.config.response_mode == "text_map"
            and self.config.text_map_generation_mode == "combined"
            and should_generate_answer
            and should_generate_reasoning_map
        )

        prompt = ""
        reasoning_map_prompt: str | None = None
        reasoning_map_raw: str | None = None
        reasoning_map: dict[str, Any] | None = None
        reasoning_map_error: str | None = None
        answer = ""

        if self.config.llm_mode == "prompt_only":
            if use_combined_text_map:
                prompt = build_combined_answer_map_prompt(
                    query=prompt_query,
                    chunks=retrieval_response.chunks,
                )
                reasoning_map_prompt = None

            else:
                if should_generate_answer:
                    prompt = build_answer_prompt(
                        query=prompt_query,
                        chunks=retrieval_response.chunks,
                        reasoning_map_json=None,
                    )

                if should_generate_reasoning_map:
                    reasoning_map_prompt = build_reasoning_map_prompt(
                        query=prompt_query,
                        chunks=retrieval_response.chunks,
                    )

            answer = self._build_prompt_only_answer(
                query_rewrite_prompt=query_rewrite_prompt,
                answer_prompt=prompt,
                reasoning_map_prompt=reasoning_map_prompt,
            )

            timings["answer_generation_seconds"] = 0.0
            timings["reasoning_map_generation_seconds"] = 0.0
            timings["combined_generation_seconds"] = 0.0

        elif self.config.llm_mode == "openai_compatible":
            if use_combined_text_map:
                prompt = build_combined_answer_map_prompt(
                    query=prompt_query,
                    chunks=retrieval_response.chunks,
                )

                step_started_at = time.perf_counter()

                try:
                    combined_raw = self._call_openai_compatible_chat_completion(
                        prompt,
                        max_tokens=self.config.max_tokens + self.config.reasoning_map_max_tokens,
                        timeout_seconds=self.config.request_timeout_seconds,
                    )
                    answer, reasoning_map = self._parse_combined_answer_map(
                        combined_raw
                    )
                    reasoning_map_raw = combined_raw

                except RuntimeError as error:
                    answer = ""
                    reasoning_map = None
                    reasoning_map_raw = None
                    reasoning_map_error = str(error)

                combined_seconds = round(time.perf_counter() - step_started_at, 3)
                timings["combined_generation_seconds"] = combined_seconds
                timings["answer_generation_seconds"] = combined_seconds
                timings["reasoning_map_generation_seconds"] = 0.0

            else:
                if should_generate_answer:
                    prompt = build_answer_prompt(
                        query=prompt_query,
                        chunks=retrieval_response.chunks,
                        reasoning_map_json=None,
                    )

                    step_started_at = time.perf_counter()

                    answer = self._call_openai_compatible_chat_completion(
                        prompt,
                        max_tokens=self.config.max_tokens,
                        timeout_seconds=self.config.request_timeout_seconds,
                    )

                    timings["answer_generation_seconds"] = round(
                        time.perf_counter() - step_started_at,
                        3,
                    )
                else:
                    answer = ""
                    timings["answer_generation_seconds"] = 0.0

                if should_generate_reasoning_map:
                    reasoning_map_prompt = build_reasoning_map_prompt(
                        query=prompt_query,
                        chunks=retrieval_response.chunks,
                    )

                    step_started_at = time.perf_counter()

                    try:
                        reasoning_map_raw = self._call_openai_compatible_chat_completion(
                            reasoning_map_prompt,
                            max_tokens=self.config.reasoning_map_max_tokens,
                            timeout_seconds=self.config.reasoning_map_timeout_seconds,
                        )
                        reasoning_map = self._parse_reasoning_map(reasoning_map_raw)
                    except RuntimeError as error:
                        reasoning_map_raw = None
                        reasoning_map = None
                        reasoning_map_error = str(error)

                    timings["reasoning_map_generation_seconds"] = round(
                        time.perf_counter() - step_started_at,
                        3,
                    )
                else:
                    timings["reasoning_map_generation_seconds"] = 0.0

                timings["combined_generation_seconds"] = 0.0

        else:
            raise ValueError(
                "Unsupported llm_mode. Expected 'prompt_only' or "
                f"'openai_compatible', got: {self.config.llm_mode}"
            )

        timings["total_seconds"] = round(
            time.perf_counter() - started_at,
            3,
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
                "response_mode": self.config.response_mode,
                "text_map_generation_mode": self.config.text_map_generation_mode,
                "answer_generated": should_generate_answer,
                "original_query": query,
                "standalone_query": standalone_query,
                "query_rewritten": standalone_query != query,
                "query_rewrite_enabled": self.config.enable_query_rewrite,
                "query_rewrite_mode": self.config.query_rewrite_mode,
                "query_rewrite_error": query_rewrite_error,
                "conversation_turn_count": len(conversation_history or []),
                "reasoning_map_enabled": should_generate_reasoning_map,
                "reasoning_map_parse_success": reasoning_map is not None,
                "reasoning_map_error": reasoning_map_error,
                "timings": timings,
            },
            reasoning_map=reasoning_map,
            reasoning_map_prompt=reasoning_map_prompt,
            reasoning_map_raw=reasoning_map_raw,
            original_query=query,
            standalone_query=standalone_query,
            query_rewrite_prompt=query_rewrite_prompt,
        )


    def _build_prompt_query(
        self,
        original_query: str,
        standalone_query: str,
    ) -> str:
        """
        Keep the user's original question visible to the answer prompt, while
        adding the standalone retrieval query only when it adds context.
        """
        if standalone_query == original_query:
            return original_query

        return (
            f"{original_query}\n\n"
            f"Standalone retrieval query for context:\n{standalone_query}"
        )

    def _rewrite_query_for_retrieval(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None,
    ) -> tuple[str, str | None, str | None]:
        """
        Rewrite a follow-up question into a standalone retrieval query.

        To control cost, this only calls the LLM when:
        - query rewriting is enabled,
        - there is conversation history,
        - llm_mode is openai_compatible,
        - and the query looks like a follow-up unless mode is "always".
        """
        query_rewrite_prompt: str | None = None

        if not self.config.enable_query_rewrite:
            return query, None, None

        if self.config.query_rewrite_mode == "disabled":
            return query, None, None

        if not conversation_history:
            return query, None, None

        query_rewrite_prompt = build_query_rewrite_prompt(
            query=query,
            conversation_history=conversation_history,
        )

        if self.config.llm_mode != "openai_compatible":
            return query, query_rewrite_prompt, None

        should_rewrite = (
            self.config.query_rewrite_mode == "always"
            or self._looks_like_follow_up(query)
        )

        if not should_rewrite:
            return query, query_rewrite_prompt, None

        try:
            rewritten_query = self._call_openai_compatible_chat_completion(
                query_rewrite_prompt,
                max_tokens=self.config.query_rewrite_max_tokens,
                timeout_seconds=self.config.query_rewrite_timeout_seconds,
            ).strip()

        except RuntimeError as error:
            return query, query_rewrite_prompt, str(error)

        rewritten_query = rewritten_query.strip().strip('"').strip("'")

        if not rewritten_query:
            return query, query_rewrite_prompt, None

        if len(rewritten_query) > 500:
            rewritten_query = rewritten_query[:500].strip()

        return rewritten_query, query_rewrite_prompt, None

    def _looks_like_follow_up(self, query: str) -> bool:
        """
        Cheap heuristic to avoid unnecessary LLM rewrite calls.

        This intentionally errs toward not rewriting standalone legal questions,
        because unnecessary rewrite calls increase cost and latency.
        """
        lowered = query.strip().lower()

        if not lowered:
            return False

        follow_up_markers = {
            "what if",
            "what about",
            "and if",
            "then",
            "in that case",
            "same situation",
            "same case",
            "that",
            "this",
            "they",
            "them",
            "it",
            "he",
            "she",
            "their",
            "those",
            "there",
        }

        if any(marker in lowered for marker in follow_up_markers):
            return True

        words = lowered.split()

        legal_context_keywords = {
            "employer",
            "employee",
            "salary",
            "dismiss",
            "dismissal",
            "terminate",
            "termination",
            "notice",
            "contract",
            "overtime",
            "vacation",
            "illness",
            "sick",
            "damages",
            "compensation",
            "swiss",
            "employment",
            "law",
        }

        has_legal_context = any(keyword in lowered for keyword in legal_context_keywords)

        if len(words) <= 7 and not has_legal_context:
            return True

        return False

    def _build_prompt_only_answer(
        self,
        query_rewrite_prompt: str | None,
        answer_prompt: str,
        reasoning_map_prompt: str | None,
    ) -> str:
        query_rewrite_section = (
            "=== QUERY REWRITE PROMPT ===\n\n"
            f"{query_rewrite_prompt}\n\n"
            if query_rewrite_prompt
            else ""
        )

        reasoning_map_section = (
            "=== REASONING MAP PROMPT ===\n\n"
            f"{reasoning_map_prompt}\n\n"
            if reasoning_map_prompt
            else ""
        )

        return (
            "PROMPT_ONLY_MODE\n\n"
            "No LLM call was made. The grounded prompts are shown below.\n\n"
            f"{query_rewrite_section}"
            "=== ANSWER PROMPT ===\n\n"
            f"{answer_prompt}\n\n"
            f"{reasoning_map_section}"
        )

    def _uses_max_completion_tokens(self, model_name: str) -> bool:
        """
        Some OpenAI reasoning/newer chat models reject max_tokens and require
        max_completion_tokens instead. Most local OpenAI-compatible servers such
        as Ollama still expect max_tokens.
        """
        normalized_model = model_name.lower()

        return normalized_model.startswith(
            (
                "gpt-5",
                "o1",
                "o3",
                "o4",
            )
        )

    def _supports_custom_temperature(self, model_name: str) -> bool:
        """
        Some newer OpenAI models only support the default temperature.
        For those models, do not send the temperature field at all.
        """
        normalized_model = model_name.lower()

        return not normalized_model.startswith(
            (
                "gpt-5",
                "o1",
                "o3",
                "o4",
            )
        )

    def _call_openai_compatible_chat_completion(
        self,
        prompt: str,
        max_tokens: int | None = None,
        timeout_seconds: int | None = None,
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

        token_limit = max_tokens or self.config.max_tokens

        uses_max_completion_tokens = self._uses_max_completion_tokens(model_name)

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
        }

        if uses_max_completion_tokens:
            payload["max_completion_tokens"] = token_limit
        else:
            payload["max_tokens"] = token_limit

        if self._supports_custom_temperature(model_name):
            payload["temperature"] = self.config.temperature


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
                timeout=timeout_seconds or self.config.request_timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8")

        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8")
            raise RuntimeError(
                f"LLM request failed with status {error.code}: {error_body}"
            ) from error

        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            raise RuntimeError(f"LLM request failed or timed out: {error}") from error

        data = json.loads(response_body)

        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError(f"LLM response did not contain choices: {data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if not content:
            raise RuntimeError(f"LLM response did not contain content: {data}")

        return str(content).strip()
    
    
    def _parse_combined_answer_map(
        self,
        raw_content: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Parse a combined JSON response containing both answer and reasoning_map.

        If the JSON is malformed, treat the raw response as answer text and return
        no reasoning map. This prevents the UI from failing completely.
        """
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            extracted_json = self._extract_json_object(raw_content)

            if extracted_json is None:
                return raw_content.strip(), None

            try:
                parsed = json.loads(extracted_json)
            except json.JSONDecodeError:
                return raw_content.strip(), None

        if not isinstance(parsed, dict):
            return raw_content.strip(), None

        answer = parsed.get("answer")
        if not isinstance(answer, str):
            answer = ""

        raw_reasoning_map = parsed.get("reasoning_map")
        reasoning_map: dict[str, Any] | None = None

        if isinstance(raw_reasoning_map, dict):
            reasoning_map = self._parse_reasoning_map(
                json.dumps(raw_reasoning_map, ensure_ascii=False)
            )

        return answer.strip(), reasoning_map
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

        if "reasoning_map" in parsed and isinstance(parsed["reasoning_map"], dict):
            parsed = parsed["reasoning_map"]

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
                    "article_number": self._safe_optional_string(
                        node.get("article_number"),
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

        title = parsed.get("title")
        if not isinstance(title, str) or not title.strip():
            title = "Legal reasoning map"

        map_type = parsed.get("map_type")
        if not isinstance(map_type, str) or not map_type.strip():
            map_type = "mixed"

        map_type = map_type.strip().lower()

        if map_type not in KNOWN_MAP_TYPES:
            map_type = "mixed"

        summary = parsed.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "Legal reasoning map."

        return {
            "title": title.strip(),
            "map_type": map_type,
            "summary": summary.strip(),
            "nodes": normalized_nodes,
            "edges": normalized_edges,
        }

    def _extract_json_object(self, text: str) -> str | None:
        """
        Extract the first likely JSON object from an LLM response.

        Handles raw JSON, fenced JSON, and responses with extra text around JSON.
        """
        cleaned_text = text.strip()

        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned_text,
            re.DOTALL,
        )

        if fenced_match:
            fenced_content = fenced_match.group(1).strip()
            start = fenced_content.find("{")
            end = fenced_content.rfind("}")

            if start != -1 and end != -1 and end > start:
                return fenced_content[start : end + 1]

        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return None

        return cleaned_text[start : end + 1]


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