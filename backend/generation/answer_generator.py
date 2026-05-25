from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from backend.generation.prompts import SYSTEM_PROMPT, build_answer_prompt
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.schemas import RetrievalConfig, RetrievalResponse


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


class AnswerGenerator:
    """
    Source-grounded answer generator for the law_rag project.

    This class deliberately separates retrieval from generation:
    - HybridRetriever retrieves and reranks legal chunks.
    - AnswerGenerator builds a grounded prompt.
    - Optional LLM call generates the final answer.

    In prompt_only mode, no external API call is made. This is useful for
    debugging and for portfolio demos where retrieval quality is the focus.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        config: AnswerGenerationConfig | None = None,
    ) -> None:
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

        prompt = build_answer_prompt(
            query=query,
            chunks=retrieval_response.chunks,
        )

        if self.config.llm_mode == "prompt_only":
            answer = self._build_prompt_only_answer(prompt)

        elif self.config.llm_mode == "openai_compatible":
            answer = self._call_openai_compatible_chat_completion(prompt)

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
            },
        )

    def _build_prompt_only_answer(self, prompt: str) -> str:
        return (
            "PROMPT_ONLY_MODE\n\n"
            "No LLM call was made. The grounded prompt is shown below.\n\n"
            f"{prompt}"
        )

    def _call_openai_compatible_chat_completion(self, prompt: str) -> str:
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
            "max_tokens": self.config.max_tokens,
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