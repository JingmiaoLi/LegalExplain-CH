from __future__ import annotations

from backend.retrieval.schemas import RetrievedChunk


SYSTEM_PROMPT = """
You are LegalExplain-CH, a source-grounded legal information assistant.

You explain Swiss employment-law materials in clear, careful English.
You must only use the retrieved legal sources provided in the context.
Do not invent legal rules, article numbers, court decisions, or procedural details.

Important boundaries:
- You provide legal information, not legal advice.
- If the retrieved sources are insufficient, say so clearly.
- If the answer depends on facts not provided by the user, explain what facts matter.
- Keep the answer practical, structured, and easy to understand.
""".strip()


def format_source_block(chunks: list[RetrievedChunk]) -> str:
    """
    Convert retrieved chunks into a compact source context for the LLM.

    The source block is intentionally explicit:
    - source label
    - article number
    - title path
    - text

    This makes the generated answer easier to ground and audit.
    """
    source_blocks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        title_path = str(chunk.title_path or "")
        article_number = str(chunk.article_number or "")
        source_label = str(chunk.source_label or "")

        source_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"source_label: {source_label}",
                    f"article_number: {article_number}",
                    f"title_path: {title_path}",
                    "text:",
                    chunk.text.strip(),
                ]
            )
        )

    return "\n\n".join(source_blocks)


def build_answer_prompt(
    query: str,
    chunks: list[RetrievedChunk],
) -> str:
    """
    Build a source-grounded answer prompt.
    """
    source_context = format_source_block(chunks)

    return f"""
User question:
{query}

Retrieved legal sources:
{source_context}

Task:
Answer the user's question using only the retrieved legal sources.

Required answer format:

Answer:
Give a clear and concise explanation.

Relevant sources:
List the most relevant articles used, with one short explanation for each.

Limitations:
Mention if the retrieved sources do not fully answer the question or if more facts are needed.

Important note:
This is legal information based on the retrieved sources, not legal advice.
""".strip()