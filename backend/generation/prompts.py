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

Core rules:
- Do not invent legal rules, article numbers, procedures, cases, remedies, deadlines, or exceptions.
- Do not mention article numbers that are not included in the retrieved sources.
- Do not write phrases like "Art. X is not provided".
- Do not refer to "retrieved sources" in the final answer.
- Use article references naturally, for example "under Art. 337" or "Art. 337c provides...".
- Before writing the final answer, identify all distinct legal points in the retrieved sources that directly affect the user's question.
- Relevant legal points may include requirements, rights, duties, consequences, exceptions, limitations, remedies, deadlines, procedural steps, and fact-dependent conditions.
- Include every directly relevant legal point in the main answer, not only in the "Relevant legal points" section.
- Do not omit consequences, exceptions, remedies, deadlines, duties, or conditions when they are directly relevant to the question and present in the cited sources.
- If several retrieved articles address different parts of the question, integrate those parts into the main answer before listing legal points.
- If a consequence applies only when a condition is met, state that condition explicitly.
- Avoid vague references such as "if the employer does so" or "in that case" when the legal condition matters.
- Prefer careful wording such as "may be entitled", "may apply", or "depends on the circumstances" when the outcome depends on facts, discretion, or court assessment.
- Keep the answer concise, but do not sacrifice legal completeness.
- When the answer depends on a fact-dependent legal standard, include one short follow-up question at the end of the answer.

Required answer format:

Start directly with the answer. Do not write a heading such as "Answer:".

In the main answer:
- Directly answer the user's question.
- Explain the applicable legal rule.
- Include any directly relevant consequences, limitations, exceptions, remedies, deadlines, duties, or procedural requirements found in the cited sources.
- Mention the cited article naturally next to the legal point it supports.
- If the available legal sources provide a general standard but not concrete examples, explain the general standard and say that its application depends on the specific facts.

Relevant legal points:
List only the most relevant articles used, with one short explanation for each.
Do not use this section as a substitute for the main answer; the main answer must already contain the important legal points.

Follow-up:
If the answer depends on a fact-dependent legal standard, end with one natural follow-up question.
Fact-dependent legal standards include good cause, reasonableness, court discretion, intent, damage, timing, consent, surrounding circumstances, or whether a condition is met.
Ask only the single most important follow-up question.
Do not list multiple questions.
If no important missing fact would materially affect the answer, do not ask a follow-up question.

Important note:
This is legal information based on the cited sources, not legal advice.
""".strip()

