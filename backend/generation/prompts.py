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
    Convert retrieved chunks into a source context for answer generation.

    This version is intentionally richer than the reasoning-map source block,
    because the final answer needs enough legal detail to remain grounded.
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


def format_compact_source_block(
    chunks: list[RetrievedChunk],
    max_sources: int = 3,
    max_chars_per_source: int = 700,
) -> str:
    """
    Convert retrieved chunks into a compact source context for graph generation.

    Reasoning maps only need the central legal points, not the full article text.
    Keeping this compact reduces latency and API cost.
    """
    source_blocks: list[str] = []

    for index, chunk in enumerate(chunks[:max_sources], start=1):
        title_path = str(chunk.title_path or "")
        article_number = str(chunk.article_number or "")
        source_label = str(chunk.source_label or "")
        text = chunk.text.strip()

        if len(text) > max_chars_per_source:
            text = f"{text[:max_chars_per_source]}..."

        source_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"source_label: {source_label}",
                    f"article_number: {article_number}",
                    f"title_path: {title_path}",
                    "text_excerpt:",
                    text,
                ]
            )
        )

    return "\n\n".join(source_blocks)


def format_conversation_history(
    conversation_history: list[dict[str, str]] | None,
    max_turns: int = 4,
    max_chars_per_message: int = 700,
) -> str:
    """
    Format recent conversation history for query rewriting.

    Query rewriting only needs recent context, so this is intentionally compact.
    """
    if not conversation_history:
        return "No previous conversation."

    recent_messages = conversation_history[-max_turns:]
    lines: list[str] = []

    for message in recent_messages:
        role = message.get("role", "unknown")
        content = message.get("content", "").strip()

        if not content:
            continue

        if len(content) > max_chars_per_message:
            content = f"{content[:max_chars_per_message]}..."

        lines.append(f"{role}: {content}")

    return "\n".join(lines) if lines else "No previous conversation."


def build_query_rewrite_prompt(
    query: str,
    conversation_history: list[dict[str, str]] | None,
) -> str:
    """
    Build a compact prompt that rewrites a follow-up question into a standalone
    retrieval query.

    This prompt is only used when the backend heuristic thinks the current query
    is likely to be a follow-up.
    """
    history_context = format_conversation_history(conversation_history)

    return f"""
Conversation history:
{history_context}

Current user question:
{query}

Rewrite the current question into one standalone search query for Swiss employment-law retrieval.

Rules:
- Return only the rewritten query.
- Do not include explanations, markdown, quotes, or labels.
- Preserve the legal actor and action.
- Do not add facts not provided by the user.
- Add only the minimum context needed from the conversation.
- If the question is already standalone, return it unchanged or only lightly clarified.
- Keep it concise.

Examples:
Previous topic: employer immediate dismissal.
Current question: What if I was late twice?
Rewritten query: Can an employer dismiss an employee immediately without notice for being late twice under Swiss employment law?

Previous topic: employee immediate resignation.
Current question: What if they stopped paying me?
Rewritten query: Can an employee terminate an employment contract immediately without notice if the employer stopped paying salary under Swiss employment law?

Return only the rewritten query.
""".strip()


def build_reasoning_map_prompt(
    query: str,
    chunks: list[RetrievedChunk],
) -> str:
    """
    Build a compact prompt for generating a legal reasoning map as JSON.

    This prompt uses compact source context to reduce LLM cost and latency.
    """
    source_context = format_compact_source_block(chunks)

    return f"""
User question:
{query}

Compact legal sources:
{source_context}

Task:
Create a compact legal reasoning map as valid JSON.

The map should help a non-lawyer see the legal structure behind the answer.
Use only the compact legal sources.

Return only valid JSON. Do not include markdown or explanations outside JSON.

Grounding rules:
- Do not invent legal rules, article numbers, remedies, deadlines, exceptions, or procedures.
- Use article_label only when clearly supported by the sources.
- If no article clearly supports a node, use null for article_label.
- Use null for source_url unless explicitly available.
- Preserve the legal actor in the user's question.
- Never reverse employer and employee roles.

Choose one map_type:
- decision: whether someone can, may, must, is allowed to, or is entitled to do something.
- overview: general rules about a topic.
- consequence: what happens if something occurs, what liability follows, or what can be claimed.
- duty: what an employer or employee must do.
- mixed: only when the question clearly combines more than one type.

Recommended structures:
- decision: issue -> condition -> test -> yes outcome / no consequence
- overview: issue -> main rule -> variation / exception -> special case -> related consequence
- consequence: triggering event -> legal consequence -> remedy or claim -> limitation
- duty: legal relationship -> main duty -> specific duty -> limit or consequence
- mixed: issue -> key rule -> related rule -> consequence or next step

Graph rules:
- Use 3 to 6 nodes. Prefer 4 or 5.
- Keep labels short, concrete, and user-friendly.
- Use plain English.
- Do not repeat the user's question as a node.
- Use description as an empty string unless a very short clarification is necessary.
- Do not use generic role names as labels.
- Bad labels: "How to decide", "Condition", "Outcome", "Legal issue", "Rule".
- Good labels: "Good cause is required", "Paid at month end", "Compensation may be owed".
- Do not force overview, consequence, duty, or mixed questions into Yes/No branches.
- Use Yes/No branches only when there is a true condition-based legal assessment.
- Use node_type "test" only when there is a real legal standard to evaluate.

Allowed node_type values:
issue, rule, condition, test, outcome, consequence, remedy, exception, limitation, duty, related

Required JSON schema:
{{
  "title": "Legal reasoning map",
  "map_type": "decision | overview | consequence | duty | mixed",
  "summary": "short sentence describing the map",
  "nodes": [
    {{
      "id": "snake_case_id",
      "node_type": "issue | rule | condition | test | outcome | consequence | remedy | exception | limitation | duty | related",
      "label": "short plain-English legal content",
      "description": "",
      "article_label": "Art. number or null",
      "source_url": null
    }}
  ],
  "edges": [
    {{
      "source": "source node id",
      "target": "target node id",
      "label": "Yes | No | null"
    }}
  ]
}}

Good decision example:
{{
  "title": "Legal reasoning map",
  "map_type": "decision",
  "summary": "A short path showing when immediate dismissal may be justified.",
  "nodes": [
    {{
      "id": "issue",
      "node_type": "issue",
      "label": "Immediate dismissal",
      "description": "",
      "article_label": null,
      "source_url": null
    }},
    {{
      "id": "good_cause",
      "node_type": "condition",
      "label": "Good cause is required",
      "description": "",
      "article_label": "Art. 337",
      "source_url": null
    }},
    {{
      "id": "unreasonable_to_continue",
      "node_type": "test",
      "label": "Continuing must be unreasonable",
      "description": "",
      "article_label": "Art. 337",
      "source_url": null
    }},
    {{
      "id": "employer_may_dismiss",
      "node_type": "outcome",
      "label": "Employer may dismiss immediately",
      "description": "",
      "article_label": "Art. 337",
      "source_url": null
    }},
    {{
      "id": "employee_claim",
      "node_type": "remedy",
      "label": "Employee may claim damages",
      "description": "",
      "article_label": "Art. 337c",
      "source_url": null
    }}
  ],
  "edges": [
    {{"source": "issue", "target": "good_cause", "label": null}},
    {{"source": "good_cause", "target": "unreasonable_to_continue", "label": null}},
    {{"source": "unreasonable_to_continue", "target": "employer_may_dismiss", "label": "Yes"}},
    {{"source": "unreasonable_to_continue", "target": "employee_claim", "label": "No"}}
  ]
}}

Good overview example:
{{
  "title": "Legal reasoning map",
  "map_type": "overview",
  "summary": "A short overview of salary payment rules.",
  "nodes": [
    {{
      "id": "issue",
      "node_type": "issue",
      "label": "Salary payment",
      "description": "",
      "article_label": null,
      "source_url": null
    }},
    {{
      "id": "monthly_payment",
      "node_type": "rule",
      "label": "Paid at month end",
      "description": "",
      "article_label": "Art. 323",
      "source_url": null
    }},
    {{
      "id": "custom_terms",
      "node_type": "exception",
      "label": "Custom terms may apply",
      "description": "",
      "article_label": "Art. 323",
      "source_url": null
    }},
    {{
      "id": "salary_statement",
      "node_type": "duty",
      "label": "Salary statement required",
      "description": "",
      "article_label": "Art. 323b",
      "source_url": null
    }}
  ],
  "edges": [
    {{"source": "issue", "target": "monthly_payment", "label": null}},
    {{"source": "monthly_payment", "target": "custom_terms", "label": null}},
    {{"source": "custom_terms", "target": "salary_statement", "label": null}}
  ]
}}

Return only valid JSON.
""".strip()

def build_combined_answer_map_prompt(
    query: str,
    chunks: list[RetrievedChunk],
) -> str:
    """
    Build a single prompt that returns both the user-facing answer and the
    reasoning map in one JSON object.

    This is used for text_map mode to avoid two separate LLM calls with
    duplicated source context.
    """
    source_context = format_source_block(chunks)

    return f"""
User question:
{query}

Retrieved legal sources:
{source_context}

Task:
Answer the user's question and create a compact legal reasoning map.

Use only the retrieved legal sources.

Return only valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.

Required JSON schema:
{{
  "answer": "concise plain-English answer",
  "reasoning_map": {{
    "title": "Legal reasoning map",
    "map_type": "decision | overview | consequence | duty | mixed",
    "summary": "short sentence describing the map",
    "nodes": [
      {{
        "id": "snake_case_id",
        "node_type": "issue | rule | condition | test | outcome | consequence | remedy | exception | limitation | duty | related",
        "label": "short plain-English legal content",
        "description": "",
        "article_label": "Art. number or null",
        "source_url": null
      }}
    ],
    "edges": [
      {{
        "source": "source node id",
        "target": "target node id",
        "label": "Yes | No | null"
      }}
    ]
  }}
}}

Core rules:
- Use only the retrieved legal sources.
- Do not invent legal rules, article numbers, procedures, cases, remedies, deadlines, exceptions, or consequences.
- Do not mention article numbers that are not included in the retrieved sources.
- Do not write phrases like "Art. X is not provided".
- Do not refer to "retrieved sources" in the final answer.
- Never reverse the legal actor in the user's question.
- Use article references naturally, for example "under Art. 337" or "Art. 337c provides...".
- The answer and reasoning_map must not contradict each other.

Actor and action alignment:
- Identify the legal actor in the user's question before answering.
- Keep the answer and map focused on the same actor, action, and legal direction as the user's question.
- If the user asks about an employer action, explain the employer's power, duty, liability, or risk.
- If the user asks about an employee action, explain the employee's right, duty, liability, or risk.
- Do not use a rule about the opposite party's action as the main answer or main graph path unless the user's question also asks about that opposite action.
- Do not convert an employer dismissal question into an employee resignation, absence, or no-show question.
- Do not convert an employee resignation or leaving question into an employer dismissal question.
- If a retrieved article concerns a different actor or a different legal action, do not use it as the main legal basis. Mention it only if clearly relevant as a contrast or secondary point.

Answer rules:
- Start the answer directly.
- Keep the answer concise and user-facing.
- Prefer 2 to 4 short paragraphs.
- Prioritize the legal points that directly answer the user's question.
- If several retrieved articles are relevant, group them briefly instead of explaining each one in full.
- Mention secondary or special-case rules only if they materially affect the answer.
- For broad overview questions, summarize the main rules first and avoid expanding every special case.
- Do not use headings such as "Answer:", "Direct answer:", "Explanation:", or "Follow-up:".
- Ask a follow-up question only when the user's provided facts are insufficient and one missing fact could materially change the legal outcome.

Reasoning map rules:
- Use 3 to 6 nodes. Prefer 4 or 5.
- Keep node labels short, concrete, and user-friendly.
- A node label should usually be under 8 words.
- Use plain English.
- Use description as an empty string unless a short clarification is truly necessary.
- Do not repeat the user's full question as a node.
- Do not use generic role names as labels.
- Bad labels: "How to decide", "Condition", "Outcome", "Legal issue", "Rule".
- Good labels: "Good cause is required", "Paid at month end", "Compensation may be owed".
- Use article_label only when the article clearly supports the node.
- If no article clearly supports a node, use null for article_label.
- Use null for source_url unless the source URL is explicitly available.
- Do not force every question into a decision graph.

Map type guidance:
- decision: use when the user asks whether someone can, may, must, is allowed to, or is entitled to do something.
- overview: use when the user asks for general rules about a topic.
- consequence: use when the user asks what happens if something occurs, what liability follows, or what can be claimed.
- duty: use when the user asks what an employer or employee must do.
- mixed: use only when the question clearly combines more than one type.

Recommended structures:
- decision: issue -> condition -> test -> yes outcome / no consequence
- overview: issue -> main rule -> variation / exception -> special case -> related consequence
- consequence: triggering event -> legal consequence -> remedy or claim -> limitation
- duty: legal relationship -> main duty -> specific duty -> limit or consequence
- mixed: issue -> key rule -> related rule -> consequence or next step

Branch rules:
- Use Yes/No branches only when there is a true condition-based legal assessment.
- The Yes branch should state the result if the condition is satisfied.
- The No branch should state the consequence or risk if the condition is not satisfied.
- Do not put the condition itself inside the Yes or No outcome.

Return only valid JSON.
""".strip()

def build_answer_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    reasoning_map_json: str | None = None,
) -> str:
    """
    Build a source-grounded answer prompt.

    The answer should remain concise and source-grounded. The reasoning map is
    optional; when unavailable, the answer should still be complete enough.
    """
    source_context = format_source_block(chunks)

    reasoning_map_block = (
        f"""
Reasoning map JSON:
{reasoning_map_json}
"""
        if reasoning_map_json
        else """
Reasoning map JSON:
Not available.
"""
    )

    return f"""
User question:
{query}

Retrieved legal sources:
{source_context}

{reasoning_map_block}

Task:
Answer the user's question using only the retrieved legal sources.

Core rules:
- Use only the retrieved legal sources.
- Do not invent legal rules, article numbers, procedures, cases, remedies, deadlines, exceptions, or consequences.
- Do not mention article numbers that are not included in the retrieved sources.
- Do not refer to "retrieved sources" in the final answer.
- Do not write phrases like "Art. X is not provided".
- Never reverse the legal actor in the user's question.
- The answer and reasoning map must not contradict each other when a reasoning map is available.

Citation rules:
- Mention the supporting article naturally next to each legal rule, requirement, right, duty, remedy, limitation, or consequence.
- Do not omit article references.
- Use article references naturally, for example "under Art. 337" or "Art. 337c provides...".
- If the answer discusses good cause for immediate termination, mention Art. 337 in that sentence or paragraph when Art. 337 is available.
- If the answer discusses damages or compensation after unjustified immediate dismissal, mention Art. 337c in that sentence or paragraph when Art. 337c is available.

Actor and action alignment:
- Identify the legal actor in the user's question before answering.
- Keep the answer focused on the same actor, action, and legal direction as the user's question.
- If the user asks about an employer action, explain the employer's power, duty, liability, or risk.
- If the user asks about an employee action, explain the employee's right, duty, liability, or risk.
- Do not use a rule about the opposite party's action as the main answer unless the user's question also asks about that opposite action.
- Do not convert an employer dismissal question into an employee resignation, absence, or no-show question.
- Do not convert an employee resignation or leaving question into an employer dismissal question.
- If a retrieved article concerns a different actor or legal action, do not use it as the main legal basis. Mention it only if clearly relevant as a contrast or secondary point.

Answer style:
- Start directly with the answer.
- Do not write headings such as "Answer:", "Direct answer:", "Relevant legal points:", "Key legal points:", "Explanation:", or "Follow-up:".
- Write in natural plain English for non-lawyers.
- Prefer 1 to 2 short paragraphs. Do not add a separate concluding paragraph that merely repeats the answer.
- For follow-up questions, answer only the new factual variation and avoid repeating the full rule from the previous answer.
- The first paragraph should directly answer the user's question.
- Then briefly explain the main applicable legal rule and any central consequence.
- Prioritize legal points that directly answer the question.
- Mention secondary or special-case rules only if they materially affect the answer.
- Do not list every retrieved article.
- Do not omit a condition, consequence, remedy, deadline, duty, or exception if it is central to the answer.
- If a consequence applies only when a condition is met, state that condition clearly.
- Use careful wording such as "may be entitled", "may apply", or "depends on the circumstances" when the outcome depends on facts, discretion, or court assessment.

Follow-up rule:
- Ask one short follow-up question only if the user's facts are insufficient and one missing fact could materially change the legal outcome.
- Do not ask a follow-up question merely because a retrieved article contains a special case, exception, or fact-dependent rule.
- Do not ask a follow-up question for broad overview questions when a general answer is possible.
- Do not label the question as "Follow-up:".
- If a general legal answer is possible, end without a follow-up question.
""".strip()