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

def build_reasoning_map_prompt(
    query: str,
    chunks: list[RetrievedChunk],
) -> str:
    """
    Build a prompt for generating a compact legal reasoning map as JSON.

    The reasoning map is intended for frontend visualization. The model should
    first choose the most suitable map type, then generate a compact graph.
    """
    source_context = format_source_block(chunks)

    return f"""
User question:
{query}

Retrieved legal sources:
{source_context}

Task:
Create a compact legal reasoning map as JSON.

The map should help a non-lawyer understand the legal structure behind the answer.
Use only the retrieved legal sources.

Return only valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.

Core legal-grounding rules:
- Do not invent legal rules, article numbers, procedures, cases, remedies, deadlines, or exceptions.
- Do not mention article numbers that are not included in the retrieved sources.
- Use article_label only when the article is clearly supported by the retrieved sources.
- If no article clearly supports a node, use null for article_label.
- Use null for source_url unless the source URL is explicitly available in the retrieved source metadata.
- Preserve the legal actor in the user's question.
- If the user asks whether an employer may do something, the map must describe the employer's power, duty, or liability.
- If the user asks whether an employee may do something, the map must describe the employee's right, duty, or risk.
- Never reverse the legal perspective.

Graph pattern selection:
First decide which type of legal explanation best fits the user's question.
Choose the graph structure that makes the legal reasoning easiest for a non-lawyer to understand.

Available map_type values:
- decision
- overview
- consequence
- duty
- mixed

Choose only one map_type.

Use "decision" when the user asks whether someone can, may, must, is allowed to, or is entitled to do something.
Examples:
- "Can my employer dismiss me immediately without notice?"
- "Can I leave my job immediately without notice?"
Recommended structure:
issue -> condition -> test -> yes outcome / no consequence

Use "overview" when the user asks for general rules about a topic.
Examples:
- "What does Swiss employment law say about salary payment?"
- "What are the rules for overtime?"
Recommended structure:
issue -> main rule -> variation / exception -> special case -> related consequence

Use "consequence" when the user asks what happens if something occurs, what liability follows, or what can be claimed.
Examples:
- "What happens if my employer dismisses me without good cause?"
- "What happens if I leave without good cause?"
Recommended structure:
triggering event -> legal consequence -> remedy or claim -> limitation or calculation

Use "duty" when the user asks what an employer or employee must do.
Examples:
- "What are the employee's duties of loyalty and care?"
- "What must the employer provide?"
Recommended structure:
legal relationship -> main duty -> specific duty -> limit or consequence

Use "mixed" only when the question clearly combines more than one of the above.
Recommended structure:
issue -> key rule -> related rule -> consequence or next step

Graph design rules:
- Do not repeat the user's question as a node.
- Use 3 to 6 nodes. Prefer 4 or 5 nodes.
- Keep each node label short, concrete, and user-friendly.
- A node label should usually be under 8 words.
- Use plain English.
- Avoid long explanations inside nodes.
- Use description as an empty string unless a very short clarification is truly necessary.
- The graph should simplify the legal structure, not summarize every sentence of the answer.
- Do not force every question into a decision graph.
- Do not create Yes/No branches for separate legal points that are not true alternatives.
- Use Yes/No branches only when the question truly requires a condition-based legal assessment.
- Use node_type "test" only when there is a real legal standard to evaluate.
- For overview, consequence, duty, and mixed maps, prefer a simple left-to-right path without Yes/No labels.

Important distinction between node_type and label:
- node_type describes the role of the node.
- label describes the actual legal content.
- Never use a generic role name as the node label.
- Bad labels: "How to decide", "Condition", "Outcome", "Consequence", "Legal issue", "Rule".
- Good labels: "Good cause is required", "Staying must be unreasonable", "Compensation may be owed", "Salary paid monthly".

Allowed node_type values:
- issue
- rule
- condition
- test
- outcome
- consequence
- exception
- limitation
- duty
- related

Node type guidance:
- Use "issue" for the legal topic or legal category.
- Use "rule" for the main legal rule in an overview map.
- Use "condition" for a requirement that must be satisfied.
- Use "test" only for how to decide whether a condition is met.
- Use "outcome" for a possible positive result, right, or entitlement.
- Use "consequence" for a risk, duty, loss, liability, or negative result.
- Use "exception" for an exception or special variation.
- Use "limitation" for a cap, limit, deadline, or restriction.
- Use "duty" for an employer or employee obligation.
- Use "related" only for an extra legal point that does not fit the main path.

Decision-branch rules:
- If the legal answer depends on a yes/no condition, create a branch.
- The condition should appear before the branch, not inside the Yes or No outcome.
- The Yes branch should state the result if the condition is satisfied.
- The No branch should state the consequence or risk if the condition is not satisfied.
- Do not put a requirement such as "Good cause is required" inside the Yes branch.
- Do not put the user's requested action, such as "Termination without notice", inside the No branch unless it is clearly the legal consequence.
- Use edge labels "Yes" and "No" only for branches from a condition or test node to outcome/consequence nodes.

Required JSON schema:
{{
  "title": "Legal reasoning map",
  "map_type": "decision | overview | consequence | duty | mixed",
  "summary": "short sentence describing the map",
  "nodes": [
    {{
      "id": "snake_case_id",
      "node_type": "issue | rule | condition | test | outcome | consequence | exception | limitation | duty | related",
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
      "id": "condition_good_cause",
      "node_type": "condition",
      "label": "Good cause is required",
      "description": "",
      "article_label": "Art. 337",
      "source_url": null
    }},
    {{
      "id": "test_unreasonable_to_continue",
      "node_type": "test",
      "label": "Continuing must be unreasonable",
      "description": "",
      "article_label": "Art. 337",
      "source_url": null
    }},
    {{
      "id": "outcome_employer_may_dismiss",
      "node_type": "outcome",
      "label": "Employer may dismiss immediately",
      "description": "",
      "article_label": "Art. 337",
      "source_url": null
    }},
    {{
      "id": "consequence_employee_claim",
      "node_type": "consequence",
      "label": "Employee may claim damages",
      "description": "",
      "article_label": "Art. 337c",
      "source_url": null
    }}
  ],
  "edges": [
    {{
      "source": "issue",
      "target": "condition_good_cause",
      "label": null
    }},
    {{
      "source": "condition_good_cause",
      "target": "test_unreasonable_to_continue",
      "label": null
    }},
    {{
      "source": "test_unreasonable_to_continue",
      "target": "outcome_employer_may_dismiss",
      "label": "Yes"
    }},
    {{
      "source": "test_unreasonable_to_continue",
      "target": "consequence_employee_claim",
      "label": "No"
    }}
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
      "id": "hardship_advance",
      "node_type": "related",
      "label": "Hardship advance possible",
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
    {{
      "source": "issue",
      "target": "monthly_payment",
      "label": null
    }},
    {{
      "source": "monthly_payment",
      "target": "custom_terms",
      "label": null
    }},
    {{
      "source": "custom_terms",
      "target": "hardship_advance",
      "label": null
    }},
    {{
      "source": "hardship_advance",
      "target": "salary_statement",
      "label": null
    }}
  ]
}}

Bad examples to avoid:
- A test node with label "HOW TO DECIDE".
- A Yes outcome node with label "Good cause required".
- A No consequence node with label "Termination without notice".
- A salary payment overview graph with Yes/No branches.
- A graph that uses the same phrase for node_type and label.
- A graph that reverses employer and employee roles.

Return only valid JSON.
""".strip()


def build_answer_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    reasoning_map_json: str | None = None,
) -> str:
    """
    Build a source-grounded answer prompt.

    The answer should be guided by the reasoning map when available, but the
    final text should still read naturally as a conversational legal answer.
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
- Do not write phrases like "Art. X is not provided".
- Do not refer to "retrieved sources" in the final answer.
- Never reverse the legal actor in the user's question.
- Use article references naturally, for example "under Art. 337" or "Art. 337c provides...".
- Use the reasoning map as the structure of the answer when it is available.
- The answer and the reasoning map must not contradict each other.

Answer focus:
- Start with the direct answer.
- Keep the answer concise and user-facing.
- Prefer 2 to 4 short paragraphs.
- Prioritize the legal points that directly answer the user's question.
- If several retrieved articles are relevant, group them briefly instead of explaining each one in full.
- Mention secondary or special-case rules only if they materially affect the answer.
- For broad overview questions, summarize the main rules first and avoid expanding every special case.
- Do not list every retrieved article.
- Do not omit a condition, consequence, remedy, deadline, duty, or exception if it is central to the answer.
- If a consequence applies only when a condition is met, state that condition clearly.
- Prefer careful wording such as "may be entitled", "may apply", or "depends on the circumstances" when the outcome depends on facts, discretion, or court assessment.

Follow-up rule:
- Ask a follow-up question only when the user's provided facts are insufficient to answer the question accurately.
- A follow-up is appropriate only if one missing fact could materially change the legal outcome.
- Do not ask a follow-up question merely because a retrieved article contains a special case, exception, or fact-dependent rule.
- Do not ask a follow-up question for broad overview questions when a general answer is possible.
- If several legal outcomes are possible depending on the facts, ask one concise question that identifies the most important missing fact.
- Ask only one follow-up question.
- Do not label it as "Follow-up:".
- If the answer can be accurately given as a general legal overview, end without a follow-up question.

Required answer style:
- Start directly with the answer.
- Do not write headings such as "Answer:", "Direct answer:", "Relevant legal points:", "Key legal points:", "Explanation:", or "Follow-up:".
- Write in natural plain English.
- The first paragraph should directly answer the user's question.
- Then briefly explain the main applicable legal rule and any central consequence.
- Mention the cited article naturally next to the legal point it supports.
- If the available legal sources provide a general standard but not concrete examples, explain that the result depends on the specific facts.
- If a follow-up question is necessary under the Follow-up rule, place it naturally at the end.

""".strip()


