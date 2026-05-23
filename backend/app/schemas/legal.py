from pydantic import BaseModel, Field
from typing import Literal


LegalTopic = Literal[
    "non_compete",
    "immediate_dismissal",
    "abusive_termination",
    "notice_period",
    "salary_during_illness",
]


class AnalyzeRequest(BaseModel):
    scenario: str = Field(
        ...,
        min_length=10,
        description="A short employment-law scenario to analyze.",
    )


class LegalIssue(BaseModel):
    id: LegalTopic
    label: str
    description: str
    confidence: Literal["high", "medium", "low"]


class LegalArticle(BaseModel):
    id: str
    article_number: str
    title: str
    text: str
    source_url: str
    topics: list[LegalTopic]


class LegalGraphNode(BaseModel):
    id: str
    label: str
    type: Literal[
        "scenario",
        "issue",
        "article",
        "condition",
        "risk",
        "missing_fact",
        "human_review",
    ]
    description: str | None = None
    article_refs: list[str] = []


class LegalGraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str | None = None


class LegalGraph(BaseModel):
    nodes: list[LegalGraphNode]
    edges: list[LegalGraphEdge]


class LegalExplanation(BaseModel):
    summary: str
    reasoning_path: list[str]
    missing_facts: list[str]
    human_review_notes: list[str]


class AnalyzeResponse(BaseModel):
    scenario: str
    issues: list[LegalIssue]
    articles: list[LegalArticle]
    graph: LegalGraph
    explanation: LegalExplanation
