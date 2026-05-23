from app.schemas.legal import (
    LegalArticle,
    LegalGraph,
    LegalGraphEdge,
    LegalGraphNode,
    LegalIssue,
)


def build_graph(
    scenario: str,
    issues: list[LegalIssue],
    articles: list[LegalArticle],
) -> LegalGraph:
    nodes: list[LegalGraphNode] = [
        LegalGraphNode(
            id="scenario",
            label="User scenario",
            type="scenario",
            description=scenario,
        )
    ]

    edges: list[LegalGraphEdge] = []

    for issue in issues:
        issue_node_id = f"issue_{issue.id}"

        nodes.append(
            LegalGraphNode(
                id=issue_node_id,
                label=issue.label,
                type="issue",
                description=issue.description,
            )
        )

        edges.append(
            LegalGraphEdge(
                id=f"scenario_to_{issue_node_id}",
                source="scenario",
                target=issue_node_id,
                label="may involve",
            )
        )

        relevant_articles = [
            article for article in articles if issue.id in article.topics
        ]

        for article in relevant_articles:
            article_node_id = f"article_{article.id}"

            nodes.append(
                LegalGraphNode(
                    id=article_node_id,
                    label=article.article_number,
                    type="article",
                    description=article.title,
                    article_refs=[article.id],
                )
            )

            edges.append(
                LegalGraphEdge(
                    id=f"{issue_node_id}_to_{article_node_id}",
                    source=issue_node_id,
                    target=article_node_id,
                    label="grounded in",
                )
            )

        review_node_id = f"review_{issue.id}"

        nodes.append(
            LegalGraphNode(
                id=review_node_id,
                label="Human review needed",
                type="human_review",
                description="A legal professional should review the facts and documents.",
            )
        )

        edges.append(
            LegalGraphEdge(
                id=f"{issue_node_id}_to_{review_node_id}",
                source=issue_node_id,
                target=review_node_id,
                label="requires",
            )
        )

    return LegalGraph(nodes=nodes, edges=edges)