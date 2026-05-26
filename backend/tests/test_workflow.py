from backend.app.services.workflow import run_legal_workflow


def test_workflow_returns_structured_response():
    scenario = "My employer dismissed me immediately without warning."

    result = run_legal_workflow(scenario)

    assert result.scenario == scenario
    assert len(result.issues) >= 1
    assert len(result.articles) >= 1
    assert len(result.graph.nodes) >= 1
    assert len(result.graph.edges) >= 1
    assert result.explanation.summary


def test_workflow_immediate_dismissal_contains_art_337():
    scenario = "My employer dismissed me immediately without warning."

    result = run_legal_workflow(scenario)

    issue_ids = [issue.id for issue in result.issues]
    article_numbers = [article.article_number for article in result.articles]

    assert "immediate_dismissal" in issue_ids
    assert "Art. 337 OR" in article_numbers


def test_workflow_graph_contains_human_review_node():
    scenario = "My employer dismissed me immediately without warning."

    result = run_legal_workflow(scenario)

    node_types = [node.type for node in result.graph.nodes]

    assert "human_review" in node_types