from backend.app.schemas.legal import AnalyzeResponse
from backend.app.services.article_retriever import retrieve_articles
from backend.app.services.explanation_generator import generate_explanation
from backend.app.services.graph_builder import build_graph
from backend.app.services.issue_classifier import classify_issues


def run_legal_workflow(scenario: str) -> AnalyzeResponse:
    issues = classify_issues(scenario)
    articles = retrieve_articles(issues)
    graph = build_graph(scenario, issues, articles)
    explanation = generate_explanation(scenario, issues, articles)

    return AnalyzeResponse(
        scenario=scenario,
        issues=issues,
        articles=articles,
        graph=graph,
        explanation=explanation,
    )