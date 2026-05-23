from app.schemas.legal import AnalyzeResponse
from app.services.article_retriever import retrieve_articles
from app.services.explanation_generator import generate_explanation
from app.services.graph_builder import build_graph
from app.services.issue_classifier import classify_issues


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