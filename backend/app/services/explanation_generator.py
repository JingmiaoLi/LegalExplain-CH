from app.schemas.legal import LegalArticle, LegalExplanation, LegalIssue


def generate_explanation(
    scenario: str,
    issues: list[LegalIssue],
    articles: list[LegalArticle],
) -> LegalExplanation:
    issue_labels = ", ".join(issue.label for issue in issues)
    article_labels = ", ".join(article.article_number for article in articles)

    return LegalExplanation(
        summary=(
            f"This scenario may involve the following Swiss employment-law "
            f"issue(s): {issue_labels}."
        ),
        reasoning_path=[
            "The system identifies possible legal issues from the scenario.",
            f"It retrieves relevant Swiss Code of Obligations provisions: {article_labels}.",
            "It builds a legal issue graph linking the scenario, issues, sources, and review points.",
            "The result is designed for human legal review, not as final legal advice.",
        ],
        missing_facts=[
            "What exactly happened?",
            "What does the employment contract say?",
            "Were there written warnings or communications?",
            "What are the relevant dates?",
            "Is a collective employment agreement applicable?",
        ],
        human_review_notes=[
            "Review the exact facts and timeline.",
            "Check the employment contract and any written policies.",
            "Verify whether the cited OR articles apply to the full context.",
            "Human legal review is required before making any decision.",
        ],
    )