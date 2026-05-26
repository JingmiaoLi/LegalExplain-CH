from backend.app.legal_data.articles import LEGAL_ARTICLES
from backend.app.schemas.legal import LegalArticle, LegalIssue


def retrieve_articles(issues: list[LegalIssue]) -> list[LegalArticle]:
    issue_ids = {issue.id for issue in issues}

    return [
        article
        for article in LEGAL_ARTICLES
        if any(topic in issue_ids for topic in article.topics)
    ]