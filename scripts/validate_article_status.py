import json
from pathlib import Path


ARTICLES_PATH = Path("data/processed/articles_or_title_10_en.json")


def main() -> None:
    articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))

    empty_articles = [
        article for article in articles
        if not article["text"] or not article["paragraphs"]
    ]

    print(f"Total articles: {len(articles)}")
    print(f"Empty-text articles: {len(empty_articles)}")

    for article in empty_articles:
        footnote_text = " ".join(
            footnote.get("text", "") for footnote in article.get("footnotes", [])
        )

        print("\n" + "-" * 80)
        print(article["article_number"])
        print(f"status: {article.get('status')}")
        print(f"footnotes: {len(article.get('footnotes', []))}")
        print(footnote_text[:500])

        assert article.get("status") == "repealed", (
            f"{article['article_number']} has empty text but is not marked as repealed"
        )

        assert "Repealed by" in footnote_text, (
            f"{article['article_number']} has empty text but no 'Repealed by' footnote"
        )

    active_articles = [
        article for article in articles
        if article.get("status") == "active"
    ]

    repealed_articles = [
        article for article in articles
        if article.get("status") == "repealed"
    ]

    print("\nSummary")
    print(f"active: {len(active_articles)}")
    print(f"repealed: {len(repealed_articles)}")

    print("\nArticle status validation passed ✅")


if __name__ == "__main__":
    main()