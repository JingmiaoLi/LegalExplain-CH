import json
from pathlib import Path


ARTICLES_PATH = Path("data/processed/articles_or_title_10_en.json")


REQUIRED_ARTICLES = [
    "Art. 319",
    "Art. 324a",
    "Art. 335c",
    "Art. 336",
    "Art. 336a",
    "Art. 336b",
    "Art. 337",
    "Art. 340",
    "Art. 340a",
    "Art. 340b",
    "Art. 340c",
    "Art. 362",
]


def get_article(articles: list[dict], article_number: str) -> dict:
    return next(article for article in articles if article["article_number"] == article_number)


def main() -> None:
    if not ARTICLES_PATH.exists():
        raise FileNotFoundError(f"Missing parsed articles file: {ARTICLES_PATH}")

    articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))

    print(f"Loaded articles: {len(articles)}")

    assert len(articles) == 140, f"Expected 140 articles, got {len(articles)}"

    article_numbers = {article["article_number"] for article in articles}

    for article_number in REQUIRED_ARTICLES:
        assert article_number in article_numbers, f"Missing required article: {article_number}"

    assert "Art. 363" not in article_numbers, "Parsed data should not include Art. 363"

    for article in articles:
        assert article["id"], f"{article['article_number']} has empty id"
        assert article["fedlex_anchor"], f"{article['article_number']} missing fedlex_anchor"
        assert article["article_number"], "Article has empty article_number"
        assert article["language"] == "en", f"{article['article_number']} has wrong language"
        assert article["source_url"], f"{article['article_number']} missing source_url"
        assert article["source_type"] == "fedlex_print_view_snapshot"
        assert article["title_path"], f"{article['article_number']} missing title_path"
        assert article["text"], f"{article['article_number']} has empty text"
        assert article["paragraphs"], f"{article['article_number']} has no paragraphs"

    art_319 = get_article(articles, "Art. 319")
    assert len(art_319["paragraphs"]) == 2
    assert art_319["title_path"] == [
        "Title Ten: The Employment Contract",
        "Section One: The Individual Employment Contract",
        "A. Definition and conclusion",
        "I. Definition",
    ]

    art_337 = get_article(articles, "Art. 337")
    assert len(art_337["paragraphs"]) == 3
    assert len(art_337["footnotes"]) >= 1
    assert art_337["title_path"] == [
        "Title Ten: The Employment Contract",
        "Section One: The Individual Employment Contract",
        "G. End of the employment relationship",
        "IV. Termination with immediate effect",
        "1. Requirements",
        "a. For good cause",
    ]

    art_340 = get_article(articles, "Art. 340")
    assert len(art_340["paragraphs"]) == 2
    assert art_340["title_path"] == [
        "Title Ten: The Employment Contract",
        "Section One: The Individual Employment Contract",
        "G. End of the employment relationship",
        "VII. Prohibition of competition",
        "1. Requirements",
    ]

    art_362 = get_article(articles, "Art. 362")
    assert len(art_362["footnotes"]) >= 1
    assert art_362["title_path"] == [
        "Title Ten: The Employment Contract",
        "Section Four: Mandatory Provisions",
        "B. Provisions from which no derogation is permissible to the detriment of the employee",
    ]

    print("Validation passed ✅")


if __name__ == "__main__":
    main()