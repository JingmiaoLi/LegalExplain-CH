# scripts/build_article_chunks.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "articles_or_title_10_en.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "article_chunks.json"


def normalize_article_number(value: str) -> str:
    """
    Normalize article numbers so both '337' and 'Art. 337' become '337'.
    """
    value = str(value).strip()

    if value.lower().startswith("art."):
        value = value[4:].strip()

    return value


def clean_title_path(raw_title_path: Any) -> list[str]:
    """
    Ensure title_path is a clean list of non-empty strings.
    """
    if not isinstance(raw_title_path, list):
        return []

    return [
        str(item).strip()
        for item in raw_title_path
        if str(item).strip()
    ]


def paragraph_text(paragraph: Any) -> str:
    """
    Extract paragraph text from either a string paragraph or a parsed paragraph dict.

    Expected parsed paragraph format:
        {"number": "1", "text": "..."}
    """
    if isinstance(paragraph, dict):
        text = paragraph.get("text", "")
    else:
        text = paragraph

    return str(text).strip()


def paragraph_number(paragraph: Any, fallback_index: int) -> str:
    """
    Extract paragraph number if available.
    """
    if isinstance(paragraph, dict):
        number = paragraph.get("number")

        if number is not None and str(number).strip():
            return str(number).strip()

    return str(fallback_index)


def load_articles(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of article objects.")

    return data


def build_article_body(paragraphs: list[Any]) -> tuple[str, list[dict[str, str]]]:
    """
    Build one article-level body from all non-empty paragraphs.

    Returns:
        article_text:
            Clean full article text without paragraph numbers.
            This is suitable for retrieval, embedding, and LLM context.

        paragraph_items:
            Structured paragraph info for display/citation.
            Paragraph numbers are preserved here instead of being mixed into text.
    """
    paragraph_items: list[dict[str, str]] = []
    text_parts: list[str] = []

    for index, paragraph in enumerate(paragraphs, start=1):
        text = paragraph_text(paragraph)

        # Fedlex may contain placeholder paragraphs such as "..."
        # They are not useful for retrieval.
        if not text or text == "...":
            continue

        number = paragraph_number(paragraph, index)

        paragraph_items.append(
            {
                "number": number,
                "text": text,
            }
        )

        # Keep article text clean. Paragraph numbers remain in paragraph_items.
        text_parts.append(text)

    return "\n".join(text_parts).strip(), paragraph_items


def format_paragraphs_for_display(paragraphs: list[dict[str, str]]) -> str:
    """
    Rebuild a display-friendly version with paragraph numbers.

    This is not stored as the main text, but can be used later in the UI
    or answer formatting if paragraph-level display is needed.
    """
    return "\n".join(
        f"{paragraph['number']}. {paragraph['text']}"
        for paragraph in paragraphs
    )


def build_article_chunks(articles: list[dict[str, Any]]) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    skipped_articles: list[dict[str, str]] = []

    for article in articles:
        article_number = normalize_article_number(
            article.get("article_number", "")
        )

        if not article_number:
            skipped_articles.append(
                {
                    "article_number": "",
                    "reason": "missing_article_number",
                }
            )
            continue

        title_path = clean_title_path(article.get("title_path", []))
        paragraphs = article.get("paragraphs", [])

        if not isinstance(paragraphs, list):
            paragraphs = []

        article_text, paragraph_items = build_article_body(paragraphs)

        if not article_text:
            skipped_articles.append(
                {
                    "article_number": article_number,
                    "reason": "empty_article_text",
                }
            )
            continue

        source_label = article.get("source_label") or f"Art. {article_number}"

        chunk = {
            "chunk_id": f"art_{article_number}",
            "chunk_type": "article",
            "article_number": article_number,
            "source_label": source_label,
            "title_path": title_path,
            "paragraph_count": len(paragraph_items),
            "word_count": len(article_text.split()),
            "char_count": len(article_text),
            "text": article_text,
            "paragraphs": paragraph_items,
            "footnotes": article.get("footnotes", []),
            "source_url": article.get("source_url", ""),
            "source_type": article.get("source_type", ""),
            "status": article.get("status", ""),
        }

        chunks.append(chunk)

    return {
        "metadata": {
            "source_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
            "chunking_strategy": "article_level",
            "description": (
                "One retrieval chunk is created per non-empty article. "
                "The main text stores clean article text without paragraph numbers. "
                "Paragraph numbers are preserved separately in the paragraphs field "
                "for display and citation."
            ),
            "chunk_count": len(chunks),
            "skipped_count": len(skipped_articles),
            "skipped_articles": skipped_articles,
        },
        "chunks": chunks,
    }


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_sample(output: dict[str, Any]) -> None:
    print(f"Chunk count: {output['metadata']['chunk_count']}")
    print(f"Skipped count: {output['metadata']['skipped_count']}")

    if output["metadata"]["skipped_articles"]:
        print("\nSkipped articles:")
        for item in output["metadata"]["skipped_articles"]:
            print(item)

    print("\nSample chunks:")

    sample_articles = {"319", "321", "321a", "330a", "337", "337c", "362"}

    for chunk in output["chunks"]:
        if chunk["article_number"] not in sample_articles:
            continue

        print("\n" + "-" * 80)
        print(f"chunk_id: {chunk['chunk_id']}")
        print(f"source_label: {chunk['source_label']}")
        print(f"title_path: {chunk['title_path']}")
        print(f"paragraph_count: {chunk['paragraph_count']}")
        print(f"word_count: {chunk['word_count']}")
        print("clean text preview:")
        print(chunk["text"][:500])

        print("\nparagraph display preview:")
        print(format_paragraphs_for_display(chunk["paragraphs"])[:500])


def main() -> None:
    articles = load_articles(INPUT_PATH)
    output = build_article_chunks(articles)
    save_json(output, OUTPUT_PATH)

    print(f"Loaded articles: {len(articles)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print_sample(output)


if __name__ == "__main__":
    main()