import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag


HTML_PATH = Path("data/raw/fedlex/or_title_10_en_print_view.html")
OUTPUT_PATH = Path("data/processed/articles_or_title_10_en.json")

BASE_SOURCE_URL = "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en"


def normalize_text(text: str) -> str:
    return " ".join(text.split())

def infer_article_status(paragraphs: list[dict], footnotes: list[dict]) -> str:
    footnote_text = " ".join(
        footnote.get("text", "") for footnote in footnotes
    ).lower()

    if not paragraphs and "repealed by" in footnote_text:
        return "repealed"

    return "active"

def article_id_to_number(article_id: str) -> str:
    """
    Convert Fedlex anchor IDs into readable article numbers.

    Examples:
    art_319 -> Art. 319
    art_324_a -> Art. 324a
    art_329_g_bis -> Art. 329gbis
    """
    raw = article_id.removeprefix("art_")
    parts = raw.split("_")

    number = parts[0]
    suffix = "".join(parts[1:])

    return f"Art. {number}{suffix}"


def remove_footnote_container(article: Tag) -> None:
    for footnote_div in article.select("div.footnotes"):
        footnote_div.decompose()


def extract_footnotes(article: Tag) -> list[dict]:
    footnotes: list[dict] = []

    for footnote in article.select("div.footnotes p[id^='fn-']"):
        text = normalize_text(footnote.get_text(" ", strip=True))
        if not text:
            continue

        marker_match = re.match(r"^(\d+)\s+(.*)$", text)
        if marker_match:
            marker = marker_match.group(1)
            body = marker_match.group(2)
        else:
            marker = None
            body = text

        footnotes.append(
            {
                "marker": marker,
                "text": body,
                "type": "amendment_note",
            }
        )

    return footnotes


def clean_paragraph(paragraph: Tag) -> tuple[str | None, str]:
    """
    Extract paragraph number and text from a Fedlex paragraph.

    Paragraph numbers are usually stored in the first <sup>.
    Footnote markers are also <sup>, so we remove all <sup> after
    reading the first numeric paragraph marker.
    """
    paragraph_copy = BeautifulSoup(str(paragraph), "lxml")

    p = paragraph_copy.find("p")
    if p is None:
        return None, ""

    paragraph_number = None

    first_sup = p.find("sup")
    if first_sup is not None:
        first_sup_text = normalize_text(first_sup.get_text(" ", strip=True))
        if first_sup_text.isdigit():
            paragraph_number = first_sup_text

    # Remove all superscripts and footnote backlink anchors from main text.
    for sup in p.find_all("sup"):
        sup.decompose()

    for anchor in p.find_all("a"):
        anchor_id = str(anchor.get("id", ""))
        href = str(anchor.get("href", ""))

        if anchor_id.startswith("fnbck-") or "#fn-" in href:
            anchor.decompose()

    text = normalize_text(p.get_text(" ", strip=True))

    return paragraph_number, text


def extract_paragraphs(article: Tag) -> list[dict]:
    paragraphs: list[dict] = []

    # Only parse direct article text paragraphs. Exclude footnotes.
    for paragraph in article.find_all("p"):
        if paragraph.find_parent("div", class_="footnotes"):
            continue

        classes = paragraph.get("class") or []
        classes = [str(class_name) for class_name in classes]

        # Main article paragraphs and list-like legal paragraphs.
        if "absatz" not in classes and "man-template-tab-krpr" not in classes:
            continue

        number, text = clean_paragraph(paragraph)

        if not text:
            continue

        paragraphs.append(
            {
                "number": number,
                "text": text,
            }
        )

    return paragraphs


TOP_LEVEL_TITLE = "Title Ten: The Employment Contract"


def get_own_section_heading(section: Tag) -> str | None:
    """
    Return the heading that belongs directly to this section.

    Fedlex uses both:
    - h3.heading for higher-level sections
    - div.heading for nested legal headings such as A., I., IV., 1., a.

    We only want headings whose nearest parent <section> is the current section.
    """
    candidates = section.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "div"],
        class_="heading",
    )

    for heading in candidates:
        if heading.find_parent("section") is not section:
            continue

        heading_text = normalize_text(heading.get_text(" ", strip=True))

        if not heading_text:
            continue

        if heading_text.startswith("Art."):
            continue

        return heading_text

    return None

def deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)

    return result


def get_title_path(article: Tag) -> list[str]:
    """
    Build the full legal hierarchy path for an article.

    Example target:
    [
      "Title Ten: The Employment Contract",
      "Section One: The Individual Employment Contract",
      "G. Termination of the employment relationship",
      "IV. Termination with immediate effect",
      "1. Requirements",
      "a. For good cause"
    ]
    """
    sections: list[Tag] = []

    parent = article.parent
    while parent is not None:
        if isinstance(parent, Tag) and parent.name == "section":
            sections.append(parent)
        parent = parent.parent

    title_path = [TOP_LEVEL_TITLE]

    for section in reversed(sections):
        heading = get_own_section_heading(section)
        if heading:
            title_path.append(heading)

    return deduplicate_preserve_order(title_path)

def parse_articles() -> list[dict]:
    html = HTML_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    articles: list[dict] = []

    for article in soup.find_all("article", id=True):
        article_id = str(article.get("id", ""))

        if not article_id.startswith("art_"):
            continue

        article_number = article_id_to_number(article_id)

        footnotes = extract_footnotes(article)

        # Work on a copy so footnote removal does not affect future logic.
        article_copy = BeautifulSoup(str(article), "lxml").find("article")
        if article_copy is None:
            continue

        remove_footnote_container(article_copy)
        paragraphs = extract_paragraphs(article_copy)

        full_text = normalize_text(
            " ".join(paragraph["text"] for paragraph in paragraphs if paragraph["text"])
        )

        articles.append(
            {
                "id": f"or_{article_id.removeprefix('art_')}_en",
                "fedlex_anchor": article_id,
                "article_number": article_number,
                "language": "en",
                "source_url": f"{BASE_SOURCE_URL}#{article_id}",
                "source_type": "fedlex_print_view_snapshot",
                "title_path": get_title_path(article),
                "paragraphs": paragraphs,
                "text": full_text,
                "footnotes": footnotes,
                "status": infer_article_status(paragraphs, footnotes),
            }
        )

    return articles


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    articles = parse_articles()

    OUTPUT_PATH.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Parsed articles: {len(articles)}")
    print(f"Saved to: {OUTPUT_PATH}")

    for target in ["Art. 319", "Art. 337", "Art. 340", "Art. 362"]:
        matches = [article for article in articles if article["article_number"] == target]
        if not matches:
            print(f"{target}: NOT FOUND")
            continue

        article = matches[0]
        print(f"\n{target}")
        print(f"title_path: {article['title_path']}")
        print(f"paragraphs: {len(article['paragraphs'])}")
        print(f"footnotes: {len(article['footnotes'])}")
        print(article["text"][:500])


if __name__ == "__main__":
    main()