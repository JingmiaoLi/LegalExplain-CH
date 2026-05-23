from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Tag

HTML_PATH = Path("data/raw/fedlex/or_title_10_en_print_view.html")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def describe(tag: Tag) -> str:
    tag_id = tag.get("id", "")
    classes = " ".join(tag.get("class", []))
    text = normalize_text(tag.get_text(" ", strip=True))
    return f"<{tag.name} id='{tag_id}' class='{classes}'> {text[:220]}"


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    for article_id in ["art_319", "art_337", "art_340", "art_362"]:
        article = soup.find("article", id=article_id)
        if article is None:
            print(f"{article_id}: not found")
            continue

        print("\n" + "#" * 100)
        print(f"ARTICLE: {article_id}")
        print("#" * 100)

        sections = []
        parent = article.parent

        while parent is not None:
            if isinstance(parent, Tag) and parent.name == "section":
                sections.append(parent)
            parent = parent.parent

        for index, section in enumerate(reversed(sections), start=1):
            print("\n" + "=" * 80)
            print(f"SECTION LEVEL {index}")
            print(describe(section))

            print("\nDirect children:")
            for child in section.children:
                if isinstance(child, Tag):
                    print("  ", describe(child))

            print("\nDirect headings only:")
            for child in section.children:
                if isinstance(child, Tag) and child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    print("  ", describe(child))


if __name__ == "__main__":
    main()