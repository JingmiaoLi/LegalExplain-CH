from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString


HTML_PATH = Path("data/raw/fedlex/or_title_10_en_print_view.html")


def short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def describe(node) -> str:
    if isinstance(node, NavigableString):
        text = short(str(node))
        return f"TEXT: {text}" if text else ""

    if isinstance(node, Tag):
        attrs = []
        if node.get("id"):
            attrs.append(f"id={node.get('id')}")
        if node.get("class"):
            attrs.append(f"class={' '.join(node.get('class'))}")
        attr_text = " ".join(attrs)
        text = short(node.get_text(" ", strip=True))
        return f"<{node.name} {attr_text}> {text}"

    return repr(node)


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    for article_id in ["art_319", "art_337", "art_340", "art_362"]:
        article = soup.find("article", id=article_id)
        if article is None:
            print(f"{article_id}: not found")
            continue

        print("\n" + "=" * 80)
        print(f"ARTICLE: {article_id}")
        print("=" * 80)

        print("\nDirect children:")
        for i, child in enumerate(article.children):
            desc = describe(child)
            if desc:
                print(f"{i}: {desc}")

        print("\nNested tags:")
        for i, tag in enumerate(article.find_all(True)[:80]):
            print(f"{i}: {describe(tag)}")


if __name__ == "__main__":
    main()