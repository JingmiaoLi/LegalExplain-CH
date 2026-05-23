from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString


HTML_PATH = Path("data/raw/fedlex/or_title_10_en_print_view.html")


def short(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def describe_node(node) -> str:
    if isinstance(node, NavigableString):
        text = short(str(node))
        if not text:
            return ""
        return f"TEXT: {text}"

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

    target = soup.find(id="art_319")
    if target is None:
        raise RuntimeError("art_319 not found")

    print("TARGET:")
    print(describe_node(target))

    print("\nPARENTS:")
    parent = target.parent
    depth = 0
    while parent is not None and depth < 8:
        print(f"{depth}: {describe_node(parent)}")
        parent = parent.parent
        depth += 1

    print("\nNEXT SIBLINGS:")
    parent = target.parent
    siblings = list(parent.next_siblings)

    count = 0
    for sibling in siblings:
        desc = describe_node(sibling)
        if desc:
            print(f"{count}: {desc}")
            count += 1
        if count >= 40:
            break


if __name__ == "__main__":
    main()
    