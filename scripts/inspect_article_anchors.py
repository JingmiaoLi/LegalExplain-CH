from pathlib import Path
from bs4 import BeautifulSoup


HTML_PATH = Path("data/raw/fedlex/or_title_10_en_print_view.html")


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    article_anchors = []

    for element in soup.find_all(id=True):
        element_id = element.get("id", "")
        if element_id.startswith("art_"):
            article_anchors.append(element_id)

    unique_anchors = sorted(set(article_anchors))

    print(f"Found {len(unique_anchors)} unique article anchors:")
    for anchor in unique_anchors:
        print(anchor)


if __name__ == "__main__":
    main()