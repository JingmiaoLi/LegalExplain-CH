from pathlib import Path
from bs4 import BeautifulSoup


HTML_PATH = Path("data/raw/fedlex/or_title_10_en_print_view.html")
OUTPUT_PATH = Path("data/processed/fedlex_visible_text_preview.txt")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    if not HTML_PATH.exists():
        raise FileNotFoundError(f"HTML file not found: {HTML_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    html = HTML_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")

    # Remove scripts/styles/noisy elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text_lines = []
    for element in soup.find_all(text=True):
        text = normalize_text(element)
        if text:
            text_lines.append(text)

    OUTPUT_PATH.write_text("\n".join(text_lines), encoding="utf-8")

    print(f"HTML length: {len(html)} characters")
    print(f"Extracted text lines: {len(text_lines)}")
    print(f"Saved preview to: {OUTPUT_PATH}")

    print("\nFirst 80 lines:")
    for line in text_lines[:80]:
        print(line)

    print("\nSearch checks:")
    for keyword in [
        "Title Ten",
        "The Employment Contract",
        "Art. 319",
        "Art. 337",
        "Art. 340",
        "Art. 362",
        "Art. 363",
    ]:
        found = any(keyword.lower() in line.lower() for line in text_lines)
        print(f"{keyword}: {'FOUND' if found else 'not found'}")


if __name__ == "__main__":
    main()