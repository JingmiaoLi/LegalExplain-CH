from pathlib import Path

import requests


FEDLEX_PRINT_URL = (
    "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en"
    "?print=true&printId=%23part_2%2Ftit_10"
)

OUTPUT_PATH = Path("data/raw/fedlex/or_title_10_en_print_snapshot.html")


def fetch_fedlex_print_snapshot() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": (
            "law_rag data discovery script "
            "(educational portfolio project; source: Fedlex)"
        )
    }

    response = requests.get(FEDLEX_PRINT_URL, headers=headers, timeout=30)
    response.raise_for_status()

    OUTPUT_PATH.write_text(response.text, encoding="utf-8")

    print(f"Saved Fedlex print snapshot to: {OUTPUT_PATH}")
    print(f"Status code: {response.status_code}")
    print(f"HTML length: {len(response.text)} characters")

    preview = response.text[:500].replace("\n", " ")
    print("\nPreview:")
    print(preview)


if __name__ == "__main__":
    fetch_fedlex_print_snapshot()