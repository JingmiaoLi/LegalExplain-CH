from pathlib import Path

from playwright.sync_api import sync_playwright


FEDLEX_PRINT_URL = (
    "https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en"
    "?print=true&printId=%23part_2%2Ftit_10"
)

OUTPUT_PATH = Path("data/raw/fedlex/or_title_10_en_print_browser_snapshot.html")


def fetch_with_browser() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Opening: {FEDLEX_PRINT_URL}")
        page.goto(FEDLEX_PRINT_URL, wait_until="networkidle", timeout=60000)

        # Give Fedlex extra time to render the legal text.
        page.wait_for_timeout(5000)

        html = page.content()
        OUTPUT_PATH.write_text(html, encoding="utf-8")

        print(f"Saved browser-rendered snapshot to: {OUTPUT_PATH}")
        print(f"HTML length: {len(html)} characters")

        if "Art. 319" in html:
            print("Found Art. 319 in rendered HTML ✅")
        else:
            print("Art. 319 not found in rendered HTML ❌")

        if "The Employment Contract" in html:
            print("Found The Employment Contract in rendered HTML ✅")
        else:
            print("The Employment Contract not found in rendered HTML ❌")

        browser.close()


if __name__ == "__main__":
    fetch_with_browser()