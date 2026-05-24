# scripts/build_article_structure_index.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "articles_or_title_10_en.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "article_structure_index.json"


def normalize_article_number(value: str) -> str:
    """
    Normalize article numbers so both '337' and 'Art. 337' become '337'.
    """
    value = value.strip()

    if value.lower().startswith("art."):
        value = value[4:].strip()

    return value


def title_path_key(title_path: list[str]) -> str:
    """
    Convert a title path list into a stable string key.
    """
    return " > ".join(title_path)


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


def load_articles(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of article objects.")

    return data


def sorted_article_numbers(article_numbers: list[str]) -> list[str]:
    """
    Sort Swiss article numbers such as 335, 335a, 335b, 337, 337c.

    This is not a universal legal citation sorter, but works well enough
    for article numbers in the current dataset.
    """

    def sort_key(article_number: str) -> tuple[int, str]:
        number_part = ""
        suffix_part = ""

        for char in article_number:
            if char.isdigit():
                number_part += char
            else:
                suffix_part += char

        numeric = int(number_part) if number_part else 0
        return numeric, suffix_part

    return sorted(set(article_numbers), key=sort_key)


def heading_category(heading: str) -> str:
    """
    Classify Fedlex-style headings.

    Examples:
        B. Obligations of the employee          -> uppercase_letter
        IV. Termination with immediate effect  -> roman
        1. Requirements                        -> numeric
        a. For good cause                      -> lowercase_letter
    """
    heading = heading.strip()

    # Roman numerals must be checked before uppercase letters,
    # because "IV." also starts with an uppercase letter.
    if re.match(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+", heading):
        return "roman"

    if re.match(r"^[A-Z]\.\s+", heading):
        return "uppercase_letter"

    if re.match(r"^\d+\.\s+", heading):
        return "numeric"

    if re.match(r"^[a-z]\.\s+", heading):
        return "lowercase_letter"

    return "other"


def is_meaningful_expansion_heading(heading: str) -> bool:
    """
    Decide whether a heading is broad enough to use as default expansion scope.

    In Fedlex, headings such as:
        B. Obligations of the employee
        IV. Termination with immediate effect

    are usually better expansion scopes than very narrow headings such as:
        1. Requirements
        a. For good cause
    """
    return heading_category(heading) in {"uppercase_letter", "roman"}


def add_article_to_tree(
    tree: dict[str, Any],
    title_path: list[str],
    article_number: str,
) -> None:
    """
    Add one article to the nested tree structure.

    Tree format:
    {
      "label": "root",
      "children": [
        {
          "label": "Title Ten: ...",
          "children": [...],
          "articles": [...]
        }
      ]
    }
    """
    current_node = tree

    for level, title in enumerate(title_path):
        children = current_node.setdefault("children", [])

        matching_child = None
        for child in children:
            if child.get("label") == title:
                matching_child = child
                break

        if matching_child is None:
            matching_child = {
                "label": title,
                "level": level,
                "category": heading_category(title),
                "children": [],
                "articles": [],
            }
            children.append(matching_child)

        current_node = matching_child

    current_node.setdefault("articles", [])

    if article_number not in current_node["articles"]:
        current_node["articles"].append(article_number)


def build_by_title_path(
    article_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Build lookup:
    title_path_key -> articles under exactly this title_path.
    """
    by_title_path: dict[str, dict[str, Any]] = {}

    for record in article_records:
        article_number = record["article_number"]
        title_path = record["title_path"]

        key = title_path_key(title_path)

        if key not in by_title_path:
            heading = title_path[-1] if title_path else "root"

            by_title_path[key] = {
                "title_path": title_path,
                "heading": heading,
                "level": len(title_path) - 1 if title_path else None,
                "category": heading_category(heading),
                "articles": [],
            }

        by_title_path[key]["articles"].append(article_number)

    for item in by_title_path.values():
        item["articles"] = sorted_article_numbers(item["articles"])

    return by_title_path


def get_parent_title_path(title_path: list[str]) -> list[str]:
    """
    Return parent path by removing the lowest-level heading.
    """
    if len(title_path) <= 1:
        return []

    return title_path[:-1]


def collect_articles_under_path(
    path_prefix: list[str],
    by_title_path: dict[str, dict[str, Any]],
) -> list[str]:
    """
    Collect all articles whose title_path starts with the given path_prefix.
    """
    if not path_prefix:
        return []

    prefix_key = title_path_key(path_prefix)
    articles: list[str] = []

    for key, section_info in by_title_path.items():
        if key == prefix_key or key.startswith(prefix_key + " > "):
            articles.extend(section_info.get("articles", []))

    return sorted_article_numbers(articles)


def build_ancestor_section_articles(
    title_path: list[str],
    by_title_path: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    For each ancestor level of the title_path, collect all articles under that ancestor.

    Example:
        Art. 337 title_path:
            Title Ten
            Section One
            G. End of the employment relationship
            IV. Termination with immediate effect
            1. Requirements
            a. For good cause

        This function gives article lists under each of those levels.
    """
    ancestor_section_articles: dict[str, dict[str, Any]] = {}

    for level in range(len(title_path)):
        ancestor_path = title_path[: level + 1]
        heading = ancestor_path[-1]
        articles_under_ancestor = collect_articles_under_path(
            ancestor_path,
            by_title_path,
        )

        ancestor_section_articles[f"level_{level}"] = {
            "level": level,
            "heading": heading,
            "category": heading_category(heading),
            "title_path": ancestor_path,
            "articles": sorted_article_numbers(articles_under_ancestor),
        }

    return ancestor_section_articles


def choose_recommended_expansion_articles(
    article_number: str,
    title_path: list[str],
    ancestor_section_articles: dict[str, dict[str, Any]],
    min_articles: int = 2,
    max_articles: int = 15,
) -> tuple[list[str], dict[str, Any]]:
    """
    Select a useful expansion set for RAG.

    Strategy:
    1. Prefer the nearest meaningful ancestor heading.
       Meaningful headings are usually Roman numerals or uppercase-letter headings.
       Example:
           IV. Termination with immediate effect
           B. Obligations of the employee

    2. Avoid very narrow headings such as:
           1. Requirements
           a. For good cause

       These are useful for explanation paths, but usually too narrow as default
       multi-hop expansion scopes.

    3. Avoid overly broad sections with too many articles.

    4. If no meaningful ancestor is suitable, fall back to the nearest reasonable
       ancestor with a manageable number of articles.

    5. If nothing works, return the article itself.
    """
    # First pass: choose nearest meaningful ancestor.
    for level in reversed(range(len(title_path))):
        heading = title_path[level]

        if not is_meaningful_expansion_heading(heading):
            continue

        level_key = f"level_{level}"
        section_info = ancestor_section_articles.get(level_key, {})
        candidate_articles = section_info.get("articles", [])

        if min_articles <= len(candidate_articles) <= max_articles:
            return (
                sorted_article_numbers(candidate_articles),
                {
                    "level": level,
                    "heading": heading,
                    "category": heading_category(heading),
                    "title_path": section_info.get(
                        "title_path",
                        title_path[: level + 1],
                    ),
                    "article_count": len(candidate_articles),
                    "selection_reason": "nearest_meaningful_ancestor",
                },
            )

    # Second pass: choose nearest reasonable ancestor regardless of category.
    for level in reversed(range(len(title_path))):
        level_key = f"level_{level}"
        section_info = ancestor_section_articles.get(level_key, {})
        candidate_articles = section_info.get("articles", [])

        if min_articles <= len(candidate_articles) <= max_articles:
            heading = title_path[level]

            return (
                sorted_article_numbers(candidate_articles),
                {
                    "level": level,
                    "heading": heading,
                    "category": heading_category(heading),
                    "title_path": section_info.get(
                        "title_path",
                        title_path[: level + 1],
                    ),
                    "article_count": len(candidate_articles),
                    "selection_reason": "nearest_reasonable_ancestor",
                },
            )

    # Final fallback: article only.
    fallback_heading = title_path[-1] if title_path else None

    return (
        [article_number],
        {
            "level": len(title_path) - 1 if title_path else None,
            "heading": fallback_heading,
            "category": heading_category(fallback_heading) if fallback_heading else None,
            "title_path": title_path,
            "article_count": 1,
            "selection_reason": "fallback_article_only",
        },
    )


def build_by_article(
    article_records: list[dict[str, Any]],
    by_title_path: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Build lookup:
    article_number -> title_path and structure-aware expansion metadata.
    """
    by_article: dict[str, dict[str, Any]] = {}

    for record in article_records:
        article_number = record["article_number"]
        title_path = record["title_path"]

        same_section_key = title_path_key(title_path)
        exact_leaf_articles = by_title_path.get(
            same_section_key,
            {},
        ).get("articles", [])

        parent_title_path = get_parent_title_path(title_path)
        direct_parent_articles: list[str] = []

        if parent_title_path:
            direct_parent_articles = collect_articles_under_path(
                parent_title_path,
                by_title_path,
            )

        ancestor_section_articles = build_ancestor_section_articles(
            title_path,
            by_title_path,
        )

        recommended_expansion_articles, recommended_expansion_basis = (
            choose_recommended_expansion_articles(
                article_number=article_number,
                title_path=title_path,
                ancestor_section_articles=ancestor_section_articles,
            )
        )

        by_article[article_number] = {
            "article_number": article_number,
            "title_path": title_path,

            # Exact leaf = articles under the exact same full title_path.
            # Often this is only the article itself, because Fedlex headings are very fine-grained.
            "exact_leaf_articles": sorted_article_numbers(exact_leaf_articles),

            # Direct parent = articles under the immediate parent heading.
            # Example for Art. 337:
            #   parent = "1. Requirements"
            #   articles = ["337", "337a"]
            "parent_title_path": parent_title_path,
            "direct_parent_articles": sorted_article_numbers(direct_parent_articles),

            # All ancestor levels with their article sets.
            # This is the most important data for structure-aware expansion.
            "ancestor_section_articles": ancestor_section_articles,

            # Recommended default expansion scope for multi-hop RAG.
            # Example for Art. 337:
            #   basis = "IV. Termination with immediate effect"
            #   articles = ["337", "337a", "337b", "337c", "337d"]
            "recommended_expansion_basis": recommended_expansion_basis,
            "recommended_expansion_articles": recommended_expansion_articles,
        }

    return by_article


def build_article_structure_index(
    articles: list[dict[str, Any]],
) -> dict[str, Any]:
    tree: dict[str, Any] = {
        "label": "root",
        "level": None,
        "category": "root",
        "children": [],
        "articles": [],
    }

    article_records: list[dict[str, Any]] = []
    skipped_count = 0

    for article in articles:
        article_number = normalize_article_number(
            str(article.get("article_number", ""))
        )

        if not article_number:
            skipped_count += 1
            continue

        title_path = clean_title_path(article.get("title_path", []))

        record = {
            "article_number": article_number,
            "title_path": title_path,
        }

        article_records.append(record)
        add_article_to_tree(tree, title_path, article_number)

    by_title_path = build_by_title_path(article_records)
    by_article = build_by_article(article_records, by_title_path)

    return {
        "metadata": {
            "source_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
            "article_count": len(article_records),
            "skipped_count": skipped_count,
            "title_path_count": len(by_title_path),
            "description": (
                "Official legal structure index derived from article title_path metadata. "
                "The tree view supports navigation and visualization. "
                "The by_article view supports structure-aware retrieval and multi-hop expansion. "
                "Recommended expansion articles are selected from meaningful ancestor headings, "
                "such as Roman numeral and uppercase-letter Fedlex headings."
            ),
        },
        "tree": tree,
        "by_article": by_article,
        "by_title_path": by_title_path,
    }


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_sample(index: dict[str, Any]) -> None:
    by_article = index["by_article"]

    print("\nSample article entries:")

    for article_number in ["319", "321", "321a", "330a", "337", "337c"]:
        if article_number not in by_article:
            continue

        item = by_article[article_number]

        print(f"\nArt. {article_number}")
        print(f"title_path: {item['title_path']}")
        print(f"exact_leaf_articles: {item['exact_leaf_articles']}")
        print(f"direct_parent_articles: {item['direct_parent_articles']}")
        print(f"recommended_expansion_basis: {item['recommended_expansion_basis']}")
        print(f"recommended_expansion_articles: {item['recommended_expansion_articles']}")


def main() -> None:
    articles = load_articles(INPUT_PATH)
    index = build_article_structure_index(articles)
    save_json(index, OUTPUT_PATH)

    print(f"Loaded articles: {len(articles)}")
    print(f"Indexed articles: {index['metadata']['article_count']}")
    print(f"Skipped articles: {index['metadata']['skipped_count']}")
    print(f"Title paths: {index['metadata']['title_path_count']}")
    print(f"Saved to: {OUTPUT_PATH}")

    print_sample(index)


if __name__ == "__main__":
    main()