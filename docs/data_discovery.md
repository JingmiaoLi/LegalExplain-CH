# Fedlex Data Discovery Notes

## 1. Goal

This document records the initial data discovery process for the `law_rag` project.

The purpose is to understand the structure of the official Fedlex legal text before finalizing the legal article schema, parsing pipeline, chunking strategy, embedding strategy, and knowledge graph design.

The project should not finalize the official `LegalArticle` JSON structure before inspecting the real Fedlex text, because the source contains structural elements such as titles, sections, article numbers, paragraph numbers, footnote markers, amendment notes, and possibly additional metadata.

---

## 2. Project scope

The V1 legal scope is Swiss employment contract law under the Swiss Code of Obligations / Obligationenrecht.

Target scope:

```text
Swiss Code of Obligations
Title Ten: The Employment Contract
Art. 319-362 OR / CO
```

The V1 focus topics are:

- non-compete clause
- immediate dismissal
- abusive termination
- notice period
- salary during illness

The project should explicitly exclude:

```text
Title Eleven: The Work Contract
Art. 363 onwards
```

Important terminology distinction:

- `employment contract` refers to an employer-employee relationship.
- `work contract` refers to a contractor-customer relationship where the contractor undertakes to produce a piece of work.

The project should consistently use:

```text
Swiss employment contract law
employment contract
employment-law scenario
```

and avoid using `work contract` for the V1 scope.

---

## 3. Primary source

Primary source candidate:

```text
Fedlex official web page for the Swiss Code of Obligations
https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en
```

The English version is useful for the V1 demo because the interface, explanations, and portfolio presentation are currently planned in English.

However, English is not an official language of Switzerland. Future versions should support at least one official-language source, such as German, French, or Italian.

Recommended future source direction:

```text
V1:
English Fedlex text for accessibility and demo clarity.

V2:
German official-language Fedlex text.

V3:
Multilingual source support: DE / FR / IT / EN.
```

---

## 4. Fedlex print view discovery

Fedlex provides a print view for selected structural units.

Candidate print view URL:

```text
https://www.fedlex.admin.ch/eli/cc/27/317_321_377/en?print=true&printId=%23part_2%2Ftit_10
```

This URL appears to isolate:

```text
Title Ten: The Employment Contract
```

This is much closer to the project scope than the full Code of Obligations page.

Observed print view options:

```text
Text
Additional information
Links as footnote
```

Initial V1 decision:

- `Text` should be enabled.
- `Additional information` should remain disabled for the first parsing target.
- `Links as footnote` should remain disabled for the first parsing target.

Reason:

The first parser should focus on the clean legal text structure. Additional information and link-footnote expansion may introduce extra noise into the main text and should be handled separately later.

---

## 5. Observed legal text structure

The Fedlex print view exposes a clear legal hierarchy.

Observed structure:

```text
Title Ten: The Employment Contract
  Section One: The Individual Employment Contract
    A. Definition and conclusion
      I. Definition
        Art. 319
          paragraph 1
          paragraph 2
```

Example from Art. 319:

```text
Art. 319

1 By means of an individual employment contract, the employee undertakes to work in the service of the employer for a limited or unlimited period and the employer undertakes to pay him a salary based on the amount of time he works (time wage) or the tasks he performs (piece work).

2 A contract whereby an employee undertakes to work regularly in the employer's service by hours, half-days or days (part-time work) is likewise deemed to be an individual employment contract.
```

This means the future article schema should preserve not only article text but also the structural path.

Potential structure:

```json
{
  "id": "or_319_en",
  "law": "Code of Obligations",
  "language": "en",
  "title_path": [
    "Title Ten: The Employment Contract",
    "Section One: The Individual Employment Contract",
    "A. Definition and conclusion",
    "I. Definition"
  ],
  "article_number": "Art. 319",
  "article_title": null,
  "paragraphs": [
    {
      "number": "1",
      "text": "..."
    },
    {
      "number": "2",
      "text": "..."
    }
  ],
  "footnotes": [],
  "source_url": "...",
  "source_type": "fedlex_print_view"
}
```

The `title_path` field is important for both retrieval and knowledge graph construction.

---

## 6. Footnote observations

Fedlex includes footnote markers in the legal text.

Example:

```text
Title Ten:119 The Employment Contract
```

The corresponding footnote text appears below the heading:

```text
119 Amended by No I of the FA of 25 June 1971, in force since 1 Jan. 1972 ...
```

Initial decision:

Footnotes should be preserved as metadata, but should not be mixed into the main legal text used for ordinary retrieval, chunking, embedding, and explanation.

Reason:

Footnotes often contain amendment notes, historical references, or source references. If they are mixed into the main article text, retrieval quality may decrease because the model may treat historical amendment information as part of the substantive legal rule.

Proposed handling:

```json
{
  "text": "main legal rule text only",
  "paragraphs": [
    {
      "number": "1",
      "text": "main paragraph text only"
    }
  ],
  "footnotes": [
    {
      "marker": "119",
      "text": "Amended by ...",
      "type": "amendment_note"
    }
  ]
}
```

Default V1 behavior:

- Use main legal text for retrieval and explanation.
- Preserve footnotes as metadata.
- Do not show footnotes in the main visual explanation graph unless the user explicitly asks about source history or amendments.

---

## 7. Relationship to knowledge graph design

The project should distinguish two graph layers.

### 7.1 Source structure graph

This graph reflects the official structure of the Fedlex legal text.

Examples:

```text
Title Ten contains Section One
Section One contains A. Definition and conclusion
A contains I. Definition
I contains Art. 319
Art. 319 has paragraph 1
Art. 319 has paragraph 2
Title Ten has footnote 119
```

Possible edge labels:

```text
contains
has_paragraph
has_footnote
```

This graph is source-structure-oriented.

### 7.2 Legal issue mapping graph

This graph supports user-facing legal explanation.

Examples:

```text
Scenario
-> may involve -> Issue: immediate dismissal
-> grounded in -> Art. 337 OR
-> requires condition -> valid reason / important reason
-> needs fact -> what exactly happened?
-> needs fact -> was the dismissal issued promptly?
-> requires -> human legal review
```

Possible node types:

```text
scenario
issue
article
paragraph
legal_concept
condition
fact_question
risk
procedure
remedy
human_review
```

Possible edge labels:

```text
may_involve
grounded_in
requires_condition
needs_fact
has_consequence
has_procedure
has_exception
requires_review
```

The V1 visual explanation should focus on the legal issue mapping graph, while preserving the source structure graph in the backend data model for future expansion.

---

## 8. Data pipeline principle

The system should not fetch, parse, chunk, or embed Fedlex text on every user request.

Instead, the project should separate offline data processing from online query handling.

### 8.1 Offline data pipeline

```text
Fedlex web text
-> save raw source snapshot
-> parse Art. 319-362
-> separate main text from footnotes
-> preserve title path and paragraph structure
-> generate structured article JSON
-> optionally build chunks
-> optionally build embeddings
-> save local processed data
```

### 8.2 Online query workflow

```text
user scenario
-> issue classification
-> retrieve local structured articles / chunks / graph
-> build legal issue graph
-> generate source-grounded explanation
-> return structured JSON
```

The backend should use local processed data at runtime. It should not depend on live Fedlex access for every request.

---

## 9. Snapshot meaning

A source snapshot means a saved local copy of the official source at a specific point in time.

Example:

```text
data/raw/fedlex/or_title_10_en_2026-05-23.html
```

The snapshot is not a screenshot. It is a saved source file used for reproducible parsing.

Benefits:

- reproducibility
- stable tests
- no runtime dependency on external websites
- traceable legal data version
- easier debugging of parsing logic

The processed data should record the source snapshot and source URL.

---

## 10. Initial schema questions

Before finalizing the official article schema, the following questions must be answered:

- Does the print view consistently preserve article numbers?
- Does each article have an article-specific title, or only higher-level section headings?
- How are paragraphs represented in the HTML?
- How are footnote markers represented?
- Can footnote text be separated reliably from the main text?
- Does the print view end at Art. 362, or does it include additional employment-related content beyond the V1 scope?
- Are lettered articles such as Art. 340a, 340b, and 340c represented consistently?
- Should each paragraph become its own chunk?
- Should each article become a graph node?
- Should section headings become graph nodes?
- Should footnotes become graph nodes or only metadata?
- What should be the canonical source URL for each article?
- Should the first official data layer use English only, or should German be prioritized earlier?

---

## 11. Current backend data status

The current backend legal article data is seed data only.

It is used to validate:

- `/analyze` API behavior
- issue classification
- article retrieval
- graph construction
- explanation generation
- automated tests

It is not official legal text and should not be presented as such.

The official Fedlex data layer should be implemented only after completing data discovery and confirming the source structure.

---

## 12. Next engineering steps

Recommended next steps:

1. Save the Fedlex print view observation in this document.
2. Create a small manual sample for Art. 319.
3. Draft a sample article schema using the observed structure.
4. Inspect the print view HTML structure.
5. Decide whether the print view can be fetched with a script or whether browser-based snapshotting is required.
6. Build a small parser for Art. 319 first.
7. Validate parsing on a few target articles:
   - Art. 324a
   - Art. 335c
   - Art. 336-336b
   - Art. 337
   - Art. 340-340c
8. Only then generate a processed JSON file for the full Art. 319-362 range.

---

## 13. Boundary note: Employment Contract vs Work Contract

Fedlex contains both:

```text
Title Ten: The Employment Contract
Title Eleven: The Work Contract
```

These are different legal areas.

The V1 project scope is:

```text
Title Ten: The Employment Contract
Art. 319-362
```

The project should exclude:

```text
Title Eleven: The Work Contract
Art. 363 onwards
```

This distinction is important for both data extraction and legal issue classification.

---

## 14. Current engineering checkpoint

The current backend foundation is complete.

Implemented:

- FastAPI backend
- `/analyze` endpoint
- modular workflow service layer
- legal schema
- basic legal-data seed layer
- automated tests with pytest
- GitHub Actions CI

Current backend workflow:

```text
scenario input
-> /analyze API
-> issue classification
-> article retrieval
-> graph construction
-> explanation generation
-> structured JSON response
```

Current CI behavior:

```text
push / pull request to main
-> set up Python 3.12
-> install backend dependencies
-> run pytest
-> verify backend workflow
```

Current test status:

```text
pytest: 6 passed
```
