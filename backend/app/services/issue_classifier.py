from app.legal_data.topics import ISSUE_CONFIG
from app.schemas.legal import LegalIssue


def classify_issues(scenario: str) -> list[LegalIssue]:
    normalized = scenario.lower()
    issues: list[LegalIssue] = []

    for issue_id, config in ISSUE_CONFIG.items():
        score = sum(
            1 for keyword in config["keywords"] if keyword.lower() in normalized
        )

        if score > 0:
            issues.append(
                LegalIssue(
                    id=issue_id,
                    label=config["label"],
                    description=config["description"],
                    confidence="high" if score >= 2 else "medium",
                )
            )

    if not issues:
        issues.append(
            LegalIssue(
                id="notice_period",
                label=ISSUE_CONFIG["notice_period"]["label"],
                description=(
                    "The scenario may involve general employment termination rules, "
                    "but more information is needed."
                ),
                confidence="low",
            )
        )

    return issues