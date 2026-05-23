from app.services.issue_classifier import classify_issues


def test_classify_immediate_dismissal():
    scenario = "My employer dismissed me immediately without warning."

    issues = classify_issues(scenario)

    issue_ids = [issue.id for issue in issues]

    assert "immediate_dismissal" in issue_ids


def test_classify_non_compete():
    scenario = "My contract has a non-compete clause after leaving the company."

    issues = classify_issues(scenario)

    issue_ids = [issue.id for issue in issues]

    assert "non_compete" in issue_ids


def test_default_to_notice_period_when_no_clear_match():
    scenario = "My employer ended my employment contract last week."

    issues = classify_issues(scenario)

    assert len(issues) >= 1
    assert issues[0].id == "notice_period"
    assert issues[0].confidence == "low"
    