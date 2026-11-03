import pytest

from ai.result_aggregator import aggregate_reviews, _is_explicitly_clean
from ai.reviewer import ReviewResult


@pytest.mark.parametrize(
    "text",
    [
        "No issues found.",
        "No bugs found.",
        "LGTM",
        "No problems found.",
    ],
)
def test_explicit_clean_responses_are_accepted(text):
    assert _is_explicitly_clean(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The author refactored the caching layer.",
        "No issues found; one possible bug remains.",
        "No issues found. However, error handling is incomplete.",
        "Tests passed.",
        "No review generated.",
    ],
)
def test_ambiguous_or_incomplete_responses_are_not_clean(text):
    assert _is_explicitly_clean(text) is False


def test_ambiguous_review_produces_notice():
    result = aggregate_reviews([
        {
            "file": "cache.py",
            "status": "completed",
            "review": "The author refactored the caching layer.",
        }
    ])

    assert "Review Notices" in result
    assert "ambiguous" in result.lower()
    assert "cache.py" in result


def test_clean_and_ambiguous_results_do_not_claim_overall_clean():
    result = aggregate_reviews([
        {
            "file": "clean.py",
            "status": "completed",
            "review": "No issues found.",
        },
        {
            "file": "cache.py",
            "status": "completed",
            "review": "The author refactored the caching layer.",
        },
    ])

    assert "Review Notices" in result
    assert "ambiguous" in result.lower()
    assert "cache.py" in result


def test_failed_chunk_produces_notice():
    result = aggregate_reviews([
        {
            "file": "good.py",
            "status": "completed",
            "review": "No issues found.",
        },
        {
            "file": "large.py",
            "status": "failed",
            "review": "Error occurred during AI code review: provider error",
        },
    ])

    assert "Review Notices" in result
    assert "large.py" in result
    assert "failed" in result.lower()


def test_incomplete_chunk_produces_notice():
    result = aggregate_reviews([
        {
            "file": "large.py",
            "status": "incomplete",
            "review": "No review generated.",
        }
    ])

    assert "Review Notices" in result
    assert "large.py" in result
    assert "incomplete" in result.lower()


def test_valid_findings_are_preserved_when_another_chunk_fails():
    finding = (
        "**Issue:** Possible null dereference\n"
        "**File:** `src/cache.py`\n"
        "**Line:** 42\n"
        "**Reason:** The value may be None.\n"
        "**Suggestion:** Check the value before using it."
    )

    result = aggregate_reviews([
        {
            "file": "src/cache.py",
            "status": "completed",
            "review": finding,
        },
        {
            "file": "large.py",
            "status": "failed",
            "review": "Error occurred during AI code review: provider error",
        },
    ])

    assert "Possible null dereference" in result
    assert "Review Notices" in result
    assert "large.py" in result


def test_review_result_preserves_status():
    result = ReviewResult("Partial review", status="incomplete")

    assert str(result) == "Partial review"
    assert result.status == "incomplete"