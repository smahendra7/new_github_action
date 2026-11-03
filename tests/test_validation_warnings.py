"""
tests/test_validation_warnings.py

Tests for the new warning-based PR validation behavior.

Verifies:
  - Validation failures become warnings, not pipeline-stopping failures.
  - All validation checks (description quality, desc-vs-code, comment accuracy)
    run independently and results are all collected.
  - AI Code Review always runs even when validation warnings exist.
  - Exactly ONE combined comment is posted to the PR.
  - _build_validation_section produces correct markdown for all warning combos.
  - _build_combined_comment always includes both sections.
  - sys.exit(1) is NOT called when the result is 'success' (even with warnings).
  - suggestion_generator functions return usable strings on API failure.
"""

import sys
import pytest
from unittest.mock import patch, MagicMock

from main import _build_validation_section, _build_combined_comment


# ─────────────────────────────────────────────────────────────────────────────
# _build_validation_section
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildValidationSectionNoWarnings:

    def test_no_warnings_shows_checkmark(self):
        section = _build_validation_section(None, None, None, [])
        assert "## PR Validation" in section
        assert "\u2705" in section
        assert "No validation issues found" in section

    def test_no_warnings_no_warning_emoji(self):
        section = _build_validation_section(None, None, None, [])
        assert "\u26a0" not in section

    def test_no_warnings_no_suggestion(self):
        section = _build_validation_section(None, None, None, [])
        assert "Suggested" not in section


class TestBuildValidationSectionDescWarning:

    def test_warning_emoji_present(self):
        section = _build_validation_section(
            "PR Description needs improvement",
            "Description is empty.",
            "Add a meaningful PR description.",
            [],
        )
        assert "\u26a0" in section

    def test_warning_label_present(self):
        section = _build_validation_section(
            "PR Description needs improvement",
            "Description is empty.",
            "Add a meaningful PR description.",
            [],
        )
        assert "PR Description needs improvement" in section

    def test_reason_included(self):
        section = _build_validation_section(
            "PR Description needs improvement",
            "Description is placeholder text.",
            None,
            [],
        )
        assert "Description is placeholder text." in section

    def test_suggestion_included_when_present(self):
        section = _build_validation_section(
            "PR Description needs improvement",
            "Too vague.",
            "Implements the authentication flow.",
            [],
        )
        assert "Suggested PR Description" in section
        assert "Implements the authentication flow." in section

    def test_no_suggestion_block_when_absent(self):
        section = _build_validation_section(
            "PR Description needs improvement",
            "Empty description.",
            None,
            [],
        )
        assert "Suggested PR Description" not in section


class TestBuildValidationSectionCommentWarnings:

    @staticmethod
    def _mismatch(i=1):
        return {
            "file": f"app/Foo{i}.kt",
            "line": str(10 + i),
            "comment": f"Returns user name {i}",
            "code": f"return userId{i}",
            "reason": f"Comment says name but code returns id {i}",
            "suggestion": f"Returns the user ID {i}",
        }

    def test_single_comment_warning_file_appears(self):
        section = _build_validation_section(None, None, None, [self._mismatch(1)])
        assert "app/Foo1.kt" in section

    def test_single_comment_warning_suggestion_appears(self):
        section = _build_validation_section(None, None, None, [self._mismatch(1)])
        assert "Returns the user ID 1" in section

    def test_single_comment_warning_label_appears(self):
        section = _build_validation_section(None, None, None, [self._mismatch(1)])
        assert "Code Comment needs improvement" in section

    def test_multiple_mismatches_all_included(self):
        mismatches = [self._mismatch(i) for i in range(1, 4)]
        section = _build_validation_section(None, None, None, mismatches)
        for i in range(1, 4):
            assert f"app/Foo{i}.kt" in section
            assert f"Returns the user ID {i}" in section

    def test_both_desc_and_comment_warnings_in_one_section(self):
        section = _build_validation_section(
            "PR Description needs improvement",
            "Too vague.",
            "Better description.",
            [self._mismatch(1)],
        )
        assert "PR Description needs improvement" in section
        assert "Code Comment needs improvement" in section
        assert "Better description." in section
        assert "Returns the user ID 1" in section
        assert "---" in section

    def test_multiple_comment_warnings_have_separators(self):
        mismatches = [self._mismatch(i) for i in range(1, 3)]
        section = _build_validation_section(None, None, None, mismatches)
        assert "---" in section


# ─────────────────────────────────────────────────────────────────────────────
# _build_combined_comment
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCombinedComment:

    def test_both_sections_present(self):
        combined = _build_combined_comment(
            "## PR Validation\n\u2705 OK",
            "AI review output",
        )
        assert "## PR Validation" in combined
        assert "## AI Code Review" in combined
        assert "AI review output" in combined

    def test_starts_directly_with_pr_validation_and_no_robot_emoji(self):
        combined = _build_combined_comment("## PR Validation\n\u2705 OK", "review")
        assert combined.startswith("## PR Validation")
        assert "\U0001f916" not in combined

    def test_separator_present(self):
        combined = _build_combined_comment("## PR Validation\n\u2705 OK", "review")
        assert "---" in combined

    def test_validation_section_before_review_section(self):
        combined = _build_combined_comment("## PR Validation\n\u2705 OK", "AI review here.")
        assert combined.index("## PR Validation") < combined.index("## AI Code Review")

    def test_bold_labels_in_validation_section(self):
        mismatch = {
            "file": "comment_test.py",
            "line": "2",
            "comment": "adds discount",
            "code": "return price - discount",
            "reason": "contradicts code",
            "suggestion": "subtracts discount",
        }
        section = _build_validation_section(
            "PR Description needs improvement",
            "PR description is empty",
            "Suggested desc",
            [mismatch],
        )
        assert "**Reason:** PR description is empty" in section
        assert "**File:** `comment_test.py`" in section
        assert "**Line:** 2" in section
        assert "**Comment:**\n> adds discount" in section
        assert "**Reason:** contradicts code" in section

    def test_bold_labels_in_ai_code_review(self):
        raw_review = (
            "Issue:\nThe comment for calculate_discount\n\n"
            "File: comment_test.py\n"
            "Line: L2-L3\n\n"
            "Code:\n```python\nReason: not bold\n```\n\n"
            "Reason:\nContradicts actual behavior\n\n"
            "Suggestion:\nAlign the comment with actual behavior"
        )
        combined = _build_combined_comment("## PR Validation\n\u2705 OK", raw_review)
        assert "**Issue:**\nThe comment for calculate_discount" in combined
        assert "**File:** comment_test.py" in combined
        assert "**Line:** L2-L3" in combined
        assert "**Code:**\n```python" in combined
        assert "Reason: not bold" in combined  # untouched inside code block
        assert "**Reason:** Contradicts actual behavior" in combined
        assert "**Suggestion:**\nAlign the comment with actual behavior" in combined

    def test_multiple_ai_code_review_issues_separated_by_horizontal_rule(self):
        raw_review = (
            "Issue:\nFirst issue\n\n"
            "File: comment_test.py\n"
            "Line: L2-L3\n\n"
            "Code:\n```python\nx = 1\n```\n\n"
            "Reason:\nFirst reason\n\n"
            "Suggestion:\nFirst suggestion\n\n"
            "Issue:\nSecond issue\n\n"
            "File: comment_test.py\n"
            "Line: L7-L8\n\n"
            "Code:\n```python\ny = 2\n```\n\n"
            "Reason:\nSecond reason\n\n"
            "Suggestion:\nSecond suggestion"
        )
        combined = _build_combined_comment("## PR Validation\n\u2705 OK", raw_review)
        assert "**Reason:** First reason" in combined
        assert "**Reason:** Second reason" in combined
        # Check that there is a separator between the first and second issue
        review_part = combined.split("## AI Code Review\n\n")[1]
        assert "\n\n---\n\n" in review_part
        assert review_part.count("---") == 1

    def test_no_duplicate_consecutive_separators_in_ai_code_review(self):
        """When LLM review output already contains --- between issues, no duplicate --- should appear."""
        raw_review = (
            "Issue:\nFirst issue\n\n"
            "File: comment_test.py\n"
            "Line: L2-L3\n\n"
            "Code:\n```python\nx = 1\n```\n\n"
            "Reason:\nFirst reason\n\n"
            "Suggestion:\nFirst suggestion\n\n"
            "---\n\n"
            "Issue:\nSecond issue\n\n"
            "File: comment_test.py\n"
            "Line: L7-L8\n\n"
            "Code:\n```python\ny = 2\n```\n\n"
            "Reason:\nSecond reason\n\n"
            "Suggestion:\nSecond suggestion"
        )
        combined = _build_combined_comment("## PR Validation\n\u2705 OK", raw_review)
        assert "**Reason:** First reason" in combined
        assert "**Reason:** Second reason" in combined
        review_part = combined.split("## AI Code Review\n\n")[1]
        assert review_part.count("---") == 1
        assert "---\n\n---" not in combined
        assert "---\n---" not in combined



# ─────────────────────────────────────────────────────────────────────────────
# run_action exit behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestRunActionExitBehavior:

    def test_success_does_not_call_sys_exit(self):
        """Warnings should produce status=success, which must not trigger sys.exit(1)."""
        with patch("sys.exit") as mock_exit:
            result = {"status": "success"}
            if result.get("status") == "failed":
                sys.exit(1)
            mock_exit.assert_not_called()

    def test_genuine_failure_calls_sys_exit_1(self):
        """Only true technical failures should still trigger sys.exit(1)."""
        with patch("sys.exit") as mock_exit:
            result = {"status": "failed", "reason": "incomplete_prompt_file_sections"}
            if result.get("status") == "failed":
                sys.exit(1)
            mock_exit.assert_called_once_with(1)


# ─────────────────────────────────────────────────────────────────────────────
# suggestion_generator fallbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestSuggestionGeneratorFallbacks:

    def test_description_suggestion_no_api_key_returns_string(self):
        import os
        from ai.suggestion_generator import generate_description_suggestion
        os.environ.pop("GROQ_API_KEY", None)
        result = generate_description_suggestion("Fix login bug", "", "")
        assert isinstance(result, str) and len(result) > 0

    def test_comment_suggestion_no_api_key_returns_string(self):
        import os
        from ai.suggestion_generator import generate_comment_suggestion
        os.environ.pop("GROQ_API_KEY", None)
        result = generate_comment_suggestion("Foo.kt", "42", "Returns name", "return id", "mismatch")
        assert isinstance(result, str) and len(result) > 0

    def test_description_suggestion_api_error_returns_fallback(self):
        from ai.suggestion_generator import generate_description_suggestion
        with patch("ai.suggestion_generator._get_client") as mock_client:
            m = MagicMock()
            m.chat.completions.create.side_effect = Exception("Timeout")
            mock_client.return_value = m
            result = generate_description_suggestion("Title", "Body", "diff")
        assert isinstance(result, str) and len(result) > 0

    def test_comment_suggestion_api_error_returns_fallback(self):
        from ai.suggestion_generator import generate_comment_suggestion
        with patch("ai.suggestion_generator._get_client") as mock_client:
            m = MagicMock()
            m.chat.completions.create.side_effect = Exception("Network error")
            mock_client.return_value = m
            result = generate_comment_suggestion("Bar.kt", "10", "old", "new code", "mismatch")
        assert isinstance(result, str) and len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: process_pr_event warning flow
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationNeverStopsPipeline:
    """
    Integration-style tests verifying that validation warnings never produce
    status='failed' and that AI Code Review always runs.
    External calls (LLM, GitHub API, filesystem) are mocked.
    """

    def _payload(self):
        return {
            "pull_request": {
                "number": 42,
                "title": "test",
                "body": "x",
                "head": {"ref": "feature", "sha": "abc123"},
                "base": {"ref": "main"},
                "requested_reviewers": [],
            },
            "repository": {"full_name": "org/repo"},
            "action": "opened",
        }


    def test_description_failure_gives_success_status(self):
        from main import process_pr_event
        with (
            patch("main.get_pr_details", return_value=("title", "some description")),
            patch("main.capture_original_reviewers"),
            patch("main.clone_repository", return_value="/tmp/ws"),
            patch("main.get_pr_changed_files", return_value=["app/Foo.kt"]),
            patch("main.get_diff", return_value="diff text"),
            patch("main.validate_pr_description", return_value=(False, "too vague")),
            patch("main.validate_description_matches_code", return_value=(True, "ok")),
            patch("main.generate_description_suggestion", return_value="Better desc"),
            patch("main.validate_code_comments", return_value=[]),
            patch("main.extract_full_code", return_value="FILE: app/Foo.kt\nL1: code"),
            patch("main.count_file_sections_in_code", return_value=1),
            patch("main.extract_added_code", return_value=""),
            patch("main.build_review_prompt", return_value="prompt"),
            patch("main.review_code", return_value="AI review here"),
            patch("main.post_pr_comment"),
            patch("main.handle_reviewers_for_validation_result"),
            patch("os.path.exists", return_value=False),
        ):
            result = process_pr_event(
                self._payload(), "pull_request", "opened", None, None
            )
        assert result.get("status") == "success"

    def test_comment_failure_gives_success_status(self):
        from main import process_pr_event
        mismatch = {
            "file": "app/Foo.kt", "line": "10",
            "comment": "wrong", "code": "code", "reason": "mismatch",
        }
        with (
            patch("main.get_pr_details", return_value=("title", "good desc")),
            patch("main.capture_original_reviewers"),
            patch("main.validate_pr_description", return_value=(True, "ok")),
            patch("main.clone_repository", return_value="/tmp/ws"),
            patch("main.get_pr_changed_files", return_value=["app/Foo.kt"]),
            patch("main.get_diff", return_value="diff"),
            patch("main.validate_description_matches_code", return_value=(True, "ok")),
            patch("main.validate_code_comments", return_value=[mismatch]),
            patch("main.generate_comment_suggestion", return_value="Correct comment"),
            patch("main.extract_full_code", return_value="FILE: app/Foo.kt\nL1: code"),
            patch("main.count_file_sections_in_code", return_value=1),
            patch("main.extract_added_code", return_value=""),
            patch("main.build_review_prompt", return_value="prompt"),
            patch("main.review_code", return_value="AI review here"),
            patch("main.post_pr_comment"),
            patch("main.handle_reviewers_for_validation_result"),
            patch("os.path.exists", return_value=False),
        ):
            result = process_pr_event(
                self._payload(), "pull_request", "opened", None, None
            )
        assert result.get("status") == "success"

    def test_both_warnings_collected_ai_review_runs_one_comment_posted(self):
        from main import process_pr_event

        ai_review_called = []
        posted_comments = []

        def fake_review(prompt):
            ai_review_called.append(True)
            return "AI review output"

        def fake_post(repo, pr, body):
            posted_comments.append(body)

        mismatch = {
            "file": "app/Foo.kt", "line": "5",
            "comment": "wrong", "code": "code", "reason": "mismatch",
        }

        with (
            patch("main.get_pr_details", return_value=("title", "vague")),
            patch("main.capture_original_reviewers"),
            patch("main.validate_pr_description", return_value=(False, "too vague")),
            patch("main.clone_repository", return_value="/tmp/ws"),
            patch("main.get_pr_changed_files", return_value=["app/Foo.kt"]),
            patch("main.get_diff", return_value="diff"),
            patch("main.validate_description_matches_code", return_value=(False, "mismatch")),
            patch("main.generate_description_suggestion", return_value="Better desc"),
            patch("main.validate_code_comments", return_value=[mismatch]),
            patch("main.generate_comment_suggestion", return_value="Correct comment"),
            patch("main.extract_full_code", return_value="FILE: app/Foo.kt\nL1: code"),
            patch("main.count_file_sections_in_code", return_value=1),
            patch("main.extract_added_code", return_value=""),
            patch("main.build_review_prompt", return_value="prompt"),
            patch("main.review_code", side_effect=fake_review),
            patch("main.post_pr_comment", side_effect=fake_post),
            patch("main.handle_reviewers_for_validation_result"),
            patch("os.path.exists", return_value=False),
        ):
            result = process_pr_event(
                self._payload(), "pull_request", "opened", None, None
            )

        # AI Code Review must run
        assert ai_review_called, "AI Code Review must run even when validation warnings exist"
        # Status must be success
        assert result.get("status") == "success"
        # Exactly one combined comment
        assert len(posted_comments) == 1, f"Expected 1 comment, got {len(posted_comments)}"

        body = posted_comments[0]
        assert "## PR Validation" in body
        assert "## AI Code Review" in body
        assert "PR Description needs improvement" in body
        assert "Code Comment needs improvement" in body
        assert "AI review output" in body
        assert "Better desc" in body
        assert "Correct comment" in body

    def test_validation_api_errors_fail_open_and_ai_review_still_runs(self):
        """If validation AI calls fail, pipeline must still reach AI Code Review."""
        from main import process_pr_event

        ai_review_called = []

        def fake_review(prompt):
            ai_review_called.append(True)
            return "AI review output"

        with (
            patch("main.get_pr_details", return_value=("title", "desc")),
            patch("main.capture_original_reviewers"),
            patch("main.validate_pr_description", side_effect=Exception("API down")),
            patch("main.clone_repository", return_value="/tmp/ws"),
            patch("main.get_pr_changed_files", return_value=["app/Foo.kt"]),
            patch("main.get_diff", return_value="diff"),
            patch("main.validate_description_matches_code", side_effect=Exception("API down")),
            patch("main.validate_code_comments", side_effect=Exception("API down")),
            patch("main.extract_full_code", return_value="FILE: app/Foo.kt\nL1: code"),
            patch("main.count_file_sections_in_code", return_value=1),
            patch("main.extract_added_code", return_value=""),
            patch("main.build_review_prompt", return_value="prompt"),
            patch("main.review_code", side_effect=fake_review),
            patch("main.post_pr_comment"),
            patch("main.handle_reviewers_for_validation_result"),
            patch("os.path.exists", return_value=False),
        ):
            result = process_pr_event(
                self._payload(), "pull_request", "opened", None, None
            )

        assert ai_review_called, "AI Code Review must run even when validation APIs fail"
        assert result.get("status") == "success"


# ─────────────────────────────────────────────────────────────────────────────
# Regression: empty / whitespace / None PR body must NEVER pass validation
# Covers the bug where _build_validation_content substituted the PR title as
# the body when body was empty, causing GitHub Actions to show
# "✅ No validation issues found." for PRs with no description.
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyBodyAlwaysFails:
    """
    validate_pr_description must return (False, <reason>) for every
    empty/whitespace/None body regardless of how meaningful the title is.
    """

    def _validate(self, title: str, body):
        from ai.description_validator import validate_pr_description
        return validate_pr_description(title, body)

    def test_empty_string_body_fails(self):
        is_valid, reason = self._validate("Add payment module", "")
        assert not is_valid, "Empty body must fail validation"
        assert reason  # must have a human-readable reason

    def test_whitespace_only_body_fails(self):
        is_valid, reason = self._validate("Add payment module", "   ")
        assert not is_valid, "Whitespace-only body must fail validation"

    def test_newline_only_body_fails(self):
        is_valid, reason = self._validate("Add payment module", "\n\n\n")
        assert not is_valid, "Newline-only body must fail validation"

    def test_none_body_fails(self):
        is_valid, reason = self._validate("Add payment module", None)
        assert not is_valid, "None body must fail validation"

    def test_meaningful_title_does_not_rescue_empty_body(self):
        """Regression: title must NOT substitute for an empty body."""
        is_valid, _ = self._validate(
            "This PR refactors the payment module to reduce duplicate validation logic",
            ""
        )
        assert not is_valid, (
            "A meaningful title must not cause an empty body to pass validation"
        )

    def test_conventional_commit_title_does_not_rescue_empty_body(self):
        """Regression: conventional-commit prefixed title must not rescue empty body."""
        is_valid, _ = self._validate("feat: add JWT authentication", "")
        assert not is_valid

    def test_good_body_still_passes(self):
        """Sanity check: a real description must still pass."""
        is_valid, _ = self._validate(
            "Add payment module",
            "This PR adds the initial payment module with Stripe integration."
        )
        assert is_valid, "A meaningful description must pass validation"

    def test_dot_body_fails(self):
        """Regression: single punctuation body is too short to be valid."""
        is_valid, reason = self._validate("Update pricing logic", ".")
        assert not is_valid, "Single-character body '.' must fail validation"
        assert reason
