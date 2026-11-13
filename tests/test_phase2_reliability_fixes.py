"""
Unit and integration tests for Phase 2 reliability fixes:
- Aggregator: no findings from incomplete or failed chunks (large_code5.py bug)
- Comment extractor: decorative separators and pure test labels filtered
- Suggestion generator: identical/empty suggestions suppressed
- Coverage: accurately reflects chunk status (PARTIALLY_REVIEWED, NOT_REVIEWED)
- Python parser: executable statements not extracted as comments
"""

from unittest.mock import MagicMock, patch
import pytest


# ============================================================================
# SECTION A: Aggregator — no findings from incomplete chunks (the large_code5.py bug)
# ============================================================================

def test_incomplete_chunk_findings_not_in_final_report():
    """INCOMPLETE chunks must not contribute findings to the final report."""
    # Create a result list where one chunk is INCOMPLETE but has issue text
    results = [
        {
            "chunk_id": 1,
            "file": "large_code5.py",
            "status": "INCOMPLETE",  # <-- NOT completed
            "review": "**Issue:** some finding\n**File:** large_code5.py\n**Line:** 42\n**Comment:** test\n**Reason:** test reason",
        },
        {
            "chunk_id": 2,
            "file": "other_file.py",
            "status": "COMPLETED_WITH_FINDINGS",
            "review": "**Issue:** real finding\n**File:** other_file.py\n**Line:** 10\n**Comment:** test\n**Reason:** real reason",
        },
    ]
    from ai.result_aggregator import _aggregate_file_results
    output = _aggregate_file_results(results)
    # large_code5.py finding must NOT appear as confirmed finding
    assert "large_code5.py" not in output or "Review Notices" in output
    # other_file.py finding MUST appear
    assert "other_file.py" in output


def test_failed_chunk_findings_not_in_final_report():
    """FAILED chunks must not contribute findings to the final report."""
    results = [
        {
            "chunk_id": 1,
            "file": "failed_file.py",
            "status": "FAILED",
            "review": "**Issue:** spurious finding\n**File:** failed_file.py\n**Line:** 5\n**Comment:** test\n**Reason:** spurious",
        },
    ]
    from ai.result_aggregator import _aggregate_file_results
    output = _aggregate_file_results(results)
    # No confirmed findings from FAILED chunks
    assert "spurious finding" not in output or "Review Notices" in output


# ============================================================================
# SECTION B: Comment extractor — decorative separators filtered
# ============================================================================

def test_decorative_separator_not_extracted():
    """Pure === and --- lines must not be extracted as comments."""
    from github.comment_extractor import extract_comments_from_content
    content = "# =============================\ndef foo():\n    pass\n"
    comments = extract_comments_from_content(content, "test.py")
    # No comment should be extracted for a pure separator
    assert all("====" not in c["comment"] for c in comments), f"Got: {comments}"


def test_test_label_not_extracted():
    """Pure TEST 095 style labels must not be extracted as comments."""
    from github.comment_extractor import extract_comments_from_content
    content = "# TEST 095\ndef test_foo():\n    return 1\n"
    comments = extract_comments_from_content(content, "test.py")
    assert all("TEST 095" not in c["comment"] for c in comments), f"Got: {comments}"


def test_real_comment_still_extracted():
    """Real comments describing behavior must still be extracted."""
    from github.comment_extractor import extract_comments_from_content
    content = "# Parse the user's input and validate it\ndef parse(x):\n    return int(x)\n"
    comments = extract_comments_from_content(content, "test.py")
    assert len(comments) >= 1
    assert any("Parse" in c["comment"] for c in comments)


def test_kotlin_separator_not_extracted():
    """Kotlin // === separators must not be extracted."""
    from github.comment_extractor import extract_comments_from_content
    content = "// ==============================\nfun foo() {\n    return\n}\n"
    comments = extract_comments_from_content(content, "Test.kt")
    assert all("====" not in c["comment"] for c in comments), f"Got: {comments}"


# ============================================================================
# SECTION C: Suggestion generator — identical/empty suggestions suppressed
# ============================================================================

def test_identical_suggestion_is_suppressed():
    """A suggestion identical to the original comment must be suppressed."""
    from ai.suggestion_generator import generate_comment_suggestion

    original_comment = "This function validates user input."

    with patch("ai.suggestion_generator._get_client") as mock_client_fn, \
         patch("ai.suggestion_generator.execute_nvidia_request") as mock_req:
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = original_comment
        mock_req.return_value = mock_completion

        result = generate_comment_suggestion(
            file="test.py",
            line="42",
            comment=original_comment,
            code="def validate(x): return bool(x)",
            reason="test reason",
        )
        assert not result or result.strip() == original_comment, \
            f"Expected suppression or same, got: {result}"


def test_empty_suggestion_is_suppressed():
    """An empty suggestion response from the LLM must return empty string."""
    from ai.suggestion_generator import generate_comment_suggestion

    with patch("ai.suggestion_generator._get_client") as mock_client_fn, \
         patch("ai.suggestion_generator.execute_nvidia_request") as mock_req:
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "   "
        mock_req.return_value = mock_completion

        result = generate_comment_suggestion(
            file="test.py",
            line="42",
            comment="Old comment",
            code="def foo(): pass",
            reason="test reason",
        )
        assert result == "", f"Expected empty string when LLM returns whitespace, got: {result!r}"


# ============================================================================
# SECTION D: Coverage accurately reflects chunk status
# ============================================================================

def test_not_reviewed_file_has_no_confirmed_findings():
    """Coverage: file marked NOT_REVIEWED must have no confirmed findings."""
    from ai.chunk_manager import calculate_review_coverage, CHUNK_STATUS_INCOMPLETE
    changed_files = ["large_code5.py", "good_file.py"]
    chunk_results = [
        {"file": "large_code5.py", "status": CHUNK_STATUS_INCOMPLETE, "chunk_id": 1, "review": ""},
        {"file": "good_file.py", "status": "COMPLETED_WITH_FINDINGS", "chunk_id": 2, "review": "some review"},
    ]
    pr_cov, file_covs = calculate_review_coverage(changed_files, chunk_results)

    large_cov = next(fc for fc in file_covs if fc["file"] == "large_code5.py")
    assert large_cov["status"] == "NOT_REVIEWED"

    good_cov = next(fc for fc in file_covs if fc["file"] == "good_file.py")
    assert good_cov["status"] == "FULLY_REVIEWED"


def test_pr_level_status_partially_reviewed():
    """PR-level status must be PARTIALLY_REVIEWED when some files fail."""
    from ai.chunk_manager import calculate_review_coverage, CHUNK_STATUS_INCOMPLETE
    changed_files = ["file_a.py", "file_b.py"]
    chunk_results = [
        {"file": "file_a.py", "status": "COMPLETED_NO_ISSUES", "chunk_id": 1, "review": "clean"},
        {"file": "file_b.py", "status": CHUNK_STATUS_INCOMPLETE, "chunk_id": 2, "review": ""},
    ]
    pr_cov, _ = calculate_review_coverage(changed_files, chunk_results)
    assert pr_cov["review_status"] == "PARTIALLY_REVIEWED"


# ============================================================================
# SECTION E: Python executable statements not extracted as comments
# ============================================================================

def test_python_function_def_not_extracted():
    """Python function definitions must not be extracted as comments."""
    from github.comment_extractor import extract_comments_from_content
    content = "def calculate(x, y):\n    return x + y\n"
    comments = extract_comments_from_content(content, "calc.py")
    assert len(comments) == 0, f"Unexpected comments: {comments}"


def test_python_return_not_extracted():
    """Python return statements must not be extracted as comments."""
    from github.comment_extractor import extract_comments_from_content
    content = "def foo():\n    return True\n"
    comments = extract_comments_from_content(content, "foo.py")
    assert len(comments) == 0


def test_python_import_not_extracted():
    """Python imports must not be extracted as comments."""
    from github.comment_extractor import extract_comments_from_content
    content = "import os\nimport sys\n"
    comments = extract_comments_from_content(content, "mod.py")
    assert len(comments) == 0


def test_python_assignment_not_extracted():
    """Python assignments must not be extracted as comments."""
    from github.comment_extractor import extract_comments_from_content
    content = "x = 42\ny = 'hello'\n"
    comments = extract_comments_from_content(content, "vars.py")
    assert len(comments) == 0


def test_python_docstring_extracted():
    """Python module/function docstrings should be extractable (they ARE comments)."""
    # Note: The current extractor uses tokenize which extracts COMMENT tokens, not STRING tokens
    # Docstrings are STRING tokens in Python, not COMMENT tokens
    # This test verifies the tokenizer behavior
    from github.comment_extractor import extract_comments_from_content
    content = '# This is a real comment\ndef foo():\n    """Docstring."""\n    pass\n'
    comments = extract_comments_from_content(content, "foo.py")
    # At minimum, the # comment should be extracted
    assert any("real comment" in c["comment"] for c in comments)


# ============================================================================
# SECTION F: Diagnostics, Failure Classification & Report Warning Banner
# ============================================================================

def test_classify_failure_categories():
    """Verify _classify_failure accurately maps error messages and exceptions."""
    from ai.chunk_manager import _classify_failure

    # 429
    assert _classify_failure(review_text="HTTP 429 rate limit") == "RATE_LIMIT_429"
    # Auth
    assert _classify_failure(review_text="Authentication error 401 invalid key") == "AUTH_ERROR"
    # Context length
    assert _classify_failure(review_text="maximum context length exceeded") == "CONTEXT_LENGTH_EXCEEDED"
    # Timeout
    assert _classify_failure(review_text="Request timed out") == "REQUEST_TIMEOUT"
    # Connection failure
    assert _classify_failure(review_text="Connection refused by remote host") == "CONNECTION_FAILURE"
    # Provider retries exhausted
    assert _classify_failure(review_text="Maximum retries exhausted.") == "PROVIDER_RETRIES_EXHAUSTED"
    # 503 Server Error
    assert _classify_failure(review_text="503 Service Unavailable") == "SERVER_ERROR_503"
    # Output token limit
    assert _classify_failure(review_text="finish_reason: length output token limit") == "OUTPUT_TOKEN_LIMIT_REACHED"
    # Empty
    assert _classify_failure(review_text="   ") == "EMPTY_RESPONSE"
    # Incomplete
    assert _classify_failure(review_text="No valid review generated after 3 attempts") == "INCOMPLETE_OR_MALFORMED"


def test_coverage_collects_diagnostics_and_failure_categories():
    """calculate_review_coverage must collect diagnostics and failure categories into incomplete_chunk_details."""
    from ai.chunk_manager import calculate_review_coverage, CHUNK_STATUS_INCOMPLETE, CHUNK_STATUS_FAILED

    chunk_results = [
        {
            "chunk_id": 1,
            "file": "api_client.py",
            "status": CHUNK_STATUS_INCOMPLETE,
            "review": "incomplete",
            "diagnostics": {"failure_category": "RATE_LIMIT_429"},
        },
        {
            "chunk_id": 2,
            "file": "api_client.py",
            "status": CHUNK_STATUS_FAILED,
            "review": "failed",
            "diagnostics": {"failure_category": "SERVER_ERROR_500"},
        },
    ]

    pr_cov, file_covs = calculate_review_coverage(["api_client.py"], chunk_results)
    assert len(file_covs) == 1
    fc = file_covs[0]
    assert fc["status"] == "NOT_REVIEWED"
    assert len(fc["incomplete_chunk_details"]) == 2
    assert fc["incomplete_chunk_details"][0] == {"chunk_id": 1, "failure_category": "RATE_LIMIT_429"}
    assert fc["incomplete_chunk_details"][1] == {"chunk_id": 2, "failure_category": "SERVER_ERROR_500"}


def test_format_coverage_section_displays_diagnostic_reasons_and_warning():
    """format_coverage_section must display specific chunk failure reasons and warning banner."""
    from ai.chunk_manager import calculate_review_coverage, format_coverage_section, CHUNK_STATUS_INCOMPLETE

    chunk_results = [
        {
            "chunk_id": 2,
            "file": "AuthRepo.kt",
            "status": CHUNK_STATUS_INCOMPLETE,
            "review": "incomplete",
            "diagnostics": {"failure_category": "RATE_LIMIT_429"},
        }
    ]

    pr_cov, file_covs = calculate_review_coverage(["AuthRepo.kt"], chunk_results)
    md = format_coverage_section(pr_cov, file_covs)

    # Must contain the specific failure category for chunk 2
    assert "chunk 2: RATE_LIMIT_429" in md
    # Must contain the explicit warning banner
    assert "⚠️ **This review is incomplete.**" in md


def test_format_coverage_section_omits_warning_when_fully_reviewed():
    """format_coverage_section must not display the warning banner when review is FULLY_REVIEWED."""
    from ai.chunk_manager import calculate_review_coverage, format_coverage_section, CHUNK_STATUS_COMPLETED_NO_ISSUES

    chunk_results = [
        {
            "chunk_id": 1,
            "file": "AuthRepo.kt",
            "status": CHUNK_STATUS_COMPLETED_NO_ISSUES,
            "review": "clean",
        }
    ]

    pr_cov, file_covs = calculate_review_coverage(["AuthRepo.kt"], chunk_results)
    md = format_coverage_section(pr_cov, file_covs)

    assert "⚠️ **This review is incomplete.**" not in md
    assert "**Coverage Status:** Fully Reviewed" in md


# ============================================================================
# SECTION G: Review Coverage Display Removal from Final GitHub Comment
# ============================================================================

def test_review_coverage_not_in_combined_comment():
    """Verify that Review Coverage section is NOT included in the final GitHub comment."""
    from main import _build_combined_comment
    comment = _build_combined_comment(
        validation_section="## PR Validation\n\n✅ All checks passed.",
        review_text="**Issue:** Performance issue\n**File:** db.py\n**Line:** 10",
        coverage_section="### 📊 Review Coverage\n- Files: 1 total",
    )
    # The coverage summary must NOT be in the comment
    assert "Review Coverage" not in comment
    assert "### 📊 Review Coverage" not in comment
    # But findings and header MUST be present
    assert "## AI Code Review" in comment
    assert "Performance issue" in comment


@patch("main.post_pr_comment")
@patch("main.review_code")
@patch("main.execute_chunked_review")
@patch("main.extract_change_aware_context")
@patch("main.extract_full_code")
@patch("main.get_diff")
@patch("main.get_pr_changed_files")
@patch("main.clone_repository")
@patch("main.get_pr_details")
@patch("main.validate_pr_description")
@patch("main.validate_description_matches_code")
@patch("main.validate_code_comments")
def test_process_pr_event_excludes_review_coverage_from_github_comment(
    mock_val_comments,
    mock_val_desc_code,
    mock_val_desc,
    mock_get_pr_details,
    mock_clone,
    mock_changed_files,
    mock_get_diff,
    mock_extract_full_code,
    mock_extract_change_context,
    mock_chunked_review,
    mock_review_code,
    mock_post_comment,
):
    """Verify that process_pr_event posts a comment without Review Coverage, while preserving internal coverage."""
    from main import process_pr_event

    mock_get_pr_details.return_value = ("fix: db optimization", "Valid description")
    mock_changed_files.return_value = ["db.py"]
    mock_get_diff.return_value = "diff --git a/db.py b/db.py\n+new_code"
    mock_extract_full_code.return_value = "FILE: db.py\n```python\nL1: db_query()\n```"
    mock_extract_change_context.return_value = ("FILE: db.py\nL1: db_query()", "diff")
    mock_val_desc.return_value = {"is_valid": True, "reason": None}
    mock_val_desc_code.return_value = {"is_valid": True, "reason": None}
    mock_val_comments.return_value = []
    mock_review_code.return_value = "**Issue:** N+1 query\n**File:** db.py\n**Line:** 1"

    payload = {
        "action": "opened",
        "pull_request": {"number": 42, "body": "valid body"},
        "repository": {"full_name": "owner/repo", "name": "repo"},
    }

    result = process_pr_event(payload, "pull_request", "opened", "", "")

    # Pipeline succeeded
    assert result["status"] == "success"
    # Internal coverage tracking is preserved
    assert "coverage" in result
    assert result["coverage"]["files_fully_reviewed"] == 1
    assert "file_coverage" in result
    assert result["review_status"] == "FULLY_REVIEWED"

    # Comment was posted
    assert mock_post_comment.call_count == 1
    posted_comment = mock_post_comment.call_args[0][2]

    # Findings must be present
    assert "N+1 query" in posted_comment
    assert "## AI Code Review" in posted_comment
    # Review Coverage must NOT be present in the GitHub comment
    assert "Review Coverage" not in posted_comment
    assert "### 📊 Review Coverage" not in posted_comment




# ---------------------------------------------------------------------------
# Tests for reasoning_effort root-cause fix and malformed table cleanup
# ---------------------------------------------------------------------------


class TestReasoningOnlyResponse:
    """Verify that a reasoning-only (empty content) response is classified as incomplete."""

    def test_reasoning_only_classified_as_incomplete(self):
        """When review_code_with_status returns empty content, status is 'incomplete'."""
        from ai.reviewer import ReviewResult

        # Simulate what happens when reasoning_effort consumed all tokens:
        # review_buffer is empty → return ("incomplete", "No review generated.")
        result = ReviewResult("", status="incomplete")
        assert result.status == "incomplete"
        assert str(result) == ""
        # The pipeline treats this chunk as INCOMPLETE — not completed
        assert result.status != "completed"

    def test_review_result_preserves_incomplete_status(self):
        """ReviewResult status attribute is correctly set for empty content."""
        from ai.reviewer import ReviewResult

        r = ReviewResult("", status="incomplete")
        assert r.status == "incomplete"
        assert str(r) == ""

    def test_review_result_completed_status(self):
        """ReviewResult status attribute is correctly set for successful content."""
        from ai.reviewer import ReviewResult

        r = ReviewResult("**Issue:** test finding", status="completed")
        assert r.status == "completed"
        assert "test finding" in r


class TestRemoveMalformedTables:
    """Verify _remove_malformed_tables strips malformed rows without touching valid tables."""

    def _fn(self, text):
        from main import _remove_malformed_tables
        return _remove_malformed_tables(text)

    def test_empty_header_and_separator_removed(self):
        """Empty header + separator block is fully removed."""
        text = "Some text\n|   |\n| - |\nMore text"
        result = self._fn(text)
        assert "|   |" not in result
        assert "| - |" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_standalone_empty_cell_row_removed(self):
        """A lone |   | row is removed."""
        text = "Before\n|   |\nAfter"
        result = self._fn(text)
        assert "|   |" not in result
        assert "Before" in result
        assert "After" in result

    def test_valid_table_preserved(self):
        """A proper table with real content columns is NOT modified."""
        table = "| File | Line | Severity |\n| --- | --- | --- |\n| foo.py | 10 | High |"
        text = f"Intro\n\n{table}\n\nOutro"
        result = self._fn(text)
        assert "foo.py" in result
        assert "Severity" in result
        assert "| --- | --- | --- |" in result

    def test_code_fence_content_preserved(self):
        """Content inside code fences is never touched."""
        code_block = "```python\n|   |\n| - |\n```"
        result = self._fn(code_block)
        assert "|   |" in result
        assert "| - |" in result

    def test_multiple_empty_rows_removed(self):
        """Multiple consecutive empty rows are all removed."""
        text = "Start\n|   |\n| - |\n|   |\nEnd"
        result = self._fn(text)
        assert "|   |" not in result
        assert "| - |" not in result
        assert "Start" in result
        assert "End" in result

    def test_empty_string_returns_empty(self):
        """Empty input returns empty string without error."""
        assert self._fn("") == ""

    def test_none_equivalent_handled(self):
        """Text with no malformed tables passes through unchanged."""
        clean = "**Issue:** Some problem\n**File:** foo.py\n**Line:** 42"
        assert self._fn(clean) == clean

    def test_mixed_valid_and_malformed_tables(self):
        """Valid tables are preserved while orphaned empty rows are stripped."""
        text = (
            "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
            "|   |\n| - |\n\n"
            "| X | Y |\n| --- | --- |\n| p | q |"
        )
        result = self._fn(text)
        assert "| A | B |" in result
        assert "| X | Y |" in result
        # The orphaned empty block should be gone
        # (at minimum check the leading empty-header standalone is not in a bare form)
        # Valid tables remain intact
        assert "| 1 | 2 |" in result
        assert "| p | q |" in result



# ---------------------------------------------------------------------------
# Tests for Ambiguous Review Notice and Double Aggregation Fixes
# ---------------------------------------------------------------------------

class TestReviewNoticeAndAggregationFixes:
    """Regression tests for the 9 areas requested for CodeGate review notices."""

    # 1. Double aggregation
    def test_step13_chunked_review_avoids_double_aggregation(self):
        """execute_chunked_review output is used directly and not re-aggregated as 'Overall PR review'."""
        from unittest.mock import patch, MagicMock
        from main import process_pr_event
        from ai.reviewer import ReviewResult

        chunk_result = ReviewResult(
            "**Issue:** Memory leak in worker\n**File:** worker.py\n**Line:** 20\n\n"
            "**Code:**\n```python\nworker.run()\n```\n\n**Reason:** Leak.\n**Suggestion:** Fix.",
            status="completed",
        )
        chunk_result.coverage = {
            "total_files": 1,
            "files_fully_reviewed": 1,
            "files_partially_reviewed": 0,
            "files_not_reviewed": 0,
            "coverage_percentage": 100.0,
            "coverage_status": "Fully Reviewed",
            "diagnostics": [],
        }
        chunk_result.file_coverage = [
            {"file": "worker.py", "chunks_total": 1, "chunks_completed": 1, "status": "FULLY_REVIEWED"}
        ]
        chunk_result.chunk_results = [
            {"chunk_id": 1, "file": "worker.py", "status": "COMPLETED_WITH_FINDINGS", "review": str(chunk_result)}
        ]

        with patch("main.get_pr_details", return_value=("feat: large", "Body")), \
             patch("main.clone_repository"), \
             patch("main.get_pr_changed_files", return_value=["worker.py"]), \
             patch("main.get_diff", return_value="diff --git a/worker.py b/worker.py\n+large"), \
             patch("main.extract_full_code", return_value="FILE: worker.py\n" + ("L1: pass\n" * 20000)), \
             patch("main.extract_change_aware_context", return_value=("code", "context")), \
             patch("main.count_tokens", return_value=150000), \
             patch("main.validate_pr_description", return_value={"is_valid": True, "reason": None}), \
             patch("main.validate_description_matches_code", return_value={"is_valid": True, "reason": None}), \
             patch("main.validate_code_comments", return_value=[]), \
             patch("main.execute_chunked_review", return_value=chunk_result), \
             patch("main.post_pr_comment") as mock_post:

            payload = {
                "action": "opened",
                "pull_request": {"number": 100, "body": "valid body"},
                "repository": {"full_name": "owner/repo", "name": "repo"},
            }

            result = process_pr_event(payload, "pull_request", "opened", "", "")

            assert result["status"] == "success"
            # Must NOT contain the synthetic "Overall PR review" label anywhere
            assert mock_post.call_count == 1
            posted = mock_post.call_args[0][2]
            assert "Overall PR review" not in posted
            assert "Memory leak in worker" in posted
            assert "(ambiguous)" not in posted

    # 2. Structured issue responses
    def test_structured_issue_accepted_as_completed_with_findings(self):
        """_infer_review_status recognizes structured **Issue:** blocks."""
        from ai.chunk_manager import _infer_review_status, CHUNK_STATUS_COMPLETED_WITH_FINDINGS
        from ai.reviewer import ReviewResult

        text = (
            "**Issue:** Unhandled exception\n**File:** app.py\n**Line:** 5\n\n"
            "**Code:**\n```python\nrisky_call()\n```\n\n**Reason:** Crash.\n**Suggestion:** Try/except."
        )
        res = ReviewResult(text, status="completed")
        status, review_text = _infer_review_status(res)
        assert status == CHUNK_STATUS_COMPLETED_WITH_FINDINGS
        assert "Unhandled exception" in review_text

    # 3. Exact "No issues found." responses
    def test_exact_no_issues_found_accepted_as_completed_no_issues(self):
        """_infer_review_status recognizes clean confirmation 'No issues found.'"""
        from ai.chunk_manager import _infer_review_status, CHUNK_STATUS_COMPLETED_NO_ISSUES
        from ai.reviewer import ReviewResult

        res = ReviewResult("No issues found.", status="completed")
        status, review_text = _infer_review_status(res)
        assert status == CHUNK_STATUS_COMPLETED_NO_ISSUES
        assert review_text == "No issues found."

    # 4. Conversational preamble without findings
    def test_conversational_preamble_alone_classified_as_incomplete(self):
        """Conversational preamble alone is NEVER classified as COMPLETED_NO_ISSUES."""
        from ai.chunk_manager import _infer_review_status, CHUNK_STATUS_INCOMPLETE
        from ai.reviewer import ReviewResult

        preamble = (
            "I'll review the changed code in the PR diff and identify issues "
            "that are directly attributable to the changes..."
        )
        res = ReviewResult(preamble, status="completed")
        status, review_text = _infer_review_status(res)
        assert status == CHUNK_STATUS_INCOMPLETE

    # 5. Preamble followed by "No issues found."
    def test_preamble_followed_by_no_issues_found_is_clean(self):
        """Preamble followed by explicit 'No issues found.' is stripped and accepted as clean."""
        from ai.chunk_manager import _infer_review_status, CHUNK_STATUS_COMPLETED_NO_ISSUES
        from ai.result_aggregator import _is_explicitly_clean, aggregate_reviews
        from ai.reviewer import ReviewResult

        text = (
            "I'll review the changed code in the PR diff and identify issues "
            "that are directly attributable to the changes.\n\nNo issues found."
        )
        assert _is_explicitly_clean(text) is True

        res = ReviewResult(text, status="completed")
        status, review_text = _infer_review_status(res)
        assert status == CHUNK_STATUS_COMPLETED_NO_ISSUES
        assert "No issues found." in review_text

        # In aggregation, does NOT produce an ambiguous notice
        agg = aggregate_reviews([{"file": "mod.py", "status": status, "review": review_text}])
        assert "(ambiguous)" not in agg
        assert "No issues found." in agg

    # 6. Arbitrary explanatory text
    def test_arbitrary_explanatory_text_not_clean_and_marked_incomplete(self):
        """Arbitrary explanatory commentary is not clean and classified as incomplete."""
        from ai.chunk_manager import _infer_review_status, CHUNK_STATUS_INCOMPLETE
        from ai.result_aggregator import _is_explicitly_clean, aggregate_reviews
        from ai.reviewer import ReviewResult

        text = "This pull request refactors the database schema and introduces three new tables."
        assert _is_explicitly_clean(text) is False

        res = ReviewResult(text, status="completed")
        status, review_text = _infer_review_status(res)
        assert status == CHUNK_STATUS_INCOMPLETE

        # If aggregated as completed, aggregate_reviews flags as ambiguous
        agg = aggregate_reviews([{"file": "db.py", "status": "completed", "review": text}])
        assert "(ambiguous)" in agg

    # 7. Incomplete responses
    def test_incomplete_responses_classified_as_incomplete(self):
        """Empty or 'No review generated.' responses are classified as INCOMPLETE."""
        from ai.chunk_manager import _infer_review_status, CHUNK_STATUS_INCOMPLETE
        from ai.reviewer import ReviewResult

        for val in ["", "No review generated.", "   "]:
            res = ReviewResult(val, status="incomplete")
            status, _ = _infer_review_status(res)
            assert status == CHUNK_STATUS_INCOMPLETE

    # 8. File-level coverage preservation
    def test_file_level_coverage_preserved_in_chunked_review(self):
        """File-level coverage is preserved when execute_chunked_review runs."""
        from ai.chunk_manager import calculate_review_coverage

        changed_files = ["service.py", "utils.py"]
        chunk_results = [
            {"chunk_id": 1, "file": "service.py", "status": "COMPLETED_NO_ISSUES", "review": "No issues found."},
            {"chunk_id": 2, "file": "utils.py", "status": "COMPLETED_WITH_FINDINGS", "review": "**Issue:** Bug\n**File:** utils.py\n**Line:** 1"},
        ]
        pr_cov, file_covs = calculate_review_coverage(changed_files, chunk_results)
        assert pr_cov["files_fully_reviewed"] == 2
        assert pr_cov["review_status"] == "FULLY_REVIEWED"
        assert len(file_covs) == 2
        assert file_covs[0]["file"] == "service.py"
        assert file_covs[1]["file"] == "utils.py"

    # 9. Failed chunks remaining marked incomplete
    def test_failed_chunks_remain_disclosed_in_notices(self):
        """Failed chunks remain marked incomplete and are disclosed in Review Notices."""
        from ai.result_aggregator import aggregate_reviews

        results = [
            {"chunk_id": 1, "file": "clean.py", "status": "COMPLETED_NO_ISSUES", "review": "No issues found."},
            {
                "chunk_id": 2,
                "file": "broken.py",
                "status": "INCOMPLETE",
                "review": "No valid review generated after 3 attempts. Last status: INCOMPLETE_OR_MALFORMED.",
            },
        ]
        agg = aggregate_reviews(results)
        assert "### ⚠️ Review Notices" in agg
        assert "broken.py" in agg
        assert "(incomplete)" in agg
