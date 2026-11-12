"""
tests/test_chunk_review_coverage_and_retry.py

Focused tests for AI Code Review chunk processing, retries, and coverage:
Test 1: Successful review with findings -> COMPLETED_WITH_FINDINGS
Test 2: Successful review without findings -> COMPLETED_NO_ISSUES
Test 3: Empty response -> triggers retry
Test 4: Malformed response -> triggers retry
Test 5: Temporary API failure -> triggers retry
Test 6: Successful retry (fail then succeed -> marked successfully reviewed)
Test 7: All retries exhausted (all 3 fail -> marked INCOMPLETE or FAILED)
Test 8: Multiple chunks in one file (one succeeds, one fails -> PARTIALLY_REVIEWED, never FULLY_REVIEWED)
Test 9: Multiple files (mix of succeed and fail -> correct PR-level coverage stats)
Test 10: Duplicate prevention (retried chunk findings included exactly once)
"""

import unittest
from unittest.mock import MagicMock, patch

from ai.chunk_manager import (
    ReviewChunk,
    _infer_review_status,
    _review_single_chunk,
    calculate_review_coverage,
    format_coverage_section,
    execute_chunked_review,
    CHUNK_STATUS_COMPLETED_WITH_FINDINGS,
    CHUNK_STATUS_COMPLETED_NO_ISSUES,
    CHUNK_STATUS_INCOMPLETE,
    CHUNK_STATUS_FAILED,
)
from ai.reviewer import ReviewResult


def _make_dummy_chunk(chunk_id=1, file_path="test.py"):
    return ReviewChunk(
        chunk_id=chunk_id,
        total_chunks=1,
        file_path=file_path,
        changed_files=[file_path],
        code_snippet="def foo():\n    return 42\n",
        diff_snippet="+ def foo():\n+     return 42\n",
        start_line=1,
        end_line=2,
    )


class TestChunkReviewCoverageAndRetry(unittest.TestCase):

    def test_1_successful_review_with_findings(self):
        """Test 1: Successful review with findings yields COMPLETED_WITH_FINDINGS."""
        review_text = (
            "**Issue:** Potential NPE\n"
            "**File:** test.py\n"
            "**Line:** 2\n\n"
            "**Code:**\n```py\nreturn 42\n```\n\n"
            "**Reason:** Code could fail.\n"
            "**Suggestion:** Add null check."
        )
        status, text = _infer_review_status(review_text)
        self.assertEqual(status, CHUNK_STATUS_COMPLETED_WITH_FINDINGS)
        self.assertIn("Potential NPE", text)

    def test_2_successful_review_without_findings(self):
        """Test 2: Successful review without findings yields COMPLETED_NO_ISSUES."""
        clean_text = "No issues found in this chunk. The implementation looks good."
        status, text = _infer_review_status(clean_text)
        self.assertEqual(status, CHUNK_STATUS_COMPLETED_NO_ISSUES)

    @patch("ai.chunk_manager.review_code")
    def test_3_empty_response_triggers_retry(self, mock_review):
        """Test 3: Empty response triggers retry up to max_attempts."""
        # Returns empty 2 times, then succeeds on 3rd attempt
        mock_review.side_effect = [
            "",
            "   \n\t",
            "No issues found in this chunk.",
        ]
        chunk = _make_dummy_chunk()
        result = _review_single_chunk(chunk, max_attempts=3, retry_delay=0.0)

        self.assertEqual(mock_review.call_count, 3)
        self.assertEqual(result.status, CHUNK_STATUS_COMPLETED_NO_ISSUES)

    @patch("ai.chunk_manager.review_code")
    def test_4_malformed_or_truncated_response_triggers_retry(self, mock_review):
        """Test 4: Malformed/truncated response triggers retry."""
        mock_review.side_effect = [
            ReviewResult("cut off mid-sentence...", status="incomplete"),
            "No issues found in this chunk.",
        ]
        chunk = _make_dummy_chunk()
        result = _review_single_chunk(chunk, max_attempts=3, retry_delay=0.0)

        self.assertEqual(mock_review.call_count, 2)
        self.assertEqual(result.status, CHUNK_STATUS_COMPLETED_NO_ISSUES)

    @patch("ai.chunk_manager.review_code")
    def test_5_temporary_api_failure_triggers_retry(self, mock_review):
        """Test 5: Temporary API failure triggers retry."""
        mock_review.side_effect = [
            RuntimeError("503 Service Unavailable"),
            "No issues found in this chunk.",
        ]
        chunk = _make_dummy_chunk()
        result = _review_single_chunk(chunk, max_attempts=3, retry_delay=0.0)

        self.assertEqual(mock_review.call_count, 2)
        self.assertEqual(result.status, CHUNK_STATUS_COMPLETED_NO_ISSUES)

    @patch("ai.chunk_manager.review_code")
    def test_6_successful_retry_after_failure(self, mock_review):
        """Test 6: First attempt fails, second attempt succeeds -> chunk marked successfully reviewed."""
        finding_text = (
            "**Issue:** Logic Bug\n"
            "**File:** test.py\n"
            "**Line:** 1\n\n"
            "**Code:**\n```py\nx=1\n```\n\n"
            "**Reason:** Error.\n"
            "**Suggestion:** Fix."
        )
        mock_review.side_effect = [
            RuntimeError("429 Rate Limit Exceeded"),
            finding_text,
        ]
        chunk = _make_dummy_chunk()
        result = _review_single_chunk(chunk, max_attempts=3, retry_delay=0.0)

        self.assertEqual(mock_review.call_count, 2)
        self.assertEqual(result.status, CHUNK_STATUS_COMPLETED_WITH_FINDINGS)
        self.assertIn("Logic Bug", str(result))

    @patch("ai.chunk_manager.review_code")
    def test_7_all_retries_exhausted_results_in_incomplete_or_failed(self, mock_review):
        """Test 7: All 3 attempts fail -> marked INCOMPLETE or FAILED."""
        # Scenario A: empty response on all 3 attempts
        mock_review.side_effect = ["", "", ""]
        chunk = _make_dummy_chunk()
        result_empty = _review_single_chunk(chunk, max_attempts=3, retry_delay=0.0)
        self.assertEqual(mock_review.call_count, 3)
        self.assertEqual(result_empty.status, CHUNK_STATUS_INCOMPLETE)
        self.assertIn("after 3 attempts", str(result_empty))

        # Scenario B: API error on all 3 attempts
        mock_review.reset_mock()
        mock_review.side_effect = [
            RuntimeError("Connection timeout"),
            RuntimeError("Connection timeout"),
            RuntimeError("Connection timeout"),
        ]
        result_error = _review_single_chunk(chunk, max_attempts=3, retry_delay=0.0)
        self.assertEqual(mock_review.call_count, 3)
        self.assertEqual(result_error.status, CHUNK_STATUS_FAILED)

    def test_8_multiple_chunks_in_one_file_partial_review(self):
        """Test 8: Multiple chunks in one file (one succeeds, one fails) -> PARTIALLY_REVIEWED, never FULLY_REVIEWED."""
        chunk_results = [
            {
                "chunk_id": 1,
                "file": "service.py",
                "status": CHUNK_STATUS_COMPLETED_NO_ISSUES,
                "review": "No issues found in this chunk.",
            },
            {
                "chunk_id": 2,
                "file": "service.py",
                "status": CHUNK_STATUS_FAILED,
                "review": "Error occurred during AI code review: timeout",
            },
        ]
        pr_coverage, file_coverages = calculate_review_coverage(
            changed_files=["service.py"],
            chunk_results=chunk_results,
        )

        self.assertEqual(len(file_coverages), 1)
        file_rec = file_coverages[0]
        self.assertEqual(file_rec["file"], "service.py")
        self.assertEqual(file_rec["chunks_generated"], 2)
        self.assertEqual(file_rec["chunks_succeeded"], 1)
        self.assertEqual(file_rec["chunks_failed"], 1)
        self.assertEqual(file_rec["status"], "PARTIALLY_REVIEWED")
        self.assertNotEqual(file_rec["status"], "FULLY_REVIEWED")

        # PR level must also be PARTIALLY_REVIEWED
        self.assertEqual(pr_coverage["review_status"], "PARTIALLY_REVIEWED")
        self.assertEqual(pr_coverage["files_partially_reviewed"], 1)
        self.assertEqual(pr_coverage["files_fully_reviewed"], 0)

    def test_9_multiple_files_correct_pr_level_coverage(self):
        """Test 9: Some files succeed while others fail -> correct PR-level coverage statistics."""
        chunk_results = [
            # File 1: fully reviewed (2 chunks succeeded)
            {"chunk_id": 1, "file": "f1.py", "status": CHUNK_STATUS_COMPLETED_WITH_FINDINGS, "review": "**Issue:** Bug\n**File:** f1.py\n**Line:** 1\n\n**Code:**\nx\n\n**Reason:** r\n**Suggestion:** s"},
            {"chunk_id": 2, "file": "f1.py", "status": CHUNK_STATUS_COMPLETED_NO_ISSUES, "review": "No issues found."},
            # File 2: partially reviewed (1 succeeded, 1 incomplete)
            {"chunk_id": 3, "file": "f2.py", "status": CHUNK_STATUS_COMPLETED_NO_ISSUES, "review": "No issues found."},
            {"chunk_id": 4, "file": "f2.py", "status": CHUNK_STATUS_INCOMPLETE, "review": "No review generated after 3 attempts."},
            # File 3: not reviewed (1 failed)
            {"chunk_id": 5, "file": "f3.py", "status": CHUNK_STATUS_FAILED, "review": "Error occurred during AI code review: failure"},
        ]

        pr_coverage, file_coverages = calculate_review_coverage(
            changed_files=["f1.py", "f2.py", "f3.py", "f4_skipped.py"],
            chunk_results=chunk_results,
        )

        self.assertEqual(pr_coverage["total_files"], 4)
        self.assertEqual(pr_coverage["files_fully_reviewed"], 1)
        self.assertEqual(pr_coverage["files_partially_reviewed"], 1)
        self.assertEqual(pr_coverage["files_not_reviewed"], 2)  # f3.py failed, f4_skipped.py has 0 chunks
        self.assertEqual(pr_coverage["total_chunks"], 5)
        self.assertEqual(pr_coverage["chunks_succeeded"], 3)
        self.assertEqual(pr_coverage["chunks_incomplete"], 1)
        self.assertEqual(pr_coverage["chunks_failed"], 1)
        self.assertEqual(pr_coverage["review_status"], "PARTIALLY_REVIEWED")

        # Verify compact markdown coverage section
        coverage_md = format_coverage_section(pr_coverage, file_coverages)
        self.assertIn("### 📊 Review Coverage", coverage_md)
        self.assertIn("4 total (1 fully reviewed, 1 partially reviewed, 2 not reviewed)", coverage_md)
        self.assertIn("5 total (3 completed, 1 incomplete, 1 failed)", coverage_md)
        self.assertIn("Partially Reviewed", coverage_md)

    @patch("ai.chunk_manager.review_code")
    def test_10_duplicate_prevention_on_retry(self, mock_review):
        """Test 10: Chunk retried successfully after earlier failed attempt -> findings included exactly once."""
        finding = (
            "**Issue:** SQL Injection\n"
            "**File:** db.py\n"
            "**Line:** 42\n\n"
            "**Code:**\n```py\nquery = f'SELECT * FROM users WHERE id = {user_id}'\n```\n\n"
            "**Reason:** Direct interpolation.\n"
            "**Suggestion:** Use parameterized query."
        )

        # Attempt 1: temporary failure
        # Attempt 2: returns finding
        mock_review.side_effect = [
            RuntimeError("500 Internal Server Error"),
            finding,
        ]

        full_code = "FILE: db.py\n```python\nL42: query = f'SELECT * FROM users WHERE id = {user_id}'\n```"
        diff = "diff --git a/db.py b/db.py\n@@ -42,1 +42,1 @@\n+query = f'SELECT * FROM users WHERE id = {user_id}'\n"

        result = execute_chunked_review(
            full_code=full_code,
            changed_files=["db.py"],
            diff=diff,
            safe_token_limit=120_000,
        )

        # Finding must be present exactly once
        self.assertEqual(result.count("SQL Injection"), 1)
        self.assertEqual(mock_review.call_count, 2)


if __name__ == "__main__":
    unittest.main()
