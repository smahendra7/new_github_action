"""
tests/test_chunked_review.py

Comprehensive tests for large-context chunked AI review:
1. Normal PR (prompt <= SAFE_INPUT_TOKEN_LIMIT) uses single AI request (existing behavior).
2. Large PR (prompt > SAFE_INPUT_TOKEN_LIMIT) triggers chunked review.
3. Multiple chunks execute concurrently with controlled concurrency (max 3).
4. Chunk results stored in in-memory list (review_results).
5. Aggregation preserves deterministic chunk ordering regardless of completion order.
6. Overlapping chunks deduplicate findings based on normalized (File, Line).
7. Different findings on different lines (e.g. L33 vs L133) are preserved.
8. Every chunk prompt tells AI to report only PR-attributable issues and has diff scope.
9. No chunk prompt exceeds SAFE_INPUT_TOKEN_LIMIT.
10. Streaming behavior is preserved per chunk.
11. If one chunk fails, other chunks continue without fabricating results.
12. Performance test: verifies concurrent execution using time/threads (mocked AI calls).
"""

import threading
import time
from unittest.mock import MagicMock, call, patch
import pytest

from ai.context_config import (
    SAFE_INPUT_TOKEN_LIMIT,
    MAX_CONCURRENCY,
    CONTEXT_WINDOW_LINES,
    CHUNK_OVERLAP_LINES,
    SAFE_PROMPT_CHAR_LIMIT,
)
from ai.token_counter import count_tokens
from ai.chunk_manager import (
    DiffHunk,
    ReviewChunk,
    create_review_chunks,
    execute_chunked_review,
    parse_diff_by_file,
    parse_full_code_files,
)
from ai.prompt_builder import build_chunk_prompt, build_review_prompt
from ai.result_aggregator import (
    aggregate_reviews,
    extract_dedup_key,
    normalize_file_path,
    normalize_line_spec,
)
from main import _format_review, _build_combined_comment, process_pr_event


# ─────────────────────────────────────────────────────────────────────────────
# 1. Token Counting Utility
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenCounter:

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0
        assert count_tokens(None) == 0

    def test_short_code_counts_accurately(self):
        code = "def add(a, b):\n    return a + b\n"
        tokens = count_tokens(code)
        assert 5 <= tokens <= 20

    def test_token_count_increases_with_size(self):
        small = "x = 1\n"
        large = "x = 1\n" * 500
        assert count_tokens(large) > count_tokens(small) * 100


# ─────────────────────────────────────────────────────────────────────────────
# 2. Diff and Full Code Parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffParsing:

    def test_parse_diff_single_file_single_hunk(self):
        raw_diff = (
            "diff --git a/app/main.py b/app/main.py\n"
            "index 123..456 100644\n"
            "--- a/app/main.py\n"
            "+++ b/app/main.py\n"
            "@@ -50,5 +50,6 @@\n"
            " def foo():\n"
            "+    bar = 1\n"
            "     return True\n"
        )
        parsed = parse_diff_by_file(raw_diff)
        assert "app/main.py" in parsed
        hunks = parsed["app/main.py"]["hunks"]
        assert len(hunks) == 1
        assert hunks[0].new_start == 50
        assert hunks[0].new_end == 55
        assert 51 in hunks[0].modified_lines

    def test_parse_diff_multiple_hunks_and_files(self):
        raw_diff = (
            "diff --git a/file1.py b/file1.py\n"
            "@@ -10,2 +10,3 @@\n"
            " a\n"
            "+b\n"
            "@@ -100,2 +101,3 @@\n"
            " x\n"
            "+y\n"
            "diff --git a/file2.py b/file2.py\n"
            "@@ -5,2 +5,2 @@\n"
            "-old\n"
            "+new\n"
        )
        parsed = parse_diff_by_file(raw_diff)
        assert "file1.py" in parsed
        assert "file2.py" in parsed
        assert len(parsed["file1.py"]["hunks"]) == 2
        assert len(parsed["file2.py"]["hunks"]) == 1


class TestFullCodeParsing:

    def test_parse_full_code_files(self):
        full_code = (
            "FILE: src/service.py\n"
            "```python\n"
            "L1: import os\n"
            "L2: def run():\n"
            "L3:     pass\n"
            "```\n\n"
            "FILE: src/util.py\n"
            "```python\n"
            "L1: x = 1\n"
            "```"
        )
        parsed = parse_full_code_files(full_code)
        assert "src/service.py" in parsed
        assert "src/util.py" in parsed
        assert parsed["src/service.py"]["lang"] == "python"
        assert parsed["src/service.py"]["total_lines"] == 3
        assert parsed["src/service.py"]["lines_dict"][2] == "L2: def run():"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Chunk Creation and Sizing
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkCreation:

    def test_small_file_single_chunk(self):
        full_code = (
            "FILE: hello.py\n"
            "```python\n"
            "L1: print('hello')\n"
            "```"
        )
        diff = (
            "diff --git a/hello.py b/hello.py\n"
            "@@ -1,1 +1,1 @@\n"
            "+print('hello')\n"
        )
        chunks = create_review_chunks(
            full_code=full_code,
            changed_files=["hello.py"],
            diff=diff,
            safe_token_limit=120_000,
        )
        assert len(chunks) == 1
        assert chunks[0].file_path == "hello.py"
        assert "L1: print('hello')" in chunks[0].code_snippet

    def test_nearby_hunks_are_grouped(self):
        lines = [f"L{i}: line_{i}" for i in range(1, 200)]
        full_code = f"FILE: app.py\n```python\n" + "\n".join(lines) + "\n```"
        diff = (
            "diff --git a/app.py b/app.py\n"
            "@@ -10,3 +10,4 @@\n"
            " line_10\n"
            "+new_11\n"
            " line_12\n"
            "@@ -40,3 +41,4 @@\n"
            " line_40\n"
            "+new_41\n"
            " line_42\n"
        )
        chunks = create_review_chunks(
            full_code=full_code,
            changed_files=["app.py"],
            diff=diff,
            safe_token_limit=120_000,
            context_lines=150,
        )
        assert len(chunks) == 1

    def test_distant_hunks_create_separate_chunks(self):
        lines = [f"L{i}: line_{i}" for i in range(1, 5000)]
        full_code = f"FILE: app.py\n```python\n" + "\n".join(lines) + "\n```"
        diff = (
            "diff --git a/app.py b/app.py\n"
            "@@ -10,2 +10,3 @@\n"
            " line_10\n"
            "+edit_1\n"
            "@@ -4800,2 +4801,3 @@\n"
            " line_4800\n"
            "+edit_2\n"
        )
        chunks = create_review_chunks(
            full_code=full_code,
            changed_files=["app.py"],
            diff=diff,
            safe_token_limit=120_000,
            context_lines=50,
        )
        assert len(chunks) == 2
        assert chunks[0].start_line == 1
        assert chunks[1].start_line == 4751

    def test_no_chunk_exceeds_safe_token_limit(self):
        lines = [f"L{i}: " + ("x" * 80) for i in range(1, 2000)]
        full_code = f"FILE: big.py\n```python\n" + "\n".join(lines) + "\n```"
        diff = (
            "diff --git a/big.py b/big.py\n"
            "@@ -1,5 +1,1999 @@\n"
            "+added\n"
        )
        safe_token_limit = 5_000
        chunks = create_review_chunks(
            full_code=full_code,
            changed_files=["big.py"],
            diff=diff,
            safe_token_limit=safe_token_limit,
            context_lines=30,
        )
        assert len(chunks) > 1
        for chunk in chunks:
            prompt = build_chunk_prompt(chunk)
            assert count_tokens(prompt) <= safe_token_limit


# ─────────────────────────────────────────────────────────────────────────────
# 4. Prompt Building and Scope Enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkPromptBuilding:

    def test_chunk_prompt_preserves_scope_rules(self):
        chunk = ReviewChunk(
            chunk_id=1,
            total_chunks=2,
            file_path="service.py",
            changed_files=["service.py"],
            code_snippet="FILE: service.py\n```python\nL50: x = 1\n```",
            diff_snippet="+x = 1",
            start_line=50,
            end_line=50,
        )
        prompt = build_chunk_prompt(chunk)
        assert "CRITICAL RULE — REVIEW SCOPE IS STRICTLY LIMITED TO THE PR DIFF" in prompt
        assert "PR DIFF / CHANGED CODE TO REVIEW:" in prompt
        assert "+x = 1" in prompt
        assert "FULL FILE CONTEXT" in prompt
        assert "L50: x = 1" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 5. Result Aggregation, Ordering, and Deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestResultAggregator:

    def test_normalization_helpers(self):
        assert normalize_file_path("`src/foo.py`") == "src/foo.py"
        assert normalize_file_path("./src/foo.py") == "src/foo.py"
        assert normalize_file_path("src\\foo.py") == "src/foo.py"
        assert normalize_file_path("src/foo.py (lines 1-10)") == "src/foo.py"

        assert normalize_line_spec("L50") == "50"
        assert normalize_line_spec("50") == "50"
        assert normalize_line_spec("L50-L55") == "50-55"
        assert normalize_line_spec("50 - 55") == "50-55"
        assert normalize_line_spec("Line 50") == "50"

    def test_dedup_removes_identical_findings_from_overlapping_chunks(self):
        chunk1 = (
            "**Issue:** Potential NoneType dereference\n"
            "**File:** src/api.py\n"
            "**Line:** L50\n\n"
            "**Code:**\n\n"
            "```python\n"
            "user.name\n"
            "```\n\n"
            "**Reason:** user could be None.\n"
            "**Suggestion:** Add a None check."
        )
        chunk2 = (
            "**Issue:** Potential NoneType dereference\n"
            "**File:** `src/api.py`\n"
            "**Line:** 50\n\n"
            "**Code:**\n\n"
            "```python\n"
            "user.name\n"
            "```\n\n"
            "**Reason:** user could be None.\n"
            "**Suggestion:** Add a None check."
        )
        combined = aggregate_reviews([chunk1, chunk2])
        assert combined.count("**Issue:**") == 1
        assert "src/api.py" in combined

    def test_different_findings_on_different_lines_are_preserved(self):
        # Explicit user requirement: L33 and L133 must remain separate
        chunk1 = (
            "**Issue:** Bug A\n"
            "**File:** large_code.py\n"
            "**Line:** L33\n\n"
            "**Code:**\n```python\nx = 1\n```\n\n"
            "**Reason:** r1\n**Suggestion:** s1"
        )
        chunk2 = (
            "**Issue:** Bug A\n"
            "**File:** large_code.py\n"
            "**Line:** L133\n\n"
            "**Code:**\n```python\ny = 2\n```\n\n"
            "**Reason:** r2\n**Suggestion:** s2"
        )
        combined = aggregate_reviews([chunk1, chunk2])
        assert combined.count("**Issue:**") == 2
        assert "L33" in combined or "33" in combined
        assert "L133" in combined or "133" in combined

    def test_aggregation_preserves_deterministic_chunk_order(self):
        # Chunk 1 (Issue at start), Chunk 2 (Issue in middle), Chunk 3 (Issue at end)
        chunk1 = "**Issue:** First Issue\n**File:** a.py\n**Line:** 10\n\n**Code:**\n```py\na=1\n```\n\n**Reason:** r\n**Suggestion:** s"
        chunk2 = "**Issue:** Second Issue\n**File:** b.py\n**Line:** 20\n\n**Code:**\n```py\nb=2\n```\n\n**Reason:** r\n**Suggestion:** s"
        chunk3 = "**Issue:** Third Issue\n**File:** c.py\n**Line:** 30\n\n**Code:**\n```py\nc=3\n```\n\n**Reason:** r\n**Suggestion:** s"

        combined = aggregate_reviews([chunk1, chunk2, chunk3])
        idx1 = combined.find("First Issue")
        idx2 = combined.find("Second Issue")
        idx3 = combined.find("Third Issue")
        assert idx1 < idx2 < idx3

    def test_aggregate_chunk_result_dictionaries_with_status_and_chunk_id(self):
        """Verify aggregate_reviews handles chunk-result dictionaries and surfaces notices with chunk IDs."""
        results = [
            {
                "chunk_id": 1,
                "file": "app.py",
                "status": "completed",
                "review": (
                    "**Issue:** Bug 1\n**File:** app.py\n**Line:** 10\n\n"
                    "**Code:**\n```py\nx=1\n```\n\n**Reason:** r\n**Suggestion:** s"
                ),
            },
            {
                "chunk_id": 2,
                "file": "app.py",
                "status": "failed",
                "review": "Error occurred during AI code review: timeout",
            },
            {
                "chunk_id": 3,
                "file": "util.py",
                "status": "completed",
                "review": "No issues found.",
            },
        ]
        aggregated = aggregate_reviews(results)

        # Successful finding preserved
        assert "Bug 1" in aggregated
        # Failed chunk explicitly reported in Review Notices with chunk ID
        assert "app.py (Chunk 2)" in aggregated
        # Cannot claim overall "No issues found."
        assert aggregated != "No issues found."

    def test_never_returns_clean_when_chunk_fails_even_without_findings(self):
        """Never return 'No issues found.' or clean equivalent if any chunk failed or is incomplete."""
        # Chunk 1 clean, Chunk 2 failed, no findings anywhere
        results = [
            {"chunk_id": 1, "file": "clean.py", "status": "completed", "review": "No issues found."},
            {"chunk_id": 2, "file": "failed.py", "status": "failed", "review": "Error occurred during AI code review: timeout"},
        ]
        aggregated = aggregate_reviews(results)

        assert "No issues found." not in aggregated
        assert "No issues found in completed" not in aggregated
        assert "Review Notices" in aggregated
        assert "failed.py (Chunk 2)" in aggregated
        assert "(failed)" in aggregated

    def test_preserves_valid_findings_when_another_chunk_fails(self):
        """Valid findings from successful chunks must be preserved even if another chunk fails."""
        results = [
            {
                "chunk_id": 1,
                "file": "app.py",
                "status": "completed",
                "review": (
                    "**Issue:** Critical security issue\n**File:** app.py\n**Line:** 42\n\n"
                    "**Code:**\n```py\neval(user_input)\n```\n\n**Reason:** Injection.\n**Suggestion:** Remove eval."
                ),
            },
            {"chunk_id": 2, "file": "broken.py", "status": "failed", "review": "Error occurred: connection reset"},
        ]
        aggregated = aggregate_reviews(results)

        # Finding must be preserved
        assert "Critical security issue" in aggregated
        assert "eval(user_input)" in aggregated
        # Failed chunk must also be present in notices
        assert "Review Notices" in aggregated
        assert "broken.py (Chunk 2)" in aggregated
        assert "No issues found." not in aggregated

    def test_commentary_without_clean_statement_not_replaced_with_clean(self):
        """Review commentary that lacks explicit clean statement is preserved and never replaced with 'No issues found.'."""
        results = [
            {"chunk_id": 1, "file": "foo.py", "status": "completed", "review": "The author implemented caching."},
        ]
        aggregated = aggregate_reviews(results)

        assert "No issues found." not in aggregated
        assert "The author implemented caching." in aggregated

    def test_never_returns_clean_when_chunk_is_incomplete(self):
        """If a chunk has incomplete status, review notices must be present and clean result forbidden."""
        results = [
            {"chunk_id": 1, "file": "partial.py", "status": "incomplete", "review": "The response was truncated..."},
        ]
        aggregated = aggregate_reviews(results)

        assert "No issues found." not in aggregated
        assert "Review Notices" in aggregated
        assert "partial.py (Chunk 1)" in aggregated
        assert "(incomplete)" in aggregated



# ─────────────────────────────────────────────────────────────────────────────
# 6. Parallel Execution and Concurrency Performance Test
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelChunkExecution:

    @patch("ai.chunk_manager.review_code")
    def test_maximum_concurrency_capped_at_three(self, mock_review):
        max_seen = 0
        current_active = 0
        lock = threading.Lock()

        def slow_review(prompt):
            nonlocal max_seen, current_active
            with lock:
                current_active += 1
                if current_active > max_seen:
                    max_seen = current_active
            time.sleep(0.04)
            with lock:
                current_active -= 1
            return "**Issue:** Bug\n**File:** a.py\n**Line:** 1\n\n**Code:**\n```py\nx=1\n```\n\n**Reason:** r\n**Suggestion:** s"

        mock_review.side_effect = slow_review

        # Create 5 distant chunks to trigger 2 batches: [3 chunks, 2 chunks]
        full_code = "\n\n".join(f"FILE: f{i}.py\n```python\nL1: x_{i} = 1\n```" for i in range(5))
        diff = "\n".join(f"diff --git a/f{i}.py b/f{i}.py\n@@ -1,1 +1,1 @@\n+x_{i} = 1" for i in range(5))
        files = [f"f{i}.py" for i in range(5)]

        execute_chunked_review(
            full_code=full_code,
            changed_files=files,
            diff=diff,
            safe_token_limit=120_000,
            max_concurrency=3,
        )

        assert mock_review.call_count == 5
        # Must have run up to 3 concurrently, never exceeding 3
        assert max_seen == 3

    @patch("ai.chunk_manager.review_code")
    def test_concurrent_execution_is_faster_than_sequential(self, mock_review):
        # Prove chunks run concurrently rather than strictly one after another
        delay = 0.05

        def mock_impl(prompt):
            time.sleep(delay)
            return "**Issue:** Speed Bug\n**File:** speed.py\n**Line:** 1\n\n**Code:**\n```py\nx=1\n```\n\n**Reason:** r\n**Suggestion:** s"

        mock_review.side_effect = mock_impl

        full_code = "\n\n".join(f"FILE: file_{i}.py\n```python\nL1: a = {i}\n```" for i in range(3))
        diff = "\n".join(f"diff --git a/file_{i}.py b/file_{i}.py\n@@ -1,1 +1,1 @@\n+a = {i}" for i in range(3))
        files = [f"file_{i}.py" for i in range(3)]

        start_time = time.time()
        execute_chunked_review(
            full_code=full_code,
            changed_files=files,
            diff=diff,
            safe_token_limit=120_000,
            max_concurrency=3,
        )
        elapsed = time.time() - start_time

        # If sequential, elapsed would be >= 3 * 0.05 = 0.15s.
        # Concurrently, it should finish well under 0.12s.
        assert elapsed < 0.12

    @patch("ai.chunk_manager.review_code")
    def test_chunk_failure_records_error_and_allows_others_to_complete(self, mock_review):
        mock_review.side_effect = [
            RuntimeError("Connection dropped to NVIDIA endpoint"),
            "**Issue:** Real Bug\n**File:** good.py\n**Line:** 50\n\n**Code:**\n```py\nok=1\n```\n\n**Reason:** r\n**Suggestion:** s",
        ]

        full_code = (
            "FILE: fail.py\n```python\nL1: x = 1\n```\n\n"
            "FILE: good.py\n```python\nL50: ok = 1\n```"
        )
        diff = (
            "diff --git a/fail.py b/fail.py\n@@ -1,1 +1,1 @@\n+x = 1\n"
            "diff --git a/good.py b/good.py\n@@ -50,1 +50,1 @@\n+ok = 1\n"
        )

        result = execute_chunked_review(
            full_code=full_code,
            changed_files=["fail.py", "good.py"],
            diff=diff,
            safe_token_limit=120_000,
            max_concurrency=3,
        )

        assert "Real Bug" in result
        assert "good.py" in result

    @patch("ai.chunk_manager.review_code")
    def test_chunk_failure_disclosed_in_final_output(self, mock_review):
        """Failed chunk must appear in the Review Notices section, not be silently dropped."""
        def mock_impl(prompt):
            if "fail.py" in prompt:
                raise RuntimeError("Connection dropped to NVIDIA endpoint")
            return (
                "**Issue:** Real Bug\n**File:** good.py\n**Line:** 50\n\n"
                "**Code:**\n```py\nok=1\n```\n\n**Reason:** r\n**Suggestion:** s"
            )
        mock_review.side_effect = mock_impl

        full_code = (
            "FILE: fail.py\n```python\nL1: x = 1\n```\n\n"
            "FILE: good.py\n```python\nL50: ok = 1\n```"
        )
        diff = (
            "diff --git a/fail.py b/fail.py\n@@ -1,1 +1,1 @@\n+x = 1\n"
            "diff --git a/good.py b/good.py\n@@ -50,1 +50,1 @@\n+ok = 1\n"
        )

        result = execute_chunked_review(
            full_code=full_code,
            changed_files=["fail.py", "good.py"],
            diff=diff,
            safe_token_limit=120_000,
            max_concurrency=3,
        )

        # The successful chunk's finding must still appear.
        assert "Real Bug" in result
        # The failed chunk must be disclosed — never silently omitted.
        assert "fail.py" in result
        assert "failed" in result.lower() or "Review Notices" in result or "could not be reviewed" in result.lower()

    @patch("ai.chunk_manager.review_code")
    def test_review_result_status_preserved_for_incomplete_response(self, mock_review):
        """If review_code returns a ReviewResult with status='incomplete', it must surface."""
        from ai.reviewer import ReviewResult

        # Chunk 1 returns an incomplete result (e.g. generation limit hit mid-response).
        incomplete = ReviewResult("Partial review text cut off here...", status="incomplete")
        # Chunk 2 returns a normal completed result with a real finding.
        complete = ReviewResult(
            "**Issue:** Logic error\n**File:** ok.py\n**Line:** 10\n\n**Code:**\n```py\nx=1\n```\n\n**Reason:** r\n**Suggestion:** s",
            status="completed",
        )
        def mock_impl(prompt):
            if "partial.py" in prompt:
                return incomplete
            return complete
        mock_review.side_effect = mock_impl

        full_code = (
            "FILE: partial.py\n```python\nL1: a = 1\n```\n\n"
            "FILE: ok.py\n```python\nL10: x = 1\n```"
        )
        diff = (
            "diff --git a/partial.py b/partial.py\n@@ -1,1 +1,1 @@\n+a = 1\n"
            "diff --git a/ok.py b/ok.py\n@@ -10,1 +10,1 @@\n+x = 1\n"
        )

        result = execute_chunked_review(
            full_code=full_code,
            changed_files=["partial.py", "ok.py"],
            diff=diff,
            safe_token_limit=120_000,
            max_concurrency=3,
        )

        # The completed finding must appear.
        assert "Logic error" in result
        # The incomplete chunk must be disclosed — never treated as clean "No issues found."
        assert "partial.py" in result
        assert "No issues found." not in result

    @patch("ai.chunk_manager.review_code")
    def test_empty_string_chunk_result_is_disclosed_not_silently_dropped(self, mock_review):
        """An empty-string return (no review text) must not be silently omitted from the output."""
        def mock_impl(prompt):
            if "empty.py" in prompt:
                return ""
            return (
                "**Issue:** Valid Finding\n**File:** b.py\n**Line:** 5\n\n"
                "**Code:**\n```py\ny=2\n```\n\n**Reason:** r\n**Suggestion:** s"
            )
        mock_review.side_effect = mock_impl

        full_code = (
            "FILE: empty.py\n```python\nL1: a = 0\n```\n\n"
            "FILE: b.py\n```python\nL5: y = 2\n```"
        )
        diff = (
            "diff --git a/empty.py b/empty.py\n@@ -1,1 +1,1 @@\n+a = 0\n"
            "diff --git a/b.py b/b.py\n@@ -5,1 +5,1 @@\n+y = 2\n"
        )

        result = execute_chunked_review(
            full_code=full_code,
            changed_files=["empty.py", "b.py"],
            diff=diff,
            safe_token_limit=120_000,
            max_concurrency=3,
        )

        # The valid finding from the second chunk must appear.
        assert "Valid Finding" in result
        # An empty chunk result must not let the aggregator claim "No issues found." for the whole PR.
        assert "No issues found." not in result
        # The file with the empty result must be disclosed.
        assert "empty.py" in result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main Pipeline Integration: Branching on SAFE_INPUT_TOKEN_LIMIT
# ─────────────────────────────────────────────────────────────────────────────

class TestMainPipelinePromptBranching:

    @patch("main.post_pr_comment")
    @patch("main.review_code")
    @patch("main.execute_chunked_review")
    @patch("main.extract_full_code")
    @patch("main.get_diff")
    @patch("main.get_pr_changed_files")
    @patch("main.clone_repository")
    @patch("main.get_pr_details")
    @patch("main.validate_pr_description")
    @patch("main.validate_description_matches_code")
    @patch("main.validate_code_comments")
    def test_prompt_within_120k_token_budget_uses_single_ai_request(
        self,
        mock_val_comments,
        mock_val_desc_code,
        mock_val_desc,
        mock_get_pr_details,
        mock_clone,
        mock_changed_files,
        mock_get_diff,
        mock_extract_full_code,
        mock_chunked_review,
        mock_review_code,
        mock_post_comment,
    ):
        mock_get_pr_details.return_value = (
            "fix: update button color",
            "This PR updates the button color to match the design system.",
        )
        mock_changed_files.return_value = ["button.py"]
        mock_get_diff.return_value = "diff --git a/button.py b/button.py\n+color = 'blue'"
        mock_extract_full_code.return_value = "FILE: button.py\n```python\nL1: color = 'blue'\n```"
        mock_val_desc.return_value = {"is_valid": True, "reason": None}
        mock_val_desc_code.return_value = {"is_valid": True, "reason": None}
        mock_val_comments.return_value = []
        mock_review_code.return_value = "No issues found."

        payload = {
            "action": "opened",
            "pull_request": {"number": 1, "body": "valid body"},
            "repository": {"full_name": "owner/repo", "name": "repo"},
        }

        result = process_pr_event(payload, "pull_request", "opened", "", "")

        assert result["status"] == "success"
        # Within 120k tokens -> exactly 1 direct review_code call
        assert mock_review_code.call_count == 1
        assert mock_chunked_review.call_count == 0
        assert mock_post_comment.call_count == 1

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
    def test_prompt_exceeding_120k_token_budget_triggers_chunked_review(
        self,
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
        mock_get_pr_details.return_value = (
            "feat: large data migration",
            "This PR performs a large database schema migration.",
        )
        mock_changed_files.return_value = ["migration.py"]
        mock_get_diff.return_value = "diff --git a/migration.py b/migration.py\n+large_change"

        # Generate full_code that exceeds 120,000 input tokens
        # Each line has ~10 tokens -> 15,000 lines > 120,000 tokens
        giant_code = "FILE: migration.py\n```python\n" + ("L1: x = some_function(arg1, arg2, param=123)\n" * 15_000) + "```"
        assert count_tokens(giant_code) > SAFE_INPUT_TOKEN_LIMIT

        mock_extract_full_code.return_value = giant_code
        mock_extract_change_context.return_value = (giant_code, giant_code)
        mock_val_desc.return_value = {"is_valid": True, "reason": None}
        mock_val_desc_code.return_value = {"is_valid": True, "reason": None}
        mock_val_comments.return_value = []
        mock_chunked_review.return_value = (
            "**Issue:** Query timeout\n**File:** migration.py\n**Line:** 50\n\n"
            "**Code:**\n```python\nmigrate()\n```\n\n**Reason:** Timeout.\n**Suggestion:** Batch."
        )

        payload = {
            "action": "opened",
            "pull_request": {"number": 2, "body": "valid body"},
            "repository": {"full_name": "owner/repo", "name": "repo"},
        }

        result = process_pr_event(payload, "pull_request", "opened", "", "")

        assert result["status"] == "success"
        # Over 120k tokens -> chunked review MUST be invoked
        assert mock_chunked_review.call_count == 1
        assert mock_review_code.call_count == 0
        assert mock_post_comment.call_count == 1

        posted_comment = mock_post_comment.call_args[0][2]
        assert "Query timeout" in posted_comment
        assert "## AI Code Review" in posted_comment

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
    def test_oversized_file_chunked_review_failure_preserved_in_final_comment(
        self,
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
        """When an oversized file is chunked and has failures, the final comment must disclose the failure."""
        from ai.reviewer import ReviewResult
        mock_get_pr_details.return_value = ("feat: big feature", "Valid description")
        mock_changed_files.return_value = ["big_service.py"]
        mock_get_diff.return_value = "diff --git a/big_service.py b/big_service.py\n+new_code"

        # Safe input token limit check for overall prompt passes, but file itself triggers chunked review
        normal_code = "FILE: big_service.py\n```python\nL1: x = 1\n```"
        mock_extract_full_code.return_value = normal_code
        mock_extract_change_context.return_value = (normal_code, normal_code)
        mock_val_desc.return_value = {"is_valid": True, "reason": None}
        mock_val_desc_code.return_value = {"is_valid": True, "reason": None}
        mock_val_comments.return_value = []

        # Patch count_tokens so file_token_count triggers the else branch (chunked review)
        with patch("main.count_tokens", side_effect=lambda p: 150_000 if "big_service.py" in p else 500):
            # Chunked review returned 1 valid issue AND a failure notice for chunk 2
            chunked_review_result = ReviewResult(
                "**Issue:** Memory leak\n**File:** big_service.py\n**Line:** 10\n\n"
                "**Code:**\n```py\ncache.append(x)\n```\n\n**Reason:** Leak.\n**Suggestion:** Limit.\n\n"
                "---\n\n"
                "### ⚠️ Review Notices\n"
                "The following file(s) could not be reviewed completely:\n"
                "- `big_service.py (Chunk 2)` (failed): Error occurred during AI code review: timeout",
                status="failed",
            )
            mock_chunked_review.return_value = chunked_review_result

            payload = {
                "action": "opened",
                "pull_request": {"number": 3, "body": "valid body"},
                "repository": {"full_name": "owner/repo", "name": "repo"},
            }

            result = process_pr_event(payload, "pull_request", "opened", "", "")
            assert result["status"] == "success"
            assert mock_chunked_review.call_count == 1
            assert mock_post_comment.call_count == 1

            posted = mock_post_comment.call_args[0][2]
            # Must preserve the valid finding
            assert "Memory leak" in posted
            # Must preserve the review notice for the failed chunk
            assert "Review Notices" in posted
            assert "big_service.py (Chunk 2)" in posted
            # Must NOT claim clean "No issues found."
            assert "No issues found." not in posted
