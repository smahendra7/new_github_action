"""
tests/test_ai_comment_code_review.py

Focused tests for AI Code Review comment & docstring behavior:
1. Changed code with no comment -> no missing-comment issue reported.
2. Changed code with matching existing comment -> no comment-vs-code issue.
3. Changed code with contradictory existing comment -> comment-vs-code issue is allowed/expected.
4. Changed comment with implementation that contradicts it -> issue is allowed/expected.
5. Changed code where associated comment is unchanged but becomes inconsistent -> issue is allowed/expected.
6. Unrelated pre-existing comment/code mismatch outside changed code -> must NOT be reported.
7. Existing normal AI review behavior remains unchanged when no comment exists.
8. Context extractor preservation of comments immediately preceding changed scopes (Python, Kotlin, Swift, decorators).
"""

from unittest.mock import patch
import pytest

from ai.context_extractor import (
    extract_change_aware_context,
    extract_file_change_context,
    _expand_start_for_comments_and_decorators,
)
from ai.prompt_builder import build_review_prompt
from ai.result_aggregator import aggregate_reviews
from main import _format_review


# ─────────────────────────────────────────────────────────────────────────────
# 1. Changed code with no comment -> no missing-comment issue
# ─────────────────────────────────────────────────────────────────────────────

class TestNoCommentBehavior:

    def test_prompt_forbids_missing_comment_issues(self):
        """Comments/docstrings are optional; the prompt must explicitly forbid reporting missing ones."""
        code = "def calculate_discount(price, discount):\n    return price - discount\n"
        diff = "@@ -1,2 +1,2 @@\n+def calculate_discount(price, discount):\n+    return price - discount"
        prompt = build_review_prompt(code=code, changed_files=["discount.py"], diff=diff)

        assert "Comments and docstrings are strictly OPTIONAL" in prompt
        assert "NEVER report an issue simply because a comment or docstring is missing" in prompt
        assert "Missing comments, missing docstrings, or lack of documentation" in prompt
        assert "1. NO COMMENT/DOCSTRING:" in prompt
        assert "Continue with the normal AI code review." in prompt

    def test_no_comment_review_does_not_flag_missing_docs(self):
        """When code has no comments and LLM reports no bugs, result is clean."""
        review_text = "No issues found."
        formatted = _format_review(review_text)
        assert "No issues found." in formatted
        assert "comment" not in formatted.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Changed code with matching existing comment -> no comment-vs-code issue
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchingCommentBehavior:

    def test_prompt_instructs_no_issue_when_comment_matches(self):
        """Prompt instructs model not to report issues when comment matches code."""
        prompt = build_review_prompt(code="", changed_files=["discount.py"], diff="")
        assert "6. NO MISMATCH:" in prompt
        assert "If the comment/docstring matches or accurately describes the implementation, do NOT report an issue." in prompt

    def test_matching_comment_context_extraction(self):
        """Verify context extractor includes matching comment and code."""
        lines = [
            "# Discount should reduce the price.",
            "def calculate_discount(price, discount):",
            "    return price - discount",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        full_code = "FILE: discount.py\n```python\n" + "\n".join(lines_dict.values()) + "\n```"
        diff = (
            "diff --git a/discount.py b/discount.py\n"
            "@@ -2,2 +2,2 @@\n"
            " def calculate_discount(price, discount):\n"
            "+    return price - discount\n"
        )

        changed_code, context = extract_change_aware_context(
            changed_files=["discount.py"],
            diff=diff,
            full_code=full_code,
        )

        assert "L1: # Discount should reduce the price." in context
        assert "L3:     return price - discount" in context


# ─────────────────────────────────────────────────────────────────────────────
# 3. Changed code with contradictory existing comment -> issue allowed/expected
# ─────────────────────────────────────────────────────────────────────────────

class TestContradictingCommentBehavior:

    def test_context_and_prompt_support_contradicting_comment_detection(self):
        """Example C: Comment says reduce price, code adds discount."""
        lines = [
            "# Discount should reduce the price.",
            "def calculate_discount(price, discount):",
            "    return price + discount",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        full_code = "FILE: discount.py\n```python\n" + "\n".join(lines_dict.values()) + "\n```"
        diff = (
            "diff --git a/discount.py b/discount.py\n"
            "@@ -2,2 +2,2 @@\n"
            " def calculate_discount(price, discount):\n"
            "+    return price + discount\n"
        )

        changed_code, context = extract_change_aware_context(
            changed_files=["discount.py"],
            diff=diff,
            full_code=full_code,
        )

        # Context must provide the comment so the model can detect the contradiction
        assert "L1: # Discount should reduce the price." in context
        assert "L3:     return price + discount" in context

        prompt = build_review_prompt(
            code=context,
            changed_files=["discount.py"],
            diff=diff,
            changed_code=changed_code,
        )

        assert "2. COMMENT/DOCSTRING EXISTS:" in prompt
        assert "If the comment contradicts, misrepresents, or is materially inconsistent with the implementation, report a comment-vs-code issue." in prompt

    def test_comment_mismatch_issue_formatting(self):
        """Verify comment-vs-code issue block formats properly through _format_review and aggregate_reviews."""
        issue_text = (
            "**Issue:** Comment contradicts implementation\n"
            "**File:** discount.py\n"
            "**Line:** L1-L3\n\n"
            "**Code:**\n\n"
            "```python\n"
            "# Discount should reduce the price.\n"
            "def calculate_discount(price, discount):\n"
            "    return price + discount\n"
            "```\n\n"
            "**Reason:** Comment states discount should reduce price but implementation uses addition (+).\n"
            "**Suggestion:** Change + to - in the return expression."
        )

        formatted = _format_review(issue_text)
        assert "**Issue:** Comment contradicts implementation" in formatted
        assert "**File:** discount.py" in formatted
        assert "**Line:** L1-L3" in formatted
        assert "**Reason:** Comment states discount should reduce price" in formatted

        aggregated = aggregate_reviews([issue_text])
        assert "**Issue:** Comment contradicts implementation" in aggregated


# ─────────────────────────────────────────────────────────────────────────────
# 4. Changed comment with implementation that contradicts it -> issue expected
# ─────────────────────────────────────────────────────────────────────────────

class TestChangedCommentContradictsCode:

    def test_changed_comment_context_extraction_and_prompt(self):
        """Example E: Comment changed from reduce to increase, code unchanged (price - discount)."""
        lines = [
            "# Discount should increase the price.",
            "discounted = price - discount",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        full_code = "FILE: discount.py\n```python\n" + "\n".join(lines_dict.values()) + "\n```"
        diff = (
            "diff --git a/discount.py b/discount.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-# Discount should reduce the price.\n"
            "+# Discount should increase the price.\n"
            " discounted = price - discount\n"
        )

        changed_code, context = extract_change_aware_context(
            changed_files=["discount.py"],
            diff=diff,
            full_code=full_code,
        )

        assert "L1: # Discount should increase the price." in changed_code or "L1: # Discount should increase the price." in context
        assert "L2: discounted = price - discount" in context

        prompt = build_review_prompt(
            code=context,
            changed_files=["discount.py"],
            diff=diff,
            changed_code=changed_code,
        )

        assert "4. COMMENT ITSELF IS CHANGED:" in prompt
        assert "compare the new comment against the implementation" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 5. Changed code where associated comment is unchanged but inconsistent
# ─────────────────────────────────────────────────────────────────────────────

class TestUnchangedCommentChangedCode:

    def test_unchanged_comment_changed_code_example_d(self):
        """Example D: Code changed from price - discount to price + discount, comment unchanged."""
        lines = [
            "# Discount should reduce the price.",
            "discounted = price + discount",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        full_code = "FILE: discount.py\n```python\n" + "\n".join(lines_dict.values()) + "\n```"
        diff = (
            "diff --git a/discount.py b/discount.py\n"
            "@@ -1,2 +1,2 @@\n"
            " # Discount should reduce the price.\n"
            "-discounted = price - discount\n"
            "+discounted = price + discount\n"
        )

        changed_code, context = extract_change_aware_context(
            changed_files=["discount.py"],
            diff=diff,
            full_code=full_code,
        )

        assert "L1: # Discount should reduce the price." in context
        assert "L2: discounted = price + discount" in changed_code

        prompt = build_review_prompt(
            code=context,
            changed_files=["discount.py"],
            diff=diff,
            changed_code=changed_code,
        )

        assert "3. COMMENT IS UNCHANGED BUT CODE CHANGES:" in prompt
        assert "you MUST report the comment-vs-code mismatch" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 6. Unrelated pre-existing mismatch outside changed code -> must NOT be reported
# ─────────────────────────────────────────────────────────────────────────────

class TestUnrelatedCommentMismatchExcluded:

    def test_unrelated_function_with_mismatch_excluded_from_context(self):
        """Unchanged function with comment mismatch far from changed code must NOT be in context."""
        lines = [
            "# This function adds numbers",
            "def multiply(a, b):",
            "    return a * b",
            "",
            "# Separator",
            "",
        ]
        # Add 50 filler lines to separate scopes clearly
        for i in range(50):
            lines.append(f"def filler_{i}(): pass")

        lines.extend([
            "",
            "def calculate_tax(amount):",
            "    tax = amount * 0.1",
            "    return tax",
        ])

        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        full_code = "FILE: math_ops.py\n```python\n" + "\n".join(lines_dict.values()) + "\n```"

        tax_line = len(lines) - 1  # line of `tax = amount * 0.1`
        diff = (
            f"diff --git a/math_ops.py b/math_ops.py\n"
            f"@@ -{tax_line},2 +{tax_line},2 @@\n"
            f"+    tax = amount * 0.1\n"
            f"     return tax\n"
        )

        changed_code, context = extract_change_aware_context(
            changed_files=["math_ops.py"],
            diff=diff,
            full_code=full_code,
        )

        # The unrelated pre-existing function and its mismatched comment must NOT be in context!
        assert "multiply" not in context
        assert "This function adds numbers" not in context

    def test_prompt_explicitly_forbids_unrelated_comment_reporting(self):
        """Verify prompt explicitly instructs not to report unrelated pre-existing comment issues."""
        prompt = build_review_prompt(code="", changed_files=["a.py"], diff="")
        assert "Do NOT independently report unrelated pre-existing issues or comment mismatches in unchanged code" in prompt
        assert "Unrelated pre-existing comments or code outside the PR changes" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 7. Normal AI review behavior remains unchanged when no comment exists
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalReviewWhenNoComment:

    def test_prompt_enforces_normal_review_continues_without_comments(self):
        """Rule 7: Normal code review must always run even when no comment exists."""
        prompt = build_review_prompt(code="def divide(a, b): return a / b", changed_files=["calc.py"])
        assert "7. NORMAL CODE REVIEW:" in prompt
        assert "The AI must still perform the normal AI Code Review (bugs, security, logic errors, etc.) of changed code even when no comment/docstring exists." in prompt

    def test_normal_bug_issue_is_formatted_identically(self):
        """Verify regular bug findings (division by zero, null deref) preserve existing formatting."""
        bug_issue = (
            "**Issue:** Potential ZeroDivisionError\n"
            "**File:** calc.py\n"
            "**Line:** L2\n\n"
            "**Code:**\n\n"
            "```python\n"
            "return a / b\n"
            "```\n\n"
            "**Reason:** Parameter b is not validated against zero.\n"
            "**Suggestion:** Add a check if b == 0 before division."
        )

        formatted = _format_review(bug_issue)
        assert "**Issue:** Potential ZeroDivisionError" in formatted
        assert "**File:** calc.py" in formatted
        assert "**Line:** L2" in formatted
        assert "**Reason:** Parameter b is not validated against zero." in formatted
        assert "**Suggestion:** Add a check if b == 0 before division." in formatted


# ─────────────────────────────────────────────────────────────────────────────
# 8. Context Extractor Leading Comment & Decorator Preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestContextExtractorLeadingComments:

    def test_expand_start_python_comments_and_decorators(self):
        lines = [
            "# Header info",
            "# Discount should reduce the price.",
            "@app.get('/discount')",
            "def calculate_discount(price, discount):",
            "    return price + discount",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}

        # start_line 4 is `def calculate_discount`
        expanded = _expand_start_for_comments_and_decorators(lines_dict, 4)
        # Should expand back through decorator (3) and comments (2, 1)
        assert expanded == 1

    def test_expand_start_stops_at_code(self):
        lines = [
            "def previous_func():",
            "    return 42",
            "# Discount comment",
            "def current_func():",
            "    pass",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        # start_line 4 is `def current_func`
        expanded = _expand_start_for_comments_and_decorators(lines_dict, 4)
        assert expanded == 3  # Line 3 is the comment, stops before Line 2 (code)

    def test_expand_start_kotlin_comments(self):
        lines = [
            "// Discount should reduce the price.",
            "fun calculateDiscount(price: Double, discount: Double): Double {",
            "    return price + discount",
            "}",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        expanded = _expand_start_for_comments_and_decorators(lines_dict, 2)
        assert expanded == 1

    def test_expand_start_swift_comments(self):
        lines = [
            "/// Discount should reduce the price.",
            "func calculateDiscount(price: Double, discount: Double) -> Double {",
            "    return price + discount",
            "}",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        expanded = _expand_start_for_comments_and_decorators(lines_dict, 2)
        assert expanded == 1
