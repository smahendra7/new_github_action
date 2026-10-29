"""
tests/test_comment_validator_diff_aware.py

Focused tests verifying diff-aware Comment vs Code validation:
1. A change to one function causes only that function's comments to be extracted and validated.
2. Comments in unchanged functions are not sent to the AI.
3. Changes to multiple functions validate each affected function separately.
4. If a function cannot be identified, only the relevant 15-lines-before/15-lines-after context is extracted.
5. The Comment vs Code validation flow does not call a whole-file comment extraction routine.
6. A changed function with no comments triggers no AI calls even if unchanged functions have comments.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure third-party SDKs are mocked before importing project modules if not installed in the environment
for mod in ["groq", "openai", "git"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from ai.comment_validator import validate_code_comments


SAMPLE_FILE_CONTENT = '''# Module header comment
import os

def calculate_discount(price, discount):
    # Discount reduces the price
    return price - discount

def calculate_tax(amount):
    # Tax adds 10 percent
    return amount * 0.1

def calculate_shipping(weight):
    # Shipping fee based on weight
    if weight > 10:
        return 20
    return 5

# Top-level standalone configuration
CONFIG_VALUE = 42  # Config comment
'''

DIFF_TOUCHING_DISCOUNT_ONLY = '''diff --git a/pricing.py b/pricing.py
index 1234567..89abcdef 100644
--- a/pricing.py
+++ b/pricing.py
@@ -5,2 +5,2 @@
 def calculate_discount(price, discount):
-    # Discount reduces the price
+    # Discount reduces the price accurately
     return price - discount
'''

DIFF_TOUCHING_DISCOUNT_AND_SHIPPING = '''diff --git a/pricing.py b/pricing.py
index 1234567..89abcdef 100644
--- a/pricing.py
+++ b/pricing.py
@@ -5,2 +5,2 @@
 def calculate_discount(price, discount):
-    # Discount reduces the price
+    # Discount reduces the price accurately
     return price - discount
@@ -14,2 +14,2 @@
 def calculate_shipping(weight):
-    # Shipping fee based on weight
+    # Updated shipping calculation
     if weight > 10:
'''

DIFF_TOUCHING_TOP_LEVEL = '''diff --git a/pricing.py b/pricing.py
index 1234567..89abcdef 100644
--- a/pricing.py
+++ b/pricing.py
@@ -19,1 +19,1 @@
-CONFIG_VALUE = 42  # Config comment
+CONFIG_VALUE = 100  # Config comment
'''

DIFF_TOUCHING_NO_COMMENT_FUNC = '''diff --git a/pricing_no_comment.py b/pricing_no_comment.py
index 1234567..89abcdef 100644
--- a/pricing_no_comment.py
+++ b/pricing_no_comment.py
@@ -1,2 +1,2 @@
 def no_comment_func(x):
-    return x + 1
+    return x + 2
'''

SAMPLE_FILE_NO_COMMENT_FUNC = '''def no_comment_func(x):
    return x + 2

def other_func_with_comment(y):
    # This comment belongs to other_func
    return y * 2
'''


class TestDiffAwareCommentValidation(unittest.TestCase):

    def setUp(self):
        self.mock_env = patch.dict("os.environ", {"NVIDIA_API_KEY": "fake_test_key"})
        self.mock_env.start()

    def tearDown(self):
        self.mock_env.stop()

    @patch("ai.comment_validator._validate_file_comments", return_value=[])
    @patch("ai.comment_validator.get_file_content", return_value=SAMPLE_FILE_CONTENT)
    def test_change_to_one_function_validates_only_that_function(self, mock_get_content, mock_validate):
        """1. A change to one function causes only that function's comments to be validated.
           2. Comments in unchanged functions are NOT sent to the AI."""
        validate_code_comments(
            repo_path="/dummy/repo",
            changed_files=["pricing.py"],
            source_branch="feature",
            diff=DIFF_TOUCHING_DISCOUNT_ONLY
        )

        # AI validation should be called exactly once for calculate_discount
        self.assertEqual(mock_validate.call_count, 1)

        # Inspect the comments passed to the AI
        call_args = mock_validate.call_args_list[0]
        passed_comments = call_args[0][2]
        file_path = call_args[0][1]

        self.assertEqual(file_path, "pricing.py")
        self.assertEqual(len(passed_comments), 1)

        comment_item = passed_comments[0]
        # Verify the comment is from calculate_discount
        self.assertIn("Discount reduces", comment_item["comment"])
        self.assertEqual(comment_item["function_name"], "calculate_discount")
        # Verify complete function code is provided
        self.assertIn("def calculate_discount(price, discount):", comment_item["code"])
        self.assertIn("return price - discount", comment_item["code"])

        # Crucial check: Comments from calculate_tax and calculate_shipping must NOT be present
        all_comments_text = " ".join(c["comment"] for c in passed_comments)
        self.assertNotIn("Tax adds", all_comments_text)
        self.assertNotIn("Shipping fee", all_comments_text)
        self.assertNotIn("Config comment", all_comments_text)

    @patch("ai.comment_validator._validate_file_comments", return_value=[])
    @patch("ai.comment_validator.get_file_content", return_value=SAMPLE_FILE_CONTENT)
    def test_changes_to_multiple_functions_validated_independently(self, mock_get_content, mock_validate):
        """3. Changes to multiple functions validate each affected function independently."""
        validate_code_comments(
            repo_path="/dummy/repo",
            changed_files=["pricing.py"],
            source_branch="feature",
            diff=DIFF_TOUCHING_DISCOUNT_AND_SHIPPING
        )

        # AI validation should be called twice (once for discount, once for shipping)
        self.assertEqual(mock_validate.call_count, 2)

        first_call_comments = mock_validate.call_args_list[0][0][2]
        second_call_comments = mock_validate.call_args_list[1][0][2]

        funcs_validated = {
            first_call_comments[0]["function_name"],
            second_call_comments[0]["function_name"]
        }
        self.assertEqual(funcs_validated, {"calculate_discount", "calculate_shipping"})

        # Unchanged calculate_tax must still be excluded
        combined_text = " ".join(
            c["comment"] for c in first_call_comments + second_call_comments
        )
        self.assertNotIn("Tax adds", combined_text)

    @patch("ai.comment_validator._validate_file_comments", return_value=[])
    @patch("ai.comment_validator.get_file_content", return_value=SAMPLE_FILE_CONTENT)
    def test_fallback_when_function_cannot_be_identified(self, mock_get_content, mock_validate):
        """4. If function cannot be identified, only 15-before/15-after context is extracted
           and comments in unchanged functions are excluded."""
        validate_code_comments(
            repo_path="/dummy/repo",
            changed_files=["pricing.py"],
            source_branch="feature",
            diff=DIFF_TOUCHING_TOP_LEVEL
        )

        self.assertEqual(mock_validate.call_count, 1)
        passed_comments = mock_validate.call_args_list[0][0][2]

        # Should only contain top-level comments, NOT comments in functions
        self.assertTrue(len(passed_comments) >= 1)
        all_passed_comments_text = " ".join(c["comment"] for c in passed_comments)
        self.assertIn("Config comment", all_passed_comments_text)

        # In fallback, function_name is None
        for item in passed_comments:
            self.assertIsNone(item.get("function_name"))
            self.assertIn("CONFIG_VALUE", item["code"])

        # Unchanged functions comments must be excluded
        self.assertNotIn("Discount reduces", all_passed_comments_text)
        self.assertNotIn("Tax adds", all_passed_comments_text)
        self.assertNotIn("Shipping fee", all_passed_comments_text)
        # Comments more than 15 lines away (line 1: # Module header comment) must be excluded
        self.assertNotIn("Module header comment", all_passed_comments_text)

    @patch("ai.comment_validator._validate_file_comments", return_value=[])
    @patch("ai.comment_validator.get_file_content", return_value=SAMPLE_FILE_NO_COMMENT_FUNC)
    def test_changed_function_with_no_comments_skips_ai(self, mock_get_content, mock_validate):
        """A changed function with no comments triggers no AI calls; unchanged function comments ignored."""
        validate_code_comments(
            repo_path="/dummy/repo",
            changed_files=["pricing_no_comment.py"],
            source_branch="feature",
            diff=DIFF_TOUCHING_NO_COMMENT_FUNC
        )

        # AI should NOT be called at all because the changed function has no comments
        mock_validate.assert_not_called()

    def test_whole_file_comment_extraction_is_not_imported_or_called(self):
        """5. Verify that extract_comments_from_files is not called by validate_code_comments."""
        import ai.comment_validator as cv
        # Confirm extract_comments_from_files is not even in comment_validator's module namespace
        self.assertFalse(hasattr(cv, "extract_comments_from_files"))

    def test_comment_extractor_context_window_15_lines(self):
        """Verify _get_context_window extracts up to 15 lines before and 15 lines after."""
        from github.comment_extractor import _get_context_window
        lines = [f"line {i}" for i in range(1, 101)]
        # Target comment at line 50
        context = _get_context_window(lines, 50)
        context_lines = context.splitlines()
        # Should be lines 35 (50-15) to 65 (50+15), inclusive -> 31 lines
        self.assertEqual(len(context_lines), 31)
        self.assertEqual(context_lines[0], "line 35")
        self.assertEqual(context_lines[-1], "line 65")

        # Boundary clamping: target near top of file (line 5)
        top_context = _get_context_window(lines, 5)
        top_lines = top_context.splitlines()
        self.assertEqual(top_lines[0], "line 1")
        self.assertEqual(top_lines[-1], "line 20")

        # Boundary clamping: target near bottom of file (line 95)
        bottom_context = _get_context_window(lines, 95)
        bottom_lines = bottom_context.splitlines()
        self.assertEqual(bottom_lines[0], "line 80")
        self.assertEqual(bottom_lines[-1], "line 100")


if __name__ == "__main__":
    unittest.main()
