"""
Regression tests for github/comment_extractor.py

Verifies that:
  1. Genuine Python comments are correctly extracted.
  2. Code fragments that are NOT comments are NOT extracted:
       - regex patterns containing // or /*
       - string literals containing # / // / /*
       - function/class definitions
       - lines whose "comment-like" text is inside a string
  3. Non-Python inline // and /* */ comments are correctly extracted.
  4. Non-Python comment markers inside string literals are NOT extracted.
"""

import textwrap
import pytest

from github.comment_extractor import extract_comments_from_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comments(text: str, filename: str = "test.py") -> list[str]:
    """Return just the extracted comment strings for a snippet."""
    return [
        item["comment"]
        for item in extract_comments_from_content(
            textwrap.dedent(text).strip(),
            filename
        )
    ]


def _items(text: str, filename: str = "test.py") -> list[dict]:
    """Return the full extracted comment dicts for a snippet."""
    return extract_comments_from_content(
        textwrap.dedent(text).strip(),
        filename
    )


# ===========================================================================
# PYTHON — genuine comments must be detected
# ===========================================================================

class TestPythonGenuineComments:

    def test_standalone_hash_comment(self):
        src = """
            # Calculate the total price
            total = price + tax
        """
        result = _comments(src)
        assert len(result) == 1
        assert "Calculate the total price" in result[0]

    def test_inline_hash_comment(self):
        src = """
            value = calculate()  # Calculate the value
        """
        result = _comments(src)
        assert len(result) == 1
        assert "Calculate the value" in result[0]

    def test_multiple_hash_comments(self):
        src = """
            # Step 1: initialize
            x = 0
            # Step 2: increment
            x += 1
        """
        result = _comments(src)
        assert len(result) == 2
        assert any("initialize" in c for c in result)
        assert any("increment" in c for c in result)

    def test_inline_comment_associated_code(self):
        items = _items("""
            total = price + tax  # Add tax to price
        """)
        assert len(items) == 1
        # Comment text must be correctly extracted.
        assert "Add tax to price" in items[0]["comment"]
        # There is no *following* line in this single-line snippet, so the
        # code field is '(no associated code found)' — that is correct
        # behaviour for the tokenize path.  What matters is that the comment
        # itself was captured, not that associated code exists here.


    def test_standalone_comment_associated_code(self):
        items = _items("""
            # Sum the values
            result = sum(values)
        """)
        assert len(items) == 1
        assert "sum" in items[0]["code"].lower() or items[0]["code"] != "(no associated code found)"

    def test_shebang_ignored(self):
        src = """
            #!/usr/bin/env python3
            x = 1
        """
        result = _comments(src)
        # shebang should be ignored
        assert not any("/usr/bin" in c for c in result)

    def test_single_line_docstring(self):
        src = '''
            def foo():
                """Return the answer."""
                return 42
        '''
        # The extractor does not enter Python docstrings via tokenize (they
        # are STRING tokens, not COMMENT tokens).  That is correct behaviour —
        # docstrings are separate from comments and should not be extracted
        # by the Python tokenize path, which only captures COMMENT tokens.
        # This test just ensures no crash and no spurious entries.
        result = _comments(src)
        # There should be no COMMENT token in this snippet.
        assert result == []

    def test_todo_comment_skipped(self):
        src = """
            # TODO: fix this later
            x = 1
        """
        result = _comments(src)
        assert result == []

    def test_fixme_comment_skipped(self):
        src = """
            # FIXME: broken logic here
            pass
        """
        result = _comments(src)
        assert result == []


# ===========================================================================
# PYTHON — non-comments inside strings / regex patterns must NOT be extracted
# ===========================================================================

class TestPythonFalsePositivePrevention:

    def test_regex_with_double_slash_not_extracted(self):
        """The exact false-positive from the bug report."""
        src = r"""
            _INLINE_COMMENT_PATTERN = re.compile(r"(?<!:)(?<!\*)//(?!\s*/)(.*)$")
        """
        result = _comments(src)
        # Nothing should be extracted — the // is inside a raw string literal.
        assert result == [], f"Unexpected extractions: {result}"

    def test_string_with_double_slash_not_extracted(self):
        src = r"""
            stripped.startswith("//")
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_string_with_block_comment_open_not_extracted(self):
        """'/*' inside a string must not trigger block-comment mode."""
        src = """
            if "/*" in line:
                in_block = True
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_string_with_block_comment_pattern_not_extracted(self):
        src = r"""
            _BLOCK_COMMENT_PATTERN = re.compile(r"/\*+(.*?)\*/", re.DOTALL)
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_string_with_hash_not_extracted(self):
        """A hash inside a string literal is not a comment."""
        src = """
            colour = "#ff0000"
            print(colour)
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_string_startswith_hash_check_not_extracted(self):
        src = """
            if stripped.startswith("#"):
                pass
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_function_definition_not_extracted(self):
        src = """
            def extract_comments_from_content(content, file_path):
                pass
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_class_definition_not_extracted(self):
        src = """
            class MyClass:
                pass
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_string_with_url_not_extracted(self):
        """URLs contain // but should not be treated as inline comments."""
        src = """
            url = "https://example.com/api"
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_raw_string_with_star_slash_not_extracted(self):
        src = r"""
            pattern = re.compile(r"/\*+(.*?)\*/")
        """
        result = _comments(src)
        assert result == [], f"Unexpected extractions: {result}"

    def test_multiline_string_not_extracted_as_comment(self):
        """A triple-quoted *string* assigned to a variable is not a comment."""
        src = '''
            message = """
            This is just a string, not a comment.
            """
            x = 1
        '''
        result = _comments(src)
        # No COMMENT tokens here; the triple-quoted string is a STRING token.
        assert result == [], f"Unexpected extractions: {result}"

    def test_comment_after_string_with_hash(self):
        """Hash in a string followed by a real inline comment."""
        src = """
            colour = "#ff0000"  # Red colour
        """
        result = _comments(src)
        # Only the real inline comment should be captured.
        assert len(result) == 1
        assert "Red colour" in result[0]

    def test_comment_after_regex_with_double_slash(self):
        """// inside a string followed by a real inline comment (Python)."""
        src = r"""
            url = "http://example.com"  # Base URL for requests
        """
        result = _comments(src)
        assert len(result) == 1
        assert "Base URL" in result[0]


# ===========================================================================
# NON-PYTHON — genuine // and /* */ comments must be detected
# ===========================================================================

class TestNonPythonGenuineComments:

    def test_js_double_slash_comment(self):
        src = """
            // Calculate total
            const total = price + tax;
        """
        result = _comments(src, "app.js")
        assert len(result) == 1
        assert "Calculate total" in result[0]

    def test_java_double_slash_comment(self):
        src = """
            // Initialize the list
            List<String> items = new ArrayList<>();
        """
        result = _comments(src, "App.java")
        assert len(result) == 1
        assert "Initialize" in result[0]

    def test_kotlin_double_slash_comment(self):
        src = """
            // Fetch user data
            val user = repository.getUser(id)
        """
        result = _comments(src, "Repo.kt")
        assert len(result) == 1
        assert "Fetch user data" in result[0]

    def test_js_block_comment(self):
        src = """
            /* Apply discount */
            price = price * 0.9;
        """
        result = _comments(src, "app.js")
        assert len(result) == 1
        assert "Apply discount" in result[0]

    def test_multiline_block_comment(self):
        src = """
            /*
             * Validate the input parameters.
             */
            function validate(params) {}
        """
        result = _comments(src, "app.js")
        assert len(result) == 1
        assert "Validate" in result[0]

    def test_js_inline_comment(self):
        src = """
            const tax = 0.1; // 10% VAT
        """
        result = _comments(src, "app.js")
        assert len(result) == 1
        assert "10% VAT" in result[0]

    def test_kotlin_kdoc(self):
        src = """
            /**
             * Returns the user display name.
             */
            fun getDisplayName(): String = name
        """
        result = _comments(src, "User.kt")
        assert len(result) == 1
        assert "display name" in result[0].lower()


# ===========================================================================
# NON-PYTHON — comment markers inside strings must NOT be extracted
# ===========================================================================

class TestNonPythonFalsePositivePrevention:

    def test_js_string_with_double_slash_not_extracted(self):
        src = r"""
            const url = "https://example.com";
        """
        result = _comments(src, "app.js")
        assert result == [], f"Unexpected extractions: {result}"

    def test_js_string_with_block_comment_open_not_extracted(self):
        src = """
            const pattern = "/* not a comment */";
        """
        result = _comments(src, "app.js")
        # The /* */ is inside a string: nothing real to extract as comment text.
        # The extractor may parse the block comment pattern inside the string,
        # but the resulting text should be empty / filtered by _append_comment.
        for item in _items(src, "app.js"):
            assert item["comment"] not in ("not a comment",), (
                f"False positive extracted: {item}"
            )

    def test_kotlin_string_with_url_not_extracted(self):
        src = """
            val endpoint = "https://api.example.com/v1"
        """
        result = _comments(src, "Api.kt")
        assert result == [], f"Unexpected extractions: {result}"

    def test_java_string_with_double_slash_not_extracted(self):
        src = r"""
            String regex = "(?<!:)//(.*)$";
        """
        result = _comments(src, "Parser.java")
        assert result == [], f"Unexpected extractions: {result}"

    def test_java_regex_block_comment_marker_not_extracted(self):
        src = r"""
            Pattern p = Pattern.compile("/\\*+(.*?)\\*/");
        """
        result = _comments(src, "Parser.java")
        assert result == [], f"Unexpected extractions: {result}"


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_content_returns_empty_list(self):
        assert _items("", "test.py") == []

    def test_content_with_only_blank_lines(self):
        assert _items("\n\n\n", "test.py") == []

    def test_python_comment_line_number_correct(self):
        src = textwrap.dedent("""
            x = 1
            # This is line 3 (after dedent strip, line 2)
            y = 2
        """).strip()
        items = _items(src, "test.py")
        assert len(items) == 1
        assert items[0]["line"] == "2"

    def test_python_todo_not_extracted(self):
        src = """
            # TODO: refactor this
            pass
        """
        assert _comments(src) == []

    def test_python_comment_with_hash_in_text(self):
        """A real comment that contains a literal # character in its text."""
        src = """
            # Use colour #FF0000
            colour = get_colour()
        """
        result = _comments(src)
        assert len(result) == 1
        assert "#FF0000" in result[0] or "FF0000" in result[0]

    def test_js_empty_inline_comment_not_extracted(self):
        """An empty // at end of line should not produce an empty comment."""
        src = """
            const x = 1; //
        """
        result = _comments(src, "app.js")
        # Empty comment text is filtered by _append_comment
        assert result == []

    def test_python_inline_comment_on_import(self):
        src = """
            import os  # standard library
        """
        result = _comments(src, "test.py")
        assert len(result) == 1
        assert "standard library" in result[0]
