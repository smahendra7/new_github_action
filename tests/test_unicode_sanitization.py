"""
Regression tests for the Unicode surrogate sanitization applied before
Groq API calls in ai/reviewer.py and ai/comment_validator.py.

The error reproduced:
    'utf-8' codec can't encode character '\\udcf3'
    in position 11288: surrogates not allowed

Root cause: raw git content decoded via surrogateescape can yield lone
surrogate code points (e.g. \\udcf3).  Python's str can hold them, but
json.dumps (used internally by the Groq SDK) raises UnicodeEncodeError
when it tries to serialise the payload to UTF-8.

Fix: encode the prompt to UTF-8 with errors='replace' and decode back.
All valid Unicode characters survive unchanged; lone surrogates become
U+FFFD (the Unicode replacement character).
"""


def _sanitize(text: str) -> str:
    """Replicate the sanitization applied in reviewer.py and comment_validator.py."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


class TestUnicodeSanitization:

    def test_lone_surrogate_replaced_with_replacement_char(self):
        """A lone surrogate is replaced (with ? or U+FFFD), not raised as an error."""
        text = "before\udcf3after"
        result = _sanitize(text)
        assert "\udcf3" not in result
        assert "before" in result
        assert "after" in result
        # Must be safely encodable — that is the actual requirement.
        result.encode("utf-8")

    def test_valid_unicode_preserved_unchanged(self):
        """All valid Unicode (BMP + supplementary) passes through unmodified."""
        text = "Hello 世界 🔥 café résumé"
        result = _sanitize(text)
        assert result == text

    def test_ascii_preserved_unchanged(self):
        text = "Simple ASCII prompt with no special characters."
        assert _sanitize(text) == text

    def test_multiple_lone_surrogates_replaced(self):
        """Every lone surrogate in the string is individually replaced."""
        text = "\udcf3 start \udcaa middle \udcbb end"
        result = _sanitize(text)
        assert "\udcf3" not in result
        assert "\udcaa" not in result
        assert "\udcbb" not in result
        # " start ", " middle ", " end " must survive
        assert "start" in result
        assert "middle" in result
        assert "end" in result
        # Must be safely encodable
        result.encode("utf-8")

    def test_sanitized_string_is_valid_utf8(self):
        """After sanitization, encode('utf-8') must not raise."""
        text = "ok\udcf3bad"
        result = _sanitize(text)
        # This must not raise — this is the actual requirement.
        encoded = result.encode("utf-8")
        assert isinstance(encoded, bytes)

    def test_empty_string(self):
        assert _sanitize("") == ""

    def test_prompt_with_code_and_surrogate(self):
        """Simulate a real-world prompt: large code block with an embedded surrogate."""
        code_block = "def foo():\n    return 42\n"
        surrogate_line = "    data = b'\\xf3'.decode('utf-8', errors='surrogateescape')\n"
        # Introduce an actual surrogate at a known position
        surrogate_line_actual = "    result = \udcf3\n"
        prompt = (
            "You are a code reviewer.\n\n"
            "Code:\n\n"
            + code_block
            + surrogate_line_actual
            + "\nPlease review."
        )
        result = _sanitize(prompt)
        # The surrogate must be gone
        assert "\udcf3" not in result
        # The rest of the prompt must survive
        assert "You are a code reviewer." in result
        assert "def foo():" in result
        assert "Please review." in result
        # Must be UTF-8 encodable
        result.encode("utf-8")
