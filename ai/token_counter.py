"""
ai/token_counter.py

Token counting utility for AI Code Review prompt sizing.
Uses tiktoken (cl100k_base) when available, and provides a reliable
code-aware fallback tokenizer if tiktoken is not present.
"""

import re
from typing import Optional

_ENCODER = None
_TIKTOKEN_INITIALIZED = False


def _get_encoder():
    global _ENCODER, _TIKTOKEN_INITIALIZED
    if not _TIKTOKEN_INITIALIZED:
        _TIKTOKEN_INITIALIZED = True
        try:
            import tiktoken
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the given text string.
    Represents the complete input sent to the model.

    Uses tiktoken cl100k_base when available. If tiktoken is not installed
    or encounters an error, falls back to a code-aware subword estimator.
    """
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    # Fallback token estimator:
    # Subword/BPE regex that splits on code identifiers, symbols, and whitespace.
    # Code with indentation and line numbers averages ~2.8-3.5 chars per token.
    pieces = re.findall(r"[a-zA-Z0-9_]+|[^\s\w]|\n|\s+", text)
    count = 0
    for p in pieces:
        if p.isspace():
            count += max(1, len(p) // 4)
        elif len(p) > 4:
            count += (len(p) + 3) // 4
        else:
            count += 1
    return max(1, count)
