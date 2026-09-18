"""
suggestion_generator.py

Generates AI-powered improvement suggestions for PR validation warnings.

Two public functions:
    generate_description_suggestion(pr_title, pr_body, diff)
        -> str   (a suggested PR description)

    generate_comment_suggestion(file, line, comment, code, reason)
        -> str   (a suggested corrected comment)

Both call Groq. On any failure (API error, timeout, parse error) they
return a safe human-readable fallback string so the caller can always
display *something* useful without hard-failing the pipeline.
"""

import os
import re

from groq import Groq

_MAX_DIFF_CHARS = 8000  # Keep prompts manageable


def _truncate_diff(diff: str) -> str:
    """Truncate diff to avoid exceeding token limits."""
    if not diff:
        return ""
    if len(diff) <= _MAX_DIFF_CHARS:
        return diff
    return diff[:_MAX_DIFF_CHARS] + "\n\n[... diff truncated for length ...]"


def _get_client() -> Groq | None:
    """Return a Groq client if GROQ_API_KEY is available, else None."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[suggestion_generator] GROQ_API_KEY not set — skipping AI suggestion.")
        return None
    return Groq(api_key=api_key)


def generate_description_suggestion(
    pr_title: str,
    pr_body: str,
    diff: str,
) -> str:
    """
    Generate a suggested PR description based on the PR title, the existing
    description (if any), and the actual code diff.

    Returns a plain-text suggested description string.
    Falls back to a safe placeholder on any error.
    """
    truncated_diff = _truncate_diff(diff)

    prompt = f"""You are a helpful Pull Request description assistant.

The developer has submitted a PR whose description needs improvement.
Based on the PR title, the existing (possibly empty or unclear) description,
and the actual code changes shown in the diff, write a clear and concise
PR description that:
- Explains the overall purpose and intent of this PR in plain language.
- Summarises what was changed at a high level (no need to list every file).
- Is accurate to the actual code changes shown.
- Is written as a developer would write it (first or third person, professional tone).
- Is 2–5 sentences long — no bullet lists unless naturally appropriate.

PR Title: {pr_title or "(no title)"}

Existing PR Description (may be empty, vague, or inaccurate):
{pr_body or "(no description provided)"}

Code Diff (actual changes):
{truncated_diff or "(no diff available)"}

Respond with ONLY the suggested PR description text — no preamble, no labels,
no "Here is the suggested description:" prefix. Just the description itself."""

    try:
        client = _get_client()
        if not client:
            return (
                "Please provide a clear description explaining the purpose "
                "and main changes of this PR."
            )

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            top_p=1,
            stream=False,
        )

        suggestion = (
            completion.choices[0].message.content or ""
        ).strip()

        # Strip any accidental "Suggested description:" prefix the model might add
        suggestion = re.sub(
            r"^(?:suggested\s+(?:pr\s+)?description\s*:?\s*)",
            "",
            suggestion,
            flags=re.IGNORECASE,
        ).strip()

        print("\n===== DESCRIPTION SUGGESTION GENERATED =====")
        print(suggestion)
        print("============================================\n")

        return suggestion if suggestion else (
            "Please provide a clear description explaining the purpose "
            "and main changes of this PR."
        )

    except Exception as e:
        print(f"[suggestion_generator] Description suggestion error: {e}. Using fallback.")
        return (
            "Please provide a clear description explaining the purpose "
            "and main changes of this PR."
        )


def generate_comment_suggestion(
    file: str,
    line: str,
    comment: str,
    code: str,
    reason: str,
) -> str:
    """
    Generate a suggested corrected comment for a code comment that does not
    accurately describe the associated code.

    Returns a plain-text suggested comment string (the comment text only,
    not the full comment syntax with # or //).
    Falls back to a safe placeholder on any error.
    """
    prompt = f"""You are a helpful code comment improvement assistant.

A code comment has been identified as inaccurate or misleading.
Write a corrected comment that accurately describes what the associated code does.

File: {file or "(unknown)"}
Line: {line or "(unknown)"}
Original (inaccurate) comment: "{comment or "(no comment text)"}"
Associated code: {code or "(no code snippet)"}
Reason the original comment is wrong: {reason or "(no reason given)"}

Rules:
- Write ONLY the replacement comment text (the words inside the comment, not the #, //, or /* markers).
- Be concise and accurate — describe what the code actually does.
- Match the style and detail level of the original comment.
- Do not add any explanation or preamble.

Respond with ONLY the corrected comment text."""

    try:
        client = _get_client()
        if not client:
            return f"Accurately describes what the code at line {line} does."

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150,
            top_p=1,
            stream=False,
        )

        suggestion = (
            completion.choices[0].message.content or ""
        ).strip()

        # Strip quotes if the model wrapped the suggestion
        suggestion = suggestion.strip('"\'')

        print(f"\n===== COMMENT SUGGESTION GENERATED (line {line}) =====")
        print(suggestion)
        print("======================================================\n")

        return suggestion if suggestion else (
            f"Accurately describes what the code at line {line} does."
        )

    except Exception as e:
        print(
            f"[suggestion_generator] Comment suggestion error for line {line}: {e}. "
            "Using fallback."
        )
        return f"Accurately describes what the code at line {line} does."
