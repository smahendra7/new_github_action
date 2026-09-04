import os
import re
from groq import Groq

# Maximum diff size (in characters) sent to the LLM.
# Stays consistent with the size guard applied in ai/reviewer.py (max_tokens=2048)
# and the way large inputs are handled elsewhere in the codebase.
_MAX_DIFF_CHARS = 12000


def _truncate_diff(diff: str) -> str:
    """Truncate the diff if it exceeds _MAX_DIFF_CHARS, appending a notice."""
    if len(diff) <= _MAX_DIFF_CHARS:
        return diff
    return (
        diff[:_MAX_DIFF_CHARS]
        + "\n\n[... diff truncated for length ...]"
    )


def validate_description_matches_code(
    pr_title: str,
    pr_body: str,
    diff: str
) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Checks whether the PR description's claimed changes are actually
    reflected in the diff. This assumes pr_body has ALREADY passed the
    separate "is this meaningful text" gate — this function only checks
    truthfulness/accuracy against the diff, not meaningfulness.
    """

    truncated_diff = _truncate_diff(diff or "")

    prompt = f"""You are a Pull Request accuracy validator. Your job is to decide whether the PR description accurately reflects what is shown in the diff.

PR Title (context only):
{pr_title or "(no title)"}

PR Description (this is what you must evaluate for accuracy):
{pr_body or "(no description)"}

Diff (the actual code changes):
{truncated_diff or "(no diff available)"}

Rules — apply ONLY to whether the description's CLAIMS are supported by the diff:
1. FAIL only when the description makes a specific claim that the diff does NOT support.
   Examples of failing cases:
   - Description says "added input validation" but no validation logic appears in the diff.
   - Description says "renamed variable X to Y" but the diff shows an unrelated change.
   - Description says "fixed login bug" but the diff only changes a completely unrelated module with no plausible connection to login.
2. PASS if the description's main claimed change is plausibly reflected in the diff, even if:
   - The description is broader or higher-level than the diff (e.g. "fixed login bug" is fine if the diff touches login-related code).
   - The description omits minor incidental changes such as formatting, imports, or whitespace.
   - The description is accurate but less detailed than the code.
   - The description does not mention every file or every change in the diff.
3. PASS if the diff is empty or unavailable — do not block PRs when there is nothing to compare against.
4. Do NOT fail just because the description is short, generic, or could be written more clearly — only fail for outright inaccurate claims.

Respond in exactly this two-line format with no markdown:
Verdict: PASS or FAIL
Reason: <one short sentence>
"""

    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print(
                "[WARNING] GROQ_API_KEY not found. "
                "Skipping PR description-vs-code match validation gate."
            )
            return True, "GROQ_API_KEY unavailable — gate skipped."

        client = Groq(api_key=api_key)

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=150,
            top_p=1,
            stream=False
        )

        response_text = (
            completion.choices[0].message.content or ""
        ).strip()

        print("\n===== DESCRIPTION-VS-CODE MATCH LLM RESPONSE =====")
        print(response_text)
        print("===================================================\n")

        verdict_match = re.search(
            r"Verdict\s*:\s*(PASS|FAIL)",
            response_text,
            re.IGNORECASE
        )
        reason_match = re.search(
            r"Reason\s*:\s*(.+)",
            response_text,
            re.IGNORECASE | re.DOTALL
        )

        if verdict_match and reason_match:
            verdict = verdict_match.group(1).upper()
            reason = reason_match.group(1).strip()
            reason = reason.splitlines()[0].strip()
            reason = re.sub(r"^\*+|\*+$", "", reason).strip()
            is_valid = (verdict == "PASS")
            return is_valid, reason

        # If we can't parse the response, fail open — don't block the PR.
        print(
            "[WARNING] Could not parse LLM response for description-vs-code "
            "match gate. Failing open."
        )
        return True, "Could not parse LLM response — gate skipped."

    except Exception as e:
        print(
            f"[WARNING] PR Description-vs-Code Match Validation Error: {e}. "
            "Failing open — allowing review to continue."
        )
        return True, f"Validation error ({e}) — gate skipped."
