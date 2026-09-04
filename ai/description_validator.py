import os
import re
from groq import Groq

_PLACEHOLDER_VALUES = {
    "test", "asdf", "qwerty", "n/a", "todo", "fix",
    "update", "fix bug", "done", "fixed", "changes", "wip",
    "no description", "tbd", "none", "na", "lorem ipsum"
}

_CONVENTIONAL_TITLE_PREFIX = re.compile(
    r"^(?:fix|feat|feature|chore|docs|refactor|test|ci|build|style|perf|revert)"
    r"(?:\([^)]+\))?!?\s*:?\s*",
    re.IGNORECASE
)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"<!--.*?-->", "", text or "", flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _word_count(text: str) -> int:
    return len(_normalize_text(text).split())


def _is_obvious_placeholder(text: str) -> bool:
    cleaned = _normalize_text(text).lower()
    if not cleaned:
        return False
    if cleaned in _PLACEHOLDER_VALUES:
        return True
    words = cleaned.split()
    return len(words) == 1 and words[0] in _PLACEHOLDER_VALUES


def _is_obviously_meaningful(text: str) -> bool:
    cleaned = _normalize_text(text)
    if not cleaned or _is_obvious_placeholder(cleaned):
        return False
    return _word_count(cleaned) >= 4


def _title_without_conventional_prefix(title: str) -> str:
    return _CONVENTIONAL_TITLE_PREFIX.sub("", title or "").strip()


def _build_validation_content(title: str, body: str) -> tuple[str, str]:
    pr_title = _normalize_text(title)
    pr_body = _normalize_text(body)

    if pr_body:
        return pr_title, pr_body

    title_detail = _title_without_conventional_prefix(pr_title)
    if _is_obviously_meaningful(title_detail) or _is_obviously_meaningful(pr_title):
        return pr_title, pr_title

    return pr_title, ""


def validate_pr_description(
    title: str,
    body: str
) -> tuple[bool, str]:

    pr_title, pr_body = _build_validation_content(title, body)

    print("\n===== PR DESCRIPTION VALIDATION GATE =====")
    print(f"Title: {pr_title!r}")
    print(f"Body : {pr_body!r}")

    if not pr_body:
        print("Validation Result: FAIL (description is empty)")
        return (
            False,
            "PR description is empty — please describe what this PR does."
        )

    if _is_obvious_placeholder(pr_body):
        print("Validation Result: FAIL (placeholder text)")
        return (
            False,
            f"PR description '{pr_body}' is placeholder or meaningless text."
        )

    prompt = f"""You are a Pull Request description validator. Your job is to decide whether the author explained what this PR does.

PR Title (context only — do NOT fail just because the title is short or uses conventional prefixes like "fix:", "feat:", or "test:"):
{pr_title if pr_title else "(no title)"}

PR Description (this is what you must evaluate):
{pr_body}

Rules — apply ONLY to the PR Description above:
1. FAIL if the description is empty, whitespace-only, or an unfilled template with no real content.
2. FAIL if the description is gibberish, random characters, or keyboard mashing (e.g. "asdf", "qwerty").
3. FAIL if the entire description is a single meaningless placeholder word or phrase (e.g. "test", "N/A", "todo", "wip", "fix bug").
4. FAIL if the description does not contain coherent sentence(s) explaining what the PR actually changes.
5. PASS if the description contains one or more meaningful sentences that explain what the PR does, even if brief.

Examples:
- Description: "Added JWT authentication and updated the user schema." -> Verdict: PASS
- Description: "This PR refactors the payment module to reduce duplicate validation logic." -> Verdict: PASS
- Description: "asdf" -> Verdict: FAIL
- Description: "fix bug" -> Verdict: FAIL
- Description: "test" -> Verdict: FAIL

Respond in exactly this two-line format with no markdown:
Verdict: PASS or FAIL
Reason: <one short sentence>
"""

    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[WARNING] GROQ_API_KEY not found. Using heuristic validation.")
            return _local_heuristic_validation(pr_title, pr_body)

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

        print("\n===== VALIDATION LLM RESPONSE =====")
        print(response_text)
        print("===================================\n")

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

        print("[WARNING] Could not parse LLM response. Using heuristic validation.")
        return _local_heuristic_validation(pr_title, pr_body)

    except Exception as e:
        print(f"PR Description Validation Error: {e}. Using heuristic validation.")
        return _local_heuristic_validation(pr_title, pr_body)


def _local_heuristic_validation(pr_title: str, pr_body: str) -> tuple[bool, str]:
    title_clean = _normalize_text(pr_title)
    body_clean = _normalize_text(pr_body)

    if body_clean:
        if _is_obvious_placeholder(body_clean):
            return False, f"PR description '{body_clean}' is placeholder or meaningless text."
        if _is_obviously_meaningful(body_clean):
            return True, "PR description explains the changes."
        return False, f"PR description '{body_clean}' is too short — add a clear explanation."

    if title_clean:
        title_detail = _title_without_conventional_prefix(title_clean)
        if _is_obviously_meaningful(title_detail) or _is_obviously_meaningful(title_clean):
            return True, "PR title provides a clear explanation of the changes."
        if _is_obvious_placeholder(title_detail) or _is_obvious_placeholder(title_clean):
            return False, f"PR description is missing and title '{title_clean}' is not descriptive enough."
        return False, f"PR description is empty for PR titled '{title_clean}'. Please add a detailed description."

    return False, "PR title and description are both missing."
