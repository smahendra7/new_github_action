
import os
import re

from openai import OpenAI

# Groq is inactive. Keep this import commented so the Groq package
# is not required when NVIDIA is the active provider.
# from groq import Groq


_PLACEHOLDER_VALUES = {
    "test", "asdf", "qwerty", "n/a", "todo", "fix",
    "update", "fix bug", "done", "fixed", "changes", "wip",
    "no description", "tbd", "none", "na", "lorem ipsum"
}

_CONVENTIONAL_TITLE_PREFIX = re.compile(
    r"^(?:fix|feat|feature|chore|docs|refactor|test|ci|build|style|perf|revert)"
    r"(?:\([^)]*\))?!?\s*:\s*",
    re.IGNORECASE
)


def _normalize_text(text: str) -> str:
    """Remove HTML comments and normalize whitespace."""
    cleaned = re.sub(r"<!--.*?-->", "", text or "", flags=re.DOTALL)
    return re.sub(r"\s+", " ", cleaned).strip()


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


def _build_validation_content(
    title: str,
    body: str
) -> tuple[str, str]:
    pr_title = _normalize_text(title)
    pr_body = _normalize_text(body)

    return pr_title, pr_body


def _parse_llm_response(response_text: str) -> tuple[bool, str] | None:
    """
    Parse the model's verdict and reason.

    Accepts minor formatting differences, including Markdown formatting.
    A valid verdict is sufficient even if the reason is missing.
    """
    if not response_text:
        return None

    verdict_match = re.search(
        r"\bVerdict\s*:\s*\**\s*(PASS|FAIL)\b",
        response_text,
        re.IGNORECASE
    )

    if not verdict_match:
        return None

    verdict = verdict_match.group(1).upper()

    reason_match = re.search(
        r"\bReason\s*:\s*(.*)",
        response_text,
        re.IGNORECASE | re.DOTALL
    )

    if reason_match:
        reason = reason_match.group(1).strip()
        reason = re.sub(r"^\*+|\*+$", "", reason).strip()
        reason = reason.splitlines()[0].strip() if reason else ""
    else:
        reason = ""

    if not reason:
        reason = (
            "The PR description explains the changes."
            if verdict == "PASS"
            else "The PR description does not adequately explain the changes."
        )

    return verdict == "PASS", reason


def validate_pr_description(
    title: str,
    body: str
) -> tuple[bool, str]:

    pr_title, pr_body = _build_validation_content(title, body)

    print("\n===== PR DESCRIPTION VALIDATION GATE =====")
    print(f"Title: {pr_title!r}")
    print(f"Body : {pr_body!r}")

    # Deterministic checks for empty descriptions and obvious placeholders.
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

    prompt = f"""You are a Pull Request description validator.
Your job is to decide whether the author explained what this PR does.

PR Title (context only — do NOT fail just because the title is short
or uses conventional prefixes like "fix:", "feat:", or "test:"):
{pr_title if pr_title else "(no title)"}

PR Description (this is what you must evaluate):
{pr_body}

Rules — apply ONLY to the PR Description above:
1. FAIL if the description is empty, whitespace-only, or an unfilled
   template with no real content.
2. FAIL if the description is gibberish, random characters, or keyboard
   mashing (e.g. "asdf", "qwerty").
3. FAIL if the entire description is a single meaningless placeholder
   word or phrase (e.g. "test", "N/A", "todo", "wip", "fix bug").
4. FAIL if the description does not contain coherent sentences explaining
   what the PR actually changes.
5. PASS if the description contains one or more meaningful sentences
   explaining what the PR does, even if brief.
6. Judge the description's meaning, not its grammar or formatting.
7. Do not require a particular number of bullet points or a fixed template.

Examples:
Description: "Added JWT authentication and updated the user schema."
Verdict: PASS
Reason: The description identifies the changes made.

Description: "This PR refactors the payment module to reduce duplicate validation logic."
Verdict: PASS
Reason: The description explains the purpose of the refactoring.

Description: "asdf"
Verdict: FAIL
Reason: The description is meaningless text.

Description: "fix bug"
Verdict: FAIL
Reason: The description does not explain what was changed.

Respond in exactly this two-line format with no Markdown:
Verdict: PASS or FAIL
Reason: One short sentence
"""

    try:
        api_key = os.environ.get("NVIDIA_API_KEY")

        if not api_key:
            print(
                "[WARNING] NVIDIA_API_KEY not found. "
                "Using heuristic validation."
            )
            return _local_heuristic_validation(pr_title, pr_body)

        # NVIDIA OpenAI-compatible client (active provider).
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )

        # Groq client reference (inactive).
        # groq_client = Groq(
        #     api_key=os.environ.get("GROQ_API_KEY")
        # )

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=512,
            top_p=1,
            stream=False
        )

        if not completion.choices:
            print("[WARNING] NVIDIA returned no completion choices.")
            print("Usage:", completion.usage)
            return _local_heuristic_validation(pr_title, pr_body)

        choice = completion.choices[0]
        message = choice.message
        response_text = (message.content or "").strip()

        print("\n===== VALIDATION LLM RESPONSE =====")
        print(repr(response_text))
        print("Finish reason:", choice.finish_reason)
        print("Usage:", completion.usage)

        # Diagnostic information for empty or unusable responses.
        if not response_text:
            reasoning = getattr(message, "reasoning", None)

            print("[WARNING] NVIDIA returned empty message content.")
            print("Reasoning field:", repr(reasoning))

        print("===================================\n")

        parsed_result = _parse_llm_response(response_text)

        if parsed_result is not None:
            is_valid, reason = parsed_result

            print(
                f"[VALIDATION] PR Description quality: "
                f"{'PASS' if is_valid else 'FAIL'}"
            )
            print(f"[VALIDATION] Reason: {reason}")

            return is_valid, reason

        print(
            "[WARNING] Could not parse NVIDIA response. "
            "Using heuristic validation."
        )

        return _local_heuristic_validation(pr_title, pr_body)

    except Exception as e:
        print(
            f"[WARNING] PR Description validation API error: "
            f"{type(e).__name__}: {e}"
        )
        print("Using heuristic validation.")

        return _local_heuristic_validation(pr_title, pr_body)


def _local_heuristic_validation(
    pr_title: str,
    pr_body: str
) -> tuple[bool, str]:
    """Fallback validation when the AI response cannot be used."""

    title_clean = _normalize_text(pr_title)
    body_clean = _normalize_text(pr_body)

    if body_clean:
        if _is_obvious_placeholder(body_clean):
            return (
                False,
                f"PR description '{body_clean}' is placeholder "
                "or meaningless text."
            )

        if _is_obviously_meaningful(body_clean):
            return True, "PR description explains the changes."

        return (
            False,
            f"PR description '{body_clean}' is too short — "
            "add a clear explanation."
        )

    if title_clean:
        return (
            False,
            "The PR description is empty and does not explain the changes."
        )

    return False, "PR title and description are both missing."
