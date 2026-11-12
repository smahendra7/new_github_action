import os
import re

from openai import OpenAI

from ai.context_config import (
    SAFE_INPUT_TOKEN_LIMIT,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
)

from ai.nvidia_request_lock import execute_nvidia_request

from ai.token_counter import count_tokens


# NVIDIA model settings. The input budget comes from the shared code-review
# configuration and already leaves room below the endpoint's context limit.
_MAX_OUTPUT_TOKENS = 512


def _normalize_text(text: str) -> str:
    """Normalize whitespace for consistent response handling."""
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_llm_response(
    response_text: str,
) -> tuple[bool, str] | None:
    """Parse PASS/FAIL and an optional reason from the model response."""

    if not response_text:
        return None

    verdict_match = re.search(
        r"\bVerdict\s*:\s*\**\s*(PASS|FAIL)\b",
        response_text,
        re.IGNORECASE,
    )

    if not verdict_match:
        return None

    verdict = verdict_match.group(1).upper()

    reason_match = re.search(
        r"\bReason\s*:\s*(.*)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )

    reason = ""
    if reason_match:
        reason = reason_match.group(1).strip()
        reason = re.sub(r"^\*+|\*+$", "", reason).strip()
        reason = reason.splitlines()[0].strip() if reason else ""

    if not reason:
        reason = (
            "The description's claims are supported by the diff."
            if verdict == "PASS"
            else "The description contains claims not supported by the diff."
        )

    return verdict == "PASS", reason


def _build_prompt(
    pr_title: str,
    pr_body: str,
    diff: str,
    file_path: str = "",
    associated_code: str = "",
    chunk_number: int = 1,
    total_chunks: int = 1,
) -> str:
    """Build the exact prompt whose tokens are measured before sending."""
    context_section = ""
    if file_path:
        context_section += f"\nFile path/context:\n{file_path}\n"

    if associated_code.strip():
        context_section += (
            "\nSupporting code context (for understanding only):\n"
            f"{associated_code}\n"
        )

    chunk_note = (
        "\nThe diff supplied below may be only part of a larger PR diff. Do not assume that changes not shown here are absent from the complete PR.\n"
    )
    if total_chunks > 1:
        chunk_note += (
            f"This is diff chunk {chunk_number} of {total_chunks}. "
            "Do not fail a PR-level claim solely because it is not implemented in this chunk; it may be evidenced in another chunk. "
            "Fail only when changes in this chunk provide material evidence that contradicts a relevant claim or clearly show that the described behavior is not implemented in the relevant changed code.\n"
        )

    return f"""You are a strict but fair Pull Request description accuracy
validator.

Your task is to determine whether the developer's claims in the PR
description are supported by the actual code changes.

PR title (context only):
{pr_title or "(no title)"}

PR description:
{pr_body}

{context_section}
{chunk_note}
Actual code diff (authoritative evidence of changes):
{diff}

Rules:
1. Identify the specific material changes claimed in the PR description.
2. Compare each material claim with the actual code diff supplied in this request.
3. PASS when the main claims are supported by the diff and no relevant contradiction is found.
4. FAIL when a material claim is contradicted by the diff or the claimed implementation is clearly absent from relevant changes shown here.
5. Do not assume that a feature exists merely because the description says it exists.
6. Do not FAIL just because the description omits minor changes, uses different wording, or does not mention every changed file.
7. Consider related changes across the supplied diff and file context.
8. Do not invent implementation details or infer unsupported behavior.
9. If the diff does not provide enough evidence to judge a claim, do not fabricate a successful comparison; explain the limitation in the reason.
10. Do not review unrelated bugs, code quality, or style.
11. When this is one of several chunks, do not treat changes outside this chunk as absent from the complete PR. Apply the chunk note above.

Examples:

Description: "Added swipe-up and swipe-down gesture handling."
Diff: Implements both swipe-up and swipe-down handling.
Verdict: PASS
Reason: Both described gestures are implemented in the diff.

Description: "Added swipe-up and swipe-down gesture handling."
Diff: Implements only swipe-up handling, with no swipe-down implementation in the supplied relevant changes.
Verdict: FAIL
Reason: The supplied diff does not support the swipe-down claim.

Description: "Added input validation."
Diff: Changes an unrelated UI color without adding validation, and this is the complete relevant diff.
Verdict: FAIL
Reason: The diff does not support the input-validation claim.

Respond in exactly this format:
Verdict: PASS or FAIL
Reason: One short sentence
"""


def _split_diff_sections(diff: str) -> list[str]:
    """Group a unified diff into file sections without dropping any lines."""
    sections: list[str] = []
    current: list[str] = []

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            sections.append("".join(current))
            current = []
        current.append(line)

    if current:
        sections.append("".join(current))

    return sections


def _split_large_file_section(
    section: str,
    prompt_for_diff,
    safe_budget: int,
) -> list[str] | None:
    """Split a single large file diff by lines, repeating its file/hunk context.

    Returns None if even a single line plus required prompt context cannot fit.
    All original diff lines are retained; file and hunk headers may be repeated
    in separate chunks to keep each chunk understandable on its own.
    """
    lines = section.splitlines(keepends=True)
    first_hunk = next(
        (index for index, line in enumerate(lines) if line.startswith("@@")),
        None,
    )

    # Binary/rename-only sections normally fit by themselves. If an unusually
    # large non-hunk section cannot fit, we cannot split it safely as code.
    if first_hunk is None:
        return None

    file_header = lines[:first_hunk]
    body = lines[first_hunk:]
    current_body: list[str] = []
    active_hunk_header = ""
    output: list[str] = []

    for line in body:
        if line.startswith("@@"):
            active_hunk_header = line

        candidate_body = current_body + [line]
        candidate_diff = "".join(file_header + candidate_body)

        if count_tokens(prompt_for_diff(candidate_diff)) <= safe_budget:
            current_body = candidate_body
            continue

        if current_body:
            output.append("".join(file_header + current_body))
            # Preserve the latest hunk header when the split happens inside a
            # hunk; a new hunk line is already included as the current line.
            if line.startswith("@@"):
                current_body = [line]
            elif active_hunk_header:
                current_body = [active_hunk_header, line]
            else:
                current_body = [line]
        else:
            # No safe split point exists yet. Try this line with required
            # context and return SKIPPED if it still exceeds the budget.
            current_body = [line]

        current_diff = "".join(file_header + current_body)
        if count_tokens(prompt_for_diff(current_diff)) > safe_budget:
            return None

    if current_body:
        output.append("".join(file_header + current_body))

    return output


def _chunk_diff(diff: str, prompt_for_diff, safe_budget: int) -> list[str] | None:
    """Pack whole files when possible; split oversized file diffs by lines."""
    chunks: list[str] = []
    current_chunk = ""

    for section in _split_diff_sections(diff):
        candidate = current_chunk + section
        if count_tokens(prompt_for_diff(candidate)) <= safe_budget:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        if count_tokens(prompt_for_diff(section)) <= safe_budget:
            current_chunk = section
            continue

        section_chunks = _split_large_file_section(
            section,
            prompt_for_diff,
            safe_budget,
        )
        if not section_chunks:
            return None
        chunks.extend(section_chunks)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks or None


def validate_description_matches_code(
    pr_title: str,
    pr_body: str,
    diff: str,
    file_path: str = "",
    associated_code: str = "",
) -> tuple[bool, str]:
    """Validate one combined PR diff, directly or in token-budgeted chunks.

    The public return shape remains ``(is_valid, reason)`` for compatibility
    with main.py. The reason starts with PASS, FAIL, SKIPPED, or ERROR so the
    caller can distinguish a mismatch from an incomplete validation.
    """
    pr_title = _normalize_text(pr_title)
    pr_body = _normalize_text(pr_body)
    diff = diff or ""
    associated_code = associated_code or ""
    target_name = file_path or "PR diff"

    print("\n===== DESCRIPTION-VS-CODE VALIDATION =====")
    print(f"Target: {target_name}")
    print(f"Description length: {len(pr_body)} characters")
    print(f"Combined diff length: {len(diff)} characters")

    # Description quality is checked separately by description_validator.py.
    if not pr_body:
        return False, "SKIPPED: PR description is empty."

    if not diff.strip():
        return False, "SKIPPED: No code diff is available for comparison."

    if not os.environ.get("NVIDIA_API_KEY"):
        print("[WARNING] NVIDIA_API_KEY is missing.")
        return False, "SKIPPED: NVIDIA_API_KEY is unavailable."

    # The prompt factory is the single source of truth for both token counting
    # and API input, so title, description, rules, context, and diff are counted.
    def prompt_for_diff(
        chunk_diff: str,
        chunk_number: int = 1,
        total_chunks: int = 1,
    ) -> str:
        return _build_prompt(
            pr_title,
            pr_body,
            chunk_diff,
            file_path=file_path,
            associated_code=associated_code,
            chunk_number=chunk_number,
            total_chunks=total_chunks,
        )

    full_prompt = prompt_for_diff(diff)
    full_prompt_tokens = count_tokens(full_prompt)
    print(
        f"Complete validation prompt: {full_prompt_tokens} tokens; "
        f"safe input budget: {SAFE_INPUT_TOKEN_LIMIT}"
    )

    prompt_base_tokens = count_tokens(prompt_for_diff(""))
    if prompt_base_tokens >= SAFE_INPUT_TOKEN_LIMIT:
        return (
            False,
            "SKIPPED: PR title/description and validation instructions exceed "
            "the safe input token budget before adding the diff.",
        )

    if full_prompt_tokens <= SAFE_INPUT_TOKEN_LIMIT:
        diff_chunks = [diff]
    else:
        # Reserve room for the chunk-number annotation added to each final
        # prompt after the number of chunks is known.
        chunk_annotation_margin = count_tokens(
            "This is diff chunk 999 of 999. Do not fail a PR-level claim solely because it is not implemented in this chunk."
        ) + 32
        chunking_budget = SAFE_INPUT_TOKEN_LIMIT - chunk_annotation_margin
        if chunking_budget <= prompt_base_tokens:
            return (
                False,
                "SKIPPED: Not enough safe input budget remains for diff content "
                "and chunk metadata.",
            )
        diff_chunks = _chunk_diff(diff, prompt_for_diff, chunking_budget)
        if not diff_chunks:
            return (
                False,
                "SKIPPED: The combined diff could not be split into chunks "
                "that fit the safe input token budget.",
            )

        print(f"Large PR diff split into {len(diff_chunks)} chunk(s).")

    total_chunks = len(diff_chunks)
    prompts = [
        prompt_for_diff(chunk, index, total_chunks)
        for index, chunk in enumerate(diff_chunks, start=1)
    ]

    # The chunk note adds prompt tokens; validate the final, exact prompts
    # before making any API request.
    final_token_counts = [count_tokens(prompt) for prompt in prompts]
    oversized_chunks = [
        index
        for index, token_count in enumerate(final_token_counts, start=1)
        if token_count > SAFE_INPUT_TOKEN_LIMIT
    ]
    if oversized_chunks:
        return (
            False,
            "SKIPPED: Final validation prompt(s) exceed the safe token budget "
            f"for chunk(s) {oversized_chunks}.",
        )

    try:
        client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=os.environ["NVIDIA_API_KEY"],
            max_retries=0,
        )
    except Exception as exc:
        print(
            "[WARNING] Could not initialize NVIDIA client: "
            f"{type(exc).__name__}: {exc}"
        )
        return False, f"ERROR: Could not initialize NVIDIA client ({type(exc).__name__})."

    chunk_results: list[tuple[int, bool, str]] = []
    chunk_errors: list[str] = []

    for index, prompt in enumerate(prompts, start=1):
        print(
            f"Validating Description-vs-Code chunk {index}/{total_chunks} "
            f"({final_token_counts[index - 1]} input tokens)"
        )

        try:
            completion = execute_nvidia_request(
                lambda: client.chat.completions.create(
                    model=NVIDIA_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=_MAX_OUTPUT_TOKENS,
                    top_p=1,
                    stream=False,
                )
            )
        except Exception as exc:
            print(
                f"[WARNING] NVIDIA validation error for chunk {index}: "
                f"{type(exc).__name__}: {exc}"
            )
            chunk_errors.append(
                f"Chunk {index}: API/processing error ({type(exc).__name__})."
            )
            # Continue so a later chunk can still reveal a genuine mismatch.
            continue

        if not completion.choices:
            print(f"[WARNING] NVIDIA returned no completion choices for chunk {index}.")
            print("Usage:", getattr(completion, "usage", None))
            chunk_errors.append(f"Chunk {index}: NVIDIA returned no completion choices.")
            continue

        choice = completion.choices[0]
        response_text = (choice.message.content or "").strip()

        print("\n===== DESCRIPTION-VS-CODE MATCH LLM RESPONSE =====")
        print(repr(response_text))
        print("Finish reason:", getattr(choice, "finish_reason", None))
        print("Usage:", getattr(completion, "usage", None))
        print("===================================================\n")

        parsed_result = _parse_llm_response(response_text)
        if parsed_result is None:
            print(
                f"[WARNING] NVIDIA response for chunk {index} was empty or "
                "did not contain a recognized PASS/FAIL verdict."
            )
            chunk_errors.append(
                f"Chunk {index}: NVIDIA returned an empty or unrecognized verdict."
            )
            continue

        is_valid, reason = parsed_result
        chunk_results.append((index, is_valid, reason))
        status = "PASS" if is_valid else "FAIL"
        print(f"[VALIDATION] Description-vs-code chunk {index}: {status}")
        print(f"[VALIDATION] Reason: {reason}")

    # A confirmed mismatch takes precedence over an incomplete chunk, because
    # a failed chunk is already sufficient to make the PR validation FAIL.
    failures = [
        (index, reason)
        for index, is_valid, reason in chunk_results
        if not is_valid
    ]
    if failures:
        reason = "; ".join(
            f"Chunk {index}: {chunk_reason}"
            for index, chunk_reason in failures
        )
        if chunk_errors:
            reason += "; additional incomplete chunks: " + "; ".join(chunk_errors)
        return False, f"FAIL: {reason}"

    if chunk_errors:
        return False, "ERROR: Validation incomplete: " + "; ".join(chunk_errors)

    if len(chunk_results) != total_chunks:
        return False, "ERROR: Not all required diff chunks were validated."

    print(
        f"[VALIDATION] Description vs code: PASS "
        f"({total_chunks} chunk(s) checked; no mismatch reported)."
    )
    return True, f"PASS: All {total_chunks} validation chunk(s) were checked; no mismatch found."
