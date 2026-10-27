"""
ai/result_aggregator.py

Aggregates multiple chunk review responses and deduplicates findings based
on normalized (File, Line) pairs while preserving existing formatting.

A review is reported as clean only when every result completed successfully
and explicitly states that no issues were found. Failed, incomplete, empty,
or ambiguous results are disclosed in Review Notices.
"""

import re
from typing import Any, Dict, List, Tuple, Union


def _bold_labels(text: str) -> str:
    """
    Ensure target labels are bolded outside code blocks:
    - **Issue:**
    - **File:**
    - **Line:**
    - **Code:**
    - **Reason:**
    - **Suggestion:**
    """
    if not text:
        return ""

    parts = re.split(r"(```[\s\S]*?```)", text)
    pattern = re.compile(
        r"(?m)^(\s*(?:💡\s*|\*\s*)?)(?!\*\*)"
        r"(Issue|File|Line|Comment|Code|Reason|Suggestion):"
    )

    for i in range(0, len(parts), 2):
        parts[i] = pattern.sub(r"\1**\2:**", parts[i])

    return "".join(parts)


def normalize_file_path(file_str: str) -> str:
    """
    Normalize a file path extracted from **File:** for deduplication.

    Strips backticks, quotes, whitespace, leading ./ and trailing
    parenthetical text.
    """
    cleaned = file_str.strip("`'\" \t").replace("\\", "/")
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", cleaned).strip()
    cleaned = cleaned.lstrip("./")
    return cleaned.lower()


def normalize_line_spec(line_str: str) -> str:
    """
    Normalize a line specification extracted from **Line:** for
    deduplication.

    Examples:
        L50       -> 50
        L50-L55   -> 50-55
        50 to 55  -> 50-55
    """
    cleaned = line_str.strip("`'\" \t")
    nums = re.findall(r"\d+", cleaned)

    if len(nums) >= 2:
        return f"{nums[0]}-{nums[1]}"

    if len(nums) == 1:
        return nums[0]

    return cleaned.lower()


def extract_dedup_key(issue_block: str) -> Tuple[str, str]:
    """
    Extract a (normalized_file, normalized_line) key from an **Issue:** block.

    If File or Line is missing, fall back to available issue text.
    """
    file_match = re.search(
        r"^\*\*File:\*\*\s*(.+)$",
        issue_block,
        re.MULTILINE,
    )
    line_match = re.search(
        r"^\*\*Line:\*\*\s*(.+)$",
        issue_block,
        re.MULTILINE,
    )

    if file_match and line_match:
        file_key = normalize_file_path(file_match.group(1))
        line_key = normalize_line_spec(line_match.group(1))
        return file_key, line_key

    if file_match:
        file_key = normalize_file_path(file_match.group(1))
        issue_title = issue_block.splitlines()[0].strip()
        return file_key, issue_title.lower()

    first_line = issue_block.splitlines()[0].strip()
    return "unknown_file", first_line.lower()


# These patterns accept short, explicit statements that no issues were found.
# They deliberately do not accept arbitrary commentary about the code.
_CLEAN_STATEMENT_PATTERNS = [
    re.compile(
        r"(?:there (?:are|were) |i (?:found|see|have found) |"
        r"we (?:found|see) )?no "
        r"(?:(?:real|genuine|significant|major|critical|other|further|"
        r"additional|obvious|evidence-based) )*"
        r"(?:issues?|bugs?|problems?|defects?|"
        r"vulnerabilit(?:y|ies)|concerns?)"
        r"(?: found)?"
        r"(?: in (?:this|the) (?:change|code|diff|pr|patch))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"no (?:code )?changes? (?:are |is )?"
        r"(?:needed|required|necessary)",
        re.IGNORECASE,
    ),
    re.compile(
        r"lgtm(?:,\s*no issues found)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:the |this )?(?:code|change|changes|pr|diff|patch) )?"
        r"(?:looks|seems) good(?: to me)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"all checks pass(?:ed)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:the |this )?(?:code|review|pr|diff|changes?) "
        r"(?:is|are|looks?) clean",
        re.IGNORECASE,
    ),
]

_CLEAN_CONTRAST = re.compile(
    r"\b(?:but|however|although|though|except|unless|nevertheless|"
    r"nonetheless|yet|whereas|other than|apart from|aside from|besides)\b",
    re.IGNORECASE,
)

# Split at sentence-ending punctuation, newlines, and semicolons. A clean
# phrase followed by additional commentary must not be accepted as clean.
_CLEAN_SENTENCE_SPLIT = re.compile(r"[.!?;]+|\n+")

_CLEAN_MAX_SENTENCE_WORDS = 20


def _strip_markdown_wrappers(text: str) -> str:
    """Remove common Markdown wrappers before evaluating a clean statement."""
    text = text.strip()

    # Remove common bullet/list prefixes.
    text = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", text)

    # Remove bold, italic, and inline-code markers.
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "")
    text = text.replace("*", "")

    # Remove surrounding punctuation and decorative symbols.
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text).strip()

    return text


def _is_explicitly_clean(text: str) -> bool:
    """
    Return True only when the entire response consists of explicit,
    unambiguous clean statements.

    Examples accepted:
        No issues found.
        LGTM
        No bugs found.

    Examples rejected:
        The author refactored the caching layer.
        No issues found; one possible bug remains.
        No issues found. However, the error handling is incomplete.
        Tests passed.
        No review generated.
    """
    if not text or not text.strip():
        return False

    # A structured finding means the review is not clean.
    if re.search(r"(?m)^\s*\*\*Issue:\*\*", text):
        return False

    # A response containing review notices cannot be considered clean.
    if re.search(r"(?i)review notices|could not be reviewed completely", text):
        return False

    # Treat separators as boundaries, but evaluate every resulting piece.
    sentences = _CLEAN_SENTENCE_SPLIT.split(text)
    found_clean_statement = False

    for raw_sentence in sentences:
        sentence = _strip_markdown_wrappers(raw_sentence)

        # Ignore empty pieces created by punctuation or blank lines.
        if not sentence:
            continue

        if len(sentence.split()) > _CLEAN_MAX_SENTENCE_WORDS:
            return False

        if _CLEAN_CONTRAST.search(sentence):
            return False

        # Require a full-string match. A matching prefix is not sufficient.
        if not any(
            pattern.fullmatch(sentence)
            for pattern in _CLEAN_STATEMENT_PATTERNS
        ):
            return False

        found_clean_statement = True

    return found_clean_statement


def _extract_embedded_notices(
    review_text: str,
    file_label: str,
    status: str,
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Separate an embedded Review Notices section from review text.

    Returns:
        (review_body, notices)

    Each notice is (file_or_chunk, status, message).
    """
    notices: List[Tuple[str, str, str]] = []

    notice_match = re.search(
        r"(?im)^###\s+.*Review Notices\s*$|"
        r"^The following file\(s\) could not be reviewed completely:\s*$",
        review_text,
    )

    if not notice_match:
        return review_text.strip(), notices

    review_body = review_text[:notice_match.start()].strip()
    notices_text = review_text[notice_match.start():].strip()

    for line in notices_text.splitlines():
        match = re.match(
            r"^\s*-\s*`?([^`]+?)`?\s*"
            r"\(([^)]+)\):\s*(.+)$",
            line.strip(),
        )

        if match:
            notices.append(
                (
                    match.group(1).strip(),
                    match.group(2).strip(),
                    match.group(3).strip(),
                )
            )

    # If a notice section exists but cannot be parsed, do not silently
    # treat the response as a complete review.
    if not notices:
        notices.append(
            (
                file_label,
                "incomplete",
                "Review notices were present but could not be parsed.",
            )
        )

    return review_body, notices


def _aggregate_file_results(results: List[Dict[str, Any]]) -> str:
    """
    Aggregate per-file or per-chunk review results.

    Every result must complete and explicitly report a clean outcome before
    the overall review may be reported as clean. Failed, incomplete, empty,
    and ambiguous results are disclosed in Review Notices.

    Valid findings from other results are preserved even when one or more
    results fail or are ambiguous.
    """
    unique_issue_blocks: List[str] = []
    seen_issue_keys = set()

    notices: List[Tuple[str, str, str]] = []
    clean_non_issues: List[str] = []

    issue_split_pattern = re.compile(r"(?m)(?=^\*\*Issue:\*\*)")

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            notices.append(
                (
                    f"Result {index}",
                    "incomplete",
                    "Review result had an invalid format.",
                )
            )
            continue

        file_name = str(item.get("file") or "unknown")
        chunk_id = item.get("chunk_id")

        if chunk_id is not None:
            file_label = f"{file_name} (Chunk {chunk_id})"
        else:
            file_label = file_name

        status = str(item.get("status") or "completed").lower()
        review_text = str(item.get("review") or "").strip()

        # Preserve notices embedded by a lower-level chunked review.
        review_text, embedded_notices = _extract_embedded_notices(
            review_text,
            file_label,
            status,
        )
        notices.extend(embedded_notices)

        if status != "completed":
            if not embedded_notices:
                message = review_text or f"Review {status}."
                notices.append((file_label, status, message))

        if not review_text or review_text == "No review generated.":
            if not embedded_notices and status == "completed":
                notices.append(
                    (
                        file_label,
                        "incomplete",
                        "No review was generated.",
                    )
                )
            continue

        if (
            "Error occurred during AI code review" in review_text
            or "inference engine failure" in review_text.lower()
        ):
            if not embedded_notices and status == "completed":
                notices.append(
                    (
                        file_label,
                        "failed",
                        review_text,
                    )
                )
            continue

        bolded_text = _bold_labels(review_text)

        raw_blocks = [
            block.strip()
            for block in issue_split_pattern.split(bolded_text)
            if block.strip()
        ]

        found_issue_for_result = False

        for block in raw_blocks:
            if not block.startswith("**Issue:**"):
                continue

            cleaned_block = re.sub(r"^\s*---\s*", "", block)
            cleaned_block = re.sub(r"\s*---\s*$", "", cleaned_block).strip()

            if not cleaned_block:
                continue

            found_issue_for_result = True
            dedup_key = extract_dedup_key(cleaned_block)

            if dedup_key in seen_issue_keys:
                continue

            seen_issue_keys.add(dedup_key)
            unique_issue_blocks.append(cleaned_block)

        # A completed response with structured findings is a valid review
        # containing findings; it must never count as a clean response.
        if found_issue_for_result:
            continue

        # Only an explicitly clean response qualifies as clean. Any other
        # nonempty commentary is ambiguous and must remain visible.
        if status == "completed":
            if _is_explicitly_clean(review_text):
                clean_non_issues.append(review_text)
            else:
                collapsed = " ".join(review_text.split())

                if len(collapsed) > 500:
                    collapsed = collapsed[:497] + "..."

                notices.append(
                    (
                        file_label,
                        "ambiguous",
                        collapsed or "The review result was ambiguous.",
                    )
                )

    sections: List[str] = []

    # Preserve all valid, deduplicated findings.
    if unique_issue_blocks:
        sections.append("\n\n---\n\n".join(unique_issue_blocks))

    # Deduplicate notices while preserving their original order.
    if notices:
        notice_lines = [
            "### ⚠️ Review Notices",
            "The following file(s) could not be reviewed completely:",
        ]

        seen_notice_keys = set()

        for file_label, status, message in notices:
            notice_key = (file_label, status, message)

            if notice_key in seen_notice_keys:
                continue

            seen_notice_keys.add(notice_key)

            short_message = (message or "Review incomplete").splitlines()[0].strip()

            # Keep the notice compact and readable.
            if len(short_message) > 300:
                short_message = short_message[:297] + "..."

            notice_lines.append(
                f"- `{file_label}` ({status}): {short_message}"
            )

        sections.append("\n".join(notice_lines))

    # Report a clean result only when EVERY input result completed and
    # explicitly reported clean, with no findings or notices.
    if not unique_issue_blocks and not notices:
        if results and len(clean_non_issues) == len(results):
            return clean_non_issues[0]

        return "No review generated."

    return "\n\n---\n\n".join(sections)


def aggregate_reviews(
    review_results: Union[List[str], List[Dict[str, Any]]]
) -> str:
    """
    Combine raw chunk responses or per-file review results into one review.

    Responsibilities:
    1. Extract issue blocks from review responses.
    2. Deduplicate issues by normalized file and line.
    3. Preserve the order of discovery.
    4. Preserve findings even when other chunks fail.
    5. Surface failed, incomplete, empty, or ambiguous results.
    6. Never report a clean review unless every result explicitly qualifies.
    """
    if not review_results:
        return "No review generated."

    # Already structured per-file/per-chunk results.
    if isinstance(review_results[0], dict):
        return _aggregate_file_results(review_results)

    dict_results: List[Dict[str, Any]] = []

    for index, result in enumerate(review_results, start=1):
        if result is None:
            text = ""
        elif isinstance(result, str):
            text = result.strip()
        else:
            text = str(result).strip()

        # Preserve status metadata when the input is a str subclass such as
        # ReviewResult. Ordinary strings have no status attribute.
        status = getattr(result, "status", None)

        if status is None:
            if not text or text == "No review generated.":
                status = "incomplete"
            elif (
                "Error occurred during AI code review" in text
                or "inference engine failure" in text.lower()
            ):
                status = "failed"
            elif re.search(
                r"(?i)Review Notices|could not be reviewed completely",
                text,
            ):
                status = (
                    "failed"
                    if re.search(r"\(failed\)", text, re.IGNORECASE)
                    else "incomplete"
                )
            else:
                status = "completed"

        dict_results.append(
            {
                "chunk_id": index if len(review_results) > 1 else None,
                "file": f"Chunk {index}" if len(review_results) > 1 else "Review",
                "status": status,
                "review": text,
            }
        )

    return _aggregate_file_results(dict_results)