
"""
ai/result_aggregator.py

Aggregates multiple AI code-review responses.

Responsibilities:
- Extract structured findings.
- Deduplicate findings using normalized file and line information.
- Preserve findings from successful chunks.
- Disclose failed, incomplete, empty, and ambiguous results.
- Report a clean review only when every result explicitly confirms
  that no issues were found.
"""

import re
from typing import Any, Dict, List, Tuple, Union


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

_REVIEW_LABELS = (
    "Issue",
    "File",
    "Line",
    "Comment",
    "Code",
    "Reason",
    "Suggestion",
)

_LABEL_PATTERN = re.compile(
    r"^(\s*(?:💡\s*|\*\s*)?)(?!\*\*)"
    r"(" + "|".join(_REVIEW_LABELS) + r"):",
    re.MULTILINE,
)


def _bold_labels(text: str) -> str:
    """Bold review labels outside fenced code blocks."""

    if not text:
        return ""

    parts = re.split(r"(```[\s\S]*?```)", text)

    for index in range(0, len(parts), 2):
        parts[index] = _LABEL_PATTERN.sub(
            r"\1**\2:**",
            parts[index],
        )

    return "".join(parts)


def _split_outside_code_blocks(text: str) -> List[str]:
    """Split text into lines while preserving code-fence boundaries."""

    return text.splitlines()


# ---------------------------------------------------------------------------
# File and line normalization
# ---------------------------------------------------------------------------

def normalize_file_path(file_str: str) -> str:
    """
    Normalize a file path for finding deduplication.

    This function does not use suffix matching and does not remove
    arbitrary leading dots or slashes.
    """

    if not file_str:
        return ""

    cleaned = str(file_str).strip()

    cleaned = cleaned.strip("`'\" \t")
    cleaned = cleaned.replace("\\", "/")

    # Remove trailing parenthetical information, such as:
    # example.py (updated)
    cleaned = re.sub(
        r"\s+\([^)]*\)\s*$",
        "",
        cleaned,
    ).strip()

    # Remove only explicit relative path prefixes.
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]

    return cleaned


def normalize_line_spec(line_str: str) -> str:
    """
    Normalize a line specification.

    Examples:
        L50         -> 50
        Line 50     -> 50
        L50-L55     -> 50-55
        50 to 55    -> 50-55
        50, 52      -> 50,52
    """

    if not line_str:
        return ""

    cleaned = str(line_str).strip("`'\" \t")

    range_match = re.search(
        r"(?i)L?\s*(\d+)\s*(?:-|–|—|to)\s*L?\s*(\d+)",
        cleaned,
    )

    if range_match:
        first = int(range_match.group(1))
        second = int(range_match.group(2))

        return f"{min(first, second)}-{max(first, second)}"

    numbers = re.findall(r"\d+", cleaned)

    if numbers:
        return ",".join(numbers)

    return cleaned.lower()


# ---------------------------------------------------------------------------
# Issue extraction and deduplication
# ---------------------------------------------------------------------------

def _extract_field(
    issue_block: str,
    field_name: str,
) -> str:
    """Extract a field from a structured issue block."""

    pattern = re.compile(
        rf"(?im)^\s*(?:\*\*)?{re.escape(field_name)}"
        rf":(?:\*\*)?\s*(.*)$"
    )

    match = pattern.search(issue_block)

    if not match:
        return ""

    return match.group(1).strip()


def extract_dedup_key(issue_block: str) -> Tuple[str, str]:
    """
    Extract a normalized (file, line) key.

    If file or line information is missing, use a fallback based on
    available issue content instead of grouping every incomplete
    finding under one shared key.
    """

    file_value = _extract_field(issue_block, "File")
    line_value = _extract_field(issue_block, "Line")

    file_key = normalize_file_path(file_value)
    line_key = normalize_line_spec(line_value)

    if file_key and line_key:
        return file_key, line_key

    if file_key:
        title = issue_block.splitlines()[0].strip().lower()
        return file_key, f"unknown-line:{title}"

    title = issue_block.splitlines()[0].strip().lower()

    return "unknown_file", title


def _extract_issue_blocks(review_text: str) -> List[str]:
    """
    Extract structured Issue blocks.

    Issue labels inside fenced code blocks are ignored.
    """

    if not review_text:
        return []

    lines = _split_outside_code_blocks(review_text)

    blocks = []
    current_block = []
    inside_code_block = False

    issue_start_pattern = re.compile(
        r"^\s*\*\*Issue:\*\*",
        re.IGNORECASE,
    )

    for line in lines:

        if line.strip().startswith("```"):
            inside_code_block = not inside_code_block

        if (
            not inside_code_block
            and issue_start_pattern.match(line)
        ):

            if current_block:
                blocks.append(
                    "\n".join(current_block).strip()
                )

            current_block = [line]
            continue

        if current_block:
            current_block.append(line)

    if current_block:
        blocks.append(
            "\n".join(current_block).strip()
        )

    return [
        block
        for block in blocks
        if block
    ]


# ---------------------------------------------------------------------------
# Explicit clean-result detection
# ---------------------------------------------------------------------------

_CLEAN_STATEMENT_PATTERNS = [
    re.compile(
        r"(?:there (?:are|were) |"
        r"i (?:found|see|have found) |"
        r"we (?:found|see) )?"
        r"no "
        r"(?:(?:real|genuine|significant|major|critical|other|"
        r"further|additional|obvious|evidence-based) )*"
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
    r"\b(?:but|however|although|though|except|unless|"
    r"nevertheless|nonetheless|yet|whereas|other than|"
    r"apart from|aside from|besides)\b",
    re.IGNORECASE,
)

_CLEAN_SENTENCE_SPLIT = re.compile(
    r"[.!?;]+|\n+"
)

_CLEAN_MAX_SENTENCE_WORDS = 20


def _strip_markdown_wrappers(text: str) -> str:
    """Remove common Markdown formatting before clean-result detection."""

    text = text.strip()

    text = re.sub(
        r"^(?:[-*+]\s+|\d+[.)]\s+)",
        "",
        text,
    )

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = text.replace("*", "")

    text = re.sub(
        r"^[\W_]+|[\W_]+$",
        "",
        text,
    ).strip()

    return text


def _is_explicitly_clean(text: str) -> bool:
    """
    Return True only if the entire response explicitly reports a clean review.

    Empty, incomplete, ambiguous, or explanatory responses are not clean.
    """

    if not text or not text.strip():
        return False

    if re.search(
        r"(?m)^\s*\*\*Issue:\*\*",
        text,
    ):
        return False

    if re.search(
        r"(?i)review notices|could not be reviewed completely",
        text,
    ):
        return False

    sentences = _CLEAN_SENTENCE_SPLIT.split(text)

    found_clean_statement = False

    for raw_sentence in sentences:

        sentence = _strip_markdown_wrappers(raw_sentence)

        if not sentence:
            continue

        if len(sentence.split()) > _CLEAN_MAX_SENTENCE_WORDS:
            return False

        if _CLEAN_CONTRAST.search(sentence):
            return False

        if not any(
            pattern.fullmatch(sentence)
            for pattern in _CLEAN_STATEMENT_PATTERNS
        ):
            return False

        found_clean_statement = True

    return found_clean_statement


# ---------------------------------------------------------------------------
# Embedded review notices
# ---------------------------------------------------------------------------

def _extract_embedded_notices(
    review_text: str,
    file_label: str,
    status: str,
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Separate an embedded Review Notices section.

    Returns:
        (review_body, notices)

    Each notice contains:
        (file_or_chunk, status, message)
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

    # Do not silently ignore an unparseable notice section.
    if not notices:
        notices.append(
            (
                file_label,
                "incomplete",
                "Review notices were present but could not be parsed.",
            )
        )

    return review_body, notices


# ---------------------------------------------------------------------------
# Main aggregation logic
# ---------------------------------------------------------------------------

def _aggregate_file_results(
    results: List[Dict[str, Any]],
) -> str:
    """
    Aggregate per-file or per-chunk review results.

    Findings from successful results are preserved even when other results
    fail. An overall clean result requires every input to be completed
    and explicitly clean.
    """

    unique_issue_blocks: List[str] = []
    seen_issue_keys = set()

    notices: List[Tuple[str, str, str]] = []
    clean_non_issues: List[str] = []

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

        status = str(
            item.get("status") or "completed"
        ).lower()

        review_text = str(
            item.get("review") or ""
        ).strip()

        # Extract notices embedded by lower-level review functions.
        review_text, embedded_notices = _extract_embedded_notices(
            review_text,
            file_label,
            status,
        )

        notices.extend(embedded_notices)

        if status != "completed" and not embedded_notices:
            notices.append(
                (
                    file_label,
                    status,
                    review_text or f"Review {status}.",
                )
            )

        # Empty or missing reviews are incomplete.
        if (
            not review_text
            or review_text == "No review generated."
        ):
            if not embedded_notices and status == "completed":
                notices.append(
                    (
                        file_label,
                        "incomplete",
                        "No review was generated.",
                    )
                )

            continue

        # Explicit AI or inference errors.
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

        # Normalize labels outside code fences.
        formatted_text = _bold_labels(review_text)

        # Extract structured findings.
        issue_blocks = _extract_issue_blocks(formatted_text)

        for block in issue_blocks:

            cleaned_block = re.sub(
                r"^\s*---\s*",
                "",
                block,
            )

            cleaned_block = re.sub(
                r"\s*---\s*$",
                "",
                cleaned_block,
            ).strip()

            if not cleaned_block:
                continue

            dedup_key = extract_dedup_key(cleaned_block)

            if dedup_key in seen_issue_keys:
                continue

            seen_issue_keys.add(dedup_key)

            unique_issue_blocks.append(cleaned_block)

        # A structured finding means this result is not clean.
        if issue_blocks:
            continue

        # Only completed results can qualify as clean.
        if status == "completed":

            if _is_explicitly_clean(review_text):
                clean_non_issues.append(review_text)

            else:
                collapsed = " ".join(
                    review_text.split()
                )

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

    # Preserve all valid deduplicated findings.
    if unique_issue_blocks:
        sections.append(
            "\n\n---\n\n".join(unique_issue_blocks)
        )

    # Deduplicate notices while preserving discovery order.
    if notices:

        notice_lines = [
            "### ⚠️ Review Notices",
            "The following file(s) could not be reviewed completely:",
        ]

        seen_notice_keys = set()

        for file_label, status, message in notices:

            notice_key = (
                file_label,
                status,
                message,
            )

            if notice_key in seen_notice_keys:
                continue

            seen_notice_keys.add(notice_key)

            short_message = (
                message or "Review incomplete"
            ).splitlines()[0].strip()

            if len(short_message) > 300:
                short_message = short_message[:297] + "..."

            notice_lines.append(
                f"- `{file_label}` ({status}): {short_message}"
            )

        sections.append(
            "\n".join(notice_lines)
        )

    # A clean result is permitted only when every input completed
    # and explicitly reported a clean outcome.
    if not unique_issue_blocks and not notices:

        if (
            results
            and len(clean_non_issues) == len(results)
        ):
            return clean_non_issues[0]

        return "No review generated."

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Public aggregation function
# ---------------------------------------------------------------------------

def aggregate_reviews(
    review_results: Union[
        List[str],
        List[Dict[str, Any]],
    ],
) -> str:
    """
    Combine raw chunk responses or structured review results.

    Supports:
    - List of strings.
    - List of dictionaries containing file, status, and review.
    - ReviewResult string subclasses carrying status metadata.
    """

    if not review_results:
        return "No review generated."

    # Structured per-file or per-chunk results.
    if isinstance(review_results[0], dict):
        return _aggregate_file_results(review_results)

    dict_results: List[Dict[str, Any]] = []

    multiple_results = len(review_results) > 1

    for index, result in enumerate(review_results, start=1):

        if result is None:
            text = ""

        elif isinstance(result, str):
            text = result.strip()

        else:
            text = str(result).strip()

        # Preserve status metadata from ReviewResult string subclasses.
        status = getattr(result, "status", None)

        if status is None:

            if (
                not text
                or text == "No review generated."
            ):
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
                    if re.search(
                        r"\(failed\)",
                        text,
                        re.IGNORECASE,
                    )
                    else "incomplete"
                )

            else:
                status = "completed"

        dict_results.append(
            {
                "chunk_id": index if multiple_results else None,
                "file": (
                    f"Chunk {index}"
                    if multiple_results
                    else "Review"
                ),
                "status": status,
                "review": text,
            }
        )

    return _aggregate_file_results(dict_results)
