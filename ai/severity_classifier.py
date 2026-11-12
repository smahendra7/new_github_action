"""
severity_classifier.py

Classifies AI Code Review issues into severity categories:
🔴 High
🟠 Moderate
🟢 Low

Groups the original issue blocks under the corresponding severity headings
without modifying the issue content, wording, or structure.
Cleans up duplicate issues and self-corrected / invalidated false positives.
"""

import re

from github.diff_extractor import resolve_code_snippet

FIELD_HEADER_RE = re.compile(
    r"(?:^|\s)(Issue|File|Line|Code|Reason|Suggestion)\s*:",
    re.IGNORECASE
)

FIELD_NAME_MAP = {
    "issue": "Issue",
    "file": "File",
    "line": "Line",
    "code": "Code",
    "reason": "Reason",
    "suggestion": "Suggestion",
}

INLINE_LANG_PREFIX_RE = re.compile(
    r"^(?:xml|kotlin|java|javascript|typescript|text|json|gradle|groovy|swift|objective-c|objectivec|objc|objcpp|cpp|csharp|c)\s+",
    re.IGNORECASE
)

INVALIDATION_PATTERNS = [
    r"upon\s+closer\s+inspection",
    r"however,?\s+upon\s+closer",
    r"is\s+not\s+actually\s+(?:an?\s+)?issue",
    r"the\s+issue\s+is\s+not\s+with",
    r"the\s+issue\s+is\s+actually\s+with",
    r"is\s+actually\s+being\s+used",
    r"is\s+actually\s+in\s+the\s+companion\s+object",
    r"is\s+accessible\s+in\s+the\s+same\s+scope",
    r"is\s+not\s+necessary\s+in\s+this\s+case",
    r"disregard",
    r"false\s+positive",
    r"no\s+issue\s+here",
    r"correction\s*:",
    r"the\s+only\s+actual\s+issues?\s+found",
    r"upon\s+closer\s+examination",
    r"not\s+an\s+issue",
    r"is\s+actually\s+correct",
]


def _parse_issue_fields(block: str) -> dict[str, str]:

    if not block:
        return {}

    matches = list(FIELD_HEADER_RE.finditer(block))
    if not matches:
        return {}

    fields: dict[str, str] = {}

    for index, match in enumerate(matches):
        raw_name = match.group(1).lower()
        field_name = FIELD_NAME_MAP.get(raw_name, raw_name.capitalize())

        if field_name in fields:
            continue

        start_val = match.end()
        end_val = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(block)
        )
        fields[field_name] = block[start_val:end_val].strip()

    return fields


def _expand_inline_line_markers(code: str) -> str:

    if not code:
        return ""

    if "\n" in code:
        return code

    if re.search(r"\sL\d+\s*:", code):
        parts = re.split(r"\s+(?=L\d+\s*:)", code.strip())
        return "\n".join(part.strip() for part in parts if part.strip())

    return code


def _normalize_code_raw(code_raw: str) -> str:

    if not code_raw:
        return ""

    code_raw = code_raw.strip()
    code_raw = INLINE_LANG_PREFIX_RE.sub("", code_raw, count=1).strip()

    backtick_match = re.match(
        r"^```(\w*)\n?(.*?)\n?```\s*$",
        code_raw,
        re.DOTALL
    )
    if backtick_match:
        code_raw = backtick_match.group(2).strip("\r\n")

    return _expand_inline_line_markers(code_raw)


def classify_issue_severity(issue_block: str) -> str:
    """
    Classify an issue block into 'High', 'Moderate', or 'Low' severity.
    """
    text_lower = issue_block.lower()

    # High severity: Critical bugs, crash risks, null pointer exceptions,
    # security vulnerabilities, memory leaks, undefined variables, data loss.
    high_keywords = [
        "crash", "nullpointer", "null pointer", "npe", "security", "vulnerability",
        "memory leak", "data loss", "deadlock", "race condition", "undefined variable",
        "syntax error", "fatal", "unhandled exception", "arrayindexoutofbounds",
        "indexoutofbounds", "classcast", "typeerror", "stackoverflow", "infinite loop",
        "force unwrap", "exc_bad_access", "retain cycle", "fatalerror",
        "unowned reference", "main thread", "threading violation", "index out of range"
    ]

    # Low severity: Code style, formatting, unused imports, typos, documentation.
    low_keywords = [
        "unused import", "formatting", "indentation", "style", "naming convention",
        "typo", "spelling", "kdoc", "javadoc", "documentation comment",
        "swiftlint", "objc naming convention"
    ]

    for kw in high_keywords:
        if kw in text_lower:
            return "High"

    for kw in low_keywords:
        if kw in text_lower:
            return "Low"

    # Default: Moderate severity for general logic, resources, UI/view issues,
    # unused methods, missing error handling, etc.
    return "Moderate"


def _detect_language(file_path: str) -> str:
    if not file_path:
        return "text"
    file_path = file_path.strip().lower()
    ext_map = {
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".java": "java",
        ".xml": "xml",
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".swift": "swift",
        ".m": "objectivec",
        ".mm": "objectivecpp",
        ".storyboard": "xml",
        ".xib": "xml",
        ".plist": "xml",
        ".pbxproj": "text",
        ".entitlements": "xml",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".gradle": "groovy",
        ".properties": "properties",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".md": "markdown",
    }
    for ext, lang in ext_map.items():
        if file_path.endswith(ext):
            return lang
    return "text"


def _clean_issue_text(text: str) -> str:

    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\*+\s*", "", cleaned)
    cleaned = re.sub(r"\s*\*+\s*$", "", cleaned)
    cleaned = re.sub(r"\*\*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_line_number_prefixes(code: str) -> str:

    cleaned_lines = []
    for line in (code or "").splitlines():
        prefix_match = re.match(r"^L\d+\s*:\s?(.*)$", line)
        if prefix_match:
            cleaned_lines.append(prefix_match.group(1))
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _strip_trailing_code_na(block: str) -> str:

    trimmed = re.sub(
        r"(?:\s*Code\s*:\s*(?:\n\s*)?N/A\s*)+$",
        "",
        block,
        flags=re.IGNORECASE
    ).strip()

    return re.sub(
        r"\s+Code\s*:\s*N/A\s*$",
        "",
        trimmed,
        flags=re.IGNORECASE
    ).strip()


def _resolve_code_snippet(
    code_raw: str,
    file_path: str,
    line_spec: str,
    workspace_path: str | None,
    diff_text: str | None = None
) -> str:

    if file_path and line_spec:
        extracted = resolve_code_snippet(
            workspace_path,
            file_path,
            line_spec,
            diff_text
        )
        if extracted:
            return extracted.strip()

    code_raw = _normalize_code_raw(code_raw)

    code_lines = code_raw.splitlines() if code_raw else []
    while code_lines and not code_lines[0].strip():
        code_lines.pop(0)
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    cleaned_code = "\n".join(code_lines) if code_lines else ""
    cleaned_code = _strip_line_number_prefixes(cleaned_code).strip()

    if cleaned_code and cleaned_code.upper() != "N/A":
        return cleaned_code

    return ""


def format_issue_block(
    block: str,
    workspace_path: str | None = None,
    diff_text: str | None = None
) -> str:
    """
    Formats a single issue block with each field on its own line:

        Issue:<title>
        File:<path>
        Line:<line>
        Code:
        <code snippet>
        Reason:<reason>
        Suggestion:<suggestion>
    """
    if not block or not re.search(r"Issue\s*:", block, re.IGNORECASE):
        return block

    fields = _parse_issue_fields(block)
    if not fields:
        return block

    output_lines = []

    issue_text = _clean_issue_text(fields.get("Issue", ""))
    if issue_text:
        output_lines.append(f"Issue:{issue_text}")

    file_val = fields.get("File", "").strip()
    if file_val:
        output_lines.append(f"File:{file_val}")

    line_val = fields.get("Line", "").strip()
    if line_val:
        output_lines.append(f"Line:{line_val}")

    output_lines.append("Code:")

    cleaned_code = _resolve_code_snippet(
        fields.get("Code", ""),
        file_val,
        line_val,
        workspace_path,
        diff_text
    )

    if cleaned_code:
        output_lines.append(cleaned_code)

    reason_val = fields.get("Reason", "").strip()
    if reason_val:
        output_lines.append(f"Reason:{reason_val}")

    sug_val = fields.get("Suggestion", "").strip()
    if sug_val:
        output_lines.append(f"Suggestion:{sug_val}")

    return "\n".join(output_lines)


def _clean_issue_block(
    block: str,
    workspace_path: str | None = None,
    diff_text: str | None = None
) -> str | None:
    """
    Cleans an issue block. Returns None if the issue was invalidated or false-positive.
    Also discards issues with no supporting code evidence (empty or N/A Code field).
    """
    if not block or not re.match(r"^Issue\s*:", block.strip(), re.IGNORECASE):
        return None

    # Check for invalidation keywords anywhere in the block
    for pattern in INVALIDATION_PATTERNS:
        if re.search(pattern, block, re.IGNORECASE):
            return None

    # Trim block to end at Suggestion field if present, or end of known fields
    clean_text = _strip_trailing_code_na(block)

    fields_preview = _parse_issue_fields(clean_text)
    if fields_preview.get("Suggestion"):
        suggestion_header = re.search(
            r"(?:^|\s)Suggestion\s*:",
            clean_text,
            re.IGNORECASE
        )
        if suggestion_header:
            suggestion_end = (
                suggestion_header.end()
                + len(fields_preview["Suggestion"])
            )
            clean_text = clean_text[:suggestion_end].strip()
    else:
        suggestion_match = re.search(
            r"(Suggestion\s*:\s*.+?)(?=\n\n|\n[A-Z][a-z]+:|\Z)",
            clean_text,
            re.DOTALL | re.IGNORECASE
        )
        if suggestion_match:
            clean_text = clean_text[:suggestion_match.end()].strip()
        else:
            clean_text = re.sub(
                r"\n\s*(?:However|The only actual|Upon closer|Correction|Disregard).*",
                "",
                clean_text,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

    clean_text = _strip_trailing_code_na(clean_text)

    if not clean_text:
        return None

    fields = _parse_issue_fields(clean_text)
    file_path = fields.get("File", "").strip()
    line_spec = fields.get("Line", "").strip()
    raw_code = fields.get("Code", "")

    resolved = _resolve_code_snippet(
        raw_code,
        file_path,
        line_spec,
        workspace_path,
        diff_text
    )

    if not resolved:
        issue_label = _clean_issue_text(fields.get("Issue", "")) or "(unknown)"
        print(
            f"[EVIDENCE FILTER] Discarding issue with no code evidence: "
            f"{issue_label}"
        )
        return None

    return format_issue_block(clean_text, workspace_path, diff_text)


def _compute_issue_key(block: str) -> str:
    """
    Computes a normalized key for deduplication based on Issue title, File, Line, and Code.
    """
    fields = _parse_issue_fields(block)

    issue_str = _clean_issue_text(fields.get("Issue", "")).lower()
    file_str = fields.get("File", "").strip().lower()
    line_str = fields.get("Line", "").strip().lower()
    code_str = _normalize_code_raw(fields.get("Code", "")).lower()

    issue_str = re.sub(r"\s+", " ", issue_str)
    file_str = re.sub(r"\s+", " ", file_str)
    line_str = re.sub(r"\s+", " ", line_str)
    code_str = re.sub(r"\s+", " ", code_str)

    if issue_str and (file_str or code_str):
        return f"{issue_str}|{file_str}|{line_str}|{code_str}"

    return re.sub(r"\s+", " ", block.lower())


def classify_and_group_review(
    review_text: str,
    workspace_path: str | None = None,
    diff_text: str | None = None
) -> str:
    """
    Parses issue blocks from the raw review text, filters out invalidated/false-positive
    issues, deduplicates identical issues, classifies each block by severity,
    and returns the review text with issues grouped under severity headings:
    - ### 🔴 High Severity
    - ### 🟠 Moderate Severity
    - ### 🟢 Low Severity

    If no issue blocks are found, returns the review text as-is.
    """
    if not review_text or "Issue:" not in review_text:
        return review_text

    # Check if there is a late summary section like "The only actual issues found in the provided code are:"
    # If so, prefer issue blocks after the last occurrence of such summary marker.
    summary_marker_match = list(re.finditer(
        r"the\s+only\s+actual\s+issues?\s+found.*?:",
        review_text,
        re.IGNORECASE
    ))
    if summary_marker_match:
        last_marker = summary_marker_match[-1]
        after_text = review_text[last_marker.end():].strip()
        if "Issue:" in after_text:
            review_text = after_text

    # Split into preamble and issue blocks.
    raw_blocks = re.split(r"(?=\n?\s*Issue\s*:)", review_text, flags=re.IGNORECASE)

    preamble = ""
    issue_blocks = []
    seen_keys = set()

    for idx, block in enumerate(raw_blocks):
        block_str = block.strip()
        if not block_str:
            continue

        if re.match(r"^Issue\s*:", block_str, re.IGNORECASE):
            cleaned_block = _clean_issue_block(
                block_str,
                workspace_path,
                diff_text
            )
            if cleaned_block:
                key = _compute_issue_key(cleaned_block)
                if key not in seen_keys:
                    seen_keys.add(key)
                    issue_blocks.append(cleaned_block)
        elif idx == 0:
            # Only the first block before any Issue: block can be a preamble,
            # and it must not contain issue fields or code markers.
            if not re.search(
                r"the\s+only\s+actual\s+issues?|Code\s*:|N/A|File\s*:|Line\s*:",
                block_str,
                re.IGNORECASE
            ):
                preamble = block_str

    if preamble:
        preamble = preamble.strip()
        if preamble in ("**", "*", "AI Code Review"):
            preamble = ""
        elif re.fullmatch(r"[\*\s]+", preamble):
            preamble = ""
        elif re.fullmatch(
            r"(?:###\s*)?[🔴🟠🟢]\s*(?:High|Moderate|Low)\s+Severity",
            preamble,
            re.IGNORECASE
        ):
            preamble = ""

    if not issue_blocks:
        return review_text

    # Categorize issue blocks
    high_issues = []
    moderate_issues = []
    low_issues = []

    for block in issue_blocks:
        severity = classify_issue_severity(block)
        if severity == "High":
            high_issues.append(block)
        elif severity == "Low":
            low_issues.append(block)
        else:
            moderate_issues.append(block)

    # Build output sections
    sections = []

    if preamble and not preamble.lower().startswith("here are the issues") and not preamble.lower().startswith("the only actual"):
        sections.append(preamble)

    if high_issues:
        sections.append("### 🔴 High Severity\n\n" + "\n\n".join(high_issues))

    if moderate_issues:
        sections.append("### 🟠 Moderate Severity\n\n" + "\n\n".join(moderate_issues))

    if low_issues:
        sections.append("### 🟢 Low Severity\n\n" + "\n\n".join(low_issues))

    return "\n\n".join(sections)

