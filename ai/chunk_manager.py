
"""
ai/chunk_manager.py

Manages change-aware chunking for large PR contexts.

Responsibilities:
- Parse Git diffs by exact file path.
- Parse line-numbered full source code.
- Group nearby changed regions with surrounding context.
- Split oversized regions while respecting token and character limits.
- Execute chunk reviews concurrently.
- Preserve failed and incomplete results during aggregation.
"""

import os
import re
import shlex
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ai.context_config import (
    SAFE_INPUT_TOKEN_LIMIT,
    MAX_CONCURRENCY,
    CONTEXT_WINDOW_LINES,
    CHUNK_OVERLAP_LINES,
    SAFE_PROMPT_CHAR_LIMIT,
)

from ai.token_counter import count_tokens
from ai.prompt_builder import build_review_prompt, build_chunk_prompt
from ai.result_aggregator import (
    aggregate_reviews,
    _extract_issue_blocks,
    _bold_labels,
    _is_explicitly_clean,
    _strip_known_preamble,
)
from ai.reviewer import ReviewResult, review_code


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class DiffHunk:
    """Represents a single unified-diff hunk."""

    new_start: int
    new_end: int
    hunk_text: str
    modified_lines: List[int] = field(default_factory=list)


@dataclass
class ReviewChunk:
    """Represents one individual AI review request."""

    chunk_id: int
    total_chunks: int
    file_path: str
    changed_files: List[str]
    code_snippet: str
    diff_snippet: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


# ---------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------

def _normalize_path(path: str) -> str:
    """Normalize a repository-relative path without changing its identity."""
    if not path:
        return ""

    normalized = str(path).strip().replace("\\", "/")

    while normalized.startswith("./"):
        normalized = normalized[2:]

    return normalized.lstrip("/")


# Binary/document assets are not source code and must not be sent to the
# line-based AI review chunker. PDF files in particular can decode into one
# enormous line that cannot be subdivided safely.
_NON_REVIEWABLE_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".svgz", ".mp3", ".wav", ".ogg", ".mp4", ".mov", ".avi",
    ".zip", ".jar", ".aar", ".class", ".so", ".dll", ".dylib",
    ".ttf", ".otf", ".woff", ".woff2", ".keystore", ".jks",
}

def _is_reviewable_source_path(path: str) -> bool:
    """Return False for binary/document assets that are not code-review input."""
    normalized = _normalize_path(path).lower()
    filename = normalized.rsplit("/", 1)[-1]
    dot = filename.rfind(".")
    extension = filename[dot:] if dot >= 0 else ""
    return extension not in _NON_REVIEWABLE_EXTENSIONS


def _extract_diff_file_path(diff_header: str) -> Optional[str]:
    """
    Extract the destination path from a Git diff header.

    Supports ordinary paths and quoted paths containing spaces.
    """
    if not diff_header.startswith("diff --git "):
        return None

    try:
        parts = shlex.split(diff_header)

        if len(parts) < 4:
            return None

        old_path = parts[2]
        new_path = parts[3]

        if old_path.startswith("a/") and new_path.startswith("b/"):
            return _normalize_path(new_path[2:])

    except ValueError:
        pass

    # Fallback for ordinary unquoted Git diff headers.
    match = re.match(
        r"^diff --git a/(.+) b/(.+)$",
        diff_header,
    )

    if match:
        return _normalize_path(match.group(2))

    return None


def _unique_paths(paths: List[str]) -> List[str]:
    """Normalize paths and remove duplicates while preserving order."""
    result = []
    seen = set()

    for path in paths or []:
        normalized = _normalize_path(path)

        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


# ---------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------

def parse_diff_by_file(diff_text: str) -> Dict[str, Dict]:
    """
    Parse a raw Git diff into a dictionary keyed by exact file path.

    Example:
    {
        "src/example.py": {
            "full_diff": "...",
            "hunks": [DiffHunk(...)]
        }
    }
    """
    if not diff_text or not diff_text.strip():
        return {}

    blocks = re.split(
        r"(?=^diff --git )",
        diff_text,
        flags=re.MULTILINE,
    )

    file_diffs: Dict[str, Dict] = {}

    hunk_header_pattern = re.compile(
        r"^@@ -\d+(?:,\d+)? "
        r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
    )

    for block in blocks:
        if not block.strip():
            continue

        block_lines = block.splitlines()

        if not block_lines:
            continue

        first_line = block_lines[0]

        if not first_line.startswith("diff --git "):
            continue

        file_path = _extract_diff_file_path(first_line)

        if not file_path:
            print(
                "[CHUNK MANAGER WARNING] Could not determine "
                f"file path from diff header: {first_line}"
            )
            continue

        hunks: List[DiffHunk] = []

        hunk_blocks = re.split(
            r"(?=^@@ )",
            block,
            flags=re.MULTILINE,
        )

        for hunk in hunk_blocks:
            hunk_lines = hunk.splitlines()

            if not hunk_lines or not hunk_lines[0].startswith("@@"):
                continue

            hunk_match = hunk_header_pattern.match(hunk_lines[0])

            if not hunk_match:
                continue

            new_start = int(hunk_match.group("new_start"))

            new_count_text = hunk_match.group("new_count")
            new_count = (
                int(new_count_text)
                if new_count_text is not None
                else 1
            )

            new_end = (
                new_start + new_count - 1
                if new_count > 0
                else new_start
            )

            modified_lines: List[int] = []
            current_line = new_start

            for line in hunk_lines[1:]:
                if line.startswith("\\"):
                    continue

                if line.startswith("+"):
                    modified_lines.append(current_line)
                    current_line += 1

                elif line.startswith(" "):
                    current_line += 1

                elif line.startswith("-"):
                    # Deleted lines do not advance the new-file line number.
                    continue

            hunks.append(
                DiffHunk(
                    new_start=new_start,
                    new_end=new_end,
                    hunk_text=hunk.strip(),
                    modified_lines=modified_lines,
                )
            )

        file_diffs[file_path] = {
            "full_diff": block.strip(),
            "hunks": hunks,
        }

    return file_diffs


# ---------------------------------------------------------------------
# Full source-code parsing
# ---------------------------------------------------------------------

def parse_full_code_files(full_code: str) -> Dict[str, Dict]:
    """
    Parse the line-numbered source generated by extract_full_code().

    Expected format:

        FILE: src/example.py
        ```py
        L1: def example():
        L2:     return True
        ```

    Returns a dictionary keyed by exact normalized file path.
    """
    if not full_code or not full_code.strip():
        return {}

    blocks = re.split(
        r"(?=^FILE:\s*)",
        full_code,
        flags=re.MULTILINE,
    )

    parsed: Dict[str, Dict] = {}

    header_pattern = re.compile(
        r"^FILE:\s*([^\r\n]+)"
    )

    code_block_start = re.compile(
        r"^```([^\s`]*)"
    )

    for block in blocks:
        block_clean = block.strip()

        if not block_clean:
            continue

        header_match = header_pattern.match(block_clean)

        if not header_match:
            continue

        file_path = _normalize_path(header_match.group(1))

        if not file_path:
            continue

        lines = block_clean.splitlines()

        language = "text"
        code_lines: List[Tuple[int, str]] = []
        inside_code = False
        next_line_number = 1

        for line in lines[1:]:
            if not inside_code:
                code_match = code_block_start.match(line)

                if code_match:
                    inside_code = True
                    language = code_match.group(1) or "text"

                continue

            if line.strip() == "```":
                inside_code = False
                continue

            line_match = re.match(
                r"^L(\d+):\s?(.*)$",
                line,
            )

            if line_match:
                line_number = int(line_match.group(1))
                code_lines.append((line_number, line))
                next_line_number = line_number + 1

            else:
                code_lines.append((next_line_number, line))
                next_line_number += 1

        lines_dict = {
            line_number: line_text
            for line_number, line_text in code_lines
        }

        total_lines = (
            max(lines_dict)
            if lines_dict
            else 0
        )

        parsed[file_path] = {
            "lang": language,
            "lines_dict": lines_dict,
            "code_lines": code_lines,
            "total_lines": total_lines,
        }

    return parsed


# ---------------------------------------------------------------------
# Code and diff formatting
# ---------------------------------------------------------------------

def _format_code_block(
    file_path: str,
    lang: str,
    selected_lines: List[str],
) -> str:
    """Format selected line-numbered lines as a FILE code block."""
    inner = "\n".join(selected_lines)

    return (
        f"FILE: {file_path}\n"
        f"```{lang}\n"
        f"{inner}\n"
        "```"
    )


def _slice_diff_to_line_range(
    diff_text: str,
    start_line: Optional[int],
    end_line: Optional[int],
) -> str:
    """
    Keep only diff lines relevant to a new-file line range.

    Hunk headers are rebuilt to match the selected body. Deletions are
    associated with their insertion point so that removed lines are not
    accidentally lost at chunk boundaries.

    If the diff is malformed or incomplete, the original diff is returned
    rather than producing an unreliable partial diff.
    """
    if (
        not diff_text
        or start_line is None
        or end_line is None
    ):
        return diff_text

    if start_line > end_line:
        start_line, end_line = end_line, start_line

    hunk_blocks = re.split(
        r"(?=^@@ )",
        diff_text,
        flags=re.MULTILINE,
    )

    selected_hunks: List[str] = []
    found_hunk = False

    header_pattern = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? "
        r"\+(\d+)(?:,(\d+))? @@(.*)$"
    )

    for hunk in hunk_blocks:
        lines = hunk.splitlines()

        if not lines:
            continue

        match = header_pattern.match(lines[0])

        if not match:
            continue

        found_hunk = True

        old_start = int(match.group(1))
        old_count = (
            int(match.group(2))
            if match.group(2) is not None
            else 1
        )

        new_start = int(match.group(3))
        new_count = (
            int(match.group(4))
            if match.group(4) is not None
            else 1
        )

        trailing_header = match.group(5)

        # Validate the original hunk before attempting to slice it.
        actual_old_count = 0
        actual_new_count = 0

        for body_line in lines[1:]:
            if body_line.startswith((" ", "-")):
                actual_old_count += 1

            if body_line.startswith((" ", "+")):
                actual_new_count += 1

            if body_line.startswith("\\"):
                continue

            if not body_line.startswith((" ", "+", "-")):
                return diff_text

        if (
            actual_old_count != old_count
            or actual_new_count != new_count
        ):
            return diff_text

        old_line = old_start
        new_line = new_start

        selected_body: List[str] = []
        selected_old_start = None
        selected_new_start = None

        for body_line in lines[1:]:
            if body_line.startswith("\\"):
                if selected_body:
                    selected_body.append(body_line)
                continue

            include_line = False

            if body_line.startswith(" "):
                include_line = (
                    start_line <= new_line <= end_line
                )

                if include_line:
                    if selected_old_start is None:
                        selected_old_start = old_line

                    if selected_new_start is None:
                        selected_new_start = new_line

                old_line += 1
                new_line += 1

            elif body_line.startswith("+"):
                include_line = (
                    start_line <= new_line <= end_line
                )

                if include_line:
                    if selected_new_start is None:
                        selected_new_start = new_line

                new_line += 1

            elif body_line.startswith("-"):
                insertion_point = new_line

                include_line = (
                    start_line <= insertion_point <= end_line + 1
                )

                if include_line and selected_old_start is None:
                    selected_old_start = old_line

                if include_line and selected_new_start is None:
                    selected_new_start = insertion_point

                old_line += 1

            if include_line:
                selected_body.append(body_line)

        if not selected_body:
            continue

        selected_old_count = sum(
            1
            for line in selected_body
            if line.startswith((" ", "-"))
        )

        selected_new_count = sum(
            1
            for line in selected_body
            if line.startswith((" ", "+"))
        )

        # Git uses the original start positions when a side has zero lines.
        output_old_start = (
            selected_old_start
            if selected_old_start is not None
            else old_start
        )

        output_new_start = (
            selected_new_start
            if selected_new_start is not None
            else new_start
        )

        old_header = (
            f"{output_old_start},{selected_old_count}"
            if selected_old_count != 1
            else str(output_old_start)
        )

        new_header = (
            f"{output_new_start},{selected_new_count}"
            if selected_new_count != 1
            else str(output_new_start)
        )

        selected_header = (
            f"@@ -{old_header} +{new_header} @@{trailing_header}"
        )

        selected_hunks.append(
            "\n".join(
                [selected_header] + selected_body
            )
        )

    if not found_hunk:
        return diff_text

    if not selected_hunks:
        return ""

    return "\n".join(selected_hunks)


# ---------------------------------------------------------------------
# Prompt sizing and subdivision
# ---------------------------------------------------------------------

def _subdivide_large_region(
    file_path: str,
    lang: str,
    region_lines: List[str],
    region_diff: str,
    is_prompt_safe: Callable[[str], bool],
    overlap_lines: int = CHUNK_OVERLAP_LINES,
) -> List[Tuple[str, str, int, int]]:
    """
    Subdivide an oversized source region into smaller chunks.

    Each chunk retains a configurable overlap with the previous chunk.
    """
    if not region_lines:
        return []

    parsed_lines: List[Tuple[int, str]] = []

    for index, line in enumerate(region_lines, start=1):
        match = re.match(r"^L(\d+):", line)

        if match:
            parsed_lines.append(
                (int(match.group(1)), line)
            )
        else:
            parsed_lines.append((index, line))

    if not parsed_lines:
        return []

    chunks: List[Tuple[str, str, int, int]] = []

    start_index = 0
    total_lines = len(parsed_lines)

    while start_index < total_lines:
        best_end_index = start_index
        start_line_number = parsed_lines[start_index][0]

        for end_index in range(
            start_index + 1,
            total_lines + 1,
        ):
            selected = parsed_lines[start_index:end_index]

            end_line_number = selected[-1][0]

            code_snippet = _format_code_block(
                file_path,
                lang,
                [line for _, line in selected],
            )

            diff_snippet = _slice_diff_to_line_range(
                region_diff,
                start_line_number,
                end_line_number,
            )

            candidate_prompt = build_review_prompt(
                code=code_snippet,
                changed_files=[file_path],
                diff=diff_snippet,
            )

            if not is_prompt_safe(candidate_prompt):
                break

            best_end_index = end_index

        if best_end_index == start_index:
            # Never submit an individually oversized line to the model.
            # Skip that line and continue with the remaining source lines.
            oversized_line_number = parsed_lines[start_index][0]
            print(
                f"[CHUNK MANAGER WARNING] Skipping oversized single line "
                f"{oversized_line_number} in {file_path}; it exceeds the "
                f"configured prompt limit."
            )
            start_index += 1
            continue

        selected = parsed_lines[start_index:best_end_index]

        chunk_start_line = selected[0][0]
        chunk_end_line = selected[-1][0]

        code_snippet = _format_code_block(
            file_path,
            lang,
            [line for _, line in selected],
        )

        diff_snippet = _slice_diff_to_line_range(
            region_diff,
            chunk_start_line,
            chunk_end_line,
        )

        chunks.append(
            (
                code_snippet,
                diff_snippet,
                chunk_start_line,
                chunk_end_line,
            )
        )

        if best_end_index >= total_lines:
            break

        # Keep neighboring context without repeating the entire previous chunk.
        overlap = max(0, int(overlap_lines))

        start_index = max(
            start_index + 1,
            best_end_index - overlap,
        )

    return chunks


# ---------------------------------------------------------------------
# Review chunk creation
# ---------------------------------------------------------------------

def create_review_chunks(
    full_code: str,
    changed_files: List[str],
    diff: str,
    safe_token_limit: int = SAFE_INPUT_TOKEN_LIMIT,
    context_lines: int = CONTEXT_WINDOW_LINES,
    safe_char_limit: Optional[int] = None,
) -> List[ReviewChunk]:
    """
    Build ReviewChunk objects for the supplied changed files.

    Only exact file-path matches are accepted. Suffix matching is deliberately
    avoided because different directories can contain files with identical
    names.
    """
    if not changed_files:
        return []

    changed_files = _unique_paths(changed_files)

    if not changed_files:
        return []

    non_reviewable_files = [
        path for path in changed_files
        if not _is_reviewable_source_path(path)
    ]
    reviewable_changed_files = [
        path for path in changed_files
        if _is_reviewable_source_path(path)
    ]

    for path in non_reviewable_files:
        print(
            f"[CHUNK MANAGER] Skipping non-source asset: {path} "
            "(not sent to AI code review)."
        )

    if not reviewable_changed_files:
        print(
            "[CHUNK MANAGER WARNING] No reviewable source files remain "
            "after filtering non-source assets."
        )
        return []

    changed_files = reviewable_changed_files

    parsed_files = parse_full_code_files(full_code)
    parsed_diff = parse_diff_by_file(diff)

    effective_token_limit = max(1, int(safe_token_limit))

    effective_char_limit = (
        SAFE_PROMPT_CHAR_LIMIT
        if safe_char_limit is None
        else safe_char_limit
    )

    def is_prompt_safe(prompt_text: str) -> bool:
        if count_tokens(prompt_text) > effective_token_limit:
            return False

        if (
            effective_char_limit is not None
            and len(prompt_text) > effective_char_limit
        ):
            return False

        return True

    raw_chunk_specs: List[Dict] = []

    for rel_path in changed_files:
        norm_path = _normalize_path(rel_path)

        # Exact path lookup only.
        file_info = parsed_files.get(norm_path)
        diff_info = parsed_diff.get(norm_path)

        if not file_info:
            if diff_info:
                print(
                    f"[CHUNK MANAGER WARNING] Full source unavailable "
                    f"for {norm_path}. Using its exact diff only."
                )

                raw_chunk_specs.append(
                    {
                        "file_path": norm_path,
                        "code_snippet": (
                            f"FILE: {norm_path}\n"
                            "(Full content unavailable; review the supplied diff only.)"
                        ),
                        "diff_snippet": diff_info["full_diff"],
                        "start_line": None,
                        "end_line": None,
                    }
                )

            else:
                print(
                    f"[CHUNK MANAGER WARNING] Skipping {norm_path}: "
                    "neither full source nor an exact matching diff was found."
                )

            continue

        lang = file_info["lang"]
        lines_dict = file_info["lines_dict"]
        code_lines = file_info["code_lines"]
        total_lines = file_info["total_lines"]

        file_diff = (
            diff_info["full_diff"]
            if diff_info
            else ""
        )

        hunks = (
            diff_info["hunks"]
            if diff_info
            else []
        )

        if hunks:
            hunk_regions: List[
                Tuple[int, int, List[DiffHunk]]
            ] = []

            for hunk in hunks:
                region_start = max(
                    1,
                    hunk.new_start - max(0, context_lines),
                )

                region_end = min(
                    total_lines,
                    hunk.new_end + max(0, context_lines),
                )

                if total_lines == 0:
                    continue

                if not hunk_regions:
                    hunk_regions.append(
                        (
                            region_start,
                            region_end,
                            [hunk],
                        )
                    )

                else:
                    last_start, last_end, last_hunks = (
                        hunk_regions[-1]
                    )

                    if region_start <= last_end + max(0, context_lines):
                        hunk_regions[-1] = (
                            last_start,
                            max(last_end, region_end),
                            last_hunks + [hunk],
                        )

                    else:
                        hunk_regions.append(
                            (
                                region_start,
                                region_end,
                                [hunk],
                            )
                        )

            for (
                region_start,
                region_end,
                region_hunks,
            ) in hunk_regions:

                region_selected_lines = [
                    lines_dict[line_number]
                    for line_number in range(
                        region_start,
                        region_end + 1,
                    )
                    if line_number in lines_dict
                ]

                if not region_selected_lines:
                    continue

                region_diff_text = "\n\n".join(
                    hunk.hunk_text
                    for hunk in region_hunks
                )

                code_snippet = _format_code_block(
                    norm_path,
                    lang,
                    region_selected_lines,
                )

                candidate_prompt = build_review_prompt(
                    code=code_snippet,
                    changed_files=[norm_path],
                    diff=region_diff_text,
                )

                if is_prompt_safe(candidate_prompt):
                    raw_chunk_specs.append(
                        {
                            "file_path": norm_path,
                            "code_snippet": code_snippet,
                            "diff_snippet": region_diff_text,
                            "start_line": region_start,
                            "end_line": region_end,
                        }
                    )

                else:
                    subdivided = _subdivide_large_region(
                        file_path=norm_path,
                        lang=lang,
                        region_lines=region_selected_lines,
                        region_diff=region_diff_text,
                        is_prompt_safe=is_prompt_safe,
                        overlap_lines=CHUNK_OVERLAP_LINES,
                    )

                    for (
                        code,
                        diff_part,
                        start_line,
                        end_line,
                    ) in subdivided:
                        raw_chunk_specs.append(
                            {
                                "file_path": norm_path,
                                "code_snippet": code,
                                "diff_snippet": diff_part,
                                "start_line": start_line,
                                "end_line": end_line,
                            }
                        )

        else:
            # No parseable hunk is available for this changed file.
            # Preserve the previous behavior of reviewing its full content.
            file_code_lines = [
                item[1]
                for item in code_lines
            ]

            if not file_code_lines:
                print(
                    f"[CHUNK MANAGER WARNING] {norm_path} has no "
                    "source lines to review."
                )
                continue

            code_snippet = _format_code_block(
                norm_path,
                lang,
                file_code_lines,
            )

            candidate_prompt = build_review_prompt(
                code=code_snippet,
                changed_files=[norm_path],
                diff=file_diff,
            )

            if is_prompt_safe(candidate_prompt):
                raw_chunk_specs.append(
                    {
                        "file_path": norm_path,
                        "code_snippet": code_snippet,
                        "diff_snippet": file_diff,
                        "start_line": 1,
                        "end_line": total_lines,
                    }
                )

            else:
                subdivided = _subdivide_large_region(
                    file_path=norm_path,
                    lang=lang,
                    region_lines=file_code_lines,
                    region_diff=file_diff,
                    is_prompt_safe=is_prompt_safe,
                    overlap_lines=CHUNK_OVERLAP_LINES,
                )

                for (
                    code,
                    diff_part,
                    start_line,
                    end_line,
                ) in subdivided:
                    raw_chunk_specs.append(
                        {
                            "file_path": norm_path,
                            "code_snippet": code,
                            "diff_snippet": diff_part,
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    )

    total_chunks = len(raw_chunk_specs)
    chunks: List[ReviewChunk] = []

    for index, spec in enumerate(raw_chunk_specs, start=1):
        chunks.append(
            ReviewChunk(
                chunk_id=index,
                total_chunks=total_chunks,
                file_path=spec["file_path"],
                changed_files=[spec["file_path"]],
                code_snippet=spec["code_snippet"],
                diff_snippet=spec["diff_snippet"],
                start_line=spec.get("start_line"),
                end_line=spec.get("end_line"),
            )
        )

    print(
        f"[CHUNK MANAGER] Created {total_chunks} review chunk(s) "
        f"from {len(changed_files)} reviewable changed file(s)."
    )

    return chunks


# ---------------------------------------------------------------------
# Chunk review statuses and helpers
# ---------------------------------------------------------------------

CHUNK_STATUS_COMPLETED_WITH_FINDINGS = "COMPLETED_WITH_FINDINGS"
CHUNK_STATUS_COMPLETED_NO_ISSUES = "COMPLETED_NO_ISSUES"
CHUNK_STATUS_INCOMPLETE = "INCOMPLETE"
CHUNK_STATUS_FAILED = "FAILED"

SUCCESSFUL_CHUNK_STATUSES = {
    CHUNK_STATUS_COMPLETED_WITH_FINDINGS,
    CHUNK_STATUS_COMPLETED_NO_ISSUES,
    "completed",
    "COMPLETED",
}


def _is_permanent_error(exc: Exception) -> bool:
    """Return True if an exception represents an unrecoverable client or auth failure."""
    if exc is None:
        return False
    status_code = getattr(exc, "status_code", None)
    if status_code in {400, 401, 403, 404}:
        return True
    msg = str(exc).lower()
    if any(k in msg for k in ["unauthorized", "invalid api key", "authentication", "forbidden"]):
        return True
    if "bad request" in msg or "invalid_request_error" in msg:
        return True
    if "maximum retries exhausted" in msg:
        return True
    return False


def _classify_failure(exc: Optional[Exception] = None, status: str = "", review_text: str = "") -> str:
    """Classify the exact failure category for structured diagnostics."""
    if exc:
        status_code = getattr(exc, "status_code", None)
        msg = str(exc).lower()
        if status_code == 429 or "rate limit" in msg or "429" in msg:
            return "RATE_LIMIT_429"
        if status_code in {401, 403} or "unauthorized" in msg or "invalid api key" in msg or "authentication" in msg:
            return "AUTH_ERROR"
        if (status_code == 400 and ("context_length" in msg or "max_tokens" in msg or "too large" in msg)) or "context length" in msg:
            return "CONTEXT_LENGTH_EXCEEDED"
        if "timeout" in msg or "timed out" in msg:
            return "REQUEST_TIMEOUT"
        if "connection" in msg:
            return "CONNECTION_FAILURE"
        if "maximum retries exhausted" in msg:
            return "PROVIDER_RETRIES_EXHAUSTED"
        if status_code and status_code >= 500:
            return f"SERVER_ERROR_{status_code}"
        return f"API_ERROR_{status_code}" if status_code else "API_ERROR"

    if not review_text or not review_text.strip() or review_text.strip() == "No review generated.":
        return "EMPTY_RESPONSE"

    msg = review_text.lower()
    if "429" in msg or "rate limit" in msg:
        return "RATE_LIMIT_429"
    if "401" in msg or "403" in msg or "unauthorized" in msg or "authentication" in msg or "invalid api key" in msg:
        return "AUTH_ERROR"
    if "context length" in msg or "context_length" in msg or "too large" in msg:
        return "CONTEXT_LENGTH_EXCEEDED"
    if "timeout" in msg or "timed out" in msg:
        return "REQUEST_TIMEOUT"
    if "connection" in msg:
        return "CONNECTION_FAILURE"
    if "maximum retries exhausted" in msg:
        return "PROVIDER_RETRIES_EXHAUSTED"
    if "500" in msg or "502" in msg or "503" in msg or "504" in msg:
        for code in (500, 502, 503, 504):
            if str(code) in msg:
                return f"SERVER_ERROR_{code}"
    if "hit generation length limit" in msg or "output token limit" in msg:
        return "OUTPUT_TOKEN_LIMIT_REACHED"
    if status == CHUNK_STATUS_INCOMPLETE or "incomplete" in msg or "no valid review generated" in msg:
        return "INCOMPLETE_OR_MALFORMED"
    if status == CHUNK_STATUS_FAILED or "failed" in msg:
        return "FAILED_REVIEW"
    return "NONE"


# ---------------------------------------------------------------------
# Review result handling
# ---------------------------------------------------------------------

def _infer_review_status(
    result: object,
) -> Tuple[str, str]:
    """
    Determine the status and text of an individual chunk result.

    Returns:
        (status, review_text)
    where status is one of:
        COMPLETED_WITH_FINDINGS, COMPLETED_NO_ISSUES, INCOMPLETE, FAILED
    """
    if result is None:
        return CHUNK_STATUS_INCOMPLETE, ""

    review_text = str(result).strip()
    status_attr = getattr(result, "status", None)
    raw_status = str(status_attr).upper() if status_attr else ""

    if raw_status in {
        CHUNK_STATUS_COMPLETED_WITH_FINDINGS,
        CHUNK_STATUS_COMPLETED_NO_ISSUES,
        CHUNK_STATUS_INCOMPLETE,
        CHUNK_STATUS_FAILED,
    }:
        return raw_status, review_text

    if (
        not review_text
        or review_text == "No review generated."
        or review_text.startswith("No review generated")
    ):
        return CHUNK_STATUS_INCOMPLETE, review_text or "No review generated."

    if (
        "Error occurred during AI code review" in review_text
        or "inference engine failure" in review_text.lower()
    ):
        return CHUNK_STATUS_FAILED, review_text

    if re.search(r"(?im)^###\s+.*Review Notices\s*$", review_text):
        if re.search(r"\(failed\)", review_text, re.IGNORECASE):
            return CHUNK_STATUS_FAILED, review_text
        return CHUNK_STATUS_INCOMPLETE, review_text

    if raw_status in {"FAILED"}:
        return CHUNK_STATUS_FAILED, review_text

    if raw_status in {"INCOMPLETE"}:
        return CHUNK_STATUS_INCOMPLETE, review_text

    # 1. Accept structured findings as valid reviews
    formatted_text = _bold_labels(review_text)
    issue_blocks = _extract_issue_blocks(formatted_text)
    if issue_blocks or re.search(r"(?im)^\s*\*\*Issue:\*\*", formatted_text):
        return CHUNK_STATUS_COMPLETED_WITH_FINDINGS, review_text

    # 2. Accept explicitly clean responses (e.g. "No issues found.")
    if _is_explicitly_clean(review_text):
        clean_text = _strip_known_preamble(review_text) or "No issues found."
        return CHUNK_STATUS_COMPLETED_NO_ISSUES, clean_text

    # 3. Arbitrary conversational text, preamble alone, or unstructured
    # responses must NEVER be classified as COMPLETED_NO_ISSUES.
    # Treat as INCOMPLETE so the chunk retry mechanism can attempt
    # to obtain a valid structured or clean review.
    return CHUNK_STATUS_INCOMPLETE, review_text


# ---------------------------------------------------------------------
# Individual chunk review with retry mechanism
# ---------------------------------------------------------------------

def _review_single_chunk(
    chunk: ReviewChunk,
    max_attempts: int = 3,
    retry_delay: Optional[float] = None,
) -> ReviewResult:
    """
    Build and submit one chunk review request with a controlled retry mechanism.

    Retries up to max_attempts (default 3) on:
    - Empty or whitespace-only responses
    - Malformed or truncated responses (finish_reason != 'stop')
    - Missing required fields
    - Temporary API errors (HTTP 429, 500, timeouts, connection drops)
    - Other recoverable failures
    """
    chunk_prompt = build_chunk_prompt(chunk)
    token_count = count_tokens(chunk_prompt)
    if retry_delay is None:
        retry_delay = float(os.environ.get("CODEGATE_RETRY_DELAY", "0.5"))

    last_error = None
    last_review_text = ""
    last_status = CHUNK_STATUS_INCOMPLETE

    for attempt in range(1, max_attempts + 1):
        start_time = time.monotonic()
        print(
            f"\n[CHUNK {chunk.chunk_id}/{chunk.total_chunks}] "
            f"Attempt {attempt}/{max_attempts} for {chunk.file_path} "
            f"(Lines {chunk.start_line or 'all'}.."
            f"{chunk.end_line or 'all'}, "
            f"{token_count} input tokens)..."
        )

        try:
            result = review_code(chunk_prompt)
            duration = time.monotonic() - start_time
            status, review_text = _infer_review_status(result)
            failure_cat = _classify_failure(status=status, review_text=review_text)

            print(
                f"[CHUNK DIAGNOSTICS] File: {chunk.file_path} | Chunk: {chunk.chunk_id}/{chunk.total_chunks} | "
                f"Attempt: {attempt}/{max_attempts} | Duration: {duration:.2f}s | Status: {status} | "
                f"Tokens: {token_count} | Failure: {failure_cat}"
            )

            if status in {CHUNK_STATUS_COMPLETED_WITH_FINDINGS, CHUNK_STATUS_COMPLETED_NO_ISSUES}:
                res = ReviewResult(review_text, status=status)
                res.diagnostics = {
                    "file": chunk.file_path,
                    "chunk_id": chunk.chunk_id,
                    "total_chunks": chunk.total_chunks,
                    "attempt": attempt,
                    "duration": duration,
                    "status": status,
                    "failure_category": "NONE",
                }
                return res

            # Recoverable response failure (empty, truncated, or incomplete)
            last_review_text = review_text
            last_status = status

            if attempt < max_attempts:
                print(
                    f"[CHUNK {chunk.chunk_id}] Attempt {attempt} returned {status} ({failure_cat}): "
                    f"'{review_text[:100] if review_text else ''}...'. Retrying..."
                )
                delay = retry_delay * (2 ** (attempt - 1))
                if delay > 0:
                    time.sleep(delay)
                continue
            else:
                final_text = f"No valid review generated after {max_attempts} attempts. Last status: {status} ({failure_cat})."
                res = ReviewResult(final_text, status=status)
                res.diagnostics = {
                    "file": chunk.file_path,
                    "chunk_id": chunk.chunk_id,
                    "total_chunks": chunk.total_chunks,
                    "attempt": attempt,
                    "duration": duration,
                    "status": status,
                    "failure_category": failure_cat,
                }
                return res

        except Exception as exc:
            duration = time.monotonic() - start_time
            last_error = exc
            failure_cat = _classify_failure(exc=exc)

            print(
                f"[CHUNK DIAGNOSTICS] File: {chunk.file_path} | Chunk: {chunk.chunk_id}/{chunk.total_chunks} | "
                f"Attempt: {attempt}/{max_attempts} | Duration: {duration:.2f}s | Status: FAILED | "
                f"Tokens: {token_count} | Exception: {type(exc).__name__}: {exc} | Failure: {failure_cat}"
            )

            if _is_permanent_error(exc):
                print(f"[CHUNK {chunk.chunk_id}] Permanent error on attempt {attempt}: {exc}")
                res = ReviewResult(
                    f"Error occurred during AI code review: {exc} ({failure_cat})",
                    status=CHUNK_STATUS_FAILED,
                )
                res.diagnostics = {
                    "file": chunk.file_path,
                    "chunk_id": chunk.chunk_id,
                    "total_chunks": chunk.total_chunks,
                    "attempt": attempt,
                    "duration": duration,
                    "status": CHUNK_STATUS_FAILED,
                    "failure_category": failure_cat,
                }
                return res

            last_status = CHUNK_STATUS_FAILED
            last_review_text = f"Error occurred during AI code review: {exc} ({failure_cat})"

            if attempt < max_attempts:
                print(f"[CHUNK {chunk.chunk_id}] Temporary error on attempt {attempt}: {exc}. Retrying...")
                delay = retry_delay * (2 ** (attempt - 1))
                if delay > 0:
                    time.sleep(delay)
                continue
            else:
                print(f"[CHUNK {chunk.chunk_id}] All {max_attempts} attempts failed: {exc}")
                res = ReviewResult(
                    last_review_text,
                    status=CHUNK_STATUS_FAILED,
                )
                res.diagnostics = {
                    "file": chunk.file_path,
                    "chunk_id": chunk.chunk_id,
                    "total_chunks": chunk.total_chunks,
                    "attempt": attempt,
                    "duration": duration,
                    "status": CHUNK_STATUS_FAILED,
                    "failure_category": failure_cat,
                }
                return res

    return ReviewResult(
        last_review_text or f"No valid review generated after {max_attempts} attempts.",
        status=last_status,
    )


# ---------------------------------------------------------------------
# Parallel chunk review orchestration
# ---------------------------------------------------------------------

def execute_chunked_review(
    full_code: str,
    changed_files: List[str],
    diff: str,
    safe_token_limit: int = SAFE_INPUT_TOKEN_LIMIT,
    context_lines: int = CONTEXT_WINDOW_LINES,
    max_concurrency: int = MAX_CONCURRENCY,
    safe_char_limit: Optional[int] = None,
) -> str:
    """
    Execute chunked AI review and aggregate all chunk results.

    Every generated chunk gets a result record. Failed and incomplete
    results are retained so the aggregator can report them.
    """
    chunks = create_review_chunks(
        full_code=full_code,
        changed_files=changed_files,
        diff=diff,
        safe_token_limit=safe_token_limit,
        context_lines=context_lines,
        safe_char_limit=safe_char_limit,
    )

    if not chunks:
        print(
            "\n[CHUNK MANAGER] No review chunks could be generated."
        )

        res = ReviewResult(
            "No review generated.",
            status="incomplete",
        )
        pr_cov, f_covs = calculate_review_coverage(changed_files, [])
        res.coverage = pr_cov
        res.file_coverage = f_covs
        res.chunk_results = []
        return res

    total_chunks = len(chunks)

    concurrency = max(
        1,
        int(max_concurrency),
    )

    print(
        f"\n[CHUNK MANAGER] Created {total_chunks} chunk(s) "
        f"for AI review. Maximum concurrency: {concurrency}.\n"
    )

    # Preserve the original chunk order regardless of completion order.
    chunk_result_dicts: List[Optional[Dict]] = (
        [None] * total_chunks
    )

    for batch_start in range(
        0,
        total_chunks,
        concurrency,
    ):
        batch_chunks = chunks[
            batch_start:batch_start + concurrency
        ]

        chunk_indices = list(
            range(
                batch_start,
                batch_start + len(batch_chunks),
            )
        )

        chunk_ids = [
            chunk.chunk_id
            for chunk in batch_chunks
        ]

        print(
            f"\n[CHUNK MANAGER] Running batch "
            f"{chunk_ids} concurrently "
            f"(max concurrency: {concurrency})..."
        )

        with ThreadPoolExecutor(
            max_workers=len(batch_chunks)
        ) as executor:

            future_to_chunk_info = {
                executor.submit(
                    _review_single_chunk,
                    chunk,
                ): (index, chunk)

                for chunk, index in zip(
                    batch_chunks,
                    chunk_indices,
                )
            }

            for future in as_completed(future_to_chunk_info):
                index, chunk = future_to_chunk_info[future]

                try:
                    result = future.result()

                    status, review_text = _infer_review_status(
                        result
                    )

                    chunk_result_dicts[index] = {
                        "chunk_id": chunk.chunk_id,
                        "file": chunk.file_path,
                        "status": status,
                        "review": review_text,
                        "diagnostics": getattr(result, "diagnostics", {}),
                    }

                except Exception as chunk_error:
                    error_text = (
                        "Error occurred during AI code review: "
                        f"{chunk_error}"
                    )

                    print(
                        f"\n[CHUNK ERROR] Chunk "
                        f"{chunk.chunk_id} failed: {chunk_error}"
                    )

                    chunk_result_dicts[index] = {
                        "chunk_id": chunk.chunk_id,
                        "file": chunk.file_path,
                        "status": "failed",
                        "review": error_text,
                    }

    # Ensure every chunk has a result record.
    for index, item in enumerate(chunk_result_dicts):
        if item is None:
            chunk = chunks[index]

            chunk_result_dicts[index] = {
                "chunk_id": chunk.chunk_id,
                "file": chunk.file_path,
                "status": CHUNK_STATUS_INCOMPLETE,
                "review": "The chunk did not produce a result.",
            }

    complete_result_list: List[Dict] = [
        item
        for item in chunk_result_dicts
        if item is not None
    ]

    # Calculate file-level and PR-level coverage
    pr_coverage, file_coverages = calculate_review_coverage(
        changed_files=changed_files,
        chunk_results=complete_result_list,
    )

    # Aggregate findings and notices from all chunk results.
    aggregated_review = aggregate_reviews(
        complete_result_list
    )

    if not aggregated_review:
        aggregated_review = "No review generated."

    notices_match = re.search(
        r"(?m)^###\s+.*Review Notices",
        aggregated_review,
    )

    notices_text = (
        aggregated_review[notices_match.start():]
        if notices_match
        else ""
    )

    has_failed = any(
        str(item.get("status", "")).upper() == CHUNK_STATUS_FAILED
        for item in complete_result_list
    )

    has_incomplete = any(
        str(item.get("status", "")).upper() not in SUCCESSFUL_CHUNK_STATUSES
        for item in complete_result_list
    )

    if (
        has_failed
        or re.search(
            r"\(failed\)",
            notices_text,
            re.IGNORECASE,
        )
    ):
        overall_status = "failed"

    elif (
        has_incomplete
        or notices_match
        or aggregated_review.strip() == "No review generated."
    ):
        overall_status = "incomplete"

    else:
        overall_status = "completed"

    print(
        "\n[CHUNK MANAGER] Completed aggregation "
        "of all chunk reviews.\n"
    )

    res = ReviewResult(
        aggregated_review,
        status=overall_status,
    )
    res.coverage = pr_coverage
    res.file_coverage = file_coverages
    res.chunk_results = complete_result_list

    return res


# ---------------------------------------------------------------------
# Coverage tracking helpers
# ---------------------------------------------------------------------

def calculate_review_coverage(
    changed_files: List[str],
    chunk_results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Calculate file-level and PR-level review coverage.

    Returns:
        (pr_coverage, file_coverages)
    """
    file_map: Dict[str, Dict[str, Any]] = {}

    for f in changed_files:
        norm = _normalize_path(f)
        if norm and norm not in file_map:
            file_map[norm] = {
                "file": norm,
                "chunks_generated": 0,
                "chunks_succeeded": 0,
                "chunks_incomplete": 0,
                "chunks_failed": 0,
                "status": "NOT_REVIEWED",
                "incomplete_chunk_details": [],
            }

    for item in chunk_results:
        f_path = _normalize_path(item.get("file", ""))
        status = str(item.get("status", "")).upper()
        if f_path not in file_map:
            file_map[f_path] = {
                "file": f_path,
                "chunks_generated": 0,
                "chunks_succeeded": 0,
                "chunks_incomplete": 0,
                "chunks_failed": 0,
                "status": "NOT_REVIEWED",
                "incomplete_chunk_details": [],
            }

        rec = file_map[f_path]
        rec.setdefault("incomplete_chunk_details", [])
        rec["chunks_generated"] += 1
        if status in SUCCESSFUL_CHUNK_STATUSES:
            rec["chunks_succeeded"] += 1
        elif status == CHUNK_STATUS_INCOMPLETE:
            rec["chunks_incomplete"] += 1
            # Capture per-chunk failure reason for the report.
            diag = item.get("diagnostics") or {}
            failure_cat = diag.get("failure_category", "INCOMPLETE")
            chunk_id = item.get("chunk_id", "?")
            rec["incomplete_chunk_details"].append(
                {"chunk_id": chunk_id, "failure_category": failure_cat}
            )
        elif status == CHUNK_STATUS_FAILED:
            rec["chunks_failed"] += 1
            diag = item.get("diagnostics") or {}
            failure_cat = diag.get("failure_category", "FAILED")
            chunk_id = item.get("chunk_id", "?")
            rec.setdefault("incomplete_chunk_details", []).append(
                {"chunk_id": chunk_id, "failure_category": failure_cat}
            )
        else:
            rec["chunks_incomplete"] += 1
            diag = item.get("diagnostics") or {}
            failure_cat = diag.get("failure_category", "INCOMPLETE")
            chunk_id = item.get("chunk_id", "?")
            rec.setdefault("incomplete_chunk_details", []).append(
                {"chunk_id": chunk_id, "failure_category": failure_cat}
            )

    file_coverages = []
    for rec in file_map.values():
        gen = rec["chunks_generated"]
        succ = rec["chunks_succeeded"]
        inc = rec["chunks_incomplete"]
        fail = rec["chunks_failed"]

        if gen > 0 and succ == gen:
            rec["status"] = "FULLY_REVIEWED"
        elif succ > 0 and (inc > 0 or fail > 0):
            rec["status"] = "PARTIALLY_REVIEWED"
        else:
            rec["status"] = "NOT_REVIEWED"

        file_coverages.append(rec)

    total_files = len(file_coverages)
    files_fully = sum(1 for r in file_coverages if r["status"] == "FULLY_REVIEWED")
    files_partially = sum(1 for r in file_coverages if r["status"] == "PARTIALLY_REVIEWED")
    files_not = sum(1 for r in file_coverages if r["status"] == "NOT_REVIEWED")

    total_chunks = len(chunk_results)
    chunks_succeeded = sum(
        1 for r in chunk_results
        if str(r.get("status", "")).upper() in SUCCESSFUL_CHUNK_STATUSES
    )
    chunks_incomplete = sum(
        1 for r in chunk_results
        if str(r.get("status", "")).upper() == CHUNK_STATUS_INCOMPLETE
    )
    chunks_failed = sum(
        1 for r in chunk_results
        if str(r.get("status", "")).upper() == CHUNK_STATUS_FAILED
    )

    if total_files > 0 and files_fully == total_files:
        pr_status = "FULLY_REVIEWED"
    elif files_fully > 0 or files_partially > 0:
        pr_status = "PARTIALLY_REVIEWED"
    else:
        pr_status = "NOT_REVIEWED"

    pr_coverage = {
        "total_files": total_files,
        "files_fully_reviewed": files_fully,
        "files_partially_reviewed": files_partially,
        "files_not_reviewed": files_not,
        "total_chunks": total_chunks,
        "chunks_succeeded": chunks_succeeded,
        "chunks_incomplete": chunks_incomplete,
        "chunks_failed": chunks_failed,
        "review_status": pr_status,
    }

    return pr_coverage, file_coverages


def format_coverage_section(
    pr_coverage: Dict[str, Any],
    file_coverages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format a compact Markdown coverage summary for the PR review."""
    total_files = pr_coverage.get("total_files", 0)
    files_fully = pr_coverage.get("files_fully_reviewed", 0)
    files_partially = pr_coverage.get("files_partially_reviewed", 0)
    files_not = pr_coverage.get("files_not_reviewed", 0)
    total_chunks = pr_coverage.get("total_chunks", 0)
    chunks_succ = pr_coverage.get("chunks_succeeded", 0)
    chunks_inc = pr_coverage.get("chunks_incomplete", 0)
    chunks_fail = pr_coverage.get("chunks_failed", 0)
    status_raw = pr_coverage.get("review_status", "FULLY_REVIEWED")
    status_display = status_raw.replace("_", " ").title()

    lines = [
        "### 📊 Review Coverage\n",
        f"- **Files:** {total_files} total ({files_fully} fully reviewed, {files_partially} partially reviewed, {files_not} not reviewed)",
        f"- **Chunks:** {total_chunks} total ({chunks_succ} completed, {chunks_inc} incomplete, {chunks_fail} failed)",
        f"- **Coverage Status:** {status_display}",
    ]

    if file_coverages:
        flagged_files = [
            fc for fc in file_coverages
            if fc.get("status") in {"PARTIALLY_REVIEWED", "NOT_REVIEWED"}
        ]
        if flagged_files:
            lines.append("\n*Files with incomplete or partial review:*")
            for fc in flagged_files:
                f_name = fc.get("file", "unknown")
                f_stat = fc.get("status", "").replace("_", " ").title()
                succ = fc.get("chunks_succeeded", 0)
                gen = fc.get("chunks_generated", 0)
                chunk_detail_parts = []
                for detail in fc.get("incomplete_chunk_details", []):
                    cid = detail.get("chunk_id", "?")
                    cat = detail.get("failure_category", "")
                    if cat and cat not in ("INCOMPLETE", "FAILED", "NONE"):
                        chunk_detail_parts.append(f"chunk {cid}: {cat}")
                    else:
                        chunk_detail_parts.append(f"chunk {cid}: incomplete")
                if chunk_detail_parts:
                    detail_str = "; ".join(chunk_detail_parts)
                    lines.append(
                        f"- `{f_name}`: {f_stat} ({succ}/{gen} chunks completed)"
                        f" — {detail_str}"
                    )
                else:
                    lines.append(
                        f"- `{f_name}`: {f_stat} ({succ}/{gen} chunks completed)"
                    )

    # Append an explicit warning when the review is incomplete.
    if status_raw in ("PARTIALLY_REVIEWED", "NOT_REVIEWED"):
        lines.append(
            "\n> ⚠️ **This review is incomplete.** Some chunks could not be "
            "reviewed. The findings above may not cover all changed code."
        )

    return "\n".join(lines)
