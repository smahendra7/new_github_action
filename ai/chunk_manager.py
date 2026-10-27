"""
ai/chunk_manager.py

Manages change-aware chunking for large PR contexts.
Extracts diff hunks, groups nearby changed regions with surrounding context,
and orchestrates parallel chunk review calls with controlled concurrency and memory aggregation.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ai.context_config import (
    SAFE_INPUT_TOKEN_LIMIT,
    MAX_CONCURRENCY,
    CONTEXT_WINDOW_LINES,
    CHUNK_OVERLAP_LINES,
    SAFE_PROMPT_CHAR_LIMIT,
)
from ai.token_counter import count_tokens
from ai.prompt_builder import build_review_prompt, build_chunk_prompt
from ai.result_aggregator import aggregate_reviews
from ai.reviewer import ReviewResult, review_code


@dataclass
class DiffHunk:
    """Represents a single diff hunk in a file."""
    new_start: int
    new_end: int
    hunk_text: str
    modified_lines: List[int] = field(default_factory=list)


@dataclass
class ReviewChunk:
    """Represents an individual chunk sent to the AI reviewer."""
    chunk_id: int
    total_chunks: int
    file_path: str
    changed_files: List[str]
    code_snippet: str
    diff_snippet: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


def _normalize_path(path: str) -> str:
    """Normalize file path for consistent matching."""
    return (path or "").strip().replace("\\", "/").lstrip("./")


def parse_diff_by_file(diff_text: str) -> Dict[str, Dict]:
    """
    Parse a raw git diff into a dictionary keyed by normalized file path:
    {
        "file/path.py": {
            "full_diff": str,
            "hunks": List[DiffHunk]
        }
    }
    """
    if not diff_text or not diff_text.strip():
        return {}

    blocks = re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE)
    file_diffs: Dict[str, Dict] = {}

    hunk_header_pattern = re.compile(
        r"^@@ -\d+(?:,\d+)? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
    )

    for block in blocks:
        if not block.strip():
            continue

        first_line = block.splitlines()[0]
        if not first_line.startswith("diff --git"):
            continue

        try:
            rel_path = first_line.split(" b/")[1].strip()
        except Exception:
            continue

        norm_path = _normalize_path(rel_path)
        hunks: List[DiffHunk] = []

        hunk_chunks = re.split(r"(?=^@@ )", block, flags=re.MULTILINE)

        for hunk in hunk_chunks:
            if not hunk.startswith("@@"):
                continue

            lines = hunk.splitlines()
            hunk_match = hunk_header_pattern.match(lines[0])

            if not hunk_match:
                continue

            new_start = int(hunk_match.group("new_start"))
            new_count_str = hunk_match.group("new_count")
            new_count = int(new_count_str) if new_count_str is not None else 1

            new_end = new_start if new_count == 0 else new_start + new_count - 1

            modified_lines: List[int] = []
            current_line = new_start

            for line in lines[1:]:
                if line.startswith("+") and not line.startswith("+++"):
                    modified_lines.append(current_line)
                    current_line += 1
                elif line.startswith(" ") or line == "":
                    current_line += 1
                elif line.startswith("-") and not line.startswith("---"):
                    pass

            hunks.append(
                DiffHunk(
                    new_start=new_start,
                    new_end=new_end,
                    hunk_text=hunk,
                    modified_lines=modified_lines,
                )
            )

        file_diffs[norm_path] = {
            "full_diff": block.strip(),
            "hunks": hunks,
        }

    return file_diffs


def parse_full_code_files(full_code: str) -> Dict[str, Dict]:
    """
    Parse the line-numbered full_code string produced by extract_full_code:

    FILE: <rel_path>
    ```<lang>
    L1: ...
    L2: ...
    ```

    Returns a dictionary mapping normalized paths to language, line mapping,
    code lines, and total line count.
    """
    if not full_code or not full_code.strip():
        return {}

    blocks = re.split(r"(?=^FILE:\s*)", full_code, flags=re.MULTILINE)
    parsed: Dict[str, Dict] = {}

    header_pattern = re.compile(r"^FILE:\s*([^\n\r]+)")
    code_block_start = re.compile(r"^```(\w*)")

    for block in blocks:
        block_clean = block.strip()

        if not block_clean:
            continue

        match = header_pattern.match(block_clean)

        if not match:
            continue

        rel_path = _normalize_path(match.group(1))
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

            line_match = re.match(r"^L(\d+):\s?(.*)$", line)

            if line_match:
                line_number = int(line_match.group(1))
                code_lines.append((line_number, line))
                next_line_number = line_number + 1
            else:
                code_lines.append((next_line_number, line))
                next_line_number += 1

        total_lines = code_lines[-1][0] if code_lines else 0
        lines_dict = {line_number: line_text for line_number, line_text in code_lines}

        parsed[rel_path] = {
            "lang": language,
            "lines_dict": lines_dict,
            "code_lines": code_lines,
            "total_lines": total_lines,
        }

    return parsed


def _format_code_block(
    file_path: str,
    lang: str,
    selected_lines: List[str],
) -> str:
    """Format selected line-numbered lines as a FILE code block."""
    inner = "\n".join(selected_lines)
    return f"FILE: {file_path}\n```{lang}\n{inner}\n```"


def _slice_diff_to_line_range(
    diff_text: str,
    start_line: Optional[int],
    end_line: Optional[int],
) -> str:
    """
    Return diff hunk lines relevant to a chunk's new-file line range.

    Hunk headers are rebuilt to match the selected body. Context and added lines
    are selected by their new-file line number. Deletions are associated with
    the insertion point at which they occur, including a boundary line so
    deletions are not lost when adjacent chunks overlap.

    If the input is not parseable as unified diff hunks, return it unchanged.
    """
    if not diff_text or start_line is None or end_line is None:
        return diff_text

    hunk_blocks = re.split(r"(?=^@@ )", diff_text, flags=re.MULTILINE)
    selected_hunks: List[str] = []
    found_hunk = False

    header_pattern = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
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
        old_count = int(match.group(2)) if match.group(2) is not None else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) is not None else 1
        trailing_header = match.group(5)

        actual_old_count = 0
        actual_new_count = 0

        for body_line in lines[1:]:
            if body_line.startswith((" ", "-")):
                actual_old_count += 1
            if body_line.startswith((" ", "+")):
                actual_new_count += 1

        # Do not slice malformed or truncated hunks using unreliable counts.
        if (
            actual_old_count != old_count
            or actual_new_count != new_count
        ):
            return diff_text

        old_line = old_start
        new_line = new_start
        selected_body: List[str] = []
        selected_old_lines: List[int] = []
        selected_new_lines: List[int] = []

        for body_line in lines[1:]:
            if body_line.startswith("\\"):
                if selected_body:
                    selected_body.append(body_line)
                continue

            include_line = False

            if body_line.startswith(" "):
                selected_old_lines.append(old_line)
                selected_new_lines.append(new_line)

                include_line = start_line <= new_line <= end_line
                old_line += 1
                new_line += 1

            elif body_line.startswith("+"):
                selected_new_lines.append(new_line)
                include_line = start_line <= new_line <= end_line
                new_line += 1

            elif body_line.startswith("-"):
                # Associate deletions with the next new-file insertion point.
                insertion_point = new_line
                include_line = (
                    start_line <= insertion_point <= end_line
                    or insertion_point == start_line
                    or insertion_point == end_line + 1
                )
                old_line += 1

            else:
                # Unknown lines indicate a malformed hunk; preserve the original.
                return diff_text

            if include_line:
                selected_body.append(body_line)

        if not selected_body:
            continue

        selected_old_count = sum(
            1 for line in selected_body
            if line.startswith((" ", "-"))
        )
        selected_new_count = sum(
            1 for line in selected_body
            if line.startswith((" ", "+"))
        )

        old_header = (
            f"{old_start},{selected_old_count}"
            if selected_old_count != 1
            else str(old_start)
        )
        new_header = (
            f"{new_start},{selected_new_count}"
            if selected_new_count != 1
            else str(new_start)
        )

        selected_header = (
            f"@@ -{old_header} +{new_header} @@{trailing_header}"
        )

        selected_hunks.append(
            "\n".join([selected_header] + selected_body)
        )

    if not found_hunk:
        return diff_text

    if not selected_hunks:
        return ""

    return "\n".join(selected_hunks)


def _subdivide_large_region(
    file_path: str,
    lang: str,
    region_lines: List[str],
    region_diff: str,
    is_prompt_safe: Callable[[str], bool],
    overlap_lines: int = CHUNK_OVERLAP_LINES,
) -> List[Tuple[str, str, int, int]]:
    """
    Subdivide an oversized code region into smaller chunks.

    Each chunk includes an overlap with its neighbor to preserve context.
    The function attempts to keep each generated prompt within the configured
    token and character limits.
    """
    if not region_lines:
        return []

    parsed_lines: List[Tuple[int, str]] = []

    for index, line in enumerate(region_lines, start=1):
        match = re.match(r"^L(\d+):", line)

        if match:
            parsed_lines.append((int(match.group(1)), line))
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

        for end_index in range(start_index + 1, total_lines + 1):
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
            # Even a single line exceeds the configured limit. Keep it as a
            # chunk; the downstream reviewer can report a provider-side failure.
            best_end_index = start_index + 1

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

        # Move back by the overlap amount to retain neighboring context.
        start_index = max(
            start_index + 1,
            best_end_index - max(0, overlap_lines),
        )

    return chunks


def create_review_chunks(
    full_code: str,
    changed_files: List[str],
    diff: str,
    safe_token_limit: int = SAFE_INPUT_TOKEN_LIMIT,
    context_lines: int = CONTEXT_WINDOW_LINES,
    safe_char_limit: Optional[int] = None,
) -> List[ReviewChunk]:
    """
    Break down changed files and diffs into manageable ReviewChunk objects.

    Groups nearby hunks with surrounding context and subdivides oversized
    regions while preserving file and line context.
    """
    parsed_files = parse_full_code_files(full_code)
    parsed_diff = parse_diff_by_file(diff)

    def is_prompt_safe(prompt_text: str) -> bool:
        tokens_ok = count_tokens(prompt_text) <= safe_token_limit

        if safe_char_limit is not None:
            return tokens_ok and len(prompt_text) <= safe_char_limit

        return tokens_ok

    raw_chunk_specs: List[Dict] = []

    for rel_path in changed_files:
        norm_path = _normalize_path(rel_path)

        file_info = parsed_files.get(norm_path)
        diff_info = parsed_diff.get(norm_path)

        if not file_info:
            for path, info in parsed_files.items():
                if path.endswith(norm_path) or norm_path.endswith(path):
                    file_info = info
                    norm_path = path
                    break

        if not diff_info:
            for path, info in parsed_diff.items():
                if path.endswith(norm_path) or norm_path.endswith(path):
                    diff_info = info
                    break

        if not file_info:
            if diff_info:
                raw_chunk_specs.append(
                    {
                        "file_path": rel_path,
                        "code_snippet": (
                            f"FILE: {rel_path}\n(Full content unavailable)"
                        ),
                        "diff_snippet": diff_info["full_diff"],
                        "start_line": None,
                        "end_line": None,
                    }
                )
            else:
                print(
                    f"[CHUNK MANAGER] Skipping {rel_path}: "
                    "neither full content nor a parsed diff was available."
                )

            continue

        lang = file_info["lang"]
        lines_dict = file_info["lines_dict"]
        code_lines = file_info["code_lines"]
        total_lines = file_info["total_lines"]
        file_diff = diff_info["full_diff"] if diff_info else ""
        hunks = diff_info["hunks"] if diff_info else []

        if hunks:
            hunk_regions: List[Tuple[int, int, List[DiffHunk]]] = []

            for hunk in hunks:
                region_start = max(1, hunk.new_start - context_lines)
                region_end = min(
                    total_lines,
                    hunk.new_end + context_lines,
                )

                if not hunk_regions:
                    hunk_regions.append(
                        (region_start, region_end, [hunk])
                    )
                else:
                    last_start, last_end, last_hunks = hunk_regions[-1]

                    if region_start <= last_end + context_lines:
                        hunk_regions[-1] = (
                            last_start,
                            max(last_end, region_end),
                            last_hunks + [hunk],
                        )
                    else:
                        hunk_regions.append(
                            (region_start, region_end, [hunk])
                        )

            for region_start, region_end, region_hunks in hunk_regions:
                region_selected_lines = [
                    lines_dict[line_number]
                    for line_number in range(region_start, region_end + 1)
                    if line_number in lines_dict
                ]

                region_diff_text = "\n\n".join(
                    hunk.hunk_text for hunk in region_hunks
                )

                code_snippet = _format_code_block(
                    rel_path,
                    lang,
                    region_selected_lines,
                )

                candidate_prompt = build_review_prompt(
                    code=code_snippet,
                    changed_files=[rel_path],
                    diff=region_diff_text,
                )

                if is_prompt_safe(candidate_prompt):
                    raw_chunk_specs.append(
                        {
                            "file_path": rel_path,
                            "code_snippet": code_snippet,
                            "diff_snippet": region_diff_text,
                            "start_line": region_start,
                            "end_line": region_end,
                        }
                    )
                else:
                    subdivided = _subdivide_large_region(
                        file_path=rel_path,
                        lang=lang,
                        region_lines=region_selected_lines,
                        region_diff=region_diff_text,
                        is_prompt_safe=is_prompt_safe,
                        overlap_lines=CHUNK_OVERLAP_LINES,
                    )

                    for code, diff_part, start_line, end_line in subdivided:
                        raw_chunk_specs.append(
                            {
                                "file_path": rel_path,
                                "code_snippet": code,
                                "diff_snippet": diff_part,
                                "start_line": start_line,
                                "end_line": end_line,
                            }
                        )

        else:
            file_code_lines = [item[1] for item in code_lines]
            code_snippet = _format_code_block(
                rel_path,
                lang,
                file_code_lines,
            )

            candidate_prompt = build_review_prompt(
                code=code_snippet,
                changed_files=[rel_path],
                diff=file_diff,
            )

            if is_prompt_safe(candidate_prompt):
                raw_chunk_specs.append(
                    {
                        "file_path": rel_path,
                        "code_snippet": code_snippet,
                        "diff_snippet": file_diff,
                        "start_line": 1 if total_lines > 0 else None,
                        "end_line": total_lines if total_lines > 0 else None,
                    }
                )
            else:
                subdivided = _subdivide_large_region(
                    file_path=rel_path,
                    lang=lang,
                    region_lines=file_code_lines,
                    region_diff=file_diff,
                    is_prompt_safe=is_prompt_safe,
                    overlap_lines=CHUNK_OVERLAP_LINES,
                )

                for code, diff_part, start_line, end_line in subdivided:
                    raw_chunk_specs.append(
                        {
                            "file_path": rel_path,
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

    return chunks


def _review_single_chunk(chunk: ReviewChunk) -> str:
    """
    Process an individual chunk request using review_code with live streaming.
    """
    chunk_prompt = build_chunk_prompt(chunk)
    token_count = count_tokens(chunk_prompt)

    print(
        f"\n[CHUNK {chunk.chunk_id}/{chunk.total_chunks}] Starting AI review "
        f"for {chunk.file_path} "
        f"(Lines {chunk.start_line or 'all'}..{chunk.end_line or 'all'}, "
        f"{token_count} input tokens, {len(chunk_prompt)} chars)..."
    )

    return review_code(chunk_prompt)


def _infer_review_status(result: object) -> Tuple[str, str]:
    """
    Determine a chunk result's status and text.

    ReviewResult instances carry an explicit status. Plain strings are handled
    conservatively: empty results and explicit error messages are not complete.
    """
    if result is None:
        return "incomplete", ""

    review_text = str(result).strip()
    status = getattr(result, "status", None)

    if status:
        return str(status).lower(), review_text

    if not review_text or review_text == "No review generated.":
        return "incomplete", review_text

    if (
        "Error occurred during AI code review" in review_text
        or "inference engine failure" in review_text.lower()
    ):
        return "failed", review_text

    if re.search(
        r"(?im)^###\s+.*Review Notices\s*$",
        review_text,
    ):
        if re.search(r"\(failed\)", review_text, re.IGNORECASE):
            return "failed", review_text
        return "incomplete", review_text

    # A nonempty response is not necessarily a clean response. The result
    # aggregator determines whether its content is explicitly clean.
    return "completed", review_text


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
    Orchestrate parallel chunked AI review.

    Every chunk receives a result record. Exceptions and empty responses are
    retained as failures or incomplete results so the final aggregator can
    produce review notices without discarding findings from other chunks.
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
        print("\n[CHUNK MANAGER] No review chunks could be generated.")
        return ReviewResult("No review generated.", status="incomplete")

    total_chunks = len(chunks)
    concurrency = max(1, int(max_concurrency))

    print(
        f"\n[CHUNK MANAGER] Large context detected. Created {total_chunks} "
        f"chunk(s) for AI review (max concurrency: {concurrency}).\n"
    )

    # Store a structured result for every chunk, using its stable index so
    # results remain in chunk order even when futures finish out of order.
    chunk_result_dicts: List[Optional[Dict]] = [None] * total_chunks

    for batch_start in range(0, total_chunks, concurrency):
        batch_chunks = chunks[batch_start:batch_start + concurrency]
        chunk_indices = list(range(batch_start, batch_start + len(batch_chunks)))
        chunk_ids = [chunk.chunk_id for chunk in batch_chunks]

        print(
            f"\n[CHUNK MANAGER] Running batch: Chunks {chunk_ids} concurrently "
            f"(concurrency: {len(batch_chunks)}/{concurrency})..."
        )

        with ThreadPoolExecutor(max_workers=len(batch_chunks)) as executor:
            future_to_chunk_info = {
                executor.submit(_review_single_chunk, chunk): (index, chunk)
                for chunk, index in zip(batch_chunks, chunk_indices)
            }

            for future in as_completed(future_to_chunk_info):
                index, chunk = future_to_chunk_info[future]

                try:
                    result = future.result()
                    status, review_text = _infer_review_status(result)

                    chunk_result_dicts[index] = {
                        "chunk_id": chunk.chunk_id,
                        "file": chunk.file_path,
                        "status": status,
                        "review": review_text,
                    }

                except Exception as chunk_error:
                    error_text = (
                        f"Error occurred during AI code review: {chunk_error}"
                    )

                    print(
                        f"\n[CHUNK ERROR] Chunk {chunk.chunk_id} failed "
                        f"with error: {chunk_error}"
                    )

                    chunk_result_dicts[index] = {
                        "chunk_id": chunk.chunk_id,
                        "file": chunk.file_path,
                        "status": "failed",
                        "review": error_text,
                    }

    # Any missing record indicates an unexpected execution gap. Represent it
    # explicitly instead of silently dropping the chunk.
    for index, item in enumerate(chunk_result_dicts):
        if item is None:
            chunk = chunks[index]
            chunk_result_dicts[index] = {
                "chunk_id": chunk.chunk_id,
                "file": chunk.file_path,
                "status": "incomplete",
                "review": "The chunk did not produce a result.",
            }

    # Aggregate structured results so failed, incomplete, and ambiguous
    # chunks become visible in Review Notices while valid findings survive.
    complete_result_list: List[Dict] = [
        item for item in chunk_result_dicts if item is not None
    ]

    aggregated_review = aggregate_reviews(complete_result_list)

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
        item["status"] == "failed"
        for item in complete_result_list
    )

    has_incomplete = any(
        item["status"] != "completed"
        for item in complete_result_list
    )

    if has_failed or re.search(r"\(failed\)", notices_text, re.IGNORECASE):
        overall_status = "failed"
    elif (
        has_incomplete
        or notices_match
        or aggregated_review.strip() == "No review generated."
    ):
        overall_status = "incomplete"
    else:
        overall_status = "completed"

    print("\n[CHUNK MANAGER] Completed aggregation of all chunk reviews.\n")

    return ReviewResult(aggregated_review, status=overall_status)