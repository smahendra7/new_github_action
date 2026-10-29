import os
import re
from groq import Groq
from openai import OpenAI

from github.comment_extractor import (
    extract_comments_from_snippet,
    get_file_content,
    is_reviewable_file,
)
from ai.context_extractor import (
    parse_code_scopes,
    find_enclosing_scope,
    _expand_start_for_comments_and_decorators,
)
from ai.chunk_manager import parse_diff_by_file

_MISMATCH_BLOCK_PATTERN = re.compile(
    r"File:\s*(?P<file>.+?)\s*\n"
    r"Line:\s*(?P<line>.+?)\s*\n"
    r'Comment:\s*"(?P<comment>.+?)"\s*\n'
    r"Code:\s*(?P<code>.+?)\s*\n"
    r"Reason:\s*(?P<reason>.+?)(?=\n---|\nFile:|\Z)",
    re.DOTALL | re.IGNORECASE
)


def _build_file_validation_prompt(file_path: str, comments: list[dict]) -> str:
    comment_blocks = []

    for index, item in enumerate(comments, start=1):
        func_tag = f" (Function: '{item['function_name']}')" if item.get("function_name") else ""
        code_label = f"Associated Function{func_tag}:" if item.get("function_name") else "Associated Code:"
        comment_blocks.append(
            f"Comment {index}:\n"
            f"Line: {item['line']}\n"
            f"Comment: \"{item['comment']}\"\n"
            f"{code_label}\n{item['code']}\n"
        )

    joined_comments = "\n".join(comment_blocks)

    return f"""You are a strict code comment accuracy validator.

File: {file_path}

Review each comment below against its associated function and code.

IMPORTANT — before evaluating any comment, verify it is a genuine source-code comment:
- A real comment starts with #, //, /*, or is a docstring in the appropriate language.
- If the "comment" text looks like a code fragment, a regular expression pattern,
  a string literal, a function call, or any other piece of executable code — it is
  NOT a comment. Skip it entirely and do not flag it.
- If you cannot clearly determine what the comment is describing, do NOT flag it.

Flag a mismatch ONLY when ALL of the following are true:
- The text is clearly a genuine human-written comment (not code, not a regex, not a string).
- The comment describes behavior, logic, or a value that the associated code provably does NOT perform.
- The mismatch is clear and specific — not a matter of phrasing or level of detail.
- A @param or @return doc tag does not match actual parameters, return type, or behavior.
- The comment is demonstrably outdated relative to the current code beneath it.

Do NOT flag:
- Do NOT report an issue merely because a comment or docstring is missing.
- Minor wording or style differences that do not change the meaning.
- Comments that are accurate but less detailed than the code.
- Comments that are a reasonable high-level summary of complex code.
- Anything that looks like a regex pattern, URL, code fragment, or string content.
- TODO/FIXME comments (already excluded).
- Comments where there is insufficient context to judge correctness.

Comments to review:
{joined_comments}

If you find one or more mismatches, respond using one block per mismatch in exactly this format:
---
File: {file_path}
Line: <line number(s)>
Comment: "<exact comment text>"
Code: <short snippet or summary of what the code actually does>
Reason: <clear, specific explanation of the mismatch>
---

If every comment accurately matches the code, respond with exactly:
NO_MISMATCHES
"""


def _parse_mismatch_blocks(response_text: str) -> list[dict]:
    mismatches = []

    for match in _MISMATCH_BLOCK_PATTERN.finditer(response_text):
        mismatches.append(
            {
                "file": match.group("file").strip(),
                "line": match.group("line").strip(),
                "comment": match.group("comment").strip(),
                "code": " ".join(
                    match.group("code").strip().splitlines()
                ),
                "reason": match.group("reason").strip()
            }
        )

    return mismatches


def _validate_file_comments(
    client: OpenAI,
    file_path: str,
    comments: list[dict]
) -> list[dict]:

    prompt = _build_file_validation_prompt(file_path, comments)

    # Sanitize surrogate characters that cannot be encoded as UTF-8.
    safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": safe_prompt
            }
        ],
        temperature=0.0,
        max_tokens=2048,
        top_p=1,
        stream=False
    )

    response_text = (
        completion.choices[0].message.content or ""
    ).strip()

    print("\n===== COMMENT VALIDATION LLM RESPONSE =====")
    print(f"File: {file_path}")
    print(response_text)
    print("===========================================\n")

    if "NO_MISMATCHES" in response_text.upper():
        return []

    return _parse_mismatch_blocks(response_text)


def _get_changed_line_numbers_by_file(diff: str) -> dict[str, list[int]]:
    """Return a mapping of normalised file path -> list of changed (added/modified)
    line numbers extracted from the PR diff.
    """
    result: dict[str, list[int]] = {}
    if not diff:
        return result

    parsed = parse_diff_by_file(diff)
    for norm_path, info in parsed.items():
        lines: list[int] = []
        for hunk in info.get("hunks", []):
            if hunk.modified_lines:
                lines.extend(hunk.modified_lines)
            else:
                lines.append(hunk.new_start)
        if lines:
            result[norm_path] = sorted(list(set(lines)))
    return result


def validate_code_comments(
    repo_path: str,
    changed_files: list[str],
    source_branch: str,
    after_sha: str | None = None,
    diff: str | None = None
) -> list[dict]:

    print("\n===== CODE COMMENT VALIDATION GATE =====")

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print(
            "[WARNING] NVIDIA_API_KEY not found. "
            "Skipping comment validation gate."
        )
        return []

    # 1. Identify developer-changed code from the PR diff FIRST
    _diff_changed_map = _get_changed_line_numbers_by_file(diff or "")

    def _changed_lines_for_file(fp: str) -> list[int]:
        """Look up changed lines for fp using flexible path matching."""
        norm = fp.strip().replace("\\", "/").lstrip("./")
        if norm in _diff_changed_map:
            return _diff_changed_map[norm]
        for k, v in _diff_changed_map.items():
            if k.endswith(norm) or norm.endswith(k):
                return v
        return []

    # NVIDIA OpenAI-compatible client (active provider)
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    all_mismatches: list[dict] = []
    file_ref = after_sha if after_sha else source_branch

    try:
        for file_path in changed_files:
            if not is_reviewable_file(file_path):
                continue

            changed_lines = _changed_lines_for_file(file_path)
            if not changed_lines:
                print(f"[COMMENT VALIDATOR] No changed lines in PR diff for {file_path}. Skipping.")
                continue

            raw_content = get_file_content(repo_path, file_ref, file_path) if repo_path else None
            if not raw_content:
                continue

            lines = raw_content.splitlines()
            total_lines = len(lines)
            if total_lines == 0:
                continue

            lines_dict = {i + 1: line for i, line in enumerate(lines)}
            scopes = parse_code_scopes(raw_content, file_path) if raw_content else []

            # 2. Identify changed functions FIRST using scope parsing
            affected_functions: dict[tuple, Any] = {}
            unmatched_lines: list[int] = []

            for cl in sorted(set(changed_lines)):
                valid_line = max(1, min(total_lines, cl))
                enclosing = find_enclosing_scope(scopes, valid_line) if scopes else None
                if enclosing and enclosing.scope_type == "function":
                    key = (enclosing.name, enclosing.start_line, enclosing.end_line)
                    if key not in affected_functions:
                        affected_functions[key] = enclosing
                else:
                    unmatched_lines.append(valid_line)

            # 3. Extract comments ONLY from affected functions and validate each independently
            for (func_name, s_line, e_line), scope in affected_functions.items():
                func_start = _expand_start_for_comments_and_decorators(lines_dict, scope.start_line)
                func_end = min(total_lines, scope.end_line)
                func_lines = lines[func_start - 1 : func_end]
                func_code = "\n".join(func_lines)

                # Extract comments ONLY from this affected function
                func_comments = extract_comments_from_snippet(
                    func_code,
                    file_path,
                    start_line=func_start
                )

                if not func_comments:
                    print(f"[COMMENT VALIDATOR] No comments found in changed function '{func_name}' ({file_path}).")
                    continue

                # Provide complete function code and relevant comments to AI
                for item in func_comments:
                    item["code"] = func_code
                    item["function_name"] = func_name

                print(f"[COMMENT VALIDATOR] Validating {len(func_comments)} comment(s) in function '{func_name}' ({file_path}).")
                mismatches = _validate_file_comments(
                    client,
                    file_path,
                    func_comments
                )
                all_mismatches.extend(mismatches)

            # 4. Fallback when function cannot be identified (at most 15 lines before/after)
            if unmatched_lines:
                raw_windows = []
                for cl in sorted(set(unmatched_lines)):
                    w_start = max(1, cl - 15)
                    w_end = min(total_lines, cl + 15)
                    raw_windows.append((w_start, w_end))

                merged_windows = []
                for w_start, w_end in raw_windows:
                    if not merged_windows:
                        merged_windows.append([w_start, w_end])
                    else:
                        prev_start, prev_end = merged_windows[-1]
                        if w_start <= prev_end:
                            merged_windows[-1][1] = max(prev_end, w_end)
                        else:
                            merged_windows.append([w_start, w_end])

                for w_start, w_end in merged_windows:
                    fb_lines = lines[w_start - 1 : w_end]
                    fb_code = "\n".join(fb_lines)

                    # Extract comments ONLY from this fallback context
                    fb_comments = extract_comments_from_snippet(
                        fb_code,
                        file_path,
                        start_line=w_start
                    )

                    filtered_fb_comments = []
                    for item in fb_comments:
                        ln_str = item.get("line", "1")
                        line_match = re.search(r"\d+", str(ln_str))
                        ln = int(line_match.group()) if line_match else 1
                        enc = find_enclosing_scope(scopes, ln) if scopes else None
                        # Exclude comments that belong to unchanged functions
                        if enc and enc.scope_type == "function":
                            continue
                        item["code"] = fb_code
                        item["function_name"] = None
                        filtered_fb_comments.append(item)

                    if not filtered_fb_comments:
                        continue

                    print(f"[COMMENT VALIDATOR] Validating {len(filtered_fb_comments)} comment(s) in fallback window lines {w_start}..{w_end} ({file_path}).")
                    mismatches = _validate_file_comments(
                        client,
                        file_path,
                        filtered_fb_comments
                    )
                    all_mismatches.extend(mismatches)

    except Exception as e:
        print(
            f"[WARNING] Comment validation error: {e}. "
            "Allowing review to continue."
        )
        return []

    if all_mismatches:
        print(
            f"Comment Validation Result: FAIL ({len(all_mismatches)} mismatch(es))"
        )
    else:
        print("Comment Validation Result: PASS")

    return all_mismatches


def format_comment_validation_failure(mismatches: list[dict]) -> str:
    blocks = []

    for item in mismatches:
        blocks.append(
            f"File: {item['file']}\n"
            f"Line: {item['line']}\n"
            f"Comment: \"{item['comment']}\"\n"
            f"Code: {item['code']}\n"
            f"Reason: {item['reason']}"
        )

    mismatch_text = "\n\n".join(blocks)

    return (
        "\u274c **Comment Validation Failed**\n\n"
        f"{mismatch_text}\n\n"
        "Please update the code comment (or fix the code logic) so they match, "
        "then push a new commit to re-trigger the review."
    )
