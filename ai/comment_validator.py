import os
import re
from groq import Groq

from github.comment_extractor import extract_comments_from_files

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
        comment_blocks.append(
            f"Comment {index}:\n"
            f"Line: {item['line']}\n"
            f"Comment: \"{item['comment']}\"\n"
            f"Associated Code:\n{item['code']}\n"
        )

    joined_comments = "\n".join(comment_blocks)

    return f"""You are a strict code comment accuracy validator.

File: {file_path}

Review each comment below against the associated code that follows or sits beside it.

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
    client: Groq,
    file_path: str,
    comments: list[dict]
) -> list[dict]:

    prompt = _build_file_validation_prompt(file_path, comments)

    # Sanitize surrogate characters that cannot be encoded as UTF-8.
    safe_prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
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


def validate_code_comments(
    repo_path: str,
    changed_files: list[str],
    source_branch: str,
    after_sha: str | None = None
) -> list[dict]:

    print("\n===== CODE COMMENT VALIDATION GATE =====")

    comments = extract_comments_from_files(
        repo_path,
        changed_files,
        source_branch,
        after_sha
    )

    if not comments:
        print("No reviewable comments found in changed files. Skipping gate.")
        return []

    print(f"Found {len(comments)} comment(s) to validate across changed files.")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "[WARNING] GROQ_API_KEY not found. "
            "Skipping comment validation gate."
        )
        return []

    comments_by_file: dict[str, list[dict]] = {}
    for item in comments:
        comments_by_file.setdefault(item["file"], []).append(item)

    client = Groq(api_key=api_key)
    all_mismatches: list[dict] = []

    try:
        for file_path, file_comments in comments_by_file.items():
            mismatches = _validate_file_comments(
                client,
                file_path,
                file_comments
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
