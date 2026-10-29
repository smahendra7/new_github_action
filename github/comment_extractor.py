import io
import os
import re
import tokenize
from git import Repo

from github.changed_files import is_automation_file

_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".swift", ".m", ".mm", ".kt", ".kts", ".rs", ".scala", ".vue"
}

_TODO_FIXME_PATTERN = re.compile(r"\b(TODO|FIXME)\b", re.IGNORECASE)
_INLINE_COMMENT_PATTERN = re.compile(r"(?<!:)(?<!\*)//(?!\s*/)(.*)$")
_BLOCK_COMMENT_PATTERN = re.compile(r"/\*+(.*?)\*/", re.DOTALL)

_STRING_LITERAL_RE = re.compile(
    r'""".*?"""|'
    r"'''.*?'''|"
    r'"(?:[^"\\]|\\.)*"|'
    r"'(?:[^'\\]|\\.)*'",
    re.DOTALL
)

_PYTHON_EXTENSIONS = {".py"}


def is_reviewable_file(file_path: str) -> bool:
    if is_automation_file(file_path):
        return False

    _, ext = os.path.splitext(file_path.lower())
    return ext in _CODE_EXTENSIONS


def get_file_content(
    repo_path: str,
    file_ref: str,
    file_path: str
) -> str | None:

    try:
        repo = Repo(repo_path)
        return repo.git.show(f"{file_ref}:{file_path}")
    except Exception as e:
        print(
            f"[COMMENT EXTRACTOR] Could not read "
            f"{file_path} at {file_ref}: {e}"
        )

    # Disk fallback if repo.git.show fails (e.g. testing or local workspace)
    if repo_path and file_path:
        full_path = os.path.join(repo_path, file_path.replace("/", os.sep))
        if os.path.isfile(full_path):
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception:
                pass

    return None


def _is_todo_fixme(comment_text: str) -> bool:
    return bool(_TODO_FIXME_PATTERN.search(comment_text))


def _clean_block_comment(raw_comment: str) -> str:
    cleaned = re.sub(r"^/\*\*?", "", raw_comment.strip())
    cleaned = re.sub(r"\*/$", "", cleaned.strip())
    cleaned = re.sub(r"^\*\s?", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def _get_following_code(
    lines: list[str],
    start_idx: int,
    max_lines: int = 15
) -> str:
    """Preserve the original helper for existing internal callers."""
    collected = []

    for i in range(start_idx, min(start_idx + max_lines, len(lines))):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            if collected:
                break
            continue

        if (
            stripped.startswith("//")
            or stripped.startswith("#")
            or stripped.startswith("/*")
            or stripped.startswith("*")
        ):
            if collected:
                break
            continue

        collected.append(line.rstrip())

    return "\n".join(collected).strip()


def _get_context_window(
    lines: list[str],
    start_line: int,
    end_line: int | None = None,
    before: int = 15,
    after: int = 15
) -> str:
    """
    Return source context around a comment.

    start_line and end_line are 1-based inclusive line numbers.
    For multiline comments, end_line should be the final line of
    the comment block.

    Includes up to 15 lines before the comment and 15 lines after
    it, while retaining the complete comment line or block.
    """
    if not lines:
        return ""

    total_lines = len(lines)

    start_line = min(max(1, start_line), total_lines)

    if end_line is None or end_line < start_line:
        end_line = start_line

    end_line = min(max(start_line, end_line), total_lines)

    first_line = max(1, start_line - before)
    last_line = min(total_lines, end_line + after)

    return "\n".join(
        line.rstrip()
        for line in lines[first_line - 1:last_line]
    )


def _append_comment(
    comments: list[dict],
    file_path: str,
    line_number: int,
    comment_text: str,
    code_snippet: str
) -> None:

    comment_text = comment_text.strip()

    if not comment_text or _is_todo_fixme(comment_text):
        return

    comments.append(
        {
            "file": file_path,
            "line": str(line_number),
            "comment": comment_text,
            "code": code_snippet or "(no associated code found)"
        }
    )


def _strip_string_literals(line: str) -> str:
    """Replace quoted string content before checking comment markers."""
    return _STRING_LITERAL_RE.sub('""', line)


def _extract_comments_python(
    content: str,
    file_path: str,
    start_line: int = 1
) -> list[dict]:
    """
    Extract Python comments using Python's tokenizer.

    This prevents strings containing #, //, or /* from being
    incorrectly interpreted as comments. If tokenization fails,
    use the existing generic fallback.
    """
    comments: list[dict] = []
    lines = content.splitlines()

    try:
        readline = io.StringIO(content).readline
        tokens = list(tokenize.generate_tokens(readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        print(
            f"[COMMENT EXTRACTOR] tokenize failed for {file_path} "
            f"({exc}); falling back to generic extractor."
        )
        return _extract_comments_generic(content, file_path, start_line=start_line)

    for tok_type, tok_string, tok_start, _tok_end, _line in tokens:
        if tok_type != tokenize.COMMENT:
            continue

        # Do not validate shebangs.
        if tok_string.startswith("#!"):
            continue

        local_line_number = tok_start[0]
        line_number = start_line + local_line_number - 1
        comment_text = tok_string.lstrip("#").strip()

        # Include the entire source line and surrounding context.
        # This also retains code preceding an inline Python comment.
        code_snippet = _get_context_window(lines, local_line_number)

        _append_comment(
            comments,
            file_path,
            line_number,
            comment_text,
            code_snippet
        )

    return comments


def _extract_comments_generic(
    content: str,
    file_path: str,
    start_line: int = 1
) -> list[dict]:
    """
    Extract comments from supported non-Python languages.

    Handles line comments, block comments, and the existing generic
    docstring fallback. Comment context includes up to 15 lines before
    and after the comment, retaining complete multiline blocks.
    """
    comments: list[dict] = []
    lines = content.splitlines()

    in_block = False
    block_start_line = 0
    block_local_start = 0
    block_lines: list[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        local_line_number = idx + 1
        line_number = start_line + idx

        if in_block:
            block_lines.append(line)

            if "*/" in line:
                in_block = False

                raw_block = "\n".join(block_lines)
                comment_text = _clean_block_comment(raw_block)

                code_snippet = _get_context_window(
                    lines,
                    block_local_start,
                    local_line_number
                )

                _append_comment(
                    comments,
                    file_path,
                    block_start_line,
                    comment_text,
                    code_snippet
                )

                block_lines = []

            idx += 1
            continue

        stripped = line.strip()
        line_no_strings = _strip_string_literals(line)

        # Block comments.
        if "/*" in line_no_strings:
            if "*/" in line_no_strings:
                for match in _BLOCK_COMMENT_PATTERN.finditer(line):
                    comment_text = _clean_block_comment(match.group(0))

                    # Retain the full line, including inline code.
                    code_snippet = _get_context_window(
                        lines,
                        local_line_number
                    )

                    _append_comment(
                        comments,
                        file_path,
                        line_number,
                        comment_text,
                        code_snippet
                    )
            else:
                in_block = True
                block_start_line = line_number
                block_local_start = local_line_number
                block_lines = [line]

            idx += 1
            continue

        # Standalone // comments.
        if stripped.startswith("//"):
            comment_text = stripped[2:].strip()

            code_snippet = _get_context_window(
                lines,
                local_line_number
            )

            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )

            idx += 1
            continue

        # Standalone # comments, excluding shebangs.
        if stripped.startswith("#") and not stripped.startswith("#!"):
            comment_text = stripped[1:].strip()

            code_snippet = _get_context_window(
                lines,
                local_line_number
            )

            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )

            idx += 1
            continue

        # Inline # comments in generic languages such as Ruby and PHP.
        # Ignore # characters inside quoted strings.
        inline_hash_match = re.search(r"#(.*)$", line_no_strings)

        if (
            inline_hash_match
            and line_no_strings[:inline_hash_match.start()].strip()
        ):
            comment_text = inline_hash_match.group(1).strip()

            code_snippet = _get_context_window(
                lines,
                local_line_number
            )

            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )

            idx += 1
            continue

        # Inline // comments.
        inline_match = _INLINE_COMMENT_PATTERN.search(line_no_strings)

        if inline_match:
            comment_text = inline_match.group(1).strip()

            code_snippet = _get_context_window(
                lines,
                local_line_number
            )

            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )

            idx += 1
            continue

        # Existing single-line docstring fallback.
        docstring_match = re.match(
            r'^\s*(["\']{3})(.*?)\1\s*$',
            line
        )

        if docstring_match:
            comment_text = docstring_match.group(2).strip()

            code_snippet = _get_context_window(
                lines,
                local_line_number
            )

            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )

            idx += 1
            continue

        # Existing multiline docstring fallback.
        multiline_docstring = False

        for quote in ('"""', "'''"):
            if stripped.startswith(quote) and not stripped.endswith(quote):
                doc_lines = [line]
                end_idx = idx

                for j in range(idx + 1, len(lines)):
                    doc_lines.append(lines[j])
                    end_idx = j

                    if lines[j].strip().endswith(quote):
                        break

                raw_doc = "\n".join(doc_lines)
                comment_text = raw_doc.strip().strip(quote).strip()

                code_snippet = _get_context_window(
                    lines,
                    local_line_number,
                    end_idx + 1
                )

                _append_comment(
                    comments,
                    file_path,
                    line_number,
                    comment_text,
                    code_snippet
                )

                idx = end_idx + 1
                multiline_docstring = True
                break

        if multiline_docstring:
            continue

        idx += 1

    return comments


def extract_comments_from_content(
    content: str,
    file_path: str,
    start_line: int = 1
) -> list[dict]:
    """Extract comments from source file content or snippet."""
    _, ext = os.path.splitext(file_path.lower())

    if ext in _PYTHON_EXTENSIONS:
        return _extract_comments_python(content, file_path, start_line=start_line)

    return _extract_comments_generic(content, file_path, start_line=start_line)


def extract_comments_from_snippet(
    snippet: str,
    file_path: str,
    start_line: int = 1
) -> list[dict]:
    """
    Extract comments only from a specific code snippet (such as an affected
    function or fallback window), offsetting line numbers by start_line.
    """
    return extract_comments_from_content(snippet, file_path, start_line=start_line)


def extract_comments_from_files(
    repo_path: str,
    changed_files: list[str],
    source_branch: str,
    after_sha: str | None = None
) -> list[dict]:

    repo = Repo(repo_path)

    if not after_sha:
        try:
            repo.remotes.origin.fetch()
        except Exception as e:
            print(f"[COMMENT EXTRACTOR] Origin fetch warning: {e}")

    file_ref = after_sha if after_sha else f"origin/{source_branch}"
    all_comments: list[dict] = []

    for file_path in changed_files:
        if not is_reviewable_file(file_path):
            continue

        content = get_file_content(
            repo_path,
            file_ref,
            file_path
        )

        if not content:
            continue

        file_comments = extract_comments_from_content(
            content,
            file_path
        )

        all_comments.extend(file_comments)

    return all_comments
