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

# Matches quoted string tokens (triple-double, triple-single, double-quoted,
# and single-quoted strings with escape awareness) so we can strip strings
# before checking for comment markers.
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

    repo = Repo(repo_path)

    try:
        return repo.git.show(f"{file_ref}:{file_path}")
    except Exception as e:
        print(
            f"[COMMENT EXTRACTOR] Could not read {file_path} at {file_ref}: {e}"
        )
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
    """Replace quoted string content with empty quotes so that comment
    markers (// or /*) that appear inside strings are no longer present
    when we scan the line for those markers.

    Example:
        'if "/*" in line:'  →  'if "" in line:'
    """
    return _STRING_LITERAL_RE.sub('""', line)


def _extract_comments_python(
    content: str,
    file_path: str
) -> list[dict]:
    """Extract comments from a Python source file using the `tokenize`
    module.  This is accurate because Python's own tokenizer correctly
    handles all string literal forms (raw strings, f-strings, byte strings,
    multiline strings, etc.) and emits only genuine COMMENT tokens.

    Falls back to the language-agnostic extractor if tokenization fails
    (e.g. the file contains a syntax error).
    """
    comments: list[dict] = []
    lines = content.splitlines()

    try:
        readline = io.StringIO(content).readline
        tokens = list(tokenize.generate_tokens(readline))
    except tokenize.TokenError as exc:
        print(
            f"[COMMENT EXTRACTOR] tokenize failed for {file_path} "
            f"({exc}); falling back to generic extractor."
        )
        return _extract_comments_generic(content, file_path)

    for tok_type, tok_string, tok_start, _tok_end, _line in tokens:
        if tok_type != tokenize.COMMENT:
            continue

        # Skip shebangs — tokenize emits them as COMMENT tokens but they are
        # not source-code comments and the generic extractor already skips them.
        if tok_string.startswith("#!"):
            continue

        # tok_start[0] is the 1-based line number of the comment.
        line_number = tok_start[0]
        # Strip the leading '#' that tokenize preserves in the token string.
        comment_text = tok_string.lstrip("#").strip()

        # Start scanning for following code at the line after this comment.
        # Line N (1-based) is at lines[N-1], so the next line (N+1) is at lines[N].
        line_idx = line_number
        code_snippet = _get_following_code(lines, line_idx)

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
    file_path: str
) -> list[dict]:
    """Language-agnostic comment extractor for non-Python files.

    Handles:
      - // line comments (JS, TS, Java, Kotlin, Go, C/C++, Swift, …)
      - /* … */ block comments
      - # line comments (Ruby, PHP, shell-like)
      - \"\"\"…\"\"\" / '''…''' docstrings (Python fallback only)

    String-literal awareness:
      Before checking whether a line contains /* or a // inline comment,
      the line is passed through _strip_string_literals() which replaces
      quoted string content with empty quotes.  This prevents comment
      markers that appear *inside* string literals from being treated as
      real comment markers.
    """
    comments: list[dict] = []
    lines = content.splitlines()
    in_block = False
    block_start_line = 0
    block_lines: list[str] = []
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        line_number = idx + 1

        if in_block:
            block_lines.append(line)
            if "*/" in line:
                in_block = False
                raw_block = "\n".join(block_lines)
                comment_text = _clean_block_comment(raw_block)
                code_snippet = _get_following_code(lines, idx + 1)
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

        # Strip string literals from the line before checking for /* or //
        # so that comment markers inside strings are not matched.
        line_no_strings = _strip_string_literals(line)
        stripped_no_strings = line_no_strings.strip()

        if "/*" in line_no_strings:
            if "*/" in line_no_strings:
                for match in _BLOCK_COMMENT_PATTERN.finditer(line):
                    comment_text = _clean_block_comment(match.group(0))
                    code_before = line[:match.start()].strip()
                    code_snippet = code_before or _get_following_code(
                        lines,
                        idx + 1
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
                block_lines = [line]
            idx += 1
            continue

        if stripped.startswith("//"):
            comment_text = stripped[2:].strip()
            code_snippet = _get_following_code(lines, idx + 1)
            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )
            idx += 1
            continue

        if stripped.startswith("#") and not stripped.startswith("#!"):
            comment_text = stripped[1:].strip()
            code_snippet = _get_following_code(lines, idx + 1)
            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )
            idx += 1
            continue

        # Inline // comment — check against string-stripped line to avoid
        # matching // inside string literals.
        inline_match = _INLINE_COMMENT_PATTERN.search(line_no_strings)
        if inline_match:
            # Use the match position in the stripped line to extract the
            # comment text from the *original* line at the same offset.
            comment_text = inline_match.group(1).strip()
            code_snippet = line[:inline_match.start()].strip()
            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )
            idx += 1
            continue

        docstring_match = re.match(
            r'^\s*(["\']{3})(.*?)\1\s*$',
            line
        )
        if docstring_match:
            comment_text = docstring_match.group(2).strip()
            code_snippet = _get_following_code(lines, idx + 1)
            _append_comment(
                comments,
                file_path,
                line_number,
                comment_text,
                code_snippet
            )
            idx += 1
            continue

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
                code_snippet = _get_following_code(lines, end_idx + 1)
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
    file_path: str
) -> list[dict]:
    """Extract comments from source file content.

    Routes Python files through the tokenize-based extractor for accuracy.
    All other supported languages use the generic regex-based extractor
    with string-literal awareness.
    """
    _, ext = os.path.splitext(file_path.lower())

    if ext in _PYTHON_EXTENSIONS:
        return _extract_comments_python(content, file_path)

    return _extract_comments_generic(content, file_path)


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

        content = get_file_content(repo_path, file_ref, file_path)
        if not content:
            continue

        file_comments = extract_comments_from_content(
            content,
            file_path
        )
        all_comments.extend(file_comments)

    return all_comments
