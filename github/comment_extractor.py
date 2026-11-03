
import io
import os
import re
import tokenize

from git import Repo

from github.changed_files import (
    is_automation_file,
    _has_commit,
    _attempt_git_fetch_recovery,
)


_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php",
    ".swift", ".m", ".mm", ".kt", ".kts", ".rs", ".scala",
    ".vue", ".html", ".css", ".scss", ".sql", ".sh",
}

_PYTHON_EXTENSIONS = {".py"}
_HASH_COMMENT_EXTENSIONS = {".rb", ".php", ".sh"}

_TODO_FIXME_PATTERN = re.compile(
    r"\b(TODO|FIXME)\b",
    re.IGNORECASE,
)


def _normalize_path(file_path: str) -> str:
    """Normalize a repository-relative file path."""
    if not file_path:
        return ""

    normalized = str(file_path).strip().replace("\\", "/")

    while normalized.startswith("./"):
        normalized = normalized[2:]

    return normalized.lstrip("/")


def _is_safe_relative_path(file_path: str) -> bool:
    """Prevent paths from escaping the repository workspace."""
    normalized = _normalize_path(file_path)

    if not normalized:
        return False

    return not any(
        part == ".."
        for part in normalized.split("/")
    )


def is_reviewable_file(
    file_path: str,
    repo_name: str | None = None,
) -> bool:
    """
    Check whether a changed file is a supported source-code file.

    Automation exclusions are applied only when the repository is
    explicitly configured as the automation repository.
    """
    normalized = _normalize_path(file_path)

    if not normalized or not _is_safe_relative_path(normalized):
        return False

    if is_automation_file(normalized, repo_name=repo_name):
        return False

    _, extension = os.path.splitext(normalized.lower())

    return extension in _CODE_EXTENSIONS


def get_file_content(
    repo_path: str,
    file_ref: str,
    file_path: str,
) -> str | None:
    """
    Read a file from the exact requested Git ref.

    If file_ref is supplied and Git cannot read that version, this
    function does not fall back to the workspace copy. This prevents
    code from another commit from being sent to the AI.
    """
    normalized_path = _normalize_path(file_path)

    if not normalized_path or not _is_safe_relative_path(normalized_path):
        print(
            f"[COMMENT EXTRACTOR] Rejected invalid file path: {file_path}"
        )
        return None

    if not repo_path:
        return None

    if file_ref:
        try:
            repo = Repo(repo_path)

            content = repo.git.show(
                f"{file_ref}:{normalized_path}"
            )

            return content

        except Exception as exc:
            print(
                f"[COMMENT EXTRACTOR] Could not read "
                f"{normalized_path} at {file_ref}: {exc}"
            )

            # Never read a different version from disk when an exact
            # source ref was requested.
            return None

    # Disk access is allowed only when no Git ref was supplied.
    full_path = os.path.realpath(
        os.path.join(
            repo_path,
            normalized_path.replace("/", os.sep),
        )
    )

    repo_root = os.path.realpath(repo_path)

    try:
        if os.path.commonpath([repo_root, full_path]) != repo_root:
            return None
    except ValueError:
        return None

    if not os.path.isfile(full_path):
        return None

    try:
        with open(
            full_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as source_file:
            return source_file.read()

    except Exception as exc:
        print(
            f"[COMMENT EXTRACTOR] Disk read failed for "
            f"{normalized_path}: {exc}"
        )
        return None


def _is_todo_fixme(comment_text: str) -> bool:
    """Return True for TODO/FIXME comments."""
    return bool(
        _TODO_FIXME_PATTERN.search(comment_text or "")
    )


def _clean_block_comment(raw_comment: str) -> str:
    """Remove block-comment delimiters and leading documentation stars."""
    cleaned = (raw_comment or "").strip()

    cleaned = re.sub(
        r"^/\*\*?",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"\*/$",
        "",
        cleaned.strip(),
    )

    cleaned = re.sub(
        r"^\s*\*\s?",
        "",
        cleaned,
        flags=re.MULTILINE,
    )

    return cleaned.strip()


def _get_following_code(
    lines: list[str],
    start_idx: int,
    max_lines: int = 15,
) -> str:
    """Preserve the original helper for existing internal callers."""
    collected = []

    for index in range(
        start_idx,
        min(start_idx + max_lines, len(lines)),
    ):
        line = lines[index]
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
    after: int = 15,
) -> str:
    """
    Return source context around a comment.

    Line numbers are 1-based and inclusive.
    """
    if not lines:
        return ""

    total_lines = len(lines)

    start_line = min(
        max(1, start_line),
        total_lines,
    )

    if end_line is None or end_line < start_line:
        end_line = start_line

    end_line = min(
        max(start_line, end_line),
        total_lines,
    )

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
    code_snippet: str,
) -> None:
    """Add a comment record after applying basic validation."""
    comment_text = (comment_text or "").strip()

    if not comment_text:
        return

    if _is_todo_fixme(comment_text):
        return

    comments.append(
        {
            "file": _normalize_path(file_path),
            "line": str(line_number),
            "comment": comment_text,
            "code": code_snippet or "(no associated code found)",
        }
    )


def _extract_comments_python(
    content: str,
    file_path: str,
    start_line: int = 1,
) -> list[dict]:
    """
    Extract Python comments using Python's tokenizer.

    Tokenization avoids treating # characters inside strings as comments.
    """
    comments: list[dict] = []
    lines = content.splitlines()

    try:
        readline = io.StringIO(content).readline
        tokens = tokenize.generate_tokens(readline)

        for token_info in tokens:
            if token_info.type != tokenize.COMMENT:
                continue

            token_text = token_info.string

            # Do not validate shebangs.
            if token_text.startswith("#!"):
                continue

            local_line_number = token_info.start[0]
            actual_line_number = (
                start_line + local_line_number - 1
            )

            comment_text = token_text.lstrip("#").strip()

            code_snippet = _get_context_window(
                lines,
                local_line_number,
            )

            _append_comment(
                comments,
                file_path,
                actual_line_number,
                comment_text,
                code_snippet,
            )

    except (
        tokenize.TokenError,
        IndentationError,
        SyntaxError,
    ) as exc:
        print(
            f"[COMMENT EXTRACTOR] Python tokenization failed for "
            f"{file_path}: {exc}. Using generic extraction."
        )

        return _extract_comments_generic(
            content,
            file_path,
            start_line=start_line,
        )

    return comments


def _extract_comments_generic(
    content: str,
    file_path: str,
    start_line: int = 1,
) -> list[dict]:
    """
    Extract comments from supported non-Python languages.

    Supports:
      - // line comments
      - /* ... */ block comments
      - # comments for Ruby, PHP, and shell files

    Comment markers inside quoted strings are ignored.
    """
    comments: list[dict] = []
    lines = content.splitlines()

    _, extension = os.path.splitext(file_path.lower())

    supports_hash_comments = (
        extension in _HASH_COMMENT_EXTENSIONS
    )

    in_block_comment = False
    block_start_line = 0
    block_local_start = 0
    block_parts: list[str] = []

    # Preserve multiline template strings and backtick strings.
    multiline_quote: str | None = None

    for index, line in enumerate(lines):
        local_line_number = index + 1
        actual_line_number = start_line + index

        comment_parts: list[str] = []
        position = 0

        while position < len(line):

            # Continue an existing block comment.
            if in_block_comment:
                close_index = line.find("*/", position)

                if close_index == -1:
                    block_parts.append(
                        line[position:]
                    )
                    position = len(line)
                    break

                block_parts.append(
                    line[position:close_index]
                )

                in_block_comment = False

                raw_comment = "\n".join(block_parts)
                comment_text = _clean_block_comment(
                    "/*" + raw_comment + "*/"
                )

                code_snippet = _get_context_window(
                    lines,
                    block_local_start,
                    local_line_number,
                )

                _append_comment(
                    comments,
                    file_path,
                    block_start_line,
                    comment_text,
                    code_snippet,
                )

                block_parts = []
                position = close_index + 2
                continue

            # Continue a multiline quoted string.
            if multiline_quote:
                close_index = line.find(
                    multiline_quote,
                    position,
                )

                if close_index == -1:
                    position = len(line)
                    break

                position = close_index + len(multiline_quote)
                multiline_quote = None
                continue

            char = line[position]

            # Skip quoted strings.
            if char in ("'", '"', "`"):
                if (
                    char in ("'", '"')
                    and line[position:position + 3] == char * 3
                ):
                    quote = char * 3
                    close_index = line.find(
                        quote,
                        position + 3,
                    )

                    if close_index == -1:
                        multiline_quote = quote
                        position = len(line)
                        break

                    position = close_index + 3
                    continue

                if char == "`":
                    close_index = line.find("`", position + 1)

                    if close_index == -1:
                        multiline_quote = "`"
                        position = len(line)
                        break

                    position = close_index + 1
                    continue

                # Single-line quoted string.
                position += 1

                while position < len(line):
                    if line[position] == "\\":
                        position += 2
                        continue

                    if line[position] == char:
                        position += 1
                        break

                    position += 1

                continue

            # Start of a block comment.
            if line.startswith("/*", position):
                in_block_comment = True
                block_start_line = actual_line_number
                block_local_start = local_line_number

                position += 2

                close_index = line.find("*/", position)

                if close_index == -1:
                    block_parts = [line[position:]]
                    position = len(line)
                    break

                block_parts = [
                    line[position:close_index]
                ]

                in_block_comment = False

                comment_text = _clean_block_comment(
                    "/*" + block_parts[0] + "*/"
                )

                code_snippet = _get_context_window(
                    lines,
                    local_line_number,
                )

                _append_comment(
                    comments,
                    file_path,
                    actual_line_number,
                    comment_text,
                    code_snippet,
                )

                block_parts = []
                position = close_index + 2
                continue

            # Start of a // line comment.
            if line.startswith("//", position):
                comment_text = line[position + 2:].strip()

                comment_parts.append(comment_text)
                break

            # Hash comments are supported only for languages that use them.
            if supports_hash_comments and char == "#":
                # A shell shebang is not a review comment.
                if (
                    extension == ".sh"
                    and line.startswith("#!")
                ):
                    break

                comment_text = line[position + 1:].strip()
                comment_parts.append(comment_text)
                break

            position += 1

        if comment_parts:
            code_snippet = _get_context_window(
                lines,
                local_line_number,
            )

            _append_comment(
                comments,
                file_path,
                actual_line_number,
                "\n".join(comment_parts),
                code_snippet,
            )

    # An unclosed block comment is still useful source information.
    if in_block_comment and block_parts:
        comment_text = "\n".join(block_parts).strip()

        code_snippet = _get_context_window(
            lines,
            block_local_start,
            len(lines),
        )

        _append_comment(
            comments,
            file_path,
            block_start_line,
            comment_text,
            code_snippet,
        )

    return comments


def extract_comments_from_content(
    content: str,
    file_path: str,
    start_line: int = 1,
    repo_name: str | None = None,
) -> list[dict]:
    """Extract comments from source file content or a code snippet."""
    if content is None or not file_path:
        return []

    if not is_reviewable_file(
        file_path,
        repo_name=repo_name,
    ):
        return []

    _, extension = os.path.splitext(file_path.lower())

    if extension in _PYTHON_EXTENSIONS:
        return _extract_comments_python(
            content,
            file_path,
            start_line=start_line,
        )

    return _extract_comments_generic(
        content,
        file_path,
        start_line=start_line,
    )


def extract_comments_from_snippet(
    snippet: str,
    file_path: str,
    start_line: int = 1,
    repo_name: str | None = None,
) -> list[dict]:
    """
    Extract comments from a specific code snippet.

    start_line is the actual starting line number in the source file.
    """
    return extract_comments_from_content(
        snippet,
        file_path,
        start_line=start_line,
        repo_name=repo_name,
    )


def extract_comments_from_files(
    repo_path: str,
    changed_files: list[str],
    source_branch: str | None,
    after_sha: str | None = None,
    repo_name: str | None = None,
) -> list[dict]:
    """
    Extract comments only from the supplied changed files.

    When after_sha is provided, every file is read from that exact commit.
    There is no fallback to another branch or to the workspace copy.
    """
    if not repo_path or not changed_files:
        return []

    repo_name = (
        repo_name or os.environ.get("GITHUB_REPOSITORY", "")
    ).strip() or None

    try:
        repo = Repo(repo_path)

    except Exception as exc:
        print(
            f"[COMMENT EXTRACTOR] Could not open repository "
            f"{repo_path}: {exc}"
        )
        return []

    # If the exact commit is missing, attempt Git recovery.
    if after_sha and not _has_commit(repo, after_sha):
        print(
            f"[COMMENT EXTRACTOR] Commit {after_sha} is missing locally. "
            "Attempting Git recovery..."
        )

        _attempt_git_fetch_recovery(
            repo=repo,
            after_sha=after_sha,
            source_branch=source_branch,
        )

    if after_sha:
        if not _has_commit(repo, after_sha):
            print(
                f"[COMMENT EXTRACTOR ERROR] Exact source commit "
                f"{after_sha} is unavailable. No alternate source "
                "version will be used."
            )
            return []

        file_ref = after_sha

    else:
        if not source_branch:
            print(
                "[COMMENT EXTRACTOR ERROR] Neither after_sha nor "
                "source_branch was provided."
            )
            return []

        # Branch mode is retained for existing callers that do not
        # provide an exact commit SHA.
        try:
            repo.remotes.origin.fetch()
        except Exception as exc:
            print(
                f"[COMMENT EXTRACTOR] Origin fetch warning: {exc}"
            )

        file_ref = (
            source_branch
            if source_branch.startswith("refs/")
            else f"origin/{source_branch}"
        )

    all_comments: list[dict] = []

    for file_path in changed_files:
        normalized_path = _normalize_path(file_path)

        if not is_reviewable_file(
            normalized_path,
            repo_name=repo_name,
        ):
            continue

        content = get_file_content(
            repo_path,
            file_ref,
            normalized_path,
        )

        # None means the exact source could not be read.
        # Empty string is a valid empty file.
        if content is None:
            print(
                f"[COMMENT EXTRACTOR WARNING] Skipping unreadable file: "
                f"{normalized_path}"
            )
            continue

        file_comments = extract_comments_from_content(
            content,
            normalized_path,
            repo_name=repo_name,
        )

        all_comments.extend(file_comments)

    print(
        f"[COMMENT EXTRACTOR] Extracted "
        f"{len(all_comments)} comment(s) from "
        f"{len(changed_files)} supplied changed file(s)."
    )

    return all_comments
