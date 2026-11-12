
import os
import re
from git import Repo

from github.changed_files import (
    is_automation_file,
    filter_automation_files,
    _has_commit,
    _attempt_git_fetch_recovery,
)


class DiffExtractionError(Exception):
    """Raised when the requested Git diff cannot be extracted safely."""


def _normalize_repo_path(path: str) -> str:
    """Normalize a repository-relative path."""
    if not path:
        return ""

    normalized = str(path).strip().replace("\\", "/")

    while normalized.startswith("./"):
        normalized = normalized[2:]

    return normalized.lstrip("/")


def _get_diff_file_path(diff_header: str) -> str | None:
    """Extract the destination file path from a Git diff header."""
    if not diff_header.startswith("diff --git "):
        return None

    # Git normally formats this as:
    # diff --git a/path/to/file.py b/path/to/file.py
    match = re.match(r"^diff --git a/(.*) b/(.*)$", diff_header)

    if match:
        return _normalize_repo_path(match.group(2))

    # Handle quoted paths as a fallback.
    import shlex

    try:
        parts = shlex.split(diff_header)

        if len(parts) >= 4 and parts[3].startswith("b/"):
            return _normalize_repo_path(parts[3][2:])

    except ValueError:
        pass

    return None


def _get_repo_name(repo_name: str | None = None) -> str | None:
    """Get the repository name from the argument or GitHub environment."""
    return (
        repo_name or os.environ.get("GITHUB_REPOSITORY", "")
    ).strip() or None


def filter_diff(
    diff_text: str,
    repo_name: str | None = None,
) -> str:
    """
    Remove automation-file diff sections.

    Automation exclusions are applied only when the repository is
    explicitly configured as the automation repository.
    """
    if not diff_text:
        return ""

    repo_name = _get_repo_name(repo_name)

    blocks = re.split(
        r"(?=^diff --git )",
        diff_text,
        flags=re.MULTILINE,
    )

    kept_blocks = []

    for block in blocks:
        if not block.strip():
            continue

        first_line = block.splitlines()[0]

        if first_line.startswith("diff --git "):
            file_path = _get_diff_file_path(first_line)

            if file_path and is_automation_file(
                file_path,
                repo_name=repo_name,
            ):
                continue

        kept_blocks.append(block)

    return "\n".join(
        block.rstrip("\n")
        for block in kept_blocks
    ).strip()


def _filter_diff_to_files(
    diff_text: str,
    changed_files: list[str],
    repo_name: str | None = None,
) -> str:
    """
    Keep only diff sections whose paths exactly match changed_files.

    Exact matching is intentional. Suffix or basename matching can select
    a different file when multiple directories contain the same filename.
    """
    if not diff_text or not changed_files:
        return ""

    allowed_files = {
        _normalize_repo_path(path)
        for path in changed_files
        if path
    }

    allowed_files = {
        path
        for path in allowed_files
        if path and not is_automation_file(path, repo_name=repo_name)
    }

    if not allowed_files:
        return ""

    blocks = re.split(
        r"(?=^diff --git )",
        diff_text,
        flags=re.MULTILINE,
    )

    kept_blocks = []

    for block in blocks:
        if not block.strip():
            continue

        first_line = block.splitlines()[0]

        if not first_line.startswith("diff --git "):
            continue

        file_path = _get_diff_file_path(first_line)

        if file_path in allowed_files:
            kept_blocks.append(block)

    return "\n".join(
        block.rstrip("\n")
        for block in kept_blocks
    ).strip()


def _get_remote_ref(branch_name: str) -> str:
    """Convert a branch name to its origin remote-tracking ref."""
    branch_name = str(branch_name).strip()

    if branch_name.startswith("refs/remotes/"):
        return branch_name

    if branch_name.startswith("origin/"):
        return f"refs/remotes/{branch_name}"

    return f"refs/remotes/origin/{branch_name}"


def _run_git_diff(
    repo: Repo,
    base_ref: str,
    head_ref: str,
    changed_files: list[str] | None = None,
) -> str:
    """Run a Git diff between two exact refs, optionally restricted to files."""
    command = [
        "--no-ext-diff",
        "--no-renames",
        "--diff-filter=d",
        base_ref,
        head_ref,
    ]

    if changed_files is not None:
        normalized_files = [
            _normalize_repo_path(path)
            for path in changed_files
            if path
        ]

        if not normalized_files:
            return ""

        command.extend(["--", *normalized_files])

    else:
        command.append("--")

    return repo.git.diff(*command)


def get_diff(
    repo_path,
    target_branch,
    source_branch,
    changed_files: list[str] | None = None,
    repo_name: str | None = None,
) -> str:
    """
    Extract a complete branch-to-branch diff.

    This function is for branch comparisons. Incremental PR processing
    should use get_incremental_diff() with exact commit SHAs.
    """
    try:
        repo = Repo(repo_path)

        if "origin" not in [remote.name for remote in repo.remotes]:
            raise DiffExtractionError(
                "The repository does not have an origin remote."
            )

        repo.remotes.origin.fetch()

        target_ref = _get_remote_ref(target_branch)
        source_ref = _get_remote_ref(source_branch)

        if changed_files is not None:
            filtered_files = filter_automation_files(
                changed_files,
                repo_name=_get_repo_name(repo_name),
            )

            if not filtered_files:
                return ""

        else:
            filtered_files = None

        diff_output = _run_git_diff(
            repo=repo,
            base_ref=target_ref,
            head_ref=source_ref,
            changed_files=filtered_files,
        )

        if filtered_files is not None:
            return _filter_diff_to_files(
                diff_output,
                filtered_files,
                repo_name=repo_name,
            )

        return filter_diff(
            diff_output,
            repo_name=repo_name,
        )

    except DiffExtractionError:
        raise

    except Exception as exc:
        raise DiffExtractionError(
            f"Branch diff extraction failed: {exc}"
        ) from exc


def _fetch_github_compare_diff(
    repo_name: str,
    before_sha: str,
    after_sha: str,
    changed_files: list[str] | None = None,
) -> str | None:
    """
    Fetch the diff for the exact commit range from GitHub Compare API.

    This does not use the PR diff endpoint because that endpoint can return
    changes outside the requested incremental commit range.
    """
    import requests

    try:
        from github.github_auth import get_github_token
        token = get_github_token() or ""
    except Exception:
        token = os.environ.get("GITHUB_TOKEN", "")

    token = str(token).strip()

    if not token:
        print(
            "[GITHUB API WARNING] No GitHub token available "
            "for Compare API fallback."
        )
        return None

    url = (
        f"https://api.github.com/repos/{repo_name}/compare/"
        f"{before_sha}...{after_sha}"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=60,
        )

        if response.status_code != 200:
            print(
                f"[GITHUB API WARNING] Compare API returned "
                f"HTTP {response.status_code}: {response.text[:500]}"
            )
            return None

        if not response.text.strip():
            return ""

        print(
            "[GITHUB API] Exact incremental diff retrieved "
            "through GitHub Compare API."
        )

        return response.text

    except requests.RequestException as exc:
        print(
            f"[GITHUB API WARNING] Compare API request failed: {exc}"
        )
        return None


def get_incremental_diff(
    repo_path,
    before_sha,
    after_sha,
    changed_files: list[str] | None = None,
    repo_name: str | None = None,
    pr_number: int | None = None,
    target_branch: str | None = None,
    source_branch: str | None = None,
) -> str:
    """
    Extract the diff for exactly before_sha -> after_sha.

    Detection order:
        1. Local Git diff between the exact commit SHAs.
        2. Recover missing commits and retry the exact Git diff.
        3. GitHub Compare API for the same exact commit range.

    Important:
        This function never falls back to a complete PR diff or a branch
        diff. Those can include unrelated changes and cause the AI to
        review files outside the requested commit range.
    """
    if not before_sha or not after_sha:
        raise DiffExtractionError(
            "Both before_sha and after_sha are required."
        )

    repo_name = _get_repo_name(repo_name)

    try:
        repo = Repo(repo_path)

        if changed_files is not None:
            filtered_files = filter_automation_files(
                changed_files,
                repo_name=repo_name,
            )

            if not filtered_files:
                print(
                    "[INCREMENTAL DIFF] No eligible changed files "
                    "were provided. No diff will be generated."
                )
                return ""

        else:
            filtered_files = None

        def run_exact_diff() -> str:
            return _run_git_diff(
                repo=repo,
                base_ref=before_sha,
                head_ref=after_sha,
                changed_files=filtered_files,
            )

        # 1. Try the exact local commit comparison.
        if _has_commit(repo, before_sha) and _has_commit(repo, after_sha):
            try:
                diff_output = run_exact_diff()

                if filtered_files is not None:
                    diff_output = _filter_diff_to_files(
                        diff_output,
                        filtered_files,
                        repo_name=repo_name,
                    )
                else:
                    diff_output = filter_diff(
                        diff_output,
                        repo_name=repo_name,
                    )

                print(
                    "[INCREMENTAL DIFF] Exact local commit comparison "
                    "completed."
                )

                return diff_output

            except Exception as exc:
                print(
                    f"[INCREMENTAL DIFF WARNING] Local exact diff failed: "
                    f"{exc}"
                )

        # 2. Attempt to recover missing commits.
        print(
            f"[INCREMENTAL DIFF] Recovering commit history for "
            f"{before_sha}..{after_sha}..."
        )

        _attempt_git_fetch_recovery(
            repo=repo,
            before_sha=before_sha,
            after_sha=after_sha,
            pr_number=pr_number,
            target_branch=target_branch,
            source_branch=source_branch,
        )

        # 3. Retry the exact local comparison.
        if _has_commit(repo, before_sha) and _has_commit(repo, after_sha):
            try:
                diff_output = run_exact_diff()

                if filtered_files is not None:
                    diff_output = _filter_diff_to_files(
                        diff_output,
                        filtered_files,
                        repo_name=repo_name,
                    )
                else:
                    diff_output = filter_diff(
                        diff_output,
                        repo_name=repo_name,
                    )

                print(
                    "[INCREMENTAL DIFF] Exact local diff succeeded "
                    "after Git recovery."
                )

                return diff_output

            except Exception as exc:
                print(
                    f"[INCREMENTAL DIFF WARNING] Retried exact diff failed: "
                    f"{exc}"
                )

        # 4. GitHub Compare API fallback for the same exact commit range.
        if repo_name:
            print(
                "[INCREMENTAL DIFF] Local Git recovery failed. "
                "Trying GitHub Compare API..."
            )

            api_diff = _fetch_github_compare_diff(
                repo_name=repo_name,
                before_sha=before_sha,
                after_sha=after_sha,
                changed_files=filtered_files,
            )

            if api_diff is not None:
                if filtered_files is not None:
                    api_diff = _filter_diff_to_files(
                        api_diff,
                        filtered_files,
                        repo_name=repo_name,
                    )
                else:
                    api_diff = filter_diff(
                        api_diff,
                        repo_name=repo_name,
                    )

                return api_diff

        raise DiffExtractionError(
            f"Could not extract the exact incremental diff between "
            f"{before_sha} and {after_sha}. Local Git recovery and "
            "GitHub Compare API fallback were unsuccessful. "
            "No full-PR or branch-diff fallback was used."
        )

    except DiffExtractionError:
        raise

    except Exception as exc:
        raise DiffExtractionError(
            f"Unexpected incremental diff extraction error: {exc}"
        ) from exc


def extract_added_code(
    diff_text: str,
    repo_name: str | None = None,
) -> str:
    """
    Extract added code lines with their exact new-file line numbers.
    """
    if not diff_text:
        return ""

    repo_name = _get_repo_name(repo_name)

    extracted_lines = []
    current_file = None
    current_line = 0

    hunk_pattern = re.compile(
        r"^@@ -\d+(?:,\d+)? \+(?P<new_start>\d+)(?:,\d+)? @@"
    )

    for line in diff_text.splitlines():

        if line.startswith("diff --git "):
            current_file = _get_diff_file_path(line)
            current_line = 0

            if current_file and is_automation_file(
                current_file,
                repo_name=repo_name,
            ):
                current_file = None
                continue

            if current_file:
                extracted_lines.append(
                    f"\nFILE: {current_file}\n"
                )

            continue

        if (
            line.startswith("index ")
            or line.startswith("--- ")
            or line.startswith("+++ ")
            or line.startswith("\\ No newline")
        ):
            continue

        hunk_match = hunk_pattern.match(line)

        if hunk_match:
            current_line = int(hunk_match.group("new_start"))
            continue

        if current_file is None:
            continue

        if line.startswith("+"):
            content = line[1:]

            extracted_lines.append(
                f"L{current_line}: {content}"
            )

            current_line += 1

        elif line.startswith(" "):
            current_line += 1

        elif line.startswith("-"):
            # Deleted lines do not advance the new-file line number.
            continue

    return "\n".join(extracted_lines)


def extract_full_code(
    workspace_path: str,
    changed_files: list,
    source_ref: str | None = None,
    repo_name: str | None = None,
) -> str:
    """
    Read complete source code for the supplied changed files.

    When source_ref is supplied, content must come from that exact Git ref.
    This function never falls back to a potentially stale workspace copy.

    Best-effort extraction: an individual file that cannot be read is logged
    and skipped. Successfully extracted files are returned so the caller can
    continue reviewing those files. The caller must use exact source/diff
    matching and report skipped files; it must not claim they were reviewed.
    """
    if not workspace_path:
        raise DiffExtractionError(
            "Cannot extract changed-file source: workspace_path is empty."
        )

    if not changed_files:
        return ""

    repo_name = _get_repo_name(repo_name)

    filtered_files = filter_automation_files(
        changed_files,
        repo_name=repo_name,
    )

    if not filtered_files:
        return ""

    # Normalize and de-duplicate without changing the original order.
    normalized_files = []
    seen_files = set()
    for raw_path in filtered_files:
        rel_path = _normalize_repo_path(raw_path)
        if not rel_path:
            raise DiffExtractionError(
                f"Invalid empty path in changed_files: {raw_path!r}"
            )
        if rel_path not in seen_files:
            normalized_files.append(rel_path)
            seen_files.add(rel_path)

    repo = None
    git_dir = os.path.join(workspace_path, ".git")

    if os.path.isdir(git_dir) or os.path.isfile(git_dir):
        try:
            repo = Repo(workspace_path)
        except Exception as exc:
            if source_ref:
                raise DiffExtractionError(
                    "A source_ref was supplied, but the workspace could "
                    f"not be opened as a Git repository: {exc}"
                ) from exc
            print(
                "[FILE EXTRACTION WARNING] Could not open Git repository: "
                f"{type(exc).__name__}: {exc}"
            )

    elif source_ref:
        raise DiffExtractionError(
            "A source_ref was supplied, but the workspace has no Git "
            "repository metadata. Refusing to read a potentially stale "
            "workspace copy."
        )

    if source_ref and repo is not None:
        # Fetch the requested commit/ref when possible. Fetch failure is
        # diagnostic only; the exact source_ref lookup below remains strict.
        for remote in repo.remotes:
            if remote.name == "origin":
                try:
                    remote.fetch()
                except Exception as exc:
                    print(
                        "[FILE EXTRACTION WARNING] Git fetch failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                break

    file_blocks = []
    extracted_files = []
    failed_files = []

    print(
        "[SOURCE EXTRACTION] Attempting to extract "
        f"{len(normalized_files)} file(s) from "
        f"{source_ref or 'workspace'}"
    )

    for rel_path in normalized_files:
        print(f"[SOURCE EXTRACTION] Reading: {rel_path}")

        try:
            content = _read_changed_file_content(
                repo=repo,
                workspace_path=workspace_path,
                rel_path=rel_path,
                source_ref=source_ref,
            )

            if content is None:
                failed_files.append(
                    (rel_path, "file could not be read from requested source")
                )
                print(
                    f"[FILE EXTRACTION ERROR] Could not read {rel_path} "
                    "from the requested source."
                )
                continue

            lines = content.splitlines()
            numbered_lines = [
                f"L{index}: {line}"
                for index, line in enumerate(lines, start=1)
            ]
            formatted_code = "\n".join(numbered_lines)

            extension = os.path.splitext(rel_path)[1].lstrip(".")
            language = extension if extension else "text"

            file_blocks.append(
                f"FILE: {rel_path}\n"
                f"```{language}\n"
                f"{formatted_code}\n"
                "```"
            )
            extracted_files.append(rel_path)
            print(f"[SOURCE EXTRACTION SUCCESS] {rel_path}")

        except Exception as exc:
            failed_files.append(
                (rel_path, f"{type(exc).__name__}: {exc}")
            )
            print(
                f"[FILE EXTRACTION ERROR] {rel_path}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n===== SOURCE EXTRACTION SUMMARY =====")
    print(f"Files requested : {len(normalized_files)}")
    print(f"Files extracted : {len(extracted_files)}")
    print(f"Files failed    : {len(failed_files)}")

    if failed_files:
        print("\nFiles that could not be extracted:")
        for file_path, reason in failed_files:
            print(f"  - {file_path}: {reason}")

    print("=====================================\n")

    if failed_files:
        print(
            "[SOURCE EXTRACTION WARNING] Continuing with successfully "
            f"extracted files: {len(extracted_files)} of "
            f"{len(normalized_files)}."
        )

    if not file_blocks:
        print(
            "[SOURCE EXTRACTION WARNING] No source files could be extracted."
        )

    return "\n\n".join(file_blocks)

def _read_changed_file_content(
    repo: Repo | None,
    workspace_path: str,
    rel_path: str,
    source_ref: str | None,
) -> str | None:
    """
    Read a file from the requested Git ref or from the workspace.

    If source_ref is provided, do not use the workspace as a fallback.
    """
    rel_path = _normalize_repo_path(rel_path)

    if not rel_path:
        return None

    # Reject paths that could escape the repository workspace.
    if any(part == ".." for part in rel_path.split("/")):
        print(
            f"[FILE EXTRACTION WARNING] Rejected unsafe path: {rel_path}"
        )
        return None

    if source_ref:
        if not repo:
            print(
                "[FILE EXTRACTION WARNING] A source ref was supplied, "
                "but the workspace is not a valid Git repository."
            )
            return None

        refs_to_try = []

        if source_ref.startswith("refs/"):
            refs_to_try.append(source_ref)

        elif re.fullmatch(r"[0-9a-fA-F]{7,64}", source_ref):
            refs_to_try.append(source_ref)

        else:
            refs_to_try.extend([
                f"refs/remotes/origin/{source_ref}",
                f"origin/{source_ref}",
                source_ref,
            ])

        for git_ref in refs_to_try:
            try:
                return repo.git.show(
                    f"{git_ref}:{rel_path}"
                )
            except Exception:
                continue

        # Do not read a different workspace version when the requested
        # commit or branch could not be resolved.
        return None

    resolved_path = resolve_workspace_file_path(
        workspace_path,
        rel_path,
    )

    if not resolved_path:
        return None

    try:
        with open(
            resolved_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as source_file:
            return source_file.read()

    except Exception as exc:
        print(
            f"[FILE EXTRACTION WARNING] Disk read failed for "
            f"{rel_path}: {exc}"
        )
        return None


def count_file_sections_in_code(code_text: str) -> int:
    """Count FILE sections in extracted source code."""
    return len(
        re.findall(
            r"^FILE:\s*",
            code_text or "",
            re.MULTILINE,
        )
    )


def _paths_match(diff_path: str, rel_path: str) -> bool:
    """Compare normalized repository paths exactly."""
    diff_norm = _normalize_repo_path(diff_path)
    rel_norm = _normalize_repo_path(rel_path)

    return bool(
        diff_norm
        and rel_norm
        and diff_norm == rel_norm
    )


def resolve_workspace_file_path(
    workspace_path: str,
    rel_path: str,
) -> str | None:
    """
    Resolve a repository-relative file path.

    Direct path matching is preferred. A suffix match is accepted only
    when exactly one candidate exists.
    """
    rel_path = _normalize_repo_path(rel_path)

    if not rel_path or not workspace_path:
        return None

    if any(part == ".." for part in rel_path.split("/")):
        return None

    workspace_root = os.path.realpath(workspace_path)

    direct_path = os.path.realpath(
        os.path.join(
            workspace_root,
            rel_path.replace("/", os.sep),
        )
    )

    try:
        if os.path.commonpath(
            [workspace_root, direct_path]
        ) != workspace_root:
            return None
    except ValueError:
        return None

    if os.path.isfile(direct_path):
        return direct_path

    suffix_matches = []

    for root, dirs, files in os.walk(workspace_root):
        if ".git" in dirs:
            dirs.remove(".git")

        for filename in files:
            full_path = os.path.realpath(
                os.path.join(root, filename)
            )

            normalized = full_path.replace("\\", "/")

            if (
                normalized.endswith("/" + rel_path)
                and os.path.isfile(full_path)
            ):
                suffix_matches.append(full_path)

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    if len(suffix_matches) > 1:
        print(
            f"[FILE EXTRACTION WARNING] Ambiguous path "
            f"{rel_path}: multiple matching files found."
        )

    return None


def extract_lines_from_diff(
    diff_text: str,
    rel_path: str,
    line_spec: str,
) -> str:
    """Extract a specific new-file line range from a Git diff."""
    if not diff_text or not rel_path or not line_spec:
        return ""

    parsed = _parse_line_spec(line_spec)

    if not parsed:
        print(
            f"[CODE EXTRACT] Could not parse line spec: {line_spec!r}"
        )
        return ""

    start_line, end_line = parsed

    if start_line > end_line:
        start_line, end_line = end_line, start_line

    rel_path = _normalize_repo_path(rel_path)

    hunk_pattern = re.compile(
        r"^@@ -\d+(?:,\d+)? \+(?P<new_start>\d+)(?:,\d+)? @@"
    )

    current_file = None
    current_line = 0
    collected: dict[int, str] = {}

    for line in diff_text.splitlines():

        if line.startswith("diff --git "):
            current_file = _get_diff_file_path(line)
            current_line = 0
            continue

        if not _paths_match(current_file or "", rel_path):
            continue

        if (
            line.startswith("index ")
            or line.startswith("--- ")
            or line.startswith("+++ ")
            or line.startswith("\\ No newline")
        ):
            continue

        hunk_match = hunk_pattern.match(line)

        if hunk_match:
            current_line = int(hunk_match.group("new_start"))
            continue

        if line.startswith("+"):
            if start_line <= current_line <= end_line:
                collected[current_line] = line[1:]

            current_line += 1

        elif line.startswith(" "):
            if start_line <= current_line <= end_line:
                collected[current_line] = line[1:]

            current_line += 1

        elif line.startswith("-"):
            continue

    if not collected:
        print(
            f"[CODE EXTRACT] No matching diff lines found for "
            f"{rel_path} {line_spec}"
        )
        return ""

    return "\n".join(
        collected[line_number]
        for line_number in range(start_line, end_line + 1)
        if line_number in collected
    )


def _parse_line_spec(
    line_spec: str,
) -> tuple[int, int] | None:
    """Parse line specifications such as L10, 10, or L10-L20."""
    line_spec = (line_spec or "").strip()

    line_spec = re.sub(
        r"[\u2011\u2013\u2014\u2010]",
        "-",
        line_spec,
    )

    line_spec = re.sub(r"\s+", "", line_spec)

    range_match = re.match(
        r"^L?(\d+)-L?(\d+)$",
        line_spec,
        re.IGNORECASE,
    )

    if range_match:
        return (
            int(range_match.group(1)),
            int(range_match.group(2)),
        )

    single_match = re.match(
        r"^L?(\d+)$",
        line_spec,
        re.IGNORECASE,
    )

    if single_match:
        line_number = int(single_match.group(1))
        return line_number, line_number

    return None


def extract_source_lines(
    workspace_path: str,
    rel_path: str,
    line_spec: str,
) -> str:
    """Extract a line range from a source file in the workspace."""
    rel_path = _normalize_repo_path(rel_path)

    if not rel_path or not line_spec:
        return ""

    full_path = resolve_workspace_file_path(
        workspace_path,
        rel_path,
    )

    if not full_path:
        print(
            f"[CODE EXTRACT] Source file not found or ambiguous: "
            f"{rel_path}"
        )
        return ""

    parsed = _parse_line_spec(line_spec)

    if not parsed:
        print(
            f"[CODE EXTRACT] Could not parse line spec: {line_spec!r}"
        )
        return ""

    start_line, end_line = parsed

    if start_line > end_line:
        start_line, end_line = end_line, start_line

    try:
        with open(
            full_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as source_file:
            all_lines = source_file.read().splitlines()

    except Exception as exc:
        print(
            f"[CODE EXTRACT] Failed to read {rel_path}: {exc}"
        )
        return ""

    if start_line < 1 or start_line > len(all_lines):
        return ""

    end_line = min(end_line, len(all_lines))

    return "\n".join(
        all_lines[start_line - 1:end_line]
    )


def resolve_code_snippet(
    workspace_path: str | None,
    rel_path: str,
    line_spec: str,
    diff_text: str | None = None,
) -> str:
    """
    Resolve a code snippet from the workspace first, then from the diff.
    """
    rel_path = _normalize_repo_path(rel_path)
    line_spec = (line_spec or "").strip()

    if not rel_path or not line_spec:
        return ""

    if workspace_path:
        extracted = extract_source_lines(
            workspace_path,
            rel_path,
            line_spec,
        )

        if extracted:
            print(
                f"[CODE EXTRACT] Resolved snippet from source file: "
                f"{rel_path} {line_spec}"
            )
            return extracted

    if diff_text:
        extracted = extract_lines_from_diff(
            diff_text,
            rel_path,
            line_spec,
        )

        if extracted:
            print(
                f"[CODE EXTRACT] Resolved snippet from Git diff: "
                f"{rel_path} {line_spec}"
            )
            return extracted

    return ""
