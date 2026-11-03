
import os
import re
import requests
from git import Repo, GitCommandError


class ChangedFilesDetectionError(Exception):
    """Raised when changed files cannot be determined safely."""


# Configure this environment variable with the exact owner/repository name
# of your AI automation repository, for example:
# AI_PR_AUTOMATION_REPOSITORY=your-org/your-automation-repo
#
# If it is not configured, no files are excluded based on their folder names.
AUTOMATION_REPOSITORY = os.environ.get(
    "AI_PR_AUTOMATION_REPOSITORY", ""
).strip().lower()

AUTOMATION_PREFIXES = (
    ".github/",
    "ai/",
    "github/",
    "reports/",
    "tests/",
)

AUTOMATION_EXACT_FILES = {
    "main.py",
    "run_action.py",
    "procfile",
}


def normalize_repo_path(file_path: str) -> str:
    """Normalize a repository-relative file path."""
    if not file_path:
        return ""

    normalized = str(file_path).strip().replace("\\", "/")

    while normalized.startswith("./"):
        normalized = normalized[2:]

    return normalized.lstrip("/")


def _is_automation_repository(repo_name: str | None = None) -> bool:
    """
    Return True only when the current repository is explicitly configured
    as the AI automation repository.
    """
    active_repository = (
        repo_name or os.environ.get("GITHUB_REPOSITORY", "")
    ).strip().lower()

    if not active_repository or not AUTOMATION_REPOSITORY:
        return False

    return active_repository == AUTOMATION_REPOSITORY


def is_automation_file(
    file_path: str,
    repo_name: str | None = None,
) -> bool:
    """
    Identify automation infrastructure files.

    Important:
    These exclusions apply only to the configured automation repository.
    Application repositories are not filtered by folder names.
    """
    if not file_path:
        return False

    if not _is_automation_repository(repo_name):
        return False

    normalized = normalize_repo_path(file_path).lower()

    if not normalized:
        return False

    if any(
        normalized.startswith(prefix)
        for prefix in AUTOMATION_PREFIXES
    ):
        return True

    if normalized in AUTOMATION_EXACT_FILES:
        return True

    if normalized.startswith("requirements"):
        return True

    return False


def filter_automation_files(
    file_list: list[str],
    repo_name: str | None = None,
) -> list[str]:
    """Remove automation files only when reviewing the automation repository."""
    return [
        normalize_repo_path(file_path)
        for file_path in file_list
        if file_path
        and not is_automation_file(file_path, repo_name)
    ]


def _get_pr_number(pr_number: int | None = None) -> int | None:
    """Get the PR number from the argument or GitHub Actions environment."""
    if pr_number:
        return int(pr_number)

    github_ref = os.environ.get("GITHUB_REF", "")
    match = re.match(r"refs/pull/(\d+)/", github_ref)

    if match:
        return int(match.group(1))

    return None


def _get_github_headers() -> dict | None:
    """Build GitHub API headers using the project's token helper or environment."""
    token = ""

    try:
        from github.github_auth import get_github_token
        token = get_github_token() or ""
    except Exception:
        token = os.environ.get("GITHUB_TOKEN", "")

    token = str(token).strip()

    if not token:
        return None

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_pr_files_from_github(
    repo_name: str,
    pr_number: int,
) -> list[str]:
    """
    Retrieve the files changed in a PR using GitHub's PR Files API.

    This returns the PR's complete changed-file list, not an incremental
    commit-to-commit diff.
    """
    headers = _get_github_headers()

    if not headers:
        raise ChangedFilesDetectionError(
            "GitHub API token is missing. Set GITHUB_TOKEN or configure "
            "github.github_auth.get_github_token()."
        )

    all_files = []
    page = 1

    while True:
        url = (
            f"https://api.github.com/repos/{repo_name}"
            f"/pulls/{pr_number}/files"
        )

        try:
            response = requests.get(
                url,
                headers=headers,
                params={
                    "per_page": 100,
                    "page": page,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            raise ChangedFilesDetectionError(
                f"GitHub PR Files API request failed: {exc}"
            ) from exc

        if response.status_code != 200:
            raise ChangedFilesDetectionError(
                "GitHub PR Files API failed: "
                f"HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            page_data = response.json()
        except ValueError as exc:
            raise ChangedFilesDetectionError(
                "GitHub PR Files API returned invalid JSON."
            ) from exc

        if not isinstance(page_data, list):
            raise ChangedFilesDetectionError(
                "Unexpected response from GitHub PR Files API."
            )

        for item in page_data:
            if not isinstance(item, dict):
                continue

            filename = item.get("filename")
            status = item.get("status", "")

            # Deleted files cannot be reviewed as current source files.
            if filename and status != "removed":
                all_files.append(filename)

        if len(page_data) < 100:
            break

        page += 1

    return all_files


def get_pr_changed_files(
    repo_path=None,
    target_branch=None,
    source_branch=None,
    repo_name: str | None = None,
    pr_number: int | None = None,
) -> list[str]:
    """
    Get the complete list of files changed in a PR.

    Preferred method:
        GitHub PR Files API, using the PR number.

    Local fallback:
        Compare the target and source branches when no PR number is available.

    This function is for complete PR file detection. Do not use it as a
    replacement for incremental commit-to-commit detection.
    """
    repo_name = (
        repo_name or os.environ.get("GITHUB_REPOSITORY", "")
    ).strip()

    pr_number = _get_pr_number(pr_number)

    # Preferred method: retrieve the exact PR file list from GitHub.
    if repo_name and pr_number:
        raw_files = _get_pr_files_from_github(
            repo_name=repo_name,
            pr_number=pr_number,
        )

        filtered_files = filter_automation_files(
            raw_files,
            repo_name=repo_name,
        )

        print(
            f"[CHANGED FILES] GitHub PR API found "
            f"{len(raw_files)} file(s); "
            f"{len(filtered_files)} file(s) remain after filtering."
        )

        return filtered_files

    # Local fallback is only used when a PR number is unavailable.
    if not repo_path or not target_branch or not source_branch:
        raise ChangedFilesDetectionError(
            "Cannot determine PR changed files. Provide a PR number with "
            "GITHUB_REPOSITORY, or provide repo_path, target_branch, "
            "and source_branch for local comparison."
        )

    try:
        repo = Repo(repo_path)

        if "origin" not in [remote.name for remote in repo.remotes]:
            raise ChangedFilesDetectionError(
                "The repository does not have an 'origin' remote."
            )

        repo.remotes.origin.fetch()

        target_ref = _qualify_remote_ref(target_branch)
        source_ref = _qualify_remote_ref(source_branch)

        changed_output = repo.git.diff(
            "--name-only",
            "--diff-filter=d",
            f"{target_ref}...{source_ref}",
        )

        raw_files = [
            normalize_repo_path(line)
            for line in changed_output.splitlines()
            if line.strip()
        ]

        return filter_automation_files(
            raw_files,
            repo_name=repo_name,
        )

    except ChangedFilesDetectionError:
        raise
    except Exception as exc:
        raise ChangedFilesDetectionError(
            f"Local PR changed-file detection failed: {exc}"
        ) from exc


def _qualify_remote_ref(branch_name: str) -> str:
    """Convert a branch name into its origin remote-tracking reference."""
    branch_name = str(branch_name).strip()

    if branch_name.startswith("refs/remotes/"):
        return branch_name

    if branch_name.startswith("origin/"):
        return f"refs/remotes/{branch_name}"

    return f"refs/remotes/origin/{branch_name}"


def _has_commit(repo: Repo, sha: str | None) -> bool:
    """Return True if the commit object exists in the local repository."""
    if not sha:
        return False

    try:
        repo.git.cat_file("-e", f"{sha}^{{commit}}")
        return True
    except Exception:
        return False


def _attempt_git_fetch_recovery(
    repo: Repo,
    before_sha: str | None = None,
    after_sha: str | None = None,
    pr_number: int | None = None,
    target_branch: str | None = None,
    source_branch: str | None = None,
) -> None:
    """Attempt to retrieve missing commits and history from origin."""
    remotes = list(repo.remotes)

    if not remotes:
        print("[GIT RECOVERY WARNING] No Git remote is configured.")
        return

    origin = next(
        (remote for remote in remotes if remote.name == "origin"),
        remotes[0],
    )

    remote_name = origin.name

    # Recover shallow-clone history if possible.
    try:
        is_shallow = (
            repo.git.rev_parse("--is-shallow-repository")
            .strip()
            .lower()
            == "true"
        )

        if is_shallow:
            print(
                "[GIT RECOVERY] Repository is shallow. "
                "Attempting to unshallow..."
            )

            try:
                repo.git.fetch(remote_name, "--unshallow")
            except Exception as exc:
                print(
                    f"[GIT RECOVERY WARNING] Unshallow failed: {exc}. "
                    "Trying to deepen history..."
                )

                try:
                    repo.git.fetch(remote_name, "--deepen=100")
                except Exception as deepen_exc:
                    print(
                        f"[GIT RECOVERY WARNING] Deepen failed: "
                        f"{deepen_exc}"
                    )

    except Exception as exc:
        print(
            f"[GIT RECOVERY WARNING] Could not check shallow status: {exc}"
        )

    # Fetch missing commit SHAs directly.
    for sha in (before_sha, after_sha):
        if sha and not _has_commit(repo, sha):
            try:
                print(
                    f"[GIT RECOVERY] Fetching missing commit {sha}..."
                )
                repo.git.fetch(remote_name, sha)
            except Exception as exc:
                print(
                    f"[GIT RECOVERY WARNING] Fetching commit "
                    f"{sha} failed: {exc}"
                )

    # Fetch the PR head ref when the PR number is known.
    if pr_number:
        try:
            print(
                f"[GIT RECOVERY] Fetching PR #{pr_number} head ref..."
            )

            repo.git.fetch(
                remote_name,
                f"+refs/pull/{pr_number}/head:"
                f"refs/remotes/{remote_name}/pr/{pr_number}",
            )

        except Exception as exc:
            print(
                f"[GIT RECOVERY WARNING] Fetching PR head failed: {exc}"
            )

    # Fetch target and source branches if supplied.
    for branch in (target_branch, source_branch):
        if not branch:
            continue

        try:
            repo.git.fetch(remote_name, branch)
        except Exception as exc:
            print(
                f"[GIT RECOVERY WARNING] Fetching branch "
                f"{branch} failed: {exc}"
            )

    # Final general fetch.
    try:
        repo.git.fetch(remote_name)
    except Exception as exc:
        print(
            f"[GIT RECOVERY WARNING] General fetch failed: {exc}"
        )


def _fetch_github_metadata_changed_files(
    repo_name: str | None = None,
    before_sha: str | None = None,
    after_sha: str | None = None,
    pr_number: int | None = None,
) -> list[str] | None:
    """
    Retrieve incremental changed files using GitHub's Compare API.

    Important:
    This function deliberately does NOT use the PR Files API. That API
    returns the complete PR file list, not the incremental commit diff.
    """
    repo_name = (
        repo_name or os.environ.get("GITHUB_REPOSITORY", "")
    ).strip()

    if not repo_name or not before_sha or not after_sha:
        return None

    headers = _get_github_headers()

    if not headers:
        print(
            "[GITHUB API WARNING] No GitHub token available "
            "for Compare API fallback."
        )
        return None

    compare_url = (
        f"https://api.github.com/repos/{repo_name}/compare/"
        f"{before_sha}...{after_sha}"
    )

    all_files = []
    page = 1

    while True:
        try:
            response = requests.get(
                compare_url,
                headers=headers,
                params={
                    "per_page": 100,
                    "page": page,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            print(
                f"[GITHUB API WARNING] Compare API request failed: {exc}"
            )
            return None

        if response.status_code != 200:
            print(
                f"[GITHUB API WARNING] Compare API returned "
                f"HTTP {response.status_code}: {response.text[:500]}"
            )
            return None

        try:
            data = response.json()
        except ValueError:
            print(
                "[GITHUB API WARNING] Compare API returned invalid JSON."
            )
            return None

        if not isinstance(data, dict):
            return None

        page_files = data.get("files", [])

        if not isinstance(page_files, list):
            return None

        for item in page_files:
            if isinstance(item, dict):
                filename = item.get("filename")
                status = item.get("status", "")

                if filename and status != "removed":
                    all_files.append(filename)

        if len(page_files) < 100:
            break

        page += 1

    filtered_files = filter_automation_files(
        all_files,
        repo_name=repo_name,
    )

    print(
        f"[GITHUB API] Incremental Compare API found "
        f"{len(all_files)} file(s); "
        f"{len(filtered_files)} file(s) remain after filtering."
    )

    return filtered_files


def get_incremental_changed_files(
    repo_path,
    before_sha,
    after_sha,
    repo_name: str | None = None,
    pr_number: int | None = None,
    target_branch: str | None = None,
    source_branch: str | None = None,
    raise_on_error: bool = True,
) -> list[str] | None:
    """
    Get files changed specifically between before_sha and after_sha.

    Detection order:
        1. Local Git diff between the exact commit SHAs.
        2. Fetch missing commits/history and retry the exact Git diff.
        3. GitHub Compare API for the same exact commit range.
        4. Raise an error (or return None when raise_on_error=False).

    This function never falls back to the complete PR Files API or to a
    branch diff, because either could include files outside the requested
    incremental commit range.
    """
    try:
        repo = Repo(repo_path)

        if not before_sha or not after_sha:
            raise ChangedFilesDetectionError(
                "Both before_sha and after_sha are required "
                "for incremental changed-file detection."
            )

        repo_name = (
            repo_name or os.environ.get("GITHUB_REPOSITORY", "")
        ).strip()

        pr_number = _get_pr_number(pr_number)

        # 1. Try the exact local commit comparison.
        if _has_commit(repo, before_sha) and _has_commit(repo, after_sha):
            try:
                changed_output = repo.git.diff(
                    "--name-only",
                    "--diff-filter=d",
                    before_sha,
                    after_sha,
                )

                raw_files = [
                    normalize_repo_path(line)
                    for line in changed_output.splitlines()
                    if line.strip()
                ]

                filtered_files = filter_automation_files(
                    raw_files,
                    repo_name=repo_name,
                )

                print(
                    f"[GIT DIFF] Exact commit comparison found "
                    f"{len(raw_files)} file(s); "
                    f"{len(filtered_files)} file(s) remain after filtering."
                )

                return filtered_files

            except Exception as exc:
                print(
                    f"[GIT DIFF WARNING] Exact commit comparison failed: "
                    f"{exc}"
                )

        # 2. Attempt to recover missing commits/history.
        print(
            "[GIT RECOVERY] One or both commits are unavailable, "
            "or the local diff failed. Attempting Git recovery..."
        )

        _attempt_git_fetch_recovery(
            repo=repo,
            before_sha=before_sha,
            after_sha=after_sha,
            pr_number=pr_number,
            target_branch=target_branch,
            source_branch=source_branch,
        )

        # 3. Retry the exact comparison after recovery.
        if _has_commit(repo, before_sha) and _has_commit(repo, after_sha):
            try:
                changed_output = repo.git.diff(
                    "--name-only",
                    "--diff-filter=d",
                    before_sha,
                    after_sha,
                )

                raw_files = [
                    normalize_repo_path(line)
                    for line in changed_output.splitlines()
                    if line.strip()
                ]

                filtered_files = filter_automation_files(
                    raw_files,
                    repo_name=repo_name,
                )

                print(
                    f"[GIT RECOVERY] Exact commit comparison succeeded. "
                    f"Found {len(filtered_files)} file(s) after filtering."
                )

                return filtered_files

            except Exception as exc:
                print(
                    f"[GIT RECOVERY WARNING] Retried exact diff failed: "
                    f"{exc}"
                )

        # 4. Use GitHub Compare API for the same exact commit range.
        print(
            "[GIT RECOVERY] Local recovery was unsuccessful. "
            "Trying GitHub Compare API..."
        )

        api_files = _fetch_github_metadata_changed_files(
            repo_name=repo_name,
            before_sha=before_sha,
            after_sha=after_sha,
            pr_number=pr_number,
        )

        if api_files is not None:
            print(
                f"[GIT RECOVERY] GitHub Compare API succeeded. "
                f"Found {len(api_files)} incremental file(s)."
            )
            return api_files

        raise ChangedFilesDetectionError(
            "Incremental changed-file detection failed for "
            f"'{before_sha}'..'{after_sha}'. "
            "The exact Git diff could not be completed, and the "
            "GitHub Compare API fallback was unsuccessful. "
            "No full-PR or branch-diff fallback was used because "
            "it could return unrelated files."
        )

    except ChangedFilesDetectionError:
        if raise_on_error:
            raise

        return None

    except Exception as exc:
        error = ChangedFilesDetectionError(
            f"Unexpected incremental changed-file detection error: {exc}"
        )

        if raise_on_error:
            raise error from exc

        return None
