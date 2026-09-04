import os
from git import Repo

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


def is_automation_file(file_path: str) -> bool:
    """
    Determine whether a file belongs to the automation infrastructure
    and should be excluded from automated code reviews.
    """
    if not file_path:
        return False

    normalized = file_path.strip().replace("\\", "/").lower()
    while normalized.startswith("./") or normalized.startswith("/"):
        normalized = normalized.lstrip("./")

    # Match automation directory prefixes
    for prefix in AUTOMATION_PREFIXES:
        if normalized.startswith(prefix):
            return True

    # Match exact root-level automation files
    if normalized in AUTOMATION_EXACT_FILES:
        return True

    # Match requirements files (e.g. requirements.txt, requirements-dev.txt)
    if normalized.startswith("requirements"):
        return True

    return False


def filter_automation_files(file_list: list[str]) -> list[str]:
    """
    Filter out automation files from the list of changed files.
    """
    return [
        f for f in file_list
        if not is_automation_file(f)
    ]


def get_pr_changed_files(
    repo_path,
    target_branch,
    source_branch
):

    repo = Repo(repo_path)

    origin = repo.remotes.origin

    origin.fetch()

    try:

        changed_files = repo.git.diff(
            "--name-only",
            f"origin/{target_branch}",
            f"origin/{source_branch}"
        )

        raw_files = [
            line.strip()
            for line in changed_files.splitlines()
            if line.strip()
        ]

        return filter_automation_files(raw_files)

    except Exception as e:

        print(
            f"PR Changed Files Error: {e}"
        )

        return []


def get_incremental_changed_files(
    repo_path,
    before_sha,
    after_sha
):

    repo = Repo(repo_path)

    try:

        changed_files = repo.git.diff(
            "--name-only",
            before_sha,
            after_sha
        )

        raw_files = [
            line.strip()
            for line in changed_files.splitlines()
            if line.strip()
        ]

        return filter_automation_files(raw_files)

    except Exception as e:

        print(
            f"Incremental Changed Files Error: {e}"
        )

        return []