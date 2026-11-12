"""
Manage PR requested reviewers based on validation outcome.

This module intentionally avoids custom email or notification logic. GitHub
already notifies the PR author on status changes. By removing reviewers on
validation failure and re-requesting them on success, we rely on GitHub's
native reviewer-request notifications instead of building our own.
"""

import re

import requests

from github.github_auth import (
    generate_installation_token
)

REVIEWER_MARKER_RE = re.compile(
    r"<!--\s*original-reviewers:\s*([^-]*?)-->"
)
TEAM_MARKER_RE = re.compile(
    r"<!--\s*original-team-reviewers:\s*([^-]*?)-->"
)
PERSISTENCE_MARKER = "original-reviewers:"


def _api_headers(token: str) -> dict:

    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def _split_marker_list(value: str) -> list[str]:

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def _extract_reviewers_from_pr(pr: dict) -> tuple[list[str], list[str]]:

    reviewers = [
        user.get("login")
        for user in (pr.get("requested_reviewers") or [])
        if user.get("login")
    ]
    teams = [
        team.get("slug")
        for team in (pr.get("requested_teams") or [])
        if team.get("slug")
    ]
    return reviewers, teams


def _build_marker_comment_body(
    reviewers: list[str],
    teams: list[str]
) -> str:

    reviewer_value = ",".join(reviewers)
    team_value = ",".join(teams)
    return (
        f"<!-- original-reviewers: {reviewer_value} -->\n"
        f"<!-- original-team-reviewers: {team_value} -->"
    )


def _parse_marker_comment_body(
    body: str
) -> tuple[list[str], list[str]]:

    if not body:
        return [], []

    reviewer_match = REVIEWER_MARKER_RE.search(body)
    team_match = TEAM_MARKER_RE.search(body)

    reviewers = (
        _split_marker_list(reviewer_match.group(1))
        if reviewer_match
        else []
    )
    teams = (
        _split_marker_list(team_match.group(1))
        if team_match
        else []
    )
    return reviewers, teams


def _list_pr_comments(
    repo_name: str,
    pr_number: int,
    token: str
) -> list[dict]:

    url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/issues/{pr_number}/comments"
    )
    response = requests.get(
        url,
        headers=_api_headers(token),
        params={"per_page": 100},
    )
    response.raise_for_status()
    return response.json()


def _find_persistence_comment(
    comments: list[dict]
) -> dict | None:

    for comment in comments:
        body = comment.get("body") or ""
        if PERSISTENCE_MARKER in body:
            return comment
    return None


def _upsert_persistence_comment(
    repo_name: str,
    pr_number: int,
    body: str,
    token: str
) -> None:

    comments = _list_pr_comments(
        repo_name,
        pr_number,
        token
    )
    existing = _find_persistence_comment(comments)

    if existing:
        comment_id = existing.get("id")
        url = (
            f"https://api.github.com/repos/"
            f"{repo_name}/issues/comments/{comment_id}"
        )
        response = requests.patch(
            url,
            headers=_api_headers(token),
            json={"body": body},
        )
        response.raise_for_status()
        print(
            "[REVIEWERS] Updated persisted reviewer marker comment"
        )
        return

    url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/issues/{pr_number}/comments"
    )
    response = requests.post(
        url,
        headers=_api_headers(token),
        json={"body": body},
    )
    response.raise_for_status()
    print(
        "[REVIEWERS] Created persisted reviewer marker comment"
    )


def _fetch_current_requested_reviewers(
    repo_name: str,
    pr_number: int,
    token: str
) -> tuple[list[str], list[str]]:

    url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/pulls/{pr_number}"
    )
    response = requests.get(
        url,
        headers=_api_headers(token),
    )
    response.raise_for_status()
    data = response.json()
    return _extract_reviewers_from_pr(data)


def capture_original_reviewers(
    repo_name: str,
    pr_number: int,
    pr: dict
) -> None:
    """
    Persist the PR's current requested reviewers before any removal.

    Only writes when the PR currently has a non-empty reviewer or team list,
    so an empty list after our own removal does not overwrite the saved state.
    """

    reviewers, teams = _extract_reviewers_from_pr(pr)
    if not reviewers and not teams:
        print(
            "[REVIEWERS] Skipping capture — no requested reviewers on PR"
        )
        return

    token = generate_installation_token()
    body = _build_marker_comment_body(
        reviewers,
        teams
    )
    _upsert_persistence_comment(
        repo_name,
        pr_number,
        body,
        token
    )
    print(
        f"[REVIEWERS] Captured original reviewers: "
        f"users={reviewers}, teams={teams}"
    )


def remove_requested_reviewers(
    repo_name: str,
    pr_number: int
) -> None:
    """
    Unrequest all currently requested reviewers on the PR.

    Safe to call repeatedly — no-ops when the PR has no requested reviewers.
    """

    token = generate_installation_token()
    reviewers, teams = _fetch_current_requested_reviewers(
        repo_name,
        pr_number,
        token
    )

    if not reviewers and not teams:
        print(
            "[REVIEWERS] No requested reviewers to remove"
        )
        return

    url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/pulls/{pr_number}/requested_reviewers"
    )
    payload = {}
    if reviewers:
        payload["reviewers"] = reviewers
    if teams:
        payload["team_reviewers"] = teams

    response = requests.delete(
        url,
        headers=_api_headers(token),
        json=payload,
    )

    if response.status_code == 422:
        print(
            "[REVIEWERS] Reviewers already removed or unavailable "
            f"(HTTP 422): {response.text}"
        )
        return

    response.raise_for_status()
    print(
        f"[REVIEWERS] Removed requested reviewers: "
        f"users={reviewers}, teams={teams}"
    )


def restore_requested_reviewers(
    repo_name: str,
    pr_number: int
) -> None:
    """
    Re-request reviewers saved in the persistence marker comment.

    Skips anyone already currently requested to avoid duplicate API calls.
    """

    token = generate_installation_token()
    comments = _list_pr_comments(
        repo_name,
        pr_number,
        token
    )
    marker = _find_persistence_comment(comments)

    if not marker:
        print(
            "[REVIEWERS] No persisted reviewer marker found — "
            "nothing to restore"
        )
        return

    saved_reviewers, saved_teams = _parse_marker_comment_body(
        marker.get("body") or ""
    )

    if not saved_reviewers and not saved_teams:
        print(
            "[REVIEWERS] Persisted reviewer marker is empty — "
            "nothing to restore"
        )
        return

    current_reviewers, current_teams = (
        _fetch_current_requested_reviewers(
            repo_name,
            pr_number,
            token
        )
    )

    reviewers_to_add = [
        login
        for login in saved_reviewers
        if login not in current_reviewers
    ]
    teams_to_add = [
        slug
        for slug in saved_teams
        if slug not in current_teams
    ]

    if not reviewers_to_add and not teams_to_add:
        print(
            "[REVIEWERS] All persisted reviewers are already requested"
        )
        return

    url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/pulls/{pr_number}/requested_reviewers"
    )
    payload = {}
    if reviewers_to_add:
        payload["reviewers"] = reviewers_to_add
    if teams_to_add:
        payload["team_reviewers"] = teams_to_add

    response = requests.post(
        url,
        headers=_api_headers(token),
        json=payload,
    )

    if response.status_code == 422:
        print(
            "[REVIEWERS] Could not re-request some reviewers "
            f"(HTTP 422): {response.text}"
        )
        return

    response.raise_for_status()
    print(
        f"[REVIEWERS] Restored requested reviewers: "
        f"users={reviewers_to_add}, teams={teams_to_add}"
    )


def handle_reviewers_for_validation_result(
    repo_name: str,
    pr_number: int,
    validation_passed: bool
) -> None:
    """
    Apply reviewer management after validation completes.

    On failure: remove requested reviewers and rely on GitHub's automatic
    author notifications — no custom email logic is used.

    On success: restore the originally captured reviewers so GitHub notifies
    them through its native review-request flow.
    """

    try:
        if validation_passed:
            print(
                "[REVIEWERS] Validation passed — restoring reviewers"
            )
            restore_requested_reviewers(
                repo_name,
                pr_number
            )
        else:
            print(
                "[REVIEWERS] Validation failed — removing reviewers"
            )
            remove_requested_reviewers(
                repo_name,
                pr_number
            )
    except Exception as e:
        print(
            f"[WARNING] Reviewer management failed: {e}"
        )
