
import requests

from github.github_auth import generate_installation_token


# Keep each GitHub comment comfortably below the API body-size limit.
MAX_COMMENT_CHARS = 58_000


def _split_comment(review: str, max_chars: int = MAX_COMMENT_CHARS) -> list[str]:
    """Split a review into ordered chunks, preferring line boundaries."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    if len(review) <= max_chars:
        return [review]

    chunks = []
    remaining = review

    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n", 0, max_chars + 1)

        # Prefer a newline. Split a long line only when necessary.
        if split_at <= 0:
            split_at = max_chars
        else:
            split_at += 1

        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]

    if remaining:
        chunks.append(remaining)

    return chunks


def post_pr_comment(repo_name, pr_number, review):
    """Post a PR review as one or more GitHub comments."""

    if not isinstance(review, str):
        raise TypeError(
            f"review must be a string, got {type(review).__name__}"
        )

    token = generate_installation_token()

    url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/issues/{pr_number}/comments"
    )

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    review_parts = _split_comment(review)
    total_parts = len(review_parts)

    print(
        f"[GITHUB COMMENT] Repository: {repo_name}"
    )
    print(
        f"[GITHUB COMMENT] PR: #{pr_number}"
    )
    print(
        f"[GITHUB COMMENT] Total review characters: {len(review)}"
    )
    print(
        f"[GITHUB COMMENT] Number of comment parts: {total_parts}"
    )

    for index, part in enumerate(review_parts, start=1):

        if total_parts == 1:
            comment_body = part
        else:
            heading = (
                f"**AI Code Review — Part {index}/{total_parts}**\n\n"
            )
            comment_body = heading + part

        print(
            f"[GITHUB COMMENT] Posting part "
            f"{index}/{total_parts}"
        )
        print(
            f"[GITHUB COMMENT] Comment characters: "
            f"{len(comment_body)}"
        )

        try:
            response = requests.post(
                url,
                headers=headers,
                json={"body": comment_body},
                timeout=30,
            )

        except requests.RequestException as exc:
            print(
                f"[GITHUB COMMENT ERROR] Network error "
                f"on part {index}/{total_parts}: {exc}"
            )
            raise

        if not response.ok:

            print(
                f"[GITHUB COMMENT ERROR] GitHub rejected "
                f"part {index}/{total_parts}"
            )

            print(
                f"[GITHUB COMMENT ERROR] HTTP Status: "
                f"{response.status_code}"
            )

            print(
                "[GITHUB COMMENT ERROR] GitHub response body:"
            )

            print(response.text[:5000])

            response.raise_for_status()

        print(
            f"[GITHUB COMMENT] Successfully posted "
            f"part {index}/{total_parts}"
        )


def get_pr_details(
    repo_name,
    pr_number
) -> tuple[str, str]:

    try:
        token = generate_installation_token()

        url = (
            f"https://api.github.com/repos/"
            f"{repo_name}/pulls/{pr_number}"
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json"
        }

        response = requests.get(
            url,
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()

            title = data.get("title") or ""
            body = data.get("body") or ""

            print(
                f"[GITHUB API] Fetched PR #{pr_number} "
                f"-> Title: {title!r}, Body: {body!r}"
            )

            return title, body

        print(
            f"[GITHUB API WARNING] Failed to fetch "
            f"PR #{pr_number} details. "
            f"Status: {response.status_code}"
        )

    except Exception as e:
        print(
            f"[GITHUB API ERROR] "
            f"Error fetching live PR details: {e}"
        )

    return "", ""
