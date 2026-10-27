import requests

from github.github_auth import (
    generate_installation_token
)


def post_pr_comment(
    repo_name,
    pr_number,
    review
):

    token = (
        generate_installation_token()
    )

    url = (
        f"https://api.github.com/repos/"
        f"{repo_name}/issues/"
        f"{pr_number}/comments"
    )

    headers = {
        "Authorization":
        f"token {token}",

        "Accept":
        "application/vnd.github+json"
    }

    body = {
        "body": review
    }

    response = requests.post(
        url,
        headers=headers,
        json=body
    )

    response.raise_for_status()


def get_pr_details(
    repo_name,
    pr_number
) -> tuple[str, str]:

    try:
        token = generate_installation_token()

        # Use /pulls/ endpoint — requires pull-requests: read/write permission
        # which is granted by the workflow.
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
            print(f"[GITHUB API] Fetched PR #{pr_number} -> Title: {title!r}, Body: {body!r}")
            return title, body

        print(f"[GITHUB API WARNING] Failed to fetch PR #{pr_number} details. Status: {response.status_code}")

    except Exception as e:
        print(f"[GITHUB API ERROR] Error fetching live PR details: {e}")

    return "", ""