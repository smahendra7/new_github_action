import base64
import os
import time
import jwt
import requests


def get_github_token() -> str:
    """
    Retrieve the GitHub authentication token.

    In GitHub Actions, the built-in GITHUB_TOKEN is supplied via the environment.
    Falls back to generating a GitHub App installation token if GITHUB_TOKEN
    is not set and GitHub App credentials are provided.
    """
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token

    # Fallback: GitHub App installation token (for legacy webhook / standalone usage)
    app_id = os.environ.get("GITHUB_APP_ID", "4365050")
    installation_id = os.environ.get("GITHUB_INSTALLATION_ID", "148257662")
    private_key = os.environ.get("GITHUB_APP_PRIVATE_KEY", "")

    if private_key:
        if not private_key.strip().startswith("-----"):
            try:
                private_key = base64.b64decode(private_key).decode("utf-8")
            except Exception:
                pass
    else:
        private_key = os.environ.get("GITHUB_PRIVATE_KEY", "")

    if not private_key:
        raise ValueError(
            "GITHUB_TOKEN environment variable is not set (and no GitHub App private key provided)."
        )

    private_key = private_key.replace("\\n", "\n")

    payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,
        "iss": app_id
    }

    encoded_jwt = jwt.encode(
        payload,
        private_key,
        algorithm="RS256"
    )

    url = (
        "https://api.github.com/app/installations/"
        f"{installation_id}/access_tokens"
    )

    headers = {
        "Authorization": f"Bearer {encoded_jwt}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.post(
        url,
        headers=headers
    )

    response.raise_for_status()

    data = response.json()

    return data["token"]


# Alias for backwards compatibility with existing imports across modules
generate_installation_token = get_github_token