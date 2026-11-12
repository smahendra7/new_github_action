from git import Repo
import os
import shutil
import time

from github.github_auth import (
    generate_installation_token
)


def clone_repository(
    repo_name,
    local_path
):

    token = generate_installation_token()

    repo_url = (
        f"https://x-access-token:{token}"
        f"@github.com/{repo_name}.git"
    )

    if os.path.exists(local_path):

        try:
            shutil.rmtree(local_path)

        except Exception:

            local_path = (
                f"{local_path}_{int(time.time())}"
            )

            print(
                "Folder is locked. "
                "Creating a new folder."
            )

    Repo.clone_from(
        repo_url,
        local_path
    )

    print(
        f"Repository cloned to: {local_path}"
    )

    return local_path