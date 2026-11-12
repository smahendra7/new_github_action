"""
GitHub Actions runner for the AI PR Automation.

This script is the entry point when the automation runs via GitHub Actions.
It reads the PR event payload from the standard $GITHUB_EVENT_PATH environment
variable (the same JSON that GitHub would POST to a webhook endpoint), extracts
the relevant fields, and calls process_pr_event() - the shared processing
function in main.py.

No processing logic lives here. All validation gates, AI review, PR commenting,
reviewer management, and report generation remain in main.py unchanged.
"""

import json
import os
import sys

from main import process_pr_event


def main():

    # $GITHUB_EVENT_PATH is set by the GitHub Actions runner and points to a
    # file containing the full webhook payload JSON for the triggering event.
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    if not event_path:
        print(
            "[ERROR] GITHUB_EVENT_PATH is not set. "
            "This script must be run inside a GitHub Actions workflow."
        )
        sys.exit(1)

    with open(event_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # $GITHUB_EVENT_NAME is the event type (e.g. "pull_request").
    # This is equivalent to the X-GitHub-Event header in a webhook request.
    event = os.environ.get("GITHUB_EVENT_NAME", "pull_request")

    # These fields are present at the top level of the pull_request event
    # payload - the same fields the webhook handler reads from the JSON body.
    # For synchronize, before/after represent the commit window of the push.
    # We also provide a fallback to head.sha for after_sha if top-level after is absent.
    pr = payload.get("pull_request", {})
    action = payload.get("action")
    before_sha = payload.get("before")
    after_sha = payload.get("after") or pr.get("head", {}).get("sha")

    print("\n================================================")
    print("GitHub Actions Runner - AI PR Automation")
    print(f"Event: {event} | Action: {action}")
    print("================================================")

    result = process_pr_event(
        payload,
        event,
        action,
        before_sha,
        after_sha
    )

    print(f"\nResult: {result}")

    # Exit non-zero if the automation reported a failure so the Actions step
    # is marked as failed in the GitHub UI (makes failures visible in the PR).
    if result.get("status") == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
