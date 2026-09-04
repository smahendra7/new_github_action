import json
import hmac
import hashlib
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response, JSONResponse

from github.repository_manager import clone_repository
from github.diff_extractor import (
    get_diff,
    get_incremental_diff,
    extract_added_code,
    extract_full_code,
    count_file_sections_in_code
)

from github.changed_files import (
    get_pr_changed_files,
    get_incremental_changed_files
)

from github.pr_commenter import (
    post_pr_comment,
    get_pr_details
)

from github.reviewer_manager import (
    capture_original_reviewers,
    handle_reviewers_for_validation_result
)

from ai.prompt_builder import build_review_prompt
from ai.reviewer import review_code
from ai.description_validator import validate_pr_description
from ai.comment_validator import (
    validate_code_comments,
    format_comment_validation_failure
)
from ai.severity_classifier import classify_and_group_review
from ai.description_code_match_validator import validate_description_matches_code

from reports.report_generator import (
    generate_word_report,
    generate_mismatch_report,
    get_report_bytes
)

app = FastAPI()

REVIEWER_MANAGEMENT_ACTIONS = [
    "opened",
    "synchronize",
    "reopened",
]

GITHUB_SECRET = os.environ.get(
    "GITHUB_WEBHOOK_SECRET",
    ""
)


def verify_signature(
    payload_body: bytes,
    signature_header: str
) -> bool:

    if not signature_header:
        return False

    try:

        sha_name, signature = (
            signature_header.split("=")
        )

        if sha_name != "sha256":
            return False

        mac = hmac.new(
            GITHUB_SECRET.encode(),
            msg=payload_body,
            digestmod=hashlib.sha256
        )

        expected_signature = (
            mac.hexdigest()
        )

        return hmac.compare_digest(
            expected_signature,
            signature
        )

    except Exception as e:

        print(
            f"Signature Error: {e}"
        )

        return False


REPORTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "reports",
    "files"
)


@app.get("/")
def home():

    return {
        "message": "AI Code Reviewer V2 Running"
    }


@app.get("/reports/download/{filename}")
def download_report(filename: str):
    """
    Serve a generated Word (.docx) review report by filename.
    Checks disk, memory cache, or regenerates from GitHub on demand.
    """
    data = get_report_bytes(filename, REPORTS_DIR)

    if not data:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": f"Report '{filename}' not found."
            }
        )

    return Response(
        content=data,
        media_type=(
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


def process_pr_event(
    payload: dict,
    event: str,
    action: str,
    before_sha: str,
    after_sha: str
) -> dict:
    """
    Core PR event processing logic.

    Called by both the FastAPI webhook route and the GitHub Actions runner
    (run_action.py). The payload, event, action, and SHA values are extracted
    by the caller from either the HTTP request or $GITHUB_EVENT_PATH — the
    processing logic here is identical in both cases.
    """

    print(
        f"Event: {event} | Action: {action}"
    )

    print(
        f"Commit Window Shas -> Before: {before_sha} | After: {after_sha}"
    )

    if event == "ping":

        print(
            "GitHub App webhook verified"
        )

        return {
            "status": "success",
            "message": "pong"
        }

    if event == "pull_request":

        pr = payload.get(
            "pull_request",
            {}
        )

        repo = payload.get(
            "repository",
            {}
        )

        repo_name = repo.get(
            "full_name"
        )

        pr_number = pr.get(
            "number"
        )

        source_branch = pr.get(
            "head",
            {}
        ).get(
            "ref"
        )

        target_branch = pr.get(
            "base",
            {}
        ).get(
            "ref"
        )

        print("\n----- PR DETAILS -----")

        print(
            f"Repository: {repo_name} | PR #{pr_number}"
        )

        print(
            f"Branch Target Route: [{target_branch}] <- [{source_branch}]"
        )

        if action in [
            "opened",
            "synchronize",
            "reopened",
            "edited"
        ]:

            # Always fetch LIVE PR data from GitHub API.
            # This ensures Redeliver scenarios also use the current description.
            payload_title = (pr.get("title") or "").strip()
            payload_body = (pr.get("body") or "").strip()

            live_title, live_body = get_pr_details(repo_name, pr_number)

            # Prefer the richer of live API vs webhook payload (avoids stale/placeholder bodies).
            def _richer_text(primary: str, fallback: str) -> str:
                primary = (primary or "").strip()
                fallback = (fallback or "").strip()
                if not primary:
                    return fallback
                if not fallback:
                    return primary
                return primary if len(primary) >= len(fallback) else fallback

            pr_title = _richer_text(live_title, payload_title)
            pr_body = _richer_text(live_body, payload_body)

            print(f"[DEBUG] Payload body : {payload_body!r}")
            print(f"[DEBUG] Live API body: {live_body!r}")
            print(f"[DEBUG] Final body used for validation: {pr_body!r}")

            if action in REVIEWER_MANAGEMENT_ACTIONS:
                try:
                    capture_original_reviewers(
                        repo_name,
                        pr_number,
                        pr
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Could not capture original reviewers: {e}"
                    )

            print("\nExecuting PR Description Validation Gate...")
            is_valid, validation_reason = validate_pr_description(
                pr_title,
                pr_body
            )

            if not is_valid:
                print(
                    "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                print(
                    f"PR DESCRIPTION VALIDATION GATE >>> FAILED <<<"
                )
                print(
                    f"Reason: {validation_reason}"
                )
                print(
                    "Pipeline STOPPED. Code review will NOT run."
                )
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                try:
                    post_pr_comment(
                        repo_name,
                        pr_number,
                        f"\u274c **PR Description Validation Failed**\n\n**Reason:** {validation_reason}\n\nPlease update your PR description with a clear explanation of what this PR does, then push a new commit to re-trigger the review."
                    )
                    print(
                        "Validation Failure Comment Posted To GitHub Successfully"
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Could not post GitHub comment: {e}"
                    )
                    print(
                        "[WARNING] Check that GITHUB_TOKEN has write permissions."
                    )

                if action in REVIEWER_MANAGEMENT_ACTIONS:
                    handle_reviewers_for_validation_result(
                        repo_name,
                        pr_number,
                        validation_passed=False
                    )

                return {
                    "status": "failed",
                    "reason": validation_reason
                }

            workspace_path = (
                f"workspace/pr_{pr_number}"
            )

            print(
                "\nStarting Repository Clone..."
            )

            workspace_path = clone_repository(
                repo_name,
                workspace_path
            )

            print(
                "Repository Clone Completed"
            )

            if (
                action == "synchronize"
                and before_sha
                and after_sha
            ):

                changed_files = (
                    get_incremental_changed_files(
                        workspace_path,
                        before_sha,
                        after_sha
                    )
                )

                print(
                    "Review Type: INCREMENTAL REVIEW"
                )

                diff = get_incremental_diff(
                    workspace_path,
                    before_sha,
                    after_sha,
                    changed_files
                )

            elif action == "opened":

                changed_files = (
                    get_pr_changed_files(
                        workspace_path,
                        target_branch,
                        source_branch
                    )
                )

                print(
                    "Review Type: FULL PR REVIEW"
                )

                diff = get_diff(
                    workspace_path,
                    target_branch,
                    source_branch,
                    changed_files
                )

            else:

                changed_files = (
                    get_pr_changed_files(
                        workspace_path,
                        target_branch,
                        source_branch
                    )
                )

                print(
                    "Review Type: FALLBACK FULL REVIEW"
                )

                diff = get_diff(
                    workspace_path,
                    target_branch,
                    source_branch,
                    changed_files
                )

            print("\n===== PR DIFF =====")
            print(diff)
            print("===================\n")

            print(
                "===== CHANGED FILES ====="
            )

            for file in changed_files:
                print(file)

            print(
                "========================="
            )

            print(
                f"\nSuccessfully Parsed {len(changed_files)} Changed File(s)"
            )

            if not changed_files:
                print(
                    "\nNo reviewable application files changed "
                    "(only automation/workflow files modified). Skipping AI code review."
                )

                try:
                    post_pr_comment(
                        repo_name,
                        pr_number,
                        "ℹ️ **AI Code Review Skipped**\n\n"
                        "No application files were modified in this PR (only automation/workflow files changed)."
                    )
                    print(
                        "Notification Comment Posted Successfully"
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Could not post GitHub comment: {e}"
                    )

                if action in REVIEWER_MANAGEMENT_ACTIONS:
                    handle_reviewers_for_validation_result(
                        repo_name,
                        pr_number,
                        validation_passed=True
                    )

                try:
                    if os.path.exists(workspace_path):
                        shutil.rmtree(workspace_path)
                        print(
                            f"Workspace cleaned up: {workspace_path}"
                        )
                except Exception as cleanup_err:
                    print(
                        f"Workspace Cleanup Warning: {cleanup_err}"
                    )

                return {
                    "status": "success",
                    "message": "no_application_files_to_review"
                }

            print("\nExecuting Code Comment Validation Gate...")
            comment_ref = (
                after_sha
                if action == "synchronize" and after_sha
                else None
            )
            comment_mismatches = validate_code_comments(
                workspace_path,
                changed_files,
                source_branch,
                comment_ref
            )

            if comment_mismatches:
                print(
                    "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                print(
                    "CODE COMMENT VALIDATION GATE >>> FAILED <<<"
                )
                print(
                    f"Mismatches Found: {len(comment_mismatches)}"
                )
                print(
                    "Pipeline STOPPED. Code review will NOT run."
                )
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                try:
                    failure_message = format_comment_validation_failure(
                        comment_mismatches
                    )

                    # ---- Word Report Generation (additive) ---------------
                    # Generate a .docx report from the mismatches and append
                    # a download link to the failure message when BASE_URL is
                    # set. If anything fails the original message is used.
                    try:
                        report_file_path = generate_mismatch_report(
                            repo_name,
                            pr_number,
                            comment_mismatches,
                            REPORTS_DIR
                        )
                        report_filename = os.path.basename(
                            report_file_path
                        )
                        base_url = (
                            os.environ.get("BASE_URL", "").rstrip("/")
                        )
                        if base_url:
                            download_url = (
                                f"{base_url}/reports/download/"
                                f"{report_filename}"
                            )
                            failure_message = (
                                f"{failure_message}\n\n"
                                "---\n"
                                f"\U0001f4c4 [Download Word Report]"
                                f"({download_url})"
                            )
                            print(
                                "[REPORT] Download link appended to "
                                "comment validation failure comment: "
                                f"{download_url}"
                            )
                        else:
                            print(
                                "[REPORT] BASE_URL not set — download "
                                "link will not be included in the comment."
                            )
                    except Exception as report_err:
                        print(
                            f"[REPORT] Warning: Could not generate Word "
                            f"report: {report_err}. Continuing without report."
                        )
                    # ---- End Word Report Generation ----------------------

                    post_pr_comment(
                        repo_name,
                        pr_number,
                        failure_message
                    )
                    print(
                        "Comment Validation Failure Comment Posted Successfully"
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Could not post GitHub comment: {e}"
                    )

                if action in REVIEWER_MANAGEMENT_ACTIONS:
                    handle_reviewers_for_validation_result(
                        repo_name,
                        pr_number,
                        validation_passed=False
                    )

                try:
                    if os.path.exists(workspace_path):
                        shutil.rmtree(workspace_path)
                        print(
                            f"Workspace cleaned up: {workspace_path}"
                        )
                except Exception as cleanup_err:
                    print(
                        f"Workspace Cleanup Warning: {cleanup_err}"
                    )

                return {
                    "status": "failed",
                    "reason": "comment_validation_failed"
                }

            print("\nExecuting PR Description-vs-Code Match Validation Gate...")
            desc_match_valid, desc_match_reason = validate_description_matches_code(
                pr_title,
                pr_body,
                diff
            )

            if not desc_match_valid:
                print(
                    "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                print(
                    "PR DESCRIPTION DOES NOT MATCH CODE CHANGES >>> FAILED <<<"
                )
                print(
                    f"Reason: {desc_match_reason}"
                )
                print(
                    "Pipeline STOPPED. Code review will NOT run."
                )
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                try:
                    post_pr_comment(
                        repo_name,
                        pr_number,
                        f"\u274c **PR Description Does Not Match Code Changes**\n\n"
                        f"**Reason:** {desc_match_reason}\n\n"
                        "Please update your PR description so it accurately reflects "
                        "the code changes, then push a new commit to re-trigger the review."
                    )
                    print(
                        "Description-vs-Code Mismatch Failure Comment Posted To GitHub Successfully"
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Could not post GitHub comment: {e}"
                    )

                if action in REVIEWER_MANAGEMENT_ACTIONS:
                    handle_reviewers_for_validation_result(
                        repo_name,
                        pr_number,
                        validation_passed=False
                    )

                try:
                    if os.path.exists(workspace_path):
                        shutil.rmtree(workspace_path)
                        print(
                            f"Workspace cleaned up: {workspace_path}"
                        )
                except Exception as cleanup_err:
                    print(
                        f"Workspace Cleanup Warning: {cleanup_err}"
                    )

                return {
                    "status": "failed",
                    "reason": desc_match_reason
                }

            print(
                "Starting Local AI LLM Processing Engine..."
            )

            code_source_ref = (
                after_sha
                if action == "synchronize" and after_sha
                else source_branch
            )

            full_code = extract_full_code(
                workspace_path,
                changed_files,
                code_source_ref
            )

            if not full_code.strip():
                full_code = extract_added_code(diff)

            print(
                "\n===== SOURCE CODE SENT TO LLM ====="
            )

            print(full_code)

            print(
                "===================================\n"
            )

            expected_file_sections = len(changed_files)
            actual_file_sections = count_file_sections_in_code(full_code)

            if actual_file_sections != expected_file_sections:
                print(
                    "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                print(
                    "[PROMPT VALIDATION ERROR] Incomplete prompt: "
                    f"expected {expected_file_sections} FILE section(s), "
                    f"found {actual_file_sections}."
                )
                print(
                    "Pipeline STOPPED. LLM review will NOT run."
                )
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                try:
                    if os.path.exists(workspace_path):
                        shutil.rmtree(workspace_path)
                        print(
                            f"Workspace cleaned up: {workspace_path}"
                        )
                except Exception as cleanup_err:
                    print(
                        f"Workspace Cleanup Warning: {cleanup_err}"
                    )

                return {
                    "status": "failed",
                    "reason": "incomplete_prompt_file_sections"
                }

            prompt = build_review_prompt(
                full_code,
                changed_files
            )

            print("\n===== FINAL PROMPT SENT TO LLM =====")
            print(prompt)
            print("====================================\n")

            review = review_code(
                prompt
            )
            review = classify_and_group_review(
                review,
                workspace_path,
                diff
            )

            # ---- Word Report Generation (additive) ----------------------
            # Generate a downloadable .docx report from the review results.
            # If report generation fails for any reason, the original review
            # string is used unchanged and the PR comment is still posted.
            try:
                report_file_path = generate_word_report(
                    repo_name,
                    pr_number,
                    review,
                    REPORTS_DIR
                )
                report_filename = os.path.basename(report_file_path)

                base_url = os.environ.get("BASE_URL", "").rstrip("/")

                if base_url:
                    download_url = (
                        f"{base_url}/reports/download/{report_filename}"
                    )
                    review = (
                        f"{review}\n\n"
                        "---\n"
                        f"\U0001f4c4 [Download Word Report]({download_url})"
                    )
                    print(
                        f"[REPORT] Download link appended to review comment: "
                        f"{download_url}"
                    )
                else:
                    print(
                        "[REPORT] BASE_URL not set — download link will not "
                        "be included in the PR comment."
                    )

            except Exception as report_err:
                print(
                    f"[REPORT] Warning: Could not generate Word report: "
                    f"{report_err}. Continuing without report."
                )
            # ---- End Word Report Generation ------------------------------

            print(
                "\nPosting Comment To GitHub..."
            )

            try:

                post_pr_comment(
                    repo_name,
                    pr_number,
                    review
                )

                print(
                    "GitHub Comment Posted Successfully"
                )

            except Exception as e:

                print(
                    f"GitHub Comment Error: {e}"
                )

            finally:

                # Clean up the cloned workspace to prevent disk buildup
                try:
                    if os.path.exists(workspace_path):
                        shutil.rmtree(workspace_path)
                        print(
                            f"Workspace cleaned up: {workspace_path}"
                        )
                except Exception as cleanup_err:
                    print(
                        f"Workspace Cleanup Warning: {cleanup_err}"
                    )

            if action in REVIEWER_MANAGEMENT_ACTIONS:
                handle_reviewers_for_validation_result(
                    repo_name,
                    pr_number,
                    validation_passed=True
                )

        else:

            print(
                f"Skipping action: {action}"
            )

    print(
        "\n================================================\n"
    )

    return {
        "status": "success"
    }


@app.post("/github/webhook")
async def github_webhook(
    request: Request
):

    body = await request.body()

    signature = request.headers.get(
        "X-Hub-Signature-256"
    )

    validation_result = verify_signature(
        body,
        signature
    )

    print("\n================================================")
    print("Webhook Received")
    print("Signature Valid:", validation_result)

    if not validation_result:

        print("Webhook Signature Failed")

        return {
            "status": "error",
            "message": "Invalid signature"
        }

    payload = json.loads(
        body.decode("utf-8")
    )

    event = request.headers.get(
        "X-GitHub-Event"
    )

    action = payload.get("action")
    before_sha = payload.get("before")
    after_sha = payload.get("after")

    return process_pr_event(
        payload,
        event,
        action,
        before_sha,
        after_sha
    )