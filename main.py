import json
import hmac
import hashlib
import os
import re
import shutil
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse

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
from ai.reviewer import ReviewResult, review_code
from ai.description_validator import validate_pr_description
from ai.comment_validator import validate_code_comments
from ai.description_code_match_validator import validate_description_matches_code
from ai.suggestion_generator import (
    generate_description_suggestion,
    generate_comment_suggestion,
)
from ai.context_config import SAFE_INPUT_TOKEN_LIMIT
from ai.token_counter import count_tokens
from ai.chunk_manager import (
    execute_chunked_review,
    parse_diff_by_file,
    parse_full_code_files,
)
from ai.context_extractor import (
    extract_change_aware_context,
    extract_file_focused_review_context,
)
from ai.result_aggregator import aggregate_reviews


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




@app.get("/")
def home():

    return {
        "message": "AI Code Reviewer V2 Running"
    }


def _build_validation_section(
    desc_warning: str | None,
    desc_reason: str | None,
    desc_suggestion: str | None,
    comment_warnings: list,
) -> str:
    """
    Build the ## PR Validation section of the combined PR comment.

    desc_warning      - set if description validation produced a warning
    desc_reason       - human-readable reason text
    desc_suggestion   - AI-generated improved description
    comment_warnings  - list of dicts: {file, line, comment, code, reason, suggestion}

    Returns the full markdown section string.
    """
    has_warnings = bool(desc_warning) or bool(comment_warnings)

    if not has_warnings:
        return "## PR Validation\n\n\u2705 No validation issues found."

    blocks = []

    # ── PR Description Validation ──────────────────────────────────────────
    if desc_warning:
        desc_lines = [f"### \u26a0\ufe0f {desc_warning}\n"]
        if desc_reason:
            desc_lines.append(f"**Reason:** {desc_reason}\n")
        if desc_suggestion:
            desc_lines.append(f"\U0001f4a1 **Suggested PR Description:**\n\n> {desc_suggestion}")
        blocks.append("\n".join(desc_lines).strip())

    # ── Code Comment Accuracy ──────────────────────────────────────────────
    if comment_warnings:
        for item in comment_warnings:
            comment_lines = [
                "### \u26a0\ufe0f Code Comment needs improvement\n",
                f"**File:** `{item['file']}`\n**Line:** {item['line']}\n",
                f"**Comment:**\n> {item['comment']}\n",
                f"**Reason:** {item['reason']}",
            ]
            suggestion = item.get("suggestion", "")
            if suggestion:
                comment_lines.append(
                    f"\n\U0001f4a1 **Suggested Corrected Comment:**\n\n> {suggestion}"
                )
            blocks.append("\n".join(comment_lines).strip())

    joined_blocks = "\n\n---\n\n".join(blocks)
    return f"## PR Validation\n\n{joined_blocks}"


def _bold_labels(text: str) -> str:
    """
    Make target labels bold wherever they appear in the text (outside code blocks):
    - **Reason:**
    - **File:**
    - **Line:**
    - **Comment:**
    - **Issue:**
    - **Code:**
    - **Suggestion:**
    """
    if not text:
        return ""
    # Split by fenced code blocks so code lines inside ```...``` are never altered
    parts = re.split(r"(```[\s\S]*?```)", text)
    pattern = re.compile(
        r"(?m)^(\s*(?:💡\s*|\*\s*)?)(?!\*\*)(Issue|File|Line|Comment|Code|Reason|Suggestion):"
    )
    for i in range(0, len(parts), 2):
        parts[i] = pattern.sub(r"\1**\2:**", parts[i])
    return "".join(parts)


def _format_review(review_text: str) -> str:
    """
    Format the raw AI Code Review text for the final PR comment:
    1. Bold all labels (**Issue:**, **File:**, **Line:**, **Code:**, **Reason:**, **Suggestion:**).
    2. Make **Reason:** inline with its text (same line) — outside code blocks only.
    3. Insert '---' separators between individual **Issue:** blocks.
    """
    if not review_text:
        return ""

    bolded = _bold_labels(review_text)

    # ── Step 1: make **Reason:** inline (outside fenced code blocks) ──────────
    parts = re.split(r"(```[\s\S]*?```)", bolded)
    reason_inline = re.compile(r"(?m)^(\s*)\*\*Reason:\*\*[ \t]*\n+([^\n\s])")
    for i in range(0, len(parts), 2):
        parts[i] = reason_inline.sub(r"\1**Reason:** \2", parts[i])
    text = "".join(parts)

    # ── Step 2: insert --- between **Issue:** blocks (deduplicating existing) ──
    issue_pattern = re.compile(r"(?m)(?=^\*\*Issue:\*\*)")
    raw_blocks = [b.strip() for b in issue_pattern.split(text) if b.strip()]
    cleaned_blocks = []
    for b in raw_blocks:
        b_clean = re.sub(r"^\s*---\s*", "", b)
        b_clean = re.sub(r"\s*---\s*$", "", b_clean).strip()
        if b_clean:
            cleaned_blocks.append(b_clean)
    if len(cleaned_blocks) > 1:
        text = "\n\n---\n\n".join(cleaned_blocks)
    elif cleaned_blocks:
        text = cleaned_blocks[0]

    # Collapse any duplicate consecutive separators
    text = re.sub(r"(?m)^---\s*\n+(?:---\s*\n*)+", "---\n\n", text)

    return text.strip()


def _build_combined_comment(
    validation_section: str,
    review_text: str,
) -> str:
    """
    Combine the PR Validation section and the AI Code Review section
    into one final PR comment body.
    """
    formatted_review = _format_review(review_text) if review_text else ""
    parts = [
        validation_section,
        "",
        "---",
        "",
        "## AI Code Review",
        "",
        formatted_review,
    ]
    return "\n".join(parts)



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

    if event in ["pull_request", "pull_request_target"]:

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

            # ============================================================
            # PR VALIDATION — collects ALL warnings, never stops the pipeline
            # ============================================================

            # Warning state accumulators
            desc_warning: str | None = None      # set if description has a problem
            desc_reason: str | None = None       # human-readable reason
            desc_suggestion: str | None = None   # AI-generated improved description
            comment_warnings: list = []          # per-comment warning dicts

            # ── Step 1: PR Description text-quality validation ────────────
            print("\nExecuting PR Description Validation...")
            try:
                is_valid, validation_reason = validate_pr_description(
                    pr_title,
                    pr_body
                )

                if not is_valid:
                    print(
                        f"[VALIDATION WARNING] PR Description quality: {validation_reason}"
                    )
                    desc_warning = "PR Description needs improvement"
                    desc_reason = validation_reason
                else:
                    print("[VALIDATION] PR Description quality: PASS")

            except Exception as e:
                print(
                    f"[WARNING] PR Description validation error: {e}. "
                    "Continuing (fail-open)."
                )

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
                        "\u2139\ufe0f **AI Code Review Skipped**\n\n"
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

            # ── Source code extraction & validation ────────────────────────
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

            parsed_diff_map = parse_diff_by_file(diff)
            parsed_files_map = parse_full_code_files(full_code)

            # ── Step 2: PR Description vs Code Match (PR-level) ─────────
            # `diff` contains the combined diff returned for all changed files.
            # Validate once against the overall PR description; chunking is
            # handled inside the validator when the complete prompt is too large.
            print("\nExecuting PR-level Description-vs-Code Validation...")
            try:
                res = validate_description_matches_code(
                    pr_title=pr_title,
                    pr_body=pr_body,
                    diff=diff,
                )
                if isinstance(res, dict):
                    desc_match_valid = res.get("is_valid", False)
                    desc_match_reason = res.get("reason", "")
                elif isinstance(res, (tuple, list)) and len(res) >= 2:
                    desc_match_valid, desc_match_reason = bool(res[0]), str(res[1])
                else:
                    desc_match_valid = False
                    desc_match_reason = "ERROR: Validator returned an unrecognized result."

                if not desc_match_valid:
                    print(
                        f"[VALIDATION WARNING] Description vs Code: {desc_match_reason}"
                    )
                    if str(desc_match_reason).startswith("FAIL:"):
                        # A confirmed mismatch is a PR-description issue.
                        desc_warning = "PR Description needs improvement"
                    elif desc_warning is None:
                        # SKIPPED/ERROR is an incomplete check, not proof that
                        # the PR description itself is inaccurate.
                        desc_warning = "Description-vs-Code validation incomplete"

                    if desc_reason:
                        desc_reason = f"{desc_reason} Additionally, Description-vs-Code: {desc_match_reason}"
                    else:
                        desc_reason = f"Description-vs-Code: {desc_match_reason}"
                else:
                    print(f"[VALIDATION] Description vs Code: {desc_match_reason}")
            except Exception as e:
                print(
                    f"[WARNING] Description-vs-code validation error: {e}. "
                    "Continuing (fail-open)."
                )
                if desc_warning is None:
                    desc_warning = "Description-vs-Code validation incomplete"
                error_reason = f"ERROR: Description-vs-Code validation errored ({type(e).__name__})."
                if desc_reason:
                    desc_reason = f"{desc_reason} Additionally, {error_reason}"
                else:
                    desc_reason = error_reason

            # ── Step 3: Generate AI description suggestion (if needed) ────
            if desc_warning == "PR Description needs improvement":
                print("\nGenerating AI description suggestion...")
                try:
                    desc_suggestion = generate_description_suggestion(
                        pr_title,
                        pr_body,
                        diff
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Could not generate description suggestion: {e}."
                    )
                    desc_suggestion = (
                        "Please update the PR description to clearly explain "
                        "the purpose and main changes of this PR."
                    )

            # ── Step 4: Code Comment Accuracy Validation ──────────────────
            print("\nExecuting Code Comment Accuracy Validation...")
            comment_ref = (
                after_sha
                if action == "synchronize" and after_sha
                else None
            )
            try:
                raw_mismatches = validate_code_comments(
                    workspace_path,
                    changed_files,
                    source_branch,
                    comment_ref,
                    diff=diff
                )
            except Exception as e:
                print(
                    f"[WARNING] Comment validation error: {e}. "
                    "Continuing (fail-open)."
                )
                raw_mismatches = []

            # ── Step 5: Generate AI suggestion for every comment mismatch ─
            if raw_mismatches:
                print(
                    f"\n[VALIDATION WARNING] {len(raw_mismatches)} code comment "
                    "mismatch(es) found. Generating AI suggestions..."
                )
                for mismatch in raw_mismatches:
                    try:
                        suggestion = generate_comment_suggestion(
                            file=mismatch.get("file", ""),
                            line=mismatch.get("line", ""),
                            comment=mismatch.get("comment", ""),
                            code=mismatch.get("code", ""),
                            reason=mismatch.get("reason", ""),
                        )
                    except Exception as e:
                        print(
                            f"[WARNING] Could not generate comment suggestion: {e}."
                        )
                        suggestion = (
                            "Update the comment to accurately describe what the code does."
                        )
                    comment_warnings.append({**mismatch, "suggestion": suggestion})
            else:
                print("[VALIDATION] Code Comment Accuracy: PASS")

            # ── Validation summary ────────────────────────────────────────
            print("\n===== PR VALIDATION SUMMARY =====")
            if desc_warning:
                print(f"  [!] PR Description: {desc_reason}".encode("ascii", errors="replace").decode("ascii"))
            else:
                print("  [OK] PR Description: OK")
            if comment_warnings:
                print(f"  [!] Code Comments: {len(comment_warnings)} issue(s)")
            else:
                print("  [OK] Code Comments: OK")
            print("  -> Proceeding to AI Code Review regardless of warnings.")
            print("=================================\n")

            # Check overall prompt token budget; if the overall review prompt exceeds SAFE_INPUT_TOKEN_LIMIT
            # (e.g. giant PR or huge dataset exceeding 120,000 tokens), delegate to the parallel chunked review pipeline
            overall_ch_code, overall_ctx = extract_change_aware_context(
                changed_files=changed_files,
                diff=diff,
                full_code=full_code,
            )
            overall_prompt = build_review_prompt(
                code=overall_ctx,
                changed_files=changed_files,
                diff=diff,
                changed_code=overall_ch_code,
            )
            overall_tokens = count_tokens(overall_prompt)

            if overall_tokens > SAFE_INPUT_TOKEN_LIMIT:
                print(
                    f"\n[LARGE PROMPT DETECTED] Overall prompt input tokens ({overall_tokens}) "
                    f"exceeds safe input budget ({SAFE_INPUT_TOKEN_LIMIT}). "
                    "Initiating parallel chunked AI review pipeline..."
                )
                try:
                    review = execute_chunked_review(
                        full_code=full_code,
                        changed_files=changed_files,
                        diff=diff,
                        safe_token_limit=SAFE_INPUT_TOKEN_LIMIT
                    )
                except Exception as e:
                    # Still post a comment that discloses the failure instead of aborting.
                    print(f"[REVIEW ERROR] Chunked review failed for the whole PR: {e}")
                    review = ReviewResult(
                        f"Error occurred during AI code review: {e}", status="failed"
                    )

                # Always pass the overall chunked result through aggregation.
                # This catches ambiguous nonempty commentary even when the
                # chunk manager labels the result "completed", while preserving
                # valid findings and any notices already generated by chunks.
                _review_text = str(review).strip()
                _review_status = getattr(review, "status", None)

                review = aggregate_reviews([{
                    "file": "Overall PR review",
                    "status": _review_status or "completed",
                    "review": _review_text,
                }])
            else:
                # ============================================================
                # AI CODE REVIEW — Sequential File-by-File with In-Memory Storage
                # ============================================================
                print(
                    "Starting Local AI LLM Processing Engine (Sequential File-by-File)..."
                )

                file_review_results: list[dict] = []

                for file_path in changed_files:
                    norm_path = file_path.strip().replace("\\", "/").lstrip("./")

                    diff_info = parsed_diff_map.get(norm_path)
                    if not diff_info:
                        for p, dinfo in parsed_diff_map.items():
                            if p.endswith(norm_path) or norm_path.endswith(p):
                                diff_info = dinfo
                                break
                    file_diff = diff_info["full_diff"] if diff_info else diff

                    file_info = parsed_files_map.get(norm_path)
                    if not file_info:
                        for p, finfo in parsed_files_map.items():
                            if p.endswith(norm_path) or norm_path.endswith(p):
                                file_info = finfo
                                break

                    lang = file_info["lang"] if file_info else "text"
                    lines_dict = file_info["lines_dict"] if file_info else {}
                    total_lines = file_info["total_lines"] if file_info else 0

                    ch_code_text, rel_ctx_text, is_guaranteed = extract_file_focused_review_context(
                        file_path=file_path,
                        file_diff=file_diff,
                        lines_dict=lines_dict,
                        total_lines=total_lines,
                        lang=lang,
                    )

                    file_prompt = build_review_prompt(
                        code=rel_ctx_text,
                        changed_files=[file_path],
                        diff=file_diff,
                        changed_code=ch_code_text,
                    )

                    file_token_count = count_tokens(file_prompt)

                    if file_token_count <= SAFE_INPUT_TOKEN_LIMIT:
                        print(f"\n===== FINAL PROMPT SENT TO LLM: {file_path} =====")
                        print(file_prompt)
                        print(f"Total prompt input tokens: {file_token_count} (<= {SAFE_INPUT_TOKEN_LIMIT} budget)")
                        print("=================================================\n")

                        try:
                            review_output = review_code(file_prompt)
                            if isinstance(review_output, tuple):
                                status, review_text = review_output
                            elif hasattr(review_output, "status"):
                                status = review_output.status
                                review_text = str(review_output)
                            else:
                                review_text = str(review_output) if review_output is not None else ""
                                if not review_text or review_text == "No review generated.":
                                    status = "incomplete"
                                elif "Error occurred" in review_text or "inference engine failure" in review_text:
                                    status = "failed"
                                else:
                                    status = "completed"
                        except Exception as e:
                            print(f"[REVIEW ERROR] Exception during review for {file_path}: {e}")
                            status = "failed"
                            review_text = f"Error occurred during AI code review: {e}"
                    else:
                        print(
                            f"\n[LARGE PROMPT DETECTED] Prompt input tokens ({file_token_count}) "
                            f"for {file_path} exceeds safe input budget ({SAFE_INPUT_TOKEN_LIMIT}). "
                            "Initiating chunked AI review pipeline..."
                        )
                        file_raw_code = f"FILE: {file_path}\n```{lang}\n" + "\n".join(lines_dict.get(i, "") for i in range(1, total_lines + 1)) + "\n```"
                        try:
                            review_output = execute_chunked_review(
                                full_code=file_raw_code,
                                changed_files=[file_path],
                                diff=file_diff,
                                safe_token_limit=SAFE_INPUT_TOKEN_LIMIT,
                            )
                        except Exception as e:
                            # Record the failure so it surfaces in Review Notices and
                            # findings from the other files are still reported.
                            print(f"[REVIEW ERROR] Chunked review failed for {file_path}: {e}")
                            review_output = (
                                "failed",
                                f"Error occurred during AI code review: {e}",
                            )
                        if isinstance(review_output, tuple):
                            status, review_text = review_output
                        elif hasattr(review_output, "status"):
                            status = review_output.status
                            review_text = str(review_output)
                        else:
                            review_text = str(review_output) if review_output is not None else ""
                            if not review_text or review_text == "No review generated.":
                                status = "incomplete"
                            elif "Error occurred" in review_text or "inference engine failure" in review_text:
                                status = "failed"
                            elif "Review Notices" in review_text or "could not be reviewed completely" in review_text:
                                status = "failed" if "(failed)" in review_text else "incomplete"
                            else:
                                status = "completed"

                    # Store each result in Python memory
                    file_review_results.append({
                        "file": file_path,
                        "status": status,
                        "review": review_text,
                    })

                # ── Aggregate all file reviews in memory ──────────────────────
                review = aggregate_reviews(file_review_results)

            # ── Build and post one combined PR comment ────────────────────
            validation_section = _build_validation_section(
                desc_warning=desc_warning,
                desc_reason=desc_reason,
                desc_suggestion=desc_suggestion,
                comment_warnings=comment_warnings,
            )

            combined_comment = _build_combined_comment(
                validation_section=validation_section,
                review_text=review,
            )

            print(
                "\nPosting Combined Comment To GitHub..."
            )

            try:

                post_pr_comment(
                    repo_name,
                    pr_number,
                    combined_comment
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

            # Validation warnings do NOT count as failures — reviewers always notified.
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