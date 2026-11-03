
import json
import hmac
import hashlib
import os
import re
import shutil

from fastapi import FastAPI, Request

from github.repository_manager import clone_repository

from github.diff_extractor import (
    get_diff,
    get_incremental_diff,
    extract_added_code,
    extract_full_code,
    count_file_sections_in_code,
)

from github.changed_files import (
    get_pr_changed_files,
    get_incremental_changed_files,
    filter_reviewable_text_files,
    ChangedFilesDetectionError,
)

from github.pr_commenter import (
    post_pr_comment,
    get_pr_details,
)

from github.reviewer_manager import (
    capture_original_reviewers,
    handle_reviewers_for_validation_result,
)

from ai.prompt_builder import build_review_prompt
from ai.reviewer import ReviewResult, review_code
from ai.description_validator import validate_pr_description
from ai.comment_validator import validate_code_comments
from ai.description_code_match_validator import (
    validate_description_matches_code,
)

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
    "",
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _normalize_repo_path(path: str) -> str:
    """
    Normalize a repository-relative path.

    Only removes explicit leading './' segments.
    Does not use suffix matching.
    """

    if not path:
        return ""

    normalized = str(path).strip().replace("\\", "/")

    while normalized.startswith("./"):
        normalized = normalized[2:]

    return normalized


def _get_exact_path_value(mapping: dict, file_path: str):
    """
    Find a file using exact normalized path matching.

    Never uses:
        endswith()
        filename-only matching
        partial path matching
    """

    target = _normalize_repo_path(file_path)

    if not target:
        return None

    for candidate_path, value in mapping.items():

        if _normalize_repo_path(candidate_path) == target:
            return value

    return None


def _normalize_changed_files(changed_files: list[str]) -> list[str]:
    """
    Normalize and deduplicate changed file paths while preserving order.
    """

    result = []
    seen = set()

    for file_path in changed_files or []:

        normalized = _normalize_repo_path(file_path)

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


# ---------------------------------------------------------------------------
# GitHub webhook signature verification
# ---------------------------------------------------------------------------

def verify_signature(
    payload_body: bytes,
    signature_header: str,
) -> bool:

    if not GITHUB_SECRET:
        print("[ERROR] GITHUB_WEBHOOK_SECRET is not configured.")
        return False

    if not signature_header:
        return False

    try:
        sha_name, signature = signature_header.split("=", 1)

        if sha_name != "sha256":
            return False

        mac = hmac.new(
            GITHUB_SECRET.encode(),
            msg=payload_body,
            digestmod=hashlib.sha256,
        )

        expected_signature = mac.hexdigest()

        return hmac.compare_digest(
            expected_signature,
            signature,
        )

    except Exception as error:
        print(f"Signature Error: {error}")
        return False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "AI Code Reviewer V2 Running",
    }


# ---------------------------------------------------------------------------
# Validation comment formatting
# ---------------------------------------------------------------------------

def _build_validation_section(
    desc_warning: str | None,
    desc_reason: str | None,
    desc_suggestion: str | None,
    comment_warnings: list,
) -> str:

    has_warnings = (
        bool(desc_warning)
        or bool(comment_warnings)
    )

    if not has_warnings:
        return (
            "## PR Validation\n\n"
            "✅ No validation issues found."
        )

    blocks = []

    # PR description validation.
    if desc_warning:

        desc_lines = [
            f"### ⚠️ {desc_warning}\n"
        ]

        if desc_reason:
            desc_lines.append(
                f"**Reason:** {desc_reason}\n"
            )

        if desc_suggestion:
            desc_lines.append(
                "💡 **Suggested PR Description:**\n\n"
                f"> {desc_suggestion}"
            )

        blocks.append(
            "\n".join(desc_lines).strip()
        )

    # Code comment validation.
    if comment_warnings:

        for item in comment_warnings:

            comment_lines = [
                "### ⚠️ Code Comment needs improvement\n",
                f"**File:** `{item.get('file', 'Unknown')}`\n"
                f"**Line:** {item.get('line', 'Unknown')}\n",
                f"**Comment:**\n> {item.get('comment', '')}\n",
                f"**Reason:** {item.get('reason', '')}",
            ]

            suggestion = item.get("suggestion", "")

            if suggestion:
                comment_lines.append(
                    "\n💡 **Suggested Corrected Comment:**\n\n"
                    f"> {suggestion}"
                )

            blocks.append(
                "\n".join(comment_lines).strip()
            )

    joined_blocks = "\n\n---\n\n".join(blocks)

    return (
        f"## PR Validation\n\n{joined_blocks}"
    )


# ---------------------------------------------------------------------------
# Review formatting
# ---------------------------------------------------------------------------

def _bold_labels(text: str) -> str:
    """Bold review labels outside fenced code blocks."""

    if not text:
        return ""

    parts = re.split(
        r"(```[\s\S]*?```)",
        text,
    )

    pattern = re.compile(
        r"(?m)^(\s*(?:💡\s*|\*\s*)?)(?!\*\*)"
        r"(Issue|File|Line|Comment|Code|Reason|Suggestion):"
    )

    for index in range(0, len(parts), 2):
        parts[index] = pattern.sub(
            r"\1**\2:**",
            parts[index],
        )

    return "".join(parts)


def _format_review(review_text: str) -> str:
    """Format AI review findings for the final PR comment."""

    if not review_text:
        return ""

    bolded = _bold_labels(review_text)

    # Make Reason labels inline with their content.
    parts = re.split(
        r"(```[\s\S]*?```)",
        bolded,
    )

    reason_inline = re.compile(
        r"(?m)^(\s*)\*\*Reason:\*\*[ \t]*\n+([^\n\s])"
    )

    for index in range(0, len(parts), 2):
        parts[index] = reason_inline.sub(
            r"\1**Reason:** \2",
            parts[index],
        )

    text = "".join(parts)

    # Add separators between actual Issue blocks.
    issue_pattern = re.compile(
        r"(?m)(?=^\*\*Issue:\*\*)"
    )

    raw_blocks = [
        block.strip()
        for block in issue_pattern.split(text)
        if block.strip()
    ]

    cleaned_blocks = []

    for block in raw_blocks:

        cleaned = re.sub(
            r"^\s*---\s*",
            "",
            block,
        )

        cleaned = re.sub(
            r"\s*---\s*$",
            "",
            cleaned,
        ).strip()

        if cleaned:
            cleaned_blocks.append(cleaned)

    if len(cleaned_blocks) > 1:
        text = "\n\n---\n\n".join(cleaned_blocks)

    elif cleaned_blocks:
        text = cleaned_blocks[0]

    text = re.sub(
        r"(?m)^---\s*\n+(?:---\s*\n*)+",
        "---\n\n",
        text,
    )

    return text.strip()


def _build_combined_comment(
    validation_section: str,
    review_text: str,
) -> str:

    formatted_review = (
        _format_review(review_text)
        if review_text
        else ""
    )

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


# ---------------------------------------------------------------------------
# AI review result helpers
# ---------------------------------------------------------------------------

def _get_review_result(
    review_output,
) -> tuple[str, str]:
    """
    Normalize review_code / execute_chunked_review results.

    Returns:
        (status, review_text)
    """

    if isinstance(review_output, tuple):

        if len(review_output) >= 2:
            return (
                str(review_output[0]),
                str(review_output[1] or ""),
            )

    if hasattr(review_output, "status"):

        return (
            str(review_output.status),
            str(review_output),
        )

    review_text = (
        str(review_output)
        if review_output is not None
        else ""
    )

    if (
        not review_text
        or review_text.strip() == "No review generated."
    ):
        return "incomplete", review_text

    if (
        "Error occurred during AI code review" in review_text
        or "inference engine failure" in review_text.lower()
    ):
        return "failed", review_text

    if (
        "Review Notices" in review_text
        or "could not be reviewed completely" in review_text
    ):
        status = (
            "failed"
            if "(failed)" in review_text.lower()
            else "incomplete"
        )

        return status, review_text

    return "completed", review_text


def _build_file_source_block(
    file_path: str,
    file_info: dict,
) -> str:
    """
    Build a source block for exactly one file.

    This block is used when an individual file needs chunked review.
    """

    lang = file_info.get("lang", "text")
    lines_dict = file_info.get("lines_dict", {})
    total_lines = file_info.get("total_lines", 0)

    source_lines = []

    for line_number in range(1, total_lines + 1):

        if line_number in lines_dict:
            source_lines.append(
                lines_dict[line_number]
            )

    if not source_lines:
        return ""

    return (
        f"FILE: {file_path}\n"
        f"```{lang}\n"
        f"{chr(10).join(source_lines)}\n"
        f"```"
    )


# ---------------------------------------------------------------------------
# Main PR event processing
# ---------------------------------------------------------------------------

def process_pr_event(
    payload: dict,
    event: str,
    action: str,
    before_sha: str,
    after_sha: str,
) -> dict:
    """
    Core PR event processing logic.

    Processing flow:

    1. Read PR details.
    2. Validate PR description.
    3. Clone repository.
    4. Detect exact changed files.
    5. Extract exact PR diff.
    6. Extract source code for exact changed files.
    7. Run description-vs-code validation.
    8. Run comment validation.
    9. Run AI code review.
    10. Aggregate results and post one PR comment.

    Each file's diff and source are matched by exact repository-relative path.
    """

    print(
        f"Event: {event} | Action: {action}"
    )

    print(
        f"Commit Window SHAs -> Before: {before_sha} | After: {after_sha}"
    )

    if event == "ping":

        print("GitHub App webhook verified")

        return {
            "status": "success",
            "message": "pong",
        }

    if event not in [
        "pull_request",
        "pull_request_target",
    ]:

        print(f"Skipping event: {event}")

        return {
            "status": "success",
            "message": "event_skipped",
        }

    pr = payload.get("pull_request", {})
    repo = payload.get("repository", {})

    repo_name = repo.get("full_name")
    pr_number = pr.get("number")

    source_branch = (
        pr.get("head", {}).get("ref")
    )

    target_branch = (
        pr.get("base", {}).get("ref")
    )

    # The PR head SHA is the preferred source version.
    pr_head_sha = (
        pr.get("head", {}).get("sha")
        or after_sha
    )

    print("\n----- PR DETAILS -----")

    print(
        f"Repository: {repo_name} | PR #{pr_number}"
    )

    print(
        f"Branch Target Route: [{target_branch}] <- [{source_branch}]"
    )

    if action not in [
        "opened",
        "synchronize",
        "reopened",
        "edited",
    ]:

        print(f"Skipping action: {action}")

        return {
            "status": "success",
            "message": "action_skipped",
        }

    if not repo_name or not pr_number:
        return {
            "status": "failed",
            "reason": "missing_repository_or_pr_number",
        }

    # -------------------------------------------------------
    # Get current PR title and description
    # -------------------------------------------------------

    payload_title = (
        pr.get("title") or ""
    ).strip()

    payload_body = (
        pr.get("body") or ""
    ).strip()

    try:
        live_title, live_body = get_pr_details(
            repo_name,
            pr_number,
        )

    except Exception as error:

        print(
            f"[WARNING] Could not retrieve live PR details: {error}"
        )

        live_title = ""
        live_body = ""

    def _richer_text(
        primary: str,
        fallback: str,
    ) -> str:

        primary = (primary or "").strip()
        fallback = (fallback or "").strip()

        if not primary:
            return fallback

        if not fallback:
            return primary

        return (
            primary
            if len(primary) >= len(fallback)
            else fallback
        )

    pr_title = _richer_text(
        live_title,
        payload_title,
    )

    pr_body = _richer_text(
        live_body,
        payload_body,
    )

    print(f"[DEBUG] Final PR title: {pr_title!r}")
    print(f"[DEBUG] Final PR body: {pr_body!r}")

    # -------------------------------------------------------
    # Capture original reviewers
    # -------------------------------------------------------

    if action in REVIEWER_MANAGEMENT_ACTIONS:

        try:
            capture_original_reviewers(
                repo_name,
                pr_number,
                pr,
            )

        except Exception as error:
            print(
                f"[WARNING] Could not capture original reviewers: {error}"
            )

    # -------------------------------------------------------
    # Validation state
    # -------------------------------------------------------

    desc_warning = None
    desc_reason = None
    desc_suggestion = None

    comment_warnings = []

    # -------------------------------------------------------
    # Step 1: PR description validation
    # -------------------------------------------------------

    print("\nExecuting PR Description Validation...")

    try:
        is_valid, validation_reason = validate_pr_description(
            pr_title,
            pr_body,
        )

        if not is_valid:

            print(
                f"[VALIDATION WARNING] PR Description: {validation_reason}"
            )

            desc_warning = "PR Description needs improvement"
            desc_reason = validation_reason

        else:
            print("[VALIDATION] PR Description quality: PASS")

    except Exception as error:

        print(
            f"[WARNING] PR Description validation error: {error}. "
            "Continuing."
        )

    # -------------------------------------------------------
    # Step 2: Clone repository
    # -------------------------------------------------------

    workspace_path = (
        f"workspace/pr_{pr_number}"
    )

    print("\nStarting Repository Clone...")

    try:
        workspace_path = clone_repository(
            repo_name,
            workspace_path,
        )

        print("Repository Clone Completed")

    except Exception as error:

        print(
            f"[ERROR] Repository clone failed: {error}"
        )

        try:
            post_pr_comment(
                repo_name,
                pr_number,
                "⚠️ **AI Code Review Failed**\n\n"
                f"Unable to clone the repository: {error}",
            )
        except Exception as comment_error:
            print(
                f"[WARNING] Could not post failure comment: {comment_error}"
            )

        return {
            "status": "failed",
            "reason": "repository_clone_failed",
            "error": str(error),
        }

    # -------------------------------------------------------
    # Repository processing
    # -------------------------------------------------------

    try:

        # ---------------------------------------------------
        # Step 3: Detect changed files and extract diff
        # ---------------------------------------------------

        print("\n===== CHANGED FILE DETECTION =====")

        if (
            action == "synchronize"
            and before_sha
            and after_sha
        ):

            print("Review Type: INCREMENTAL REVIEW")

            try:
                changed_files = get_incremental_changed_files(
                    workspace_path,
                    before_sha,
                    after_sha,
                    repo_name=repo_name,
                    pr_number=pr_number,
                    target_branch=target_branch,
                    source_branch=source_branch,
                )

                detected_file_count = len(changed_files or [])
                changed_files = filter_reviewable_text_files(
                    changed_files
                )
                print(
                    f"[REVIEWABLE FILES] Incremental diff detected "
                    f"{detected_file_count} path(s); "
                    f"{len(changed_files)} text file(s) will be reviewed."
                )

            except ChangedFilesDetectionError as error:

                print(
                    f"[ERROR] Incremental changed-file detection failed: {error}"
                )

                try:
                    post_pr_comment(
                        repo_name,
                        pr_number,
                        "⚠️ **AI Code Review Failed**\n\n"
                        f"Unable to determine changed files between "
                        f"`{before_sha[:7]}` and `{after_sha[:7]}`.\n\n"
                        f"Reason: {error}",
                    )
                except Exception as comment_error:
                    print(
                        f"[WARNING] Could not post failure comment: {comment_error}"
                    )

                return {
                    "status": "failed",
                    "reason": "changed_files_detection_failed",
                    "error": str(error),
                }

            diff = get_incremental_diff(
                workspace_path,
                before_sha,
                after_sha,
                changed_files,
                repo_name=repo_name,
                pr_number=pr_number,
                target_branch=target_branch,
                source_branch=source_branch,
            )

        else:

            changed_files = get_pr_changed_files(
                workspace_path,
                target_branch,
                source_branch,
            )

            detected_file_count = len(changed_files or [])
            changed_files = filter_reviewable_text_files(
                changed_files
            )
            print(
                f"[REVIEWABLE FILES] Full PR detected "
                f"{detected_file_count} path(s); "
                f"{len(changed_files)} text file(s) will be reviewed."
            )

            print("Review Type: FULL PR REVIEW")

            diff = get_diff(
                workspace_path,
                target_branch,
                source_branch,
                changed_files,
            )

        changed_files = _normalize_changed_files(
            changed_files
        )

        print("\n===== CHANGED FILES =====")

        for file_path in changed_files:
            print(file_path)

        print("=========================")

        print(
            f"Successfully detected {len(changed_files)} changed file(s)."
        )

        print("\n===== PR DIFF =====")
        print(diff)
        print("===================\n")

        # ---------------------------------------------------
        # Step 4: No changed files
        # ---------------------------------------------------

        if not changed_files:

            print(
                "[INFO] No reviewable application files were changed."
            )

            try:
                post_pr_comment(
                    repo_name,
                    pr_number,
                    "ℹ️ **AI Code Review Skipped**\n\n"
                    "No reviewable application files were modified "
                    "in this PR.",
                )

            except Exception as error:
                print(
                    f"[WARNING] Could not post GitHub comment: {error}"
                )

            if action in REVIEWER_MANAGEMENT_ACTIONS:

                handle_reviewers_for_validation_result(
                    repo_name,
                    pr_number,
                    validation_passed=True,
                )

            return {
                "status": "success",
                "message": "no_application_files_to_review",
            }

        if not diff or not diff.strip():

            raise RuntimeError(
                "Changed files were detected, but no PR diff was produced. "
                "AI review was stopped to prevent reviewing unrelated code."
            )

        # ---------------------------------------------------
        # Step 5: Extract full source code
        # ---------------------------------------------------

        code_source_ref = (
            pr_head_sha
            or source_branch
        )

        print(
            f"[SOURCE EXTRACTION] Source ref: {code_source_ref}"
        )

        full_code = extract_full_code(
            workspace_path,
            changed_files,
            code_source_ref,
        )

        if not full_code or not full_code.strip():

            print(
                "[WARNING] Full source extraction returned empty content. "
                "Trying exact added-code extraction from the PR diff."
            )

            full_code = extract_added_code(diff)

        if not full_code or not full_code.strip():

            raise RuntimeError(
                "Could not extract source code for the changed files. "
                "AI review was stopped."
            )

        # ---------------------------------------------------
        # Step 6: Validate source file sections
        # ---------------------------------------------------

        expected_file_sections = len(changed_files)

        actual_file_sections = count_file_sections_in_code(
            full_code
        )

        print(
            f"[SOURCE VALIDATION] Expected file sections: "
            f"{expected_file_sections}"
        )

        print(
            f"[SOURCE VALIDATION] Actual file sections: "
            f"{actual_file_sections}"
        )

        if actual_file_sections < expected_file_sections:
            print(
                "[SOURCE EXTRACTION WARNING] Some changed files could not "
                "be extracted. The review will continue with files whose "
                "source is available."
            )
        elif actual_file_sections > expected_file_sections:
            print(
                "[SOURCE EXTRACTION WARNING] More FILE sections were "
                "extracted than expected. Exact-path matching will be used "
                "before reviewing any file."
            )

        # ---------------------------------------------------
        # Step 7: Parse diff and source maps
        # ---------------------------------------------------

        parsed_diff_map = parse_diff_by_file(
            diff
        )

        parsed_files_map = parse_full_code_files(
            full_code
        )

        print(
            f"[DIFF MAP] Parsed {len(parsed_diff_map)} file(s)."
        )

        print(
            f"[SOURCE MAP] Parsed {len(parsed_files_map)} file(s)."
        )

        # Keep only files that have BOTH an exact diff and an exact source
        # block. Missing files are recorded and skipped individually; they
        # must never cause the entire PR review to stop.
        reviewable_files = []
        skipped_files = []
        reviewable_diff_blocks = []
        reviewable_diff_map = {}
        reviewable_source_map = {}

        for file_path in changed_files:

            diff_info = _get_exact_path_value(
                parsed_diff_map,
                file_path,
            )
            file_info = _get_exact_path_value(
                parsed_files_map,
                file_path,
            )

            if not diff_info:
                skipped_files.append(
                    (file_path, "exact file diff was not found")
                )
                print(
                    f"[REVIEW SKIPPED] {file_path}: exact file diff was not found."
                )
                continue

            if not file_info:
                skipped_files.append(
                    (file_path, "source file could not be extracted")
                )
                print(
                    f"[REVIEW SKIPPED] {file_path}: source file could not be extracted."
                )
                continue

            file_diff = diff_info.get("full_diff", "")
            if not file_diff.strip():
                skipped_files.append(
                    (file_path, "exact file diff is empty")
                )
                print(
                    f"[REVIEW SKIPPED] {file_path}: exact file diff is empty."
                )
                continue

            reviewable_files.append(file_path)
            reviewable_diff_map[file_path] = diff_info
            reviewable_source_map[file_path] = file_info
            reviewable_diff_blocks.append(file_diff)

        if skipped_files:
            print("\n===== FILES SKIPPED FROM AI REVIEW =====")
            for skipped_path, reason in skipped_files:
                print(f"  - {skipped_path}: {reason}")
            print(
                f"[SOURCE EXTRACTION WARNING] Skipped {len(skipped_files)} "
                f"file(s); continuing with {len(reviewable_files)} reviewable file(s)."
            )
            print("========================================\n")

        if not reviewable_files:
            raise RuntimeError(
                "No changed files had both an exact diff and extractable "
                "source code. There is no safe source context to send for AI review."
            )

        # From this point onward, every review operation receives only the
        # exact files for which both source and diff were successfully found.
        changed_files = reviewable_files
        parsed_diff_map = reviewable_diff_map
        parsed_files_map = reviewable_source_map
        diff = "\n\n".join(reviewable_diff_blocks)

        # ---------------------------------------------------
        # Step 8: Description vs code validation
        # ---------------------------------------------------

        print(
            "\nExecuting PR-level Description-vs-Code Validation..."
        )

        try:
            result = validate_description_matches_code(
                pr_title=pr_title,
                pr_body=pr_body,
                diff=diff,
            )

            if isinstance(result, dict):

                desc_match_valid = result.get(
                    "is_valid",
                    False,
                )

                desc_match_reason = result.get(
                    "reason",
                    "",
                )

            elif (
                isinstance(result, (tuple, list))
                and len(result) >= 2
            ):

                desc_match_valid = bool(result[0])
                desc_match_reason = str(result[1])

            else:

                desc_match_valid = False

                desc_match_reason = (
                    "ERROR: Validator returned an unrecognized result."
                )

            if not desc_match_valid:

                print(
                    f"[VALIDATION WARNING] Description vs Code: "
                    f"{desc_match_reason}"
                )

                if str(desc_match_reason).startswith("FAIL:"):

                    desc_warning = (
                        "PR Description needs improvement"
                    )

                elif desc_warning is None:

                    desc_warning = (
                        "Description-vs-Code validation incomplete"
                    )

                if desc_reason:
                    desc_reason = (
                        f"{desc_reason} Additionally, "
                        f"Description-vs-Code: {desc_match_reason}"
                    )
                else:
                    desc_reason = (
                        f"Description-vs-Code: {desc_match_reason}"
                    )

            else:

                print(
                    f"[VALIDATION] Description vs Code: "
                    f"{desc_match_reason}"
                )

        except Exception as error:

            print(
                f"[WARNING] Description-vs-code validation error: "
                f"{error}. Continuing."
            )

            if desc_warning is None:
                desc_warning = (
                    "Description-vs-Code validation incomplete"
                )

            error_reason = (
                "ERROR: Description-vs-Code validation errored "
                f"({type(error).__name__})."
            )

            if desc_reason:
                desc_reason = (
                    f"{desc_reason} Additionally, {error_reason}"
                )
            else:
                desc_reason = error_reason

        # ---------------------------------------------------
        # Step 9: Description suggestion
        # ---------------------------------------------------

        if desc_warning == "PR Description needs improvement":

            print(
                "\nGenerating AI description suggestion..."
            )

            try:
                desc_suggestion = generate_description_suggestion(
                    pr_title,
                    pr_body,
                    diff,
                )

            except Exception as error:

                print(
                    f"[WARNING] Could not generate description suggestion: "
                    f"{error}"
                )

                desc_suggestion = (
                    "Please update the PR description to clearly explain "
                    "the purpose and main changes of this PR."
                )

        # ---------------------------------------------------
        # Step 10: Code comment validation
        # ---------------------------------------------------

        print(
            "\nExecuting Code Comment Accuracy Validation..."
        )

        comment_ref = (
            pr_head_sha
            if pr_head_sha
            else source_branch
        )

        try:
            raw_mismatches = validate_code_comments(
                workspace_path,
                changed_files,
                source_branch,
                comment_ref,
                diff=diff,
            )

        except Exception as error:

            print(
                f"[WARNING] Comment validation error: {error}. "
                "Continuing."
            )

            raw_mismatches = []

        # ---------------------------------------------------
        # Step 11: Generate comment suggestions
        # ---------------------------------------------------

        if raw_mismatches:

            print(
                f"\n[VALIDATION WARNING] {len(raw_mismatches)} "
                "code comment mismatch(es) found."
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

                except Exception as error:

                    print(
                        f"[WARNING] Could not generate comment suggestion: "
                        f"{error}"
                    )

                    suggestion = (
                        "Update the comment to accurately describe "
                        "what the code does."
                    )

                comment_warnings.append(
                    {
                        **mismatch,
                        "suggestion": suggestion,
                    }
                )

        else:

            print(
                "[VALIDATION] Code Comment Accuracy: PASS"
            )

        # ---------------------------------------------------
        # Validation summary
        # ---------------------------------------------------

        print("\n===== PR VALIDATION SUMMARY =====")

        if desc_warning:
            print(
                f"  [!] PR Description: {desc_reason}"
            )
        else:
            print(
                "  [OK] PR Description: OK"
            )

        if comment_warnings:
            print(
                f"  [!] Code Comments: {len(comment_warnings)} issue(s)"
            )
        else:
            print(
                "  [OK] Code Comments: OK"
            )

        print(
            "  -> Proceeding to AI Code Review regardless of warnings."
        )

        print("=================================\n")

        # ---------------------------------------------------
        # Step 12: Calculate overall prompt size
        # ---------------------------------------------------

        overall_changed_code, overall_context = (
            extract_change_aware_context(
                changed_files=changed_files,
                diff=diff,
                full_code=full_code,
            )
        )

        overall_prompt = build_review_prompt(
            code=overall_context,
            changed_files=changed_files,
            diff=diff,
            changed_code=overall_changed_code,
        )

        overall_tokens = count_tokens(
            overall_prompt
        )

        print(
            f"[TOKEN CHECK] Overall prompt tokens: {overall_tokens}"
        )

        # ---------------------------------------------------
        # Step 13: Large PR review
        # ---------------------------------------------------

        if overall_tokens > SAFE_INPUT_TOKEN_LIMIT:

            print(
                "\n[LARGE PROMPT DETECTED] Overall prompt input tokens "
                f"({overall_tokens}) exceed safe input budget "
                f"({SAFE_INPUT_TOKEN_LIMIT})."
            )

            print(
                "Starting parallel chunked AI review pipeline..."
            )

            try:
                review_output = execute_chunked_review(
                    full_code=full_code,
                    changed_files=changed_files,
                    diff=diff,
                    safe_token_limit=SAFE_INPUT_TOKEN_LIMIT,
                )

                review_status, review_text = _get_review_result(
                    review_output
                )

            except Exception as error:

                print(
                    f"[REVIEW ERROR] Chunked review failed: {error}"
                )

                review_status = "failed"

                review_text = (
                    f"Error occurred during AI code review: {error}"
                )

            review = aggregate_reviews(
                [
                    {
                        "file": "Overall PR review",
                        "status": review_status,
                        "review": review_text,
                    }
                ]
            )

        # ---------------------------------------------------
        # Step 14: Normal sequential file-by-file review
        # ---------------------------------------------------

        else:

            print(
                "\nStarting Local AI LLM Processing Engine "
                "(Sequential File-by-File)..."
            )

            file_review_results = []

            for file_path in changed_files:

                print(
                    f"\n========== REVIEWING FILE: {file_path} =========="
                )

                # Exact diff lookup only.
                diff_info = _get_exact_path_value(
                    parsed_diff_map,
                    file_path,
                )

                if not diff_info:

                    print(
                        f"[REVIEW SKIPPED] Exact diff not found for "
                        f"{file_path}."
                    )

                    file_review_results.append(
                        {
                            "file": file_path,
                            "status": "incomplete",
                            "review": (
                                "Exact file diff was not found. "
                                "This file was not reviewed."
                            ),
                        }
                    )

                    continue

                # IMPORTANT:
                # Never fall back to the entire PR diff here.
                file_diff = diff_info.get("full_diff", "")

                if not file_diff:

                    print(
                        f"[REVIEW SKIPPED] Empty file diff for {file_path}."
                    )

                    file_review_results.append(
                        {
                            "file": file_path,
                            "status": "incomplete",
                            "review": (
                                "The exact file diff was empty. "
                                "This file was not reviewed."
                            ),
                        }
                    )

                    continue

                # Exact source lookup only.
                file_info = _get_exact_path_value(
                    parsed_files_map,
                    file_path,
                )

                if not file_info:

                    print(
                        f"[REVIEW SKIPPED] Exact source not found for "
                        f"{file_path}."
                    )

                    file_review_results.append(
                        {
                            "file": file_path,
                            "status": "incomplete",
                            "review": (
                                "Exact source code was not found. "
                                "This file was not reviewed."
                            ),
                        }
                    )

                    continue

                lang = file_info.get(
                    "lang",
                    "text",
                )

                lines_dict = file_info.get(
                    "lines_dict",
                    {},
                )

                total_lines = file_info.get(
                    "total_lines",
                    0,
                )

                # ---------------------------------------------------
                # Extract context for this exact file and exact diff
                # ---------------------------------------------------

                changed_code_text, relevant_context_text, is_guaranteed = (
                    extract_file_focused_review_context(
                        file_path=file_path,
                        file_diff=file_diff,
                        lines_dict=lines_dict,
                        total_lines=total_lines,
                        lang=lang,
                    )
                )

                if (
                    not relevant_context_text
                    or relevant_context_text.strip()
                    == "(no source context available)"
                ):

                    print(
                        f"[REVIEW SKIPPED] No source context for {file_path}."
                    )

                    file_review_results.append(
                        {
                            "file": file_path,
                            "status": "incomplete",
                            "review": (
                                "Source context could not be extracted. "
                                "This file was not reviewed."
                            ),
                        }
                    )

                    continue

                # ---------------------------------------------------
                # Build prompt for this file only
                # ---------------------------------------------------

                file_prompt = build_review_prompt(
                    code=relevant_context_text,
                    changed_files=[file_path],
                    diff=file_diff,
                    changed_code=changed_code_text,
                )

                file_token_count = count_tokens(
                    file_prompt
                )

                print(
                    f"[TOKEN CHECK] {file_path}: "
                    f"{file_token_count} tokens"
                )

                # ---------------------------------------------------
                # Normal review
                # ---------------------------------------------------

                if file_token_count <= SAFE_INPUT_TOKEN_LIMIT:

                    print(
                        f"\n===== FINAL PROMPT SENT TO LLM: "
                        f"{file_path} ====="
                    )

                    print(file_prompt)

                    print(
                        f"Total prompt input tokens: {file_token_count} "
                        f"(<= {SAFE_INPUT_TOKEN_LIMIT} budget)"
                    )

                    print(
                        "=================================================\n"
                    )

                    try:
                        review_output = review_code(
                            file_prompt
                        )

                        status, review_text = _get_review_result(
                            review_output
                        )

                    except Exception as error:

                        print(
                            f"[REVIEW ERROR] Exception during review "
                            f"for {file_path}: {error}"
                        )

                        status = "failed"

                        review_text = (
                            "Error occurred during AI code review: "
                            f"{error}"
                        )

                # ---------------------------------------------------
                # Large individual file review
                # ---------------------------------------------------

                else:

                    print(
                        f"\n[LARGE PROMPT DETECTED] Prompt input tokens "
                        f"({file_token_count}) for {file_path} exceed "
                        f"safe input budget ({SAFE_INPUT_TOKEN_LIMIT})."
                    )

                    print(
                        "Initiating chunked AI review for this file..."
                    )

                    # Build source block for this exact file only.
                    file_raw_code = _build_file_source_block(
                        file_path,
                        file_info,
                    )

                    if not file_raw_code:

                        status = "incomplete"

                        review_text = (
                            "Could not construct source code for "
                            "individual file chunking."
                        )

                    else:

                        try:
                            review_output = execute_chunked_review(
                                full_code=file_raw_code,
                                changed_files=[file_path],
                                diff=file_diff,
                                safe_token_limit=SAFE_INPUT_TOKEN_LIMIT,
                            )

                            status, review_text = _get_review_result(
                                review_output
                            )

                        except Exception as error:

                            print(
                                f"[REVIEW ERROR] Chunked review failed "
                                f"for {file_path}: {error}"
                            )

                            status = "failed"

                            review_text = (
                                "Error occurred during AI code review: "
                                f"{error}"
                            )

                # ---------------------------------------------------
                # Store this file's result
                # ---------------------------------------------------

                file_review_results.append(
                    {
                        "file": file_path,
                        "status": status,
                        "review": review_text,
                    }
                )

                print(
                    f"[FILE REVIEW RESULT] {file_path}: {status}"
                )

                print(
                    "===================================================="
                )

            # ---------------------------------------------------
            # Aggregate all file results
            # ---------------------------------------------------

            review = aggregate_reviews(
                file_review_results
            )

        # -------------------------------------------------------
        # Step 15: Build combined PR comment
        # -------------------------------------------------------

        validation_section = _build_validation_section(
            desc_warning=desc_warning,
            desc_reason=desc_reason,
            desc_suggestion=desc_suggestion,
            comment_warnings=comment_warnings,
        )

        if skipped_files:
            skipped_summary = [
                "\n\n### Files Not Reviewed (Source Unavailable)",
                "The following changed files could not be extracted and were "
                "skipped. The remaining files were still reviewed:",
            ]
            skipped_summary.extend(
                f"- `{file_path}` — {reason}"
                for file_path, reason in skipped_files
            )
            review = (review or "") + "\n".join(skipped_summary)

        combined_comment = _build_combined_comment(
            validation_section=validation_section,
            review_text=review,
        )

        print(
            "\n===== FINAL COMBINED PR COMMENT ====="
        )

        print(combined_comment)

        print(
            "====================================="
        )

        # -------------------------------------------------------
        # Step 16: Post PR comment
        # -------------------------------------------------------

        print(
            "\nPosting Combined Comment To GitHub..."
        )

        post_pr_comment(
            repo_name,
            pr_number,
            combined_comment,
        )

        print(
            "GitHub Comment Posted Successfully"
        )

        # -------------------------------------------------------
        # Step 17: Reviewer management
        # -------------------------------------------------------

        if action in REVIEWER_MANAGEMENT_ACTIONS:

            handle_reviewers_for_validation_result(
                repo_name,
                pr_number,
                validation_passed=True,
            )

        return {
            "status": "success",
            "message": "review_completed",
            "changed_files": changed_files,
            "skipped_files": [
                {"file": file_path, "reason": reason}
                for file_path, reason in skipped_files
            ],
        }

    except Exception as error:

        print(
            f"\n[PIPELINE ERROR] {type(error).__name__}: {error}"
        )

        try:
            post_pr_comment(
                repo_name,
                pr_number,
                "⚠️ **AI Code Review Failed**\n\n"
                f"The review pipeline could not complete safely.\n\n"
                f"Reason: `{type(error).__name__}: {error}`",
            )

        except Exception as comment_error:

            print(
                f"[WARNING] Could not post pipeline failure comment: "
                f"{comment_error}"
            )

        return {
            "status": "failed",
            "reason": "pipeline_error",
            "error": str(error),
        }

    finally:

        # Always clean up the cloned workspace.
        try:

            if workspace_path and os.path.exists(workspace_path):

                shutil.rmtree(workspace_path)

                print(
                    f"[CLEANUP] Workspace cleaned up: {workspace_path}"
                )

        except Exception as cleanup_error:

            print(
                f"[WARNING] Workspace cleanup failed: {cleanup_error}"
            )


# ---------------------------------------------------------------------------
# GitHub webhook endpoint
# ---------------------------------------------------------------------------

@app.post("/github/webhook")
async def github_webhook(
    request: Request,
):

    body = await request.body()

    signature = request.headers.get(
        "X-Hub-Signature-256"
    )

    signature_valid = verify_signature(
        body,
        signature,
    )

    print("\n================================================")
    print("Webhook Received")
    print("Signature Valid:", signature_valid)

    if not signature_valid:

        print(
            "Webhook Signature Failed"
        )

        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": "Invalid signature",
            },
        )

    try:
        payload = json.loads(
            body.decode("utf-8")
        )

    except (UnicodeDecodeError, json.JSONDecodeError) as error:

        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Invalid JSON payload: {error}",
            },
        )

    event = request.headers.get(
        "X-GitHub-Event"
    )

    action = payload.get("action")

    before_sha = payload.get("before")
    after_sha = payload.get("after")

    result = process_pr_event(
        payload,
        event,
        action,
        before_sha,
        after_sha,
    )

    return result
