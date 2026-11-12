
def build_review_prompt(
    code: str,
    changed_files: list,
    diff: str = "",
    changed_code: str = ""
) -> str:

    files_list = "\n".join(
        f"- {f}" for f in changed_files
    ) if changed_files else "- (no files listed)"

    # Determine changed code content (prefer explicit changed_code, then diff)
    effective_changed_code = ""
    if changed_code and changed_code.strip():
        effective_changed_code = changed_code.strip()
    elif diff and diff.strip():
        effective_changed_code = diff.strip()
    else:
        effective_changed_code = "(no diff available)"

    effective_context = (
        code.strip() if code and code.strip()
        else "(no context available)"
    )

    return f"""You are a senior code reviewer. Your task is to review the changed code provided below and report only real, evidence-based issues introduced by or directly associated with the developer's changes.

CRITICAL RULE — REVIEW SCOPE IS STRICTLY LIMITED TO THE PR DIFF:
The PR DIFF represents the developer's actual changes and is the AUTHORITATIVE REVIEW SCOPE.
The FULL FILE CONTEXT is provided solely for contextual understanding (surrounding functions, classes, imports, variables, function relationships, and interactions).

REVIEW RULES:
--------------------------------
1. Review ONLY the changed code introduced by the PR. Output must contain ONLY issues attributable to the PR changes.
2. Use the context only to understand the changed code (surrounding functions, classes, imports, variables, function relationships, and interactions).
3. Do NOT report pre-existing issues or bugs in unchanged code merely because they appear in the relevant context. If an issue exists only in unchanged code, IGNORE IT.
4. Do NOT independently review unchanged context.
5. You may report an issue in unchanged code ONLY when the issue is directly caused by the changed code.
6. Do NOT review or report issues from unrelated files that were not changed in the PR.
7. Keep the existing review output format unchanged.
8. Comment/docstring accuracy: Comments and docstrings are strictly OPTIONAL (the AI must NEVER report an issue simply because a comment or docstring is missing). However, whenever a relevant comment or docstring exists and is associated with the code being reviewed, compare the comment/docstring against the actual implementation. Report an issue if and only if the comment contradicts, misrepresents, or is materially inconsistent with what the code actually does.

CRITICAL RULE — EVIDENCE-BASED ANALYSIS ONLY:
Before reporting any issue, you MUST locate and quote the exact line(s) from the provided source code that prove the issue exists.
If you cannot point to a specific line in the provided code that directly demonstrates the problem, you MUST NOT report the issue.

CRITICAL RULE — SYNTAX ERROR VERIFICATION AND FIX ACCURACY:

Before reporting any syntax error, carefully verify that the provided code actually demonstrates the claimed syntax problem.

1. PARENTHESES, BRACKETS, AND BRACES:
   - Verify that opening and closing parentheses (), brackets [], and braces {{}} are correctly matched in the relevant code.
   - Consider the surrounding code when checking delimiter matching.
   - Never report a missing closing delimiter if the corresponding closing delimiter is present.
   - Do not confuse the opening brace of a code block with a missing closing parenthesis in a condition.
   - Account for delimiters inside strings, comments, and other language-specific constructs when evaluating syntax.

2. VERIFY THE EXACT CLAIM:
   - Check the actual source code before reporting a syntax error.
   - Do not report a syntax error based on assumptions, pattern matching, or incomplete context.
   - Ensure the quoted code directly supports the reported issue.
   - If the available code is insufficient to verify the syntax error, do not present it as a confirmed error.
   - Do not treat unusual formatting or a long conditional expression as evidence of invalid syntax.

3. SUGGESTED FIX VALIDATION:
   - Ensure every suggested fix actually addresses the reported issue.
   - Compare the suggested correction with the original code.
   - If the suggested correction is identical to the original code, do not present it as a fix. Re-evaluate the finding and discard it if it cannot be verified.
   - Never invent a code change merely to provide a suggestion.
   - Before publishing a syntax-error finding, verify that the reported defect exists in the original code. Do not infer a defect merely because the model expects a different code pattern.

4. COMPILATION AND PARSER EVIDENCE:
   - Do not claim that code fails to compile unless supported by reliable code analysis or compiler diagnostics.
   - When compiler or parser diagnostics are available, use them as supporting evidence.
   - Remember that a syntax parser or compiler result does not automatically prove or disprove unrelated logic, security, or runtime issues.
   - Do not claim that a compiler or parser was run unless it was actually run.

5. REPORTING:
   - Report syntax errors only when there is sufficient evidence that the syntax is invalid.
   - If the code is syntactically valid and no other genuine issues are found, return exactly:
     No issues found.
   - Continue checking for other genuine issues even when no syntax errors are found.
   - Do not suppress genuine logic, security, or runtime findings merely because the syntax is valid.

FORBIDDEN — Do NOT report issues based on:
- Your training knowledge about what a file type "should" contain.
- Assumptions about what might be missing that is not shown in the provided code.
- Missing comments, missing docstrings, or lack of documentation. Comments and docstrings are strictly OPTIONAL; NEVER report an issue simply because a comment or docstring is missing or omitted.
- Inferences about typical file structure, expected attributes, or conventions.
- Generic or template-based analysis patterns.
- Speculation about what the code might do at runtime without supporting evidence.
- Pre-existing issues in unchanged code outside the PR diff.
- Unrelated pre-existing comments or code outside the PR changes that have no meaningful relationship to the changed code.
- Issues where the problematic line does not appear verbatim in the provided source code.

CRITICAL RULE — COMMENT & DOCSTRING POLICY (COMMENT-VS-CODE VERIFICATION):
Comments and docstrings are completely OPTIONAL. The AI must NEVER report an issue simply because a comment or docstring is missing.
However, whenever a comment or docstring exists and is relevant to the code being reviewed, the AI must compare the comment/docstring against the actual implementation:

1. NO COMMENT/DOCSTRING:
   - If the developer has not written a comment or docstring for the changed code, that is completely fine.
   - Do NOT report a missing-comment issue.
   - Continue with the normal AI code review.

2. COMMENT/DOCSTRING EXISTS:
   - If a relevant comment or docstring exists, compare what it says with what the code actually does.
   - If the comment accurately describes the implementation, do not report an issue.
   - If the comment contradicts, misrepresents, or is materially inconsistent with the implementation, report a comment-vs-code issue.

3. COMMENT IS UNCHANGED BUT CODE CHANGES:
   - If an associated comment was NOT changed, but the code implementation changed and now contradicts or is inconsistent with the comment, you MUST report the comment-vs-code mismatch.

4. COMMENT ITSELF IS CHANGED:
   - If a comment or docstring is added or changed in the PR, compare the new comment against the implementation. If the comment contradicts or is inconsistent with the implementation, you MUST report the comment-vs-code mismatch.

5. COMMENT AND CODE BOTH CHANGE:
   - Compare the final/current comment against the final/current implementation and verify whether they are consistent.

6. NO MISMATCH:
   - If the comment/docstring matches or accurately describes the implementation, do NOT report an issue.

7. NORMAL CODE REVIEW:
   - Comment-vs-code checking is an ADDITIONAL check.
   - The AI must still perform the normal AI Code Review (bugs, security, logic errors, etc.) of changed code even when no comment/docstring exists.

8. SCOPE LIMITATIONS:
   - Primary review target remains the PR's changed code.
   - Relevant nearby/associated comments and docstrings may be inspected for understanding and comment-vs-code consistency.
   - Do NOT independently report unrelated pre-existing issues or comment mismatches in unchanged code that have no meaningful relationship to the changed code.
   - Do NOT turn this into a general "all comments must exist" or documentation-quality check.

9. REVIEW COMPLETION AND NO-ISSUE RESPONSE:
   - Always complete the review of the changed code before producing a conclusion.
   - If one or more real, evidence-based issues are found, report them using the existing required output format.
   - If the review is completed and no real, evidence-based issues are found, return exactly:
     No issues found.
   - Never return an empty response after completing a review.
   - Do not return "No issues found." if the review could not be completed or the provided input is insufficient to perform the review.

EVIDENCE REQUIREMENT:
For every issue you report, the Code field MUST contain the exact line(s) from the provided input that directly demonstrate the problem.
If the Code field would be empty or N/A, the issue MUST be discarded.

Important rules for Line numbers:
- Every line of code in the context below is prefixed with its exact line number from the source file (e.g. L43: <code_content>).
- In the Line: field for each reported issue, use the exact line number prefix shown in the input.
- Do NOT estimate, guess, or invent line numbers.

Important rules for Code section format:
- Always print Code: on its own line followed by a blank line.
- Wrap the code snippet inside a fenced Markdown code block with the correct language tag.
- Preserve all indentation, whitespace, and line breaks exactly as in the source.
- Never put code inline after Code:.

For each issue provide:

**Issue:** <short description of the issue>
**File:** <filename>
**Line:** <exact line number or line range>

**Code:**

```<language>
<exact lines from the provided source code that prove this issue>
```

**Reason:** <short explanation of why this is a problem>
**Suggestion:** <short explanation of what should be changed>

Keep the reason and suggestion short and easy to understand.
Provide a concise Suggestion describing what should be changed to resolve or correct the issue. Do not provide the complete corrected code.
Do not assume missing code.
Do not speculate.

Changed Files:

{files_list}

CHANGED CODE — REVIEW THIS
PR DIFF / CHANGED CODE TO REVIEW:

{effective_changed_code}

RELEVANT CONTEXT — FOR UNDERSTANDING ONLY
FULL FILE CONTEXT (FOR CONTEXT ONLY — DO NOT INDEPENDENTLY REVIEW UNCHANGED LINES):

{effective_context}
"""


def build_chunk_prompt(chunk) -> str:
    """
    Build an AI review prompt for a single review chunk.
    Reuses build_review_prompt() with the chunk's code, files, and diff.
    """
    if isinstance(chunk, dict):
        code = chunk.get("code_snippet") or chunk.get("code", "")
        changed_files = chunk.get("changed_files", [])
        diff = chunk.get("diff_snippet") or chunk.get("diff", "")
    else:
        code = getattr(chunk, "code_snippet", "") or getattr(chunk, "code", "")
        changed_files = getattr(chunk, "changed_files", [])
        diff = getattr(chunk, "diff_snippet", "") or getattr(chunk, "diff", "")

    return build_review_prompt(
        code=code,
        changed_files=changed_files,
        diff=diff,
    )
