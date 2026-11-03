"""
Tests for automation file filtering in github/changed_files.py,
github/diff_extractor.py, and github/comment_extractor.py.

Verifies that:
  1. Automation infrastructure files are excluded:
       - .github/
       - ai/
       - github/
       - reports/
       - tests/
       - main.py
       - run_action.py
       - requirements*
       - Procfile
  2. Normal application files are never accidentally excluded.
  3. Mixed PR changed-file lists are filtered to retain only application files.
  4. Path format variants (Windows backslashes, leading ./) are handled cleanly.
  5. is_reviewable_file excludes automation files even with code extensions.
  6. filter_diff excludes automation files from full and incremental diffs.
  7. Automation-only PRs produce empty diffs.
  8. extract_added_code ignores automation file sections in diffs.
  9. build_review_prompt only lists and includes application files.
"""

import textwrap
from github.changed_files import is_automation_file, filter_automation_files
from github.comment_extractor import is_reviewable_file
from github.diff_extractor import filter_diff, extract_added_code
from ai.prompt_builder import build_review_prompt


class TestIsAutomationFile:

    def test_github_directory_files(self):
        assert is_automation_file(".github/workflows/ai-pr-automation.yml") is True
        assert is_automation_file(".github/actions/setup.yml") is True
        assert is_automation_file(".github\\workflows\\build.yml") is True

    def test_ai_directory_files(self):
        assert is_automation_file("ai/reviewer.py") is True
        assert is_automation_file("ai/comment_validator.py") is True
        assert is_automation_file("ai/prompt_builder.py") is True
        assert is_automation_file("ai/severity_classifier.py") is True
        assert is_automation_file("ai/description_validator.py") is True
        assert is_automation_file("ai/description_code_match_validator.py") is True
        assert is_automation_file("ai\\reviewer.py") is True

    def test_github_module_files(self):
        assert is_automation_file("github/comment_extractor.py") is True
        assert is_automation_file("github/pr_commenter.py") is True
        assert is_automation_file("github/changed_files.py") is True
        assert is_automation_file("github/github_auth.py") is True
        assert is_automation_file("github/diff_extractor.py") is True
        assert is_automation_file("github/reviewer_manager.py") is True
        assert is_automation_file("github/repository_manager.py") is True
        assert is_automation_file("github\\comment_extractor.py") is True

    def test_reports_directory_files(self):
        assert is_automation_file("reports/report_generator.py") is True
        assert is_automation_file("reports/__init__.py") is True
        assert is_automation_file("reports/files/report.docx") is True

    def test_tests_directory_files(self):
        assert is_automation_file("tests/test_automation_filter.py") is True
        assert is_automation_file("tests/test_comment_extractor.py") is True
        assert is_automation_file("tests/test_unicode_sanitization.py") is True
        assert is_automation_file("tests\\test_review.py") is True

    def test_root_automation_scripts(self):
        assert is_automation_file("main.py") is True
        assert is_automation_file("./main.py") is True
        assert is_automation_file("run_action.py") is True
        assert is_automation_file("./run_action.py") is True
        assert is_automation_file("Procfile") is True
        assert is_automation_file("procfile") is True

    def test_requirements_files(self):
        assert is_automation_file("requirements.txt") is True
        assert is_automation_file("requirements-dev.txt") is True
        assert is_automation_file("requirements.in") is True
        assert is_automation_file("requirements/base.txt") is True

    def test_normal_application_files_not_excluded(self):
        assert is_automation_file("app/src/MainActivity.kt") is False
        assert is_automation_file("app/src/UserRepository.kt") is False
        assert is_automation_file("app/src/LoginActivity.kt") is False
        assert is_automation_file("core/common/NetworkClient.kt") is False
        assert is_automation_file("feature/auth/LoginViewModel.kt") is False
        assert is_automation_file("feature/home/HomeScreen.kt") is False
        assert is_automation_file("build.gradle.kts") is False
        assert is_automation_file("settings.gradle.kts") is False
        assert is_automation_file("gradle.properties") is False
        assert is_automation_file("app/src/main.py") is False

    def test_empty_or_none_returns_false(self):
        assert is_automation_file("") is False
        assert is_automation_file(None) is False


class TestFilterAutomationFiles:

    def test_mixed_files_only_retains_application_files(self):
        changed = [
            "ai/reviewer.py",
            "github/pr_commenter.py",
            "tests/test_automation_filter.py",
            "app/src/MainActivity.kt",
            "app/src/LoginActivity.kt",
        ]
        result = filter_automation_files(changed)
        assert result == [
            "app/src/MainActivity.kt",
            "app/src/LoginActivity.kt",
        ]

    def test_only_automation_files_returns_empty(self):
        changed = [
            "ai/reviewer.py",
            "github/comment_extractor.py",
            "main.py",
            "run_action.py",
            ".github/workflows/ai-pr-automation.yml",
            "requirements.txt",
            "Procfile",
            "tests/test_automation_filter.py",
        ]
        result = filter_automation_files(changed)
        assert result == []

    def test_only_application_files_retained_in_full(self):
        changed = [
            "app/src/MainActivity.kt",
            "core/NetworkClient.kt",
            "feature/HomeScreen.kt",
        ]
        result = filter_automation_files(changed)
        assert result == changed


class TestIsReviewableFileWithAutomationExclusion:

    def test_automation_files_not_reviewable_even_with_code_extensions(self):
        assert is_reviewable_file("ai/reviewer.py") is False
        assert is_reviewable_file("github/comment_extractor.py") is False
        assert is_reviewable_file("main.py") is False
        assert is_reviewable_file("run_action.py") is False
        assert is_reviewable_file("tests/test_automation_filter.py") is False

    def test_application_code_files_are_reviewable(self):
        assert is_reviewable_file("app/src/MainActivity.kt") is True
        assert is_reviewable_file("app/src/User.java") is True
        assert is_reviewable_file("core/common/Util.kt") is True


class TestFilterDiff:

    SAMPLE_MIXED_DIFF = textwrap.dedent("""\
        diff --git a/ai/reviewer.py b/ai/reviewer.py
        index 1111111..2222222 100644
        --- a/ai/reviewer.py
        +++ b/ai/reviewer.py
        @@ -20,3 +20,4 @@
         def review_code(prompt: str) -> str:
        +    # new comment
             pass
        diff --git a/app/src/MainActivity.kt b/app/src/MainActivity.kt
        index 3333333..4444444 100644
        --- a/app/src/MainActivity.kt
        +++ b/app/src/MainActivity.kt
        @@ -10,3 +10,4 @@
         class MainActivity : ComponentActivity() {
        +    val newField = 42
         }
        diff --git a/main.py b/main.py
        index 5555555..6666666 100644
        --- a/main.py
        +++ b/main.py
        @@ -1,3 +1,4 @@
        +import sys
         import os
        diff --git a/tests/test_automation_filter.py b/tests/test_automation_filter.py
        index 7777777..8888888 100644
        --- a/tests/test_automation_filter.py
        +++ b/tests/test_automation_filter.py
        @@ -1,2 +1,3 @@
        +# new test
        """).strip()

    def test_filter_diff_removes_automation_and_keeps_app(self):
        filtered = filter_diff(self.SAMPLE_MIXED_DIFF)
        assert "ai/reviewer.py" not in filtered
        assert "main.py" not in filtered
        assert "tests/test_automation_filter.py" not in filtered
        assert "app/src/MainActivity.kt" in filtered
        assert "val newField = 42" in filtered

    def test_filter_diff_automation_only_returns_empty(self):
        automation_diff = textwrap.dedent("""\
            diff --git a/ai/reviewer.py b/ai/reviewer.py
            index 1111111..2222222 100644
            --- a/ai/reviewer.py
            +++ b/ai/reviewer.py
            @@ -1,2 +1,3 @@
            +# change
            diff --git a/github/changed_files.py b/github/changed_files.py
            index 3333333..4444444 100644
            --- a/github/changed_files.py
            +++ b/github/changed_files.py
            @@ -1,2 +1,3 @@
            +# change
            """).strip()
        filtered = filter_diff(automation_diff)
        assert filtered == ""

    def test_filter_diff_application_only_preserved(self):
        app_diff = textwrap.dedent("""\
            diff --git a/app/src/MainActivity.kt b/app/src/MainActivity.kt
            index 3333333..4444444 100644
            --- a/app/src/MainActivity.kt
            +++ b/app/src/MainActivity.kt
            @@ -10,3 +10,4 @@
             class MainActivity {
            +    fun doSomething() {}
             }
            """).strip()
        filtered = filter_diff(app_diff)
        assert filtered == app_diff

    def test_filter_diff_empty_or_none(self):
        assert filter_diff("") == ""
        assert filter_diff(None) == ""


class TestExtractAddedCodeFiltering:

    def test_extract_added_code_ignores_automation_diffs(self):
        diff = textwrap.dedent("""\
            diff --git a/ai/reviewer.py b/ai/reviewer.py
            index 1111111..2222222 100644
            --- a/ai/reviewer.py
            +++ b/ai/reviewer.py
            @@ -10,2 +10,3 @@
             def foo():
            +    x = 1
            diff --git a/app/src/MainActivity.kt b/app/src/MainActivity.kt
            index 3333333..4444444 100644
            --- a/app/src/MainActivity.kt
            +++ b/app/src/MainActivity.kt
            @@ -15,2 +15,3 @@
             class MainActivity {
            +    val appCode = 99
            """).strip()

        extracted = extract_added_code(diff)
        assert "ai/reviewer.py" not in extracted
        assert "x = 1" not in extracted
        assert "app/src/MainActivity.kt" in extracted
        assert "val appCode = 99" in extracted


class TestPromptBuildingFiltering:

    def test_build_review_prompt_only_lists_application_files(self):
        code = "FILE: app/src/MainActivity.kt\n```kotlin\nL1: class MainActivity\n```"
        changed_files = ["app/src/MainActivity.kt"]
        prompt = build_review_prompt(code, changed_files)
        assert "- app/src/MainActivity.kt" in prompt
        assert "ai/reviewer.py" not in prompt
        assert "github/comment_extractor.py" not in prompt
        assert "main.py" not in prompt

    def test_build_review_prompt_includes_diff_and_full_file_context(self):
        code = "FILE: app/src/MainActivity.kt\n```kotlin\nL1: class MainActivity\nL2: val x = 1\n```"
        diff = "diff --git a/app/src/MainActivity.kt b/app/src/MainActivity.kt\n+val x = 1"
        changed_files = ["app/src/MainActivity.kt"]
        prompt = build_review_prompt(code, changed_files, diff)

        assert "PR DIFF / CHANGED CODE TO REVIEW:" in prompt
        assert "+val x = 1" in prompt
        assert "FULL FILE CONTEXT (FOR CONTEXT ONLY — DO NOT INDEPENDENTLY REVIEW UNCHANGED LINES):" in prompt
        assert "L1: class MainActivity" in prompt

    def test_build_review_prompt_contains_scope_rules(self):
        code = "FILE: app/src/MainActivity.kt\n```kotlin\nL1: class MainActivity\n```"
        diff = "diff --git a/app/src/MainActivity.kt b/app/src/MainActivity.kt\n+val x = 1"
        prompt = build_review_prompt(code, ["app/src/MainActivity.kt"], diff)

        assert "REVIEW SCOPE IS STRICTLY LIMITED TO THE PR DIFF" in prompt
        assert "AUTHORITATIVE REVIEW SCOPE" in prompt
        assert "FULL FILE CONTEXT is provided solely for contextual understanding" in prompt
        assert "Do NOT report pre-existing issues or bugs in unchanged code" in prompt

    def test_build_review_prompt_fallback_when_no_diff(self):
        code = "FILE: app/src/MainActivity.kt\n```kotlin\nL1: class MainActivity\n```"
        prompt = build_review_prompt(code, ["app/src/MainActivity.kt"])
        assert "(no diff available)" in prompt
        assert "FULL FILE CONTEXT" in prompt
