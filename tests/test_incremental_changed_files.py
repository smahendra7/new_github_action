"""
tests/test_incremental_changed_files.py

Comprehensive tests verifying robust incremental changed-file detection and recovery:
1. Missing before SHA attempts recovery/fallback instead of returning an empty list.
2. Fetch and recovery in a real temporary git repo where before_sha is missing in a shallow clone.
3. Normal incremental comparison when both commits exist locally.
4. Recovery failure when commit cannot be fetched and reliable fallback metadata is unavailable.
5. Reliable fallback via GitHub API metadata with pagination support and automation file filtering.
6. Preservation of file eligibility rules, validation flow, and genuine skip on no reviewable files.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from git import Repo

from github.changed_files import (
    ChangedFilesDetectionError,
    _attempt_git_fetch_recovery,
    _fetch_github_metadata_changed_files,
    _has_commit,
    filter_automation_files,
    get_incremental_changed_files,
    is_automation_file,
)
from github.diff_extractor import get_incremental_diff


class TestIncrementalChangedFiles(unittest.TestCase):

    def setUp(self):
        self.temp_dirs = []

    def tearDown(self):
        for d in self.temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

    def _make_temp_dir(self):
        d = tempfile.mkdtemp()
        self.temp_dirs.append(d)
        return d

    def _init_git_repo(self, path):
        repo = Repo.init(path)
        with repo.config_writer() as config:
            config.set_value("user", "name", "Test Runner")
            config.set_value("user", "email", "test@example.com")
        return repo

    # -------------------------------------------------------------------------
    # Test 1: Missing before SHA
    # -------------------------------------------------------------------------
    def test_missing_before_sha_attempts_recovery_or_fallback(self):
        """Simulate missing before commit that causes Git to fail.

        Verify that the implementation attempts recovery or a safe fallback
        rather than returning an empty changed-file list as though comparison succeeded.
        """
        repo_dir = self._make_temp_dir()
        repo = self._init_git_repo(repo_dir)

        # Create one initial commit (after_sha)
        file_path = os.path.join(repo_dir, "app.py")
        with open(file_path, "w") as f:
            f.write("print('hello')\n")
        repo.index.add(["app.py"])
        after_commit = repo.index.commit("Initial commit")

        missing_before_sha = "000000000000000000000000000000000000dead"

        # Verify before_sha is truly missing locally
        self.assertFalse(_has_commit(repo, missing_before_sha))

        # Mock fallback to return a file list so recovery succeeds through fallback
        with patch(
            "github.changed_files._fetch_github_metadata_changed_files",
            return_value=["app.py"]
        ) as mock_fallback, patch(
            "github.changed_files._attempt_git_fetch_recovery"
        ) as mock_fetch:

            result = get_incremental_changed_files(
                repo_path=repo_dir,
                before_sha=missing_before_sha,
                after_sha=after_commit.hexsha,
                repo_name="org/repo",
                pr_number=10
            )

            # Recovery/fetch must be attempted
            mock_fetch.assert_called_once()
            # Fallback must be attempted when commit is still missing
            mock_fallback.assert_called_once()
            # Must NOT return empty list
            self.assertEqual(result, ["app.py"])

    # -------------------------------------------------------------------------
    # Test 2: Fetch and recovery in real Git repository
    # -------------------------------------------------------------------------
    def test_fetch_and_recovery_from_remote(self):
        """Create remote repo with commits, clone with depth=1 so before_sha is missing.

        Verify implementation fetches required history, retries diff, and returns
        correct changed files.
        """
        remote_dir = self._make_temp_dir()
        remote_repo = self._init_git_repo(remote_dir)

        # Commit 1 (before_sha)
        f1 = os.path.join(remote_dir, "service.py")
        with open(f1, "w") as f:
            f.write("def run():\n    return 1\n")
        remote_repo.index.add(["service.py"])
        c1 = remote_repo.index.commit("Commit 1")

        # Add 10 filler commits so that depth=1 clone won't include c1
        for i in range(10):
            with open(f1, "w") as f:
                f.write(f"def run():\n    return {i + 10}\n")
            remote_repo.index.add(["service.py"])
            remote_repo.index.commit(f"Filler commit {i}")

        # Commit after filler (after_sha)
        f2 = os.path.join(remote_dir, "handler.py")
        with open(f2, "w") as f:
            f.write("def handle():\n    pass\n")
        remote_repo.index.add(["handler.py"])
        c3 = remote_repo.index.commit("Commit after filler")

        # Shallow clone with --no-local forces Git's network transport which honours --depth,
        # ensuring that c1 (the first commit, 11 steps back) is genuinely absent in the clone.
        local_dir = self._make_temp_dir()
        local_repo = Repo.clone_from(remote_dir, local_dir, depth=1, no_local=True)

        # Confirm c1 (before_sha) is NOT in local_repo (it is 11 commits behind depth=1)
        self.assertFalse(_has_commit(local_repo, c1.hexsha))
        self.assertTrue(_has_commit(local_repo, c3.hexsha))

        # Run get_incremental_changed_files — should recover via fetch and return both files
        changed = get_incremental_changed_files(
            repo_path=local_dir,
            before_sha=c1.hexsha,
            after_sha=c3.hexsha
        )

        # Recovery must find service.py (modified in c1→c3) and handler.py (added before c3)
        self.assertIn("service.py", changed)
        self.assertIn("handler.py", changed)

        # Also verify get_incremental_diff succeeds after history is recovered
        diff_text = get_incremental_diff(
            repo_path=local_dir,
            before_sha=c1.hexsha,
            after_sha=c3.hexsha,
            changed_files=changed
        )
        self.assertIn("diff --git a/service.py", diff_text)
        self.assertIn("diff --git a/handler.py", diff_text)

    # -------------------------------------------------------------------------
    # Test 3: Normal incremental comparison when both commits are available
    # -------------------------------------------------------------------------
    def test_normal_incremental_comparison(self):
        """When both commits are present locally, normal comparison identifies changed files."""
        repo_dir = self._make_temp_dir()
        repo = self._init_git_repo(repo_dir)

        # Commit 1
        p1 = os.path.join(repo_dir, "calculator.py")
        with open(p1, "w") as f:
            f.write("def add(a, b): return a + b\n")
        repo.index.add(["calculator.py"])
        c1 = repo.index.commit("Add calculator")

        # Commit 2
        with open(p1, "w") as f:
            f.write("def add(a, b): return a + b + 0\n")
        p2 = os.path.join(repo_dir, "README.md")
        with open(p2, "w") as f:
            f.write("# Calculator\n")
        repo.index.add(["calculator.py", "README.md"])
        c2 = repo.index.commit("Update calculator and readme")

        # Both commits exist
        self.assertTrue(_has_commit(repo, c1.hexsha))
        self.assertTrue(_has_commit(repo, c2.hexsha))

        with patch("github.changed_files._attempt_git_fetch_recovery") as mock_fetch:
            changed = get_incremental_changed_files(
                repo_path=repo_dir,
                before_sha=c1.hexsha,
                after_sha=c2.hexsha
            )
            # Recovery should not be called when both commits are present
            mock_fetch.assert_not_called()
            self.assertIn("calculator.py", changed)
            self.assertIn("README.md", changed)

    # -------------------------------------------------------------------------
    # Test 4: Recovery failure reports clear error, does not return empty list
    # -------------------------------------------------------------------------
    def test_recovery_failure_raises_error(self):
        """When before commit cannot be fetched and fallback metadata is unavailable,

        changed-file detection reports a clear failure and does not return an empty list.
        """
        repo_dir = self._make_temp_dir()
        repo = self._init_git_repo(repo_dir)

        f1 = os.path.join(repo_dir, "app.py")
        with open(f1, "w") as f:
            f.write("x = 1\n")
        repo.index.add(["app.py"])
        after_commit = repo.index.commit("Commit after")

        bogus_before_sha = "ffffffffffffffffffffffffffffffffffffffff"

        # Ensure fallback returns None (unavailable)
        with patch(
            "github.changed_files._fetch_github_metadata_changed_files",
            return_value=None
        ):
            # Must raise ChangedFilesDetectionError when raise_on_error=True
            with self.assertRaises(ChangedFilesDetectionError):
                get_incremental_changed_files(
                    repo_path=repo_dir,
                    before_sha=bogus_before_sha,
                    after_sha=after_commit.hexsha,
                    raise_on_error=True
                )

            # With raise_on_error=False, returns None, NOT empty list []
            res = get_incremental_changed_files(
                repo_path=repo_dir,
                before_sha=bogus_before_sha,
                after_sha=after_commit.hexsha,
                raise_on_error=False
            )
            self.assertIsNone(res)
            self.assertNotEqual(res, [])

    # -------------------------------------------------------------------------
    # Test 5: Reliable GitHub metadata fallback with pagination
    # -------------------------------------------------------------------------
    def test_reliable_fallback_with_pagination(self):
        """Verify GitHub metadata fallback queries PR files API with pagination and filters automation files."""
        # 120 files across 2 pages (page 1: 100, page 2: 20)
        page1_files = [{"filename": f"src/module_{i}.py"} for i in range(100)]
        page2_files = [
            {"filename": f"src/module_{i}.py"} for i in range(100, 119)
        ] + [
            {"filename": ".github/workflows/ci.yml"}  # Automation file should be filtered
        ]

        def mock_requests_get(url, headers=None, timeout=None):
            resp = MagicMock()
            resp.status_code = 200
            # Use endswith to avoid "page=1" matching "page=10", "page=11", etc.
            if url.endswith("page=1"):
                resp.json.return_value = page1_files
            elif url.endswith("page=2"):
                resp.json.return_value = page2_files
            else:
                resp.json.return_value = []
            return resp

        with patch("requests.get", side_effect=mock_requests_get), \
             patch("github.github_auth.get_github_token", return_value="fake_token"):

            result = _fetch_github_metadata_changed_files(
                repo_name="boat/app",
                pr_number=99
            )

            self.assertIsNotNone(result)
            self.assertEqual(len(result), 119)
            self.assertIn("src/module_0.py", result)
            self.assertIn("src/module_118.py", result)
            # Automation file must be excluded
            self.assertNotIn(".github/workflows/ci.yml", result)

    # -------------------------------------------------------------------------
    # Test 6: No unrelated behavior changes
    # -------------------------------------------------------------------------
    def test_no_unrelated_behavior_changes(self):
        """Verify file eligibility rules and workflow error handling behavior."""
        # 1. Automation file exclusion rules
        self.assertTrue(is_automation_file(".github/workflows/run.yml"))
        self.assertTrue(is_automation_file("ai/reviewer.py"))
        self.assertTrue(is_automation_file("github/changed_files.py"))
        self.assertTrue(is_automation_file("reports/summary.py"))
        self.assertTrue(is_automation_file("tests/test_something.py"))
        self.assertTrue(is_automation_file("main.py"))
        self.assertTrue(is_automation_file("run_action.py"))
        self.assertTrue(is_automation_file("requirements.txt"))
        self.assertFalse(is_automation_file("app/models.py"))
        self.assertFalse(is_automation_file("src/index.ts"))

        # 2. Genuine empty application file changes (only automation changed) follows skip flow
        # Stub out heavyweight optional dependencies that are not installed in the test environment
        # so that `import main` does not raise ModuleNotFoundError.
        import sys
        import types
        for _mod_name in ("openai", "groq"):
            if _mod_name not in sys.modules:
                sys.modules[_mod_name] = types.ModuleType(_mod_name)
        # Stub openai.OpenAI class
        if not hasattr(sys.modules["openai"], "OpenAI"):
            sys.modules["openai"].OpenAI = MagicMock()
        # Stub groq.Groq class
        if not hasattr(sys.modules["groq"], "Groq"):
            sys.modules["groq"].Groq = MagicMock()

        from main import process_pr_event

        payload = {
            "pull_request": {
                "number": 5,
                "title": "Update CI workflow",
                "body": "Fix CI script",
                "head": {"ref": "feature", "sha": "c2c2c2c2"},
                "base": {"ref": "main"}
            },
            "repository": {"full_name": "boat/app"},
            "action": "synchronize",
            "before": "c1c1c1c1",
            "after": "c2c2c2c2"
        }

        # Succeeded comparison returning only automation file -> filtered to []
        with patch(
            "main.validate_pr_description", return_value=(True, "OK")
        ), patch(
            "main.get_pr_details", return_value=("Update CI workflow", "Fix CI script")
        ), patch(
            "main.capture_original_reviewers"
        ), patch(
            "main.clone_repository", return_value="/tmp/dummy"
        ), patch(
            "main.get_incremental_changed_files", return_value=[]
        ), patch(
            "main.get_incremental_diff", return_value=""
        ), patch(
            "main.post_pr_comment"
        ) as mock_post_comment, patch(
            "main.handle_reviewers_for_validation_result"
        ), patch(
            "os.path.exists", return_value=False
        ):

            res = process_pr_event(
                payload=payload,
                event="pull_request",
                action="synchronize",
                before_sha="c1c1c1c1",
                after_sha="c2c2c2c2"
            )

            # Genuine empty application files results in successful skip
            self.assertEqual(res.get("status"), "success")
            self.assertEqual(res.get("message"), "no_application_files_to_review")
            mock_post_comment.assert_called_once()
            self.assertIn("AI Code Review Skipped", mock_post_comment.call_args[0][2])

        # 3. Detection failure (ChangedFilesDetectionError) marks review as failed
        with patch(
            "main.validate_pr_description", return_value=(True, "OK")
        ), patch(
            "main.get_pr_details", return_value=("Update CI workflow", "Fix CI script")
        ), patch(
            "main.capture_original_reviewers"
        ), patch(
            "main.clone_repository", return_value="/tmp/dummy"
        ), patch(
            "main.get_incremental_changed_files",
            side_effect=ChangedFilesDetectionError("Commit bad object")
        ), patch(
            "main.post_pr_comment"
        ) as mock_post_comment, patch(
            "main.handle_reviewers_for_validation_result"
        ), patch(
            "os.path.exists", return_value=False
        ):

            res = process_pr_event(
                payload=payload,
                event="pull_request",
                action="synchronize",
                before_sha="c1c1c1c1",
                after_sha="c2c2c2c2"
            )

            # Detection failure must return failed status
            self.assertEqual(res.get("status"), "failed")
            self.assertEqual(res.get("reason"), "changed_files_detection_failed")
            mock_post_comment.assert_called_once()
            self.assertIn("AI Code Review Failed", mock_post_comment.call_args[0][2])


if __name__ == "__main__":
    unittest.main()
