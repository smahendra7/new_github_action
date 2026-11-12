"""
tests/test_comment_accuracy_classification.py

Focused tests for Code Comment Validation accuracy and classification:
Test 1: Accurate comment -> No finding
Test 2: Incorrect comment -> Finding is generated
Test 3: Outdated comment -> Finding is generated
Test 4: Alternative wording -> No finding
Test 5: Contradictory AI output (ACCURATE + suggestion) -> Final report generates no finding
Test 6: Genuine comment mismatch -> Finding is generated
"""

import unittest
from unittest.mock import MagicMock, patch

from ai.comment_validator import (
    _parse_mismatch_blocks,
    _is_accurate_comment,
    validate_code_comments,
    format_comment_validation_failure,
)
from main import _build_validation_section


class TestCommentAccuracyClassification(unittest.TestCase):

    def test_1_accurate_comment_produces_no_finding(self):
        """Test 1: Accurate comment matching implementation produces no finding."""
        llm_response = (
            "---\n"
            "File: ConnectivityDialogs.kt\n"
            "Line: 45\n"
            "Classification: ACCURATE\n"
            'Comment: "// Standard dialog prompting user to turn ON Wi-Fi before pairing or communicating with a camera."\n'
            "Code: fun EnableWifiDialog() { ... }\n"
            "Reason: Comment accurately describes the EnableWifiDialog function's purpose and behavior.\n"
            "---"
        )
        mismatches = _parse_mismatch_blocks(llm_response)
        self.assertEqual(mismatches, [])

        # Also verify NO_MISMATCHES response produces empty
        no_mismatch_resp = "NO_MISMATCHES"
        self.assertEqual(_parse_mismatch_blocks(no_mismatch_resp), [])

    def test_2_incorrect_comment_generates_finding(self):
        """Test 2: Incorrect comment contradicting implementation generates a finding."""
        llm_response = (
            "---\n"
            "File: math.py\n"
            "Line: 10\n"
            "Classification: MISMATCH\n"
            'Comment: "// Returns sum of a and b"\n'
            "Code: return a - b\n"
            "Reason: The comment states it returns the sum, but the implementation performs subtraction.\n"
            "---"
        )
        mismatches = _parse_mismatch_blocks(llm_response)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["classification"], "MISMATCH")
        self.assertEqual(mismatches[0]["file"], "math.py")
        self.assertEqual(mismatches[0]["line"], "10")
        self.assertIn("subtraction", mismatches[0]["reason"])

    def test_3_outdated_comment_generates_finding(self):
        """Test 3: Outdated comment describing removed behavior generates a finding."""
        llm_response = (
            "---\n"
            "File: cache.py\n"
            "Line: 15\n"
            "Classification: OUTDATED\n"
            'Comment: "// Writes to disk cache and memory cache"\n'
            "Code: memoryCache.put(key, value)\n"
            "Reason: Disk cache was removed; the function now only writes to memory cache.\n"
            "---"
        )
        mismatches = _parse_mismatch_blocks(llm_response)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["classification"], "OUTDATED")
        self.assertEqual(mismatches[0]["file"], "cache.py")
        self.assertIn("Disk cache was removed", mismatches[0]["reason"])

    def test_4_alternative_wording_produces_no_finding(self):
        """Test 4: Accurate comment that could be worded differently produces no finding."""
        llm_response = (
            "---\n"
            "File: user.py\n"
            "Line: 20\n"
            "Classification: ACCURATE\n"
            'Comment: "// Sets user status to active"\n'
            'Code: user.status = "ACTIVE"\n'
            "Reason: Comment accurately describes what the function does. Could be rephrased as 'Activates user'.\n"
            "---"
        )
        mismatches = _parse_mismatch_blocks(llm_response)
        self.assertEqual(mismatches, [])

    def test_5_contradictory_ai_output_suppresses_warning_in_report(self):
        """Test 5: Contradictory AI output (classification ACCURATE + non-empty suggestion) produces NO finding in report."""
        # 1. Parsing layer: parser discards block marked ACCURATE
        llm_response = (
            "---\n"
            "File: dialog.kt\n"
            "Line: 50\n"
            "Classification: ACCURATE\n"
            'Comment: "// Prompts user to turn on Bluetooth"\n'
            "Code: promptBluetoothEnable()\n"
            "Reason: Comment accurately describes Bluetooth prompt.\n"
            "---"
        )
        mismatches = _parse_mismatch_blocks(llm_response)
        self.assertEqual(mismatches, [])

        # 2. Final report layer: even if an item with ACCURATE and a non-empty suggestion reaches report builder,
        # it MUST NOT generate a warning in the final validation section.
        contradictory_warning = [
            {
                "file": "dialog.kt",
                "line": "50",
                "classification": "ACCURATE",
                "comment": "// Prompts user to turn on Bluetooth",
                "code": "promptBluetoothEnable()",
                "reason": "Comment accurately describes Bluetooth prompt.",
                "suggestion": "Show Bluetooth enable dialog.",
            }
        ]
        report_section = _build_validation_section(
            desc_warning=None,
            desc_reason=None,
            desc_suggestion=None,
            comment_warnings=contradictory_warning,
        )
        self.assertNotIn("⚠️ Code Comment needs improvement", report_section)
        self.assertIn("✅ No validation issues found.", report_section)

    def test_6_genuine_comment_mismatch_generates_finding(self):
        """Test 6: Genuine mismatch (comment claims unconditional cache return vs code returns null) generates finding."""
        llm_response = (
            "---\n"
            "File: UserRepository.kt\n"
            "Line: 35\n"
            "Classification: MISMATCH\n"
            'Comment: "// Returns the cached user unconditionally."\n'
            "Code: if (isCacheValid()) return cache.get() else return null\n"
            "Reason: The comment states the function returns the cached value unconditionally, while the implementation returns null when the cache is invalid.\n"
            "---"
        )
        mismatches = _parse_mismatch_blocks(llm_response)
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["classification"], "MISMATCH")
        self.assertEqual(mismatches[0]["file"], "UserRepository.kt")
        self.assertEqual(mismatches[0]["line"], "35")
        self.assertIn("returns null when the cache is invalid", mismatches[0]["reason"])


if __name__ == "__main__":
    unittest.main()
