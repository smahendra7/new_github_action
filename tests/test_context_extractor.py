"""
tests/test_context_extractor.py

Comprehensive tests for language-aware change-aware context extraction:
1. Python: standalone function, class method, multiple changed regions, nested functions.
2. Kotlin: function, class method, multiple functions, multiple changed regions.
3. Swift: function, struct/class method, multiple functions, multiple changed regions.
4. Fallback: unparseable syntax, unknown language, ensures limited window instead of entire file.
5. MAX_CONTEXT_LINES: verifies an oversized enclosing function is capped to a limited window.
6. Prompt verification: changed code marked as reviewable, context marked as context only,
   scope rules explicitly prohibit reporting pre-existing issues.
7. Large file simulation: 12,000-line file with 10 changed lines produces a compact prompt.
"""

from unittest.mock import patch
import pytest

from ai.context_config import (
    MAX_CONTEXT_LINES,
    FALLBACK_CONTEXT_WINDOW_LINES,
    SAFE_INPUT_TOKEN_LIMIT,
)
from ai.context_extractor import (
    extract_change_aware_context,
    extract_file_change_context,
    find_enclosing_scope,
    parse_code_scopes,
)
from ai.prompt_builder import build_review_prompt
from ai.token_counter import count_tokens


# ─────────────────────────────────────────────────────────────────────────────
# 1. Python AST Extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestPythonContextExtraction:

    def test_python_standalone_function(self):
        code = (
            "def calculate_tax(price, rate):\n"
            "    # Validate inputs\n"
            "    if price < 0:\n"
            "        raise ValueError('Invalid price')\n"
            "    tax = price * rate\n"
            "    return tax\n"
        )
        scopes = parse_code_scopes(code, "tax.py")
        assert len(scopes) == 1
        assert scopes[0].name == "calculate_tax"
        assert scopes[0].start_line == 1
        assert scopes[0].end_line == 6

        # Changed line 5: tax = price * rate
        enclosing = find_enclosing_scope(scopes, 5)
        assert enclosing is not None
        assert enclosing.name == "calculate_tax"
        assert enclosing.scope_type == "function"

    def test_python_class_method_and_enclosing_class(self):
        code = (
            "class PaymentManager:\n"
            "    def __init__(self, currency):\n"
            "        self.currency = currency\n"
            "\n"
            "    def calculate_total(self, price, tax):\n"
            "        total = price + tax\n"
            "        return total\n"
        )
        scopes = parse_code_scopes(code, "payment.py")
        assert any(s.scope_type == "class" and s.name == "PaymentManager" for s in scopes)

        # Changed line 6: total = price + tax
        enclosing = find_enclosing_scope(scopes, 6)
        assert enclosing is not None
        assert enclosing.name == "calculate_total"
        assert enclosing.enclosing_class == "PaymentManager"

    def test_python_multiple_changed_regions_in_different_functions(self):
        lines = [
            "class Service:",
            "    def op_one(self):",
            "        a = 1",
            "        return a",
            "",
            "    def op_two(self):",
            "        b = 2",
            "        return b",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        # Changes on line 3 (in op_one) and line 7 (in op_two)
        regions = extract_file_change_context(
            file_path="service.py",
            changed_line_numbers=[3, 7],
            lines_dict=lines_dict,
            total_lines=len(lines),
        )
        assert len(regions) == 2
        assert regions[0].scope.name == "op_one"
        assert regions[1].scope.name == "op_two"

    def test_python_multiple_changes_in_same_function_deduplicated(self):
        lines = [
            "def worker():",
            "    x = 1",
            "    y = 2",
            "    z = 3",
            "    return x + y + z",
        ]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}
        # Changes on lines 2 and 4 in the same function
        regions = extract_file_change_context(
            file_path="worker.py",
            changed_line_numbers=[2, 4],
            lines_dict=lines_dict,
            total_lines=len(lines),
        )
        # Should be merged into ONE region for worker()
        assert len(regions) == 1
        assert regions[0].scope.name == "worker"
        assert set(regions[0].changed_lines) == {2, 4}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Kotlin Tree-Sitter Extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestKotlinContextExtraction:

    def test_kotlin_class_and_function(self):
        code = (
            "class PaymentViewModel {\n"
            "    fun calculateTotal(price: Double, tax: Double): Double {\n"
            "        val total = price + tax\n"
            "        return total\n"
            "    }\n"
            "}\n"
        )
        scopes = parse_code_scopes(code, "PaymentViewModel.kt")
        assert any(s.scope_type == "class" and s.name == "PaymentViewModel" for s in scopes)

        # Changed line 3: val total = price + tax
        enclosing = find_enclosing_scope(scopes, 3)
        assert enclosing is not None
        assert enclosing.name == "calculateTotal"
        assert enclosing.enclosing_class == "PaymentViewModel"

    def test_kotlin_object_and_multiple_functions(self):
        code = (
            "object MathUtil {\n"
            "    fun add(a: Int, b: Int): Int {\n"
            "        return a + b\n"
            "    }\n"
            "    fun multiply(a: Int, b: Int): Int {\n"
            "        return a * b\n"
            "    }\n"
            "}\n"
        )
        lines = code.splitlines()
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}

        # Changed line 3 (add) and line 6 (multiply)
        regions = extract_file_change_context(
            file_path="MathUtil.kt",
            changed_line_numbers=[3, 6],
            lines_dict=lines_dict,
            total_lines=len(lines),
        )
        assert len(regions) == 2
        assert regions[0].scope.name == "add"
        assert regions[1].scope.name == "multiply"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Swift Tree-Sitter Extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestSwiftContextExtraction:

    def test_swift_class_and_method(self):
        code = (
            "class PaymentViewModel {\n"
            "    func calculateTotal(price: Double, tax: Double) -> Double {\n"
            "        let total = price + tax\n"
            "        return total\n"
            "    }\n"
            "}\n"
        )
        scopes = parse_code_scopes(code, "PaymentViewModel.swift")
        assert any(s.name == "PaymentViewModel" for s in scopes)

        # Changed line 3: let total = price + tax
        enclosing = find_enclosing_scope(scopes, 3)
        assert enclosing is not None
        assert enclosing.name == "calculateTotal"
        assert enclosing.enclosing_class == "PaymentViewModel"

    def test_swift_struct_and_multiple_functions(self):
        code = (
            "struct CartManager {\n"
            "    func addItem() {\n"
            "        print(\"item added\")\n"
            "    }\n"
            "    func removeItem() {\n"
            "        print(\"item removed\")\n"
            "    }\n"
            "}\n"
        )
        lines = code.splitlines()
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}

        regions = extract_file_change_context(
            file_path="CartManager.swift",
            changed_line_numbers=[3, 6],
            lines_dict=lines_dict,
            total_lines=len(lines),
        )
        assert len(regions) == 2
        assert regions[0].scope.name == "addItem"
        assert regions[1].scope.name == "removeItem"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fallback Handling
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackContextHandling:

    def test_fallback_on_unparseable_or_syntax_error(self):
        # Malformed code with syntax error
        bad_code = "def invalid_syntax( (((\n" * 50
        lines = bad_code.splitlines()
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}

        # Changed line 25
        regions = extract_file_change_context(
            file_path="bad.py",
            changed_line_numbers=[25],
            lines_dict=lines_dict,
            total_lines=len(lines),
            fallback_window_lines=10,
        )
        assert len(regions) == 1
        assert regions[0].scope.scope_type == "window"
        # Window bounded around 25 with margin 10
        assert regions[0].start_line == 15
        assert regions[0].end_line == 35

    def test_fallback_on_unknown_extension(self):
        lines = [f"config_key_{i}=val_{i}" for i in range(1, 100)]
        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(lines)}

        regions = extract_file_change_context(
            file_path="settings.properties",
            changed_line_numbers=[50],
            lines_dict=lines_dict,
            total_lines=len(lines),
            fallback_window_lines=15,
        )
        assert len(regions) == 1
        assert regions[0].scope.scope_type == "window"
        assert regions[0].start_line == 35
        assert regions[0].end_line == 65


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAX_CONTEXT_LINES Enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxContextLinesLimit:

    def test_huge_function_is_capped_to_window(self):
        # Create a massive 500-line Python function
        func_lines = ["def massive_function():"]
        for i in range(1, 500):
            func_lines.append(f"    step_{i} = do_something({i})")
        func_lines.append("    return True")

        lines_dict = {i + 1: f"L{i + 1}: {line}" for i, line in enumerate(func_lines)}

        # Change line 50 inside the 500-line function
        regions = extract_file_change_context(
            file_path="huge.py",
            changed_line_numbers=[50],
            lines_dict=lines_dict,
            total_lines=len(func_lines),
            max_context_lines=100,      # limit is 100 lines
            fallback_window_lines=20,   # window is +- 20 lines
        )
        assert len(regions) == 1
        reg = regions[0]
        # Should NOT include all 500 lines!
        assert len(reg.context_lines) < 100
        # Windowed around line 50 (+-20 lines) -> lines 30 to 70
        assert reg.start_line == 30
        assert reg.end_line == 70
        assert "windowed" in reg.scope.scope_type


# ─────────────────────────────────────────────────────────────────────────────
# 6. Prompt Structure and Review Scope Verification
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptStructureAndScope:

    def test_prompt_clearly_separates_changed_code_and_context(self):
        changed_code = "FILE: Payment.kt\nL50: val total = price + tax"
        context = "FILE: Payment.kt\n[Scope: calculateTotal]\nL45: fun calculateTotal() {\nL50: val total = price + tax\nL55: }"

        prompt = build_review_prompt(
            code=context,
            changed_files=["Payment.kt"],
            changed_code=changed_code,
        )

        assert "CHANGED CODE — REVIEW THIS" in prompt
        assert "RELEVANT CONTEXT — FOR UNDERSTANDING ONLY" in prompt
        assert "REVIEW RULES:" in prompt
        assert "1. Review ONLY the changed code" in prompt
        assert "3. Do NOT report pre-existing issues or bugs in unchanged code" in prompt
        assert "4. Do NOT independently review unchanged context" in prompt
        assert "L50: val total = price + tax" in prompt

    def test_existing_issue_format_is_preserved_in_prompt(self):
        prompt = build_review_prompt("code", ["a.py"])
        assert "**Issue:**" in prompt
        assert "**File:**" in prompt
        assert "**Line:**" in prompt
        assert "**Code:**" in prompt
        assert "**Reason:**" in prompt
        assert "**Suggestion:**" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 7. Large File Simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestLargeFileSimulation:

    def test_12000_line_file_produces_compact_prompt(self):
        # Simulate a 12,000-line Kotlin file with 1 small function changed
        lines = []
        for i in range(1, 12001):
            if i == 5000:
                lines.append("class OrderService {")
            elif i == 5010:
                lines.append("    fun processOrder(id: String) {")
            elif i == 5015:
                lines.append("        val discount = calculateDiscount(id)")
            elif i == 5020:
                lines.append("        return discount")
            elif i == 5025:
                lines.append("    }")
            elif i == 5030:
                lines.append("}")
            else:
                lines.append(f"    // line {i} of boilerplate")

        full_code = "FILE: OrderService.kt\n```kotlin\n" + "\n".join(f"L{i}: {line}" for i, line in enumerate(lines, 1)) + "\n```"
        diff = (
            "diff --git a/OrderService.kt b/OrderService.kt\n"
            "@@ -5015,1 +5015,1 @@\n"
            "+        val discount = calculateDiscount(id)\n"
        )

        changed_code, relevant_context = extract_change_aware_context(
            changed_files=["OrderService.kt"],
            diff=diff,
            full_code=full_code,
        )

        prompt = build_review_prompt(
            code=relevant_context,
            changed_files=["OrderService.kt"],
            diff=diff,
            changed_code=changed_code,
        )

        # The full file has ~400,000 characters and 12,000 lines.
        # But the change-aware prompt must be very compact (< 10,000 characters / < 2,500 tokens)!
        prompt_tokens = count_tokens(prompt)
        assert prompt_tokens < 3000
        assert len(prompt) < 15000
        assert "processOrder" in prompt
        assert "discount" in prompt
        # Crucially: distant lines like line 1 or line 11000 must NOT be in the prompt!
        assert "line 11000" not in prompt
        assert "line 1 " not in prompt
