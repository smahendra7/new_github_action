"""
ai/context_extractor.py

Language-aware, change-aware context extractor for AI Code Review.

Supports:
- Python (.py) using the built-in AST parser.
- Kotlin (.kt, .kts) using tree-sitter-kotlin.
- Swift (.swift) using tree-sitter-swift.
- A configurable line-window fallback for unsupported or unparseable files.

Only exact file-path matches are accepted when associating source files with
diffs. This prevents similarly named files from being mixed up.
"""

import ast
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ai.chunk_manager import parse_diff_by_file, parse_full_code_files
from ai.context_config import (
    MAX_CONTEXT_LINES,
    FALLBACK_CONTEXT_WINDOW_LINES,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CodeScope:
    """Represents a structural code scope."""

    scope_type: str
    name: str
    start_line: int
    end_line: int
    enclosing_class: Optional[str] = None


@dataclass
class ChangeContextRegion:
    """Represents an extracted context region for changed lines."""

    file_path: str
    scope: CodeScope
    changed_lines: List[int]
    start_line: int
    end_line: int
    context_lines: List[str]
    changed_lines_code: List[str]


# ---------------------------------------------------------------------------
# Path and line helpers
# ---------------------------------------------------------------------------

def _normalize_path(path: str) -> str:
    """
    Normalize a repository-relative path without performing fuzzy matching.

    Important:
    - Does not remove arbitrary leading dots or slashes.
    - Does not match by filename suffix.
    - Only removes repeated leading './' segments.
    """

    if not path:
        return ""

    normalized = str(path).strip().replace("\\", "/")

    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1]

    while normalized.startswith("./"):
        normalized = normalized[2:]

    return normalized


def _get_exact_file_info(files: Dict, file_path: str):
    """Return file information only when the normalized paths match exactly."""

    target = _normalize_path(file_path)

    if not target:
        return None

    for candidate_path, info in files.items():
        if _normalize_path(candidate_path) == target:
            return info

    return None


def _clean_line(line: str) -> str:
    """Remove the L<number>: prefix used by the full-code parser."""

    if line is None:
        return ""

    return re.sub(r"^L\d+:\s?", "", str(line))


def _get_valid_line_number(line_number: int, total_lines: int) -> int:
    """Clamp a line number to the valid source range."""

    if total_lines <= 0:
        return 1

    return max(1, min(total_lines, int(line_number)))


def _get_language(file_path: str) -> str:
    """Return a language label suitable for Markdown code fences."""

    extension = os.path.splitext(file_path)[1].lower()

    return {
        ".py": "python",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".swift": "swift",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
    }.get(extension, "text")


# ---------------------------------------------------------------------------
# Python scope parsing
# ---------------------------------------------------------------------------

def _parse_python_scopes(raw_code: str) -> List[CodeScope]:
    """Parse Python source code into class and function scopes."""

    try:
        tree = ast.parse(raw_code)
    except (SyntaxError, ValueError, TypeError):
        return []

    scopes: List[CodeScope] = []

    def visit_node(
        node: ast.AST,
        current_class: Optional[str] = None,
    ) -> None:

        for child in ast.iter_child_nodes(node):

            if isinstance(child, ast.ClassDef):
                class_scope = CodeScope(
                    scope_type="class",
                    name=child.name,
                    start_line=child.lineno,
                    end_line=getattr(child, "end_lineno", child.lineno),
                    enclosing_class=current_class,
                )

                scopes.append(class_scope)

                visit_node(child, current_class=child.name)

            elif isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                function_scope = CodeScope(
                    scope_type="function",
                    name=child.name,
                    start_line=child.lineno,
                    end_line=getattr(child, "end_lineno", child.lineno),
                    enclosing_class=current_class,
                )

                scopes.append(function_scope)

                # Continue into nested functions and classes.
                visit_node(child, current_class=current_class)

            else:
                visit_node(child, current_class=current_class)

    visit_node(tree)

    return scopes


# ---------------------------------------------------------------------------
# Tree-sitter compatibility helpers
# ---------------------------------------------------------------------------

def _create_tree_sitter_parser(language_module):
    """
    Create a tree-sitter parser.

    Supports both older and newer tree-sitter Python APIs.
    """

    from tree_sitter import Language, Parser

    language = Language(language_module.language())

    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        parser.language = language
        return parser


def _get_tree_sitter_node_name(node) -> str:
    """Extract a declaration name from a tree-sitter node."""

    name_node = node.child_by_field_name("name")

    if name_node is not None:
        return name_node.text.decode("utf-8", errors="replace")

    for child in node.children:
        if child.type in (
            "identifier",
            "simple_identifier",
            "type_identifier",
        ):
            return child.text.decode("utf-8", errors="replace")

    return "anonymous"


# ---------------------------------------------------------------------------
# Kotlin scope parsing
# ---------------------------------------------------------------------------

def _parse_kotlin_scopes(raw_code: str) -> List[CodeScope]:
    """Parse Kotlin source code using tree-sitter-kotlin."""

    try:
        import tree_sitter_kotlin as tskotlin

        parser = _create_tree_sitter_parser(tskotlin)
        tree = parser.parse(raw_code.encode("utf-8"))

    except Exception:
        return []

    scopes: List[CodeScope] = []

    class_node_types = {
        "class_declaration",
        "object_declaration",
        "companion_object",
        "interface_declaration",
    }

    function_node_types = {
        "function_declaration",
        "secondary_constructor",
    }

    def walk(node, current_class: Optional[str] = None) -> None:

        for child in node.children:

            if child.type in class_node_types:
                class_name = _get_tree_sitter_node_name(child)

                scopes.append(
                    CodeScope(
                        scope_type="class",
                        name=class_name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        enclosing_class=current_class,
                    )
                )

                walk(child, current_class=class_name)

            elif child.type in function_node_types:
                function_name = _get_tree_sitter_node_name(child)

                scopes.append(
                    CodeScope(
                        scope_type="function",
                        name=function_name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        enclosing_class=current_class,
                    )
                )

                walk(child, current_class=current_class)

            else:
                walk(child, current_class=current_class)

    walk(tree.root_node)

    return scopes


# ---------------------------------------------------------------------------
# Swift scope parsing
# ---------------------------------------------------------------------------

def _parse_swift_scopes(raw_code: str) -> List[CodeScope]:
    """Parse Swift source code using tree-sitter-swift."""

    try:
        import tree_sitter_swift as tsswift

        parser = _create_tree_sitter_parser(tsswift)
        tree = parser.parse(raw_code.encode("utf-8"))

    except Exception:
        return []

    scopes: List[CodeScope] = []

    class_node_types = {
        "class_declaration",
        "struct_declaration",
        "enum_declaration",
        "extension_declaration",
        "protocol_declaration",
    }

    function_node_types = {
        "function_declaration",
        "init_declaration",
        "deinit_declaration",
    }

    def walk(node, current_class: Optional[str] = None) -> None:

        for child in node.children:

            if child.type in class_node_types:
                declaration_name = _get_tree_sitter_node_name(child)

                if child.type == "struct_declaration":
                    scope_type = "struct"
                elif child.type == "enum_declaration":
                    scope_type = "enum"
                else:
                    scope_type = "class"

                scopes.append(
                    CodeScope(
                        scope_type=scope_type,
                        name=declaration_name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        enclosing_class=current_class,
                    )
                )

                walk(child, current_class=declaration_name)

            elif child.type in function_node_types:
                function_name = _get_tree_sitter_node_name(child)

                scopes.append(
                    CodeScope(
                        scope_type="function",
                        name=function_name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        enclosing_class=current_class,
                    )
                )

                walk(child, current_class=current_class)

            else:
                walk(child, current_class=current_class)

    walk(tree.root_node)

    return scopes


# ---------------------------------------------------------------------------
# Public scope parser
# ---------------------------------------------------------------------------

def parse_code_scopes(
    raw_code: str,
    file_path: str,
) -> List[CodeScope]:
    """Parse code structure based on the file extension."""

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".py":
        return _parse_python_scopes(raw_code)

    if extension in (".kt", ".kts"):
        return _parse_kotlin_scopes(raw_code)

    if extension == ".swift":
        return _parse_swift_scopes(raw_code)

    return []


# ---------------------------------------------------------------------------
# Find enclosing scope
# ---------------------------------------------------------------------------

def find_enclosing_scope(
    scopes: List[CodeScope],
    target_line: int,
) -> Optional[CodeScope]:
    """
    Find the smallest useful enclosing scope for a line.

    Functions are preferred over classes.
    """

    matching = [
        scope
        for scope in scopes
        if scope.start_line <= target_line <= scope.end_line
    ]

    if not matching:
        return None

    functions = [
        scope
        for scope in matching
        if scope.scope_type == "function"
    ]

    if functions:
        return min(
            functions,
            key=lambda scope: scope.end_line - scope.start_line,
        )

    return min(
        matching,
        key=lambda scope: scope.end_line - scope.start_line,
    )


# ---------------------------------------------------------------------------
# Comments and decorators
# ---------------------------------------------------------------------------

def _expand_start_for_comments_and_decorators(
    lines_dict: Dict[int, str],
    start_line: int,
    max_scan_back: int = 15,
) -> int:
    """
    Include immediately preceding comments and decorators attached to a scope.
    """

    current_line = start_line - 1
    expanded_start = start_line
    blank_lines = 0

    while (
        current_line >= 1
        and (start_line - current_line) <= max_scan_back
    ):
        line = _clean_line(lines_dict.get(current_line, "")).strip()

        if not line:
            blank_lines += 1

            if blank_lines > 1:
                break

            current_line -= 1
            continue

        is_comment = (
            line.startswith("#")
            or line.startswith("//")
            or line.startswith("/*")
            or line.startswith("*")
            or line.endswith("*/")
            or line.startswith('"""')
            or line.startswith("'''")
            or line.endswith('"""')
            or line.endswith("'''")
        )

        is_decorator = line.startswith("@")

        if is_comment or is_decorator:
            expanded_start = current_line
            blank_lines = 0
            current_line -= 1
        else:
            break

    return expanded_start


# ---------------------------------------------------------------------------
# Scope identity
# ---------------------------------------------------------------------------

def _scope_identity(scope: CodeScope) -> Tuple:
    """Return a stable identity for a scope."""

    return (
        scope.scope_type,
        scope.name,
        scope.start_line,
        scope.end_line,
        scope.enclosing_class,
    )


# ---------------------------------------------------------------------------
# Extract context for one file
# ---------------------------------------------------------------------------

def extract_file_change_context(
    file_path: str,
    changed_line_numbers: List[int],
    lines_dict: Dict[int, str],
    total_lines: int,
    max_context_lines: int = MAX_CONTEXT_LINES,
    fallback_window_lines: int = FALLBACK_CONTEXT_WINDOW_LINES,
) -> List[ChangeContextRegion]:
    """
    Extract relevant context for changed lines in one file.

    - Finds the enclosing function or class where possible.
    - Uses a bounded window for very large scopes.
    - Uses a fallback window when parsing is unavailable.
    - Merges overlapping regions only when they belong to the same scope.
    """

    if not changed_line_numbers or not lines_dict or total_lines <= 0:
        return []

    max_context_lines = max(1, int(max_context_lines))
    fallback_window_lines = max(0, int(fallback_window_lines))

    raw_lines = [
        _clean_line(lines_dict.get(line_number, ""))
        for line_number in range(1, total_lines + 1)
    ]

    raw_code = "\n".join(raw_lines)

    scopes = parse_code_scopes(raw_code, file_path)

    candidate_regions: List[
        Tuple[int, int, CodeScope, int]
    ] = []

    for line_number in sorted(set(changed_line_numbers)):

        valid_line = _get_valid_line_number(
            line_number,
            total_lines,
        )

        enclosing = (
            find_enclosing_scope(scopes, valid_line)
            if scopes
            else None
        )

        if enclosing is not None:

            expanded_start = _expand_start_for_comments_and_decorators(
                lines_dict,
                enclosing.start_line,
            )

            scope_size = enclosing.end_line - expanded_start + 1

            if scope_size <= max_context_lines:
                region_start = expanded_start
                region_end = enclosing.end_line
                scope_info = enclosing

            else:
                region_start = max(
                    enclosing.start_line,
                    valid_line - fallback_window_lines,
                )

                region_end = min(
                    enclosing.end_line,
                    valid_line + fallback_window_lines,
                )

                scope_info = CodeScope(
                    scope_type=f"{enclosing.scope_type} (windowed)",
                    name=enclosing.name,
                    start_line=region_start,
                    end_line=region_end,
                    enclosing_class=enclosing.enclosing_class,
                )

        else:
            region_start = max(
                1,
                valid_line - fallback_window_lines,
            )

            region_end = min(
                total_lines,
                valid_line + fallback_window_lines,
            )

            scope_info = CodeScope(
                scope_type="window",
                name=f"Lines {region_start}..{region_end}",
                start_line=region_start,
                end_line=region_end,
                enclosing_class=None,
            )

        candidate_regions.append(
            (
                region_start,
                region_end,
                scope_info,
                valid_line,
            )
        )

    candidate_regions.sort(
        key=lambda item: (item[0], item[1])
    )

    merged_regions: List[ChangeContextRegion] = []

    for region_start, region_end, scope_info, changed_line in candidate_regions:

        context_lines = [
            lines_dict[line_number]
            for line_number in range(region_start, region_end + 1)
            if line_number in lines_dict
        ]

        changed_code = (
            [lines_dict[changed_line]]
            if changed_line in lines_dict
            else [f"L{changed_line}: (source line unavailable)"]
        )

        if not merged_regions:
            merged_regions.append(
                ChangeContextRegion(
                    file_path=file_path,
                    scope=scope_info,
                    changed_lines=[changed_line],
                    start_line=region_start,
                    end_line=region_end,
                    context_lines=context_lines,
                    changed_lines_code=changed_code,
                )
            )
            continue

        previous = merged_regions[-1]

        same_scope = (
            _scope_identity(previous.scope)
            == _scope_identity(scope_info)
        )

        both_windows = (
            previous.scope.scope_type == "window"
            and scope_info.scope_type == "window"
        )

        overlaps = region_start <= previous.end_line

        if overlaps and (same_scope or both_windows):

            previous.end_line = max(
                previous.end_line,
                region_end,
            )

            if changed_line not in previous.changed_lines:
                previous.changed_lines.append(changed_line)
                previous.changed_lines_code.extend(changed_code)

            previous.context_lines = [
                lines_dict[line_number]
                for line_number in range(
                    previous.start_line,
                    previous.end_line + 1,
                )
                if line_number in lines_dict
            ]

        else:
            merged_regions.append(
                ChangeContextRegion(
                    file_path=file_path,
                    scope=scope_info,
                    changed_lines=[changed_line],
                    start_line=region_start,
                    end_line=region_end,
                    context_lines=context_lines,
                    changed_lines_code=changed_code,
                )
            )

    return merged_regions


# ---------------------------------------------------------------------------
# Extract changed lines from a parsed diff
# ---------------------------------------------------------------------------

def _get_changed_line_numbers(diff_info: Optional[Dict]) -> List[int]:
    """
    Get changed line numbers from parsed diff information.

    Uses new-file line numbers from the diff.

    For pure deletion hunks, uses the nearest valid new-file position.
    """

    if not diff_info:
        return []

    changed_lines: List[int] = []

    for hunk in diff_info.get("hunks", []):

        modified_lines = getattr(
            hunk,
            "modified_lines",
            None,
        )

        if modified_lines:
            changed_lines.extend(modified_lines)
            continue

        new_start = getattr(hunk, "new_start", None)

        if new_start is not None:
            changed_lines.append(max(1, int(new_start)))

    return sorted(set(changed_lines))


# ---------------------------------------------------------------------------
# Change-aware context for all changed files
# ---------------------------------------------------------------------------

def extract_change_aware_context(
    changed_files: List[str],
    diff: str,
    full_code: str,
    max_context_lines: int = MAX_CONTEXT_LINES,
    fallback_window_lines: int = FALLBACK_CONTEXT_WINDOW_LINES,
) -> Tuple[str, str]:
    """
    Extract change-aware review context for all changed files.

    Returns:
        (changed_code_text, relevant_context_text)

    File matching is exact. No filename suffix matching is performed.
    """

    parsed_files = parse_full_code_files(full_code or "")
    parsed_diff = parse_diff_by_file(diff or "")

    all_changed_blocks: List[str] = []
    all_context_blocks: List[str] = []

    for relative_path in changed_files:

        normalized_path = _normalize_path(relative_path)

        if not normalized_path:
            continue

        file_info = _get_exact_file_info(
            parsed_files,
            normalized_path,
        )

        diff_info = _get_exact_file_info(
            parsed_diff,
            normalized_path,
        )

        # Do not substitute another file with a similar name.
        if not file_info:
            all_context_blocks.append(
                f"FILE: {normalized_path}\n"
                "[Source code unavailable for this exact file path. "
                "No substitute file was selected.]"
            )
            continue

        lang = file_info.get("lang") or _get_language(normalized_path)

        lines_dict = file_info.get("lines_dict", {})
        total_lines = int(file_info.get("total_lines", 0))

        changed_line_numbers = _get_changed_line_numbers(diff_info)

        # Do not assume line 1 changed when no changed lines were found.
        if not changed_line_numbers:
            all_context_blocks.append(
                f"FILE: {normalized_path}\n"
                "[No changed line numbers were found in the supplied diff. "
                "Source context was not treated as changed code.]"
            )
            continue

        regions = extract_file_change_context(
            file_path=normalized_path,
            changed_line_numbers=changed_line_numbers,
            lines_dict=lines_dict,
            total_lines=total_lines,
            max_context_lines=max_context_lines,
            fallback_window_lines=fallback_window_lines,
        )

        for region in regions:

            changed_code_inner = "\n".join(
                region.changed_lines_code
            )

            all_changed_blocks.append(
                f"FILE: {normalized_path} "
                f"(Lines {', '.join(str(line) for line in region.changed_lines)})\n"
                f"```{lang}\n"
                f"{changed_code_inner}\n"
                f"```"
            )

            header_parts = [
                f"FILE: {normalized_path}"
            ]

            if region.scope.enclosing_class:
                header_parts.append(
                    f"[Enclosing Class: {region.scope.enclosing_class}]"
                )

            header_parts.append(
                f"[Scope: {region.scope.name} "
                f"(Lines {region.start_line}..{region.end_line})]"
            )

            header = "\n".join(header_parts)
            context_inner = "\n".join(region.context_lines)

            all_context_blocks.append(
                f"{header}\n"
                f"```{lang}\n"
                f"{context_inner}\n"
                f"```"
            )

    changed_code_text = (
        "\n\n".join(all_changed_blocks)
        if all_changed_blocks
        else "(no changed code detected)"
    )

    relevant_context_text = (
        "\n\n".join(all_context_blocks)
        if all_context_blocks
        else "(no context available)"
    )

    return changed_code_text, relevant_context_text


# ---------------------------------------------------------------------------
# Focused context for one changed file
# ---------------------------------------------------------------------------

def extract_file_focused_review_context(
    file_path: str,
    file_diff: str,
    lines_dict: Dict[int, str],
    total_lines: int,
    lang: str = "text",
) -> Tuple[str, str, bool]:
    """
    Extract focused review context for one changed file.

    When an enclosing function is found:
    - Includes the complete associated function.
    - Includes up to 15 lines before the function.
    - Includes up to 15 lines after the function.

    When function boundaries cannot be determined:
    - Uses a clearly marked fallback window.

    Returns:
        (changed_code_text, relevant_context_text, is_guaranteed_function)
    """

    if not lines_dict or total_lines <= 0:
        return (
            file_diff or "(no diff available)",
            "(no source context available)",
            False,
        )

    normalized_path = _normalize_path(file_path)

    parsed_diff = parse_diff_by_file(file_diff or "")

    diff_info = _get_exact_file_info(
        parsed_diff,
        normalized_path,
    )

    changed_line_numbers = _get_changed_line_numbers(diff_info)

    # Fallback to reading hunk headers only if the diff parser did not
    # return any changed line numbers.
    if not changed_line_numbers:
        matches = re.findall(
            r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@",
            file_diff or "",
        )

        for start_text, count_text in matches:
            start = int(start_text)
            count = int(count_text) if count_text else 1

            # A zero-length new range represents a deletion.
            if count == 0:
                changed_line_numbers.append(max(1, start))
            else:
                changed_line_numbers.extend(
                    range(start, start + count)
                )

    changed_line_numbers = sorted(
        set(changed_line_numbers)
    )

    if not changed_line_numbers:
        return (
            file_diff or "(no changed lines detected)",
            "[No changed line numbers were found. "
            "No source lines were assumed to be changed.]",
            False,
        )

    raw_lines = [
        _clean_line(lines_dict.get(line_number, ""))
        for line_number in range(1, total_lines + 1)
    ]

    raw_code = "\n".join(raw_lines)

    scopes = parse_code_scopes(
        raw_code,
        normalized_path,
    )

    function_scopes: List[CodeScope] = []
    seen_scope_keys = set()
    unmatched_lines: List[int] = []

    for line_number in changed_line_numbers:

        valid_line = _get_valid_line_number(
            line_number,
            total_lines,
        )

        enclosing = (
            find_enclosing_scope(scopes, valid_line)
            if scopes
            else None
        )

        if enclosing and enclosing.scope_type == "function":

            key = _scope_identity(enclosing)

            if key not in seen_scope_keys:
                seen_scope_keys.add(key)
                function_scopes.append(enclosing)

        else:
            unmatched_lines.append(valid_line)

    context_sections: List[str] = []
    is_guaranteed = False

    # -------------------------------------------------------
    # Complete associated functions
    # -------------------------------------------------------

    if function_scopes:

        is_guaranteed = True

        function_scopes.sort(
            key=lambda scope: scope.start_line
        )

        for scope in function_scopes:

            function_start = _expand_start_for_comments_and_decorators(
                lines_dict,
                scope.start_line,
            )

            function_end = scope.end_line

            function_lines = [
                lines_dict[line_number]
                for line_number in range(
                    function_start,
                    function_end + 1,
                )
                if line_number in lines_dict
            ]

            function_body = "\n".join(function_lines)

            # Up to 15 lines before the function.
            before_start = max(
                1,
                function_start - 15,
            )

            before_end = function_start - 1

            before_lines = [
                lines_dict[line_number]
                for line_number in range(
                    before_start,
                    before_end + 1,
                )
                if line_number in lines_dict
            ]

            # Up to 15 lines after the function.
            after_start = function_end + 1

            after_end = min(
                total_lines,
                function_end + 15,
            )

            after_lines = [
                lines_dict[line_number]
                for line_number in range(
                    after_start,
                    after_end + 1,
                )
                if line_number in lines_dict
            ]

            scope_title = (
                f"[Complete Associated Function: '{scope.name}' "
                f"(Lines {function_start}..{function_end})]"
            )

            if scope.enclosing_class:
                scope_title = (
                    f"[Enclosing Class: {scope.enclosing_class}]\n"
                    f"{scope_title}"
                )

            parts = [
                f"FILE: {normalized_path}",
                scope_title,
            ]

            if before_lines:
                parts.append(
                    f"--- Context Before Function "
                    f"(Up to 15 lines: "
                    f"Lines {before_start}..{before_end}) ---\n"
                    f"```{lang}\n"
                    f"{chr(10).join(before_lines)}\n"
                    f"```"
                )

            parts.append(
                f"--- Complete Associated Function "
                f"(Lines {function_start}..{function_end}) ---\n"
                f"```{lang}\n"
                f"{function_body}\n"
                f"```"
            )

            if after_lines:
                parts.append(
                    f"--- Context After Function "
                    f"(Up to 15 lines: "
                    f"Lines {after_start}..{after_end}) ---\n"
                    f"```{lang}\n"
                    f"{chr(10).join(after_lines)}\n"
                    f"```"
                )

            context_sections.append(
                "\n\n".join(parts)
            )

    # -------------------------------------------------------
    # Changes outside functions
    # -------------------------------------------------------

    if not function_scopes:

        target_lines = unmatched_lines or changed_line_numbers

        window_start = max(
            1,
            min(target_lines) - 15,
        )

        window_end = min(
            total_lines,
            max(target_lines) + 15,
        )

        window_lines = [
            lines_dict[line_number]
            for line_number in range(
                window_start,
                window_end + 1,
            )
            if line_number in lines_dict
        ]

        fallback_block = (
            f"FILE: {normalized_path}\n"
            f"[Scope: Lines {window_start}..{window_end} "
            f"(Fallback Window - Exact Function Boundaries "
            f"Not Determined)]\n"
            f"```{lang}\n"
            f"{chr(10).join(window_lines)}\n"
            f"```"
        )

        context_sections.append(fallback_block)
        is_guaranteed = False

    elif unmatched_lines:

        window_start = max(
            1,
            min(unmatched_lines) - 15,
        )

        window_end = min(
            total_lines,
            max(unmatched_lines) + 15,
        )

        window_lines = [
            lines_dict[line_number]
            for line_number in range(
                window_start,
                window_end + 1,
            )
            if line_number in lines_dict
        ]

        fallback_block = (
            f"FILE: {normalized_path}\n"
            f"[Scope: Lines {window_start}..{window_end} "
            f"(Fallback Window - Additional Changes Outside Functions)]\n"
            f"```{lang}\n"
            f"{chr(10).join(window_lines)}\n"
            f"```"
        )

        context_sections.append(fallback_block)

    # -------------------------------------------------------
    # Changed code
    # -------------------------------------------------------

    changed_code_lines = [
        lines_dict[line_number]
        for line_number in changed_line_numbers
        if line_number in lines_dict
    ]

    if changed_code_lines:

        changed_code_text = (
            f"FILE: {normalized_path}\n"
            f"```{lang}\n"
            f"{chr(10).join(changed_code_lines)}\n"
            f"```"
        )

    else:
        changed_code_text = (
            file_diff or "(changed source lines unavailable)"
        )

    relevant_context_text = (
        "\n\n========================================\n\n".join(
            context_sections
        )
    )

    return (
        changed_code_text,
        relevant_context_text,
        is_guaranteed,
    )
