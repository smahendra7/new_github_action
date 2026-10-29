"""
ai/context_extractor.py

Language-aware change-aware context extractor for AI Code Review.
Supports Python (.py), Kotlin (.kt, .kts), and Swift (.swift) using AST parsing,
with a configurable line-window fallback for unparseable or other file types.

Determines the relevant enclosing function/method and class before calling the AI,
ensuring the model receives only the changed code and minimal necessary context.
"""

import ast
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ai.chunk_manager import parse_diff_by_file, parse_full_code_files
from ai.context_config import (
    MAX_CONTEXT_LINES,
    FALLBACK_CONTEXT_WINDOW_LINES,
)


@dataclass
class CodeScope:
    """Represents a structural code scope (function, class, etc.)."""
    scope_type: str        # 'function', 'class', 'struct', 'enum', 'window'
    name: str              # e.g. 'calculateTotal'
    start_line: int        # 1-based start line
    end_line: int          # 1-based end line
    enclosing_class: Optional[str] = None


@dataclass
class ChangeContextRegion:
    """Represents an extracted context region for one or more changed lines."""
    file_path: str
    scope: CodeScope
    changed_lines: List[int]
    start_line: int
    end_line: int
    context_lines: List[str]      # formatted line strings e.g. "L45: val total = ..."
    changed_lines_code: List[str] # changed code line strings


def _parse_python_scopes(raw_code: str) -> List[CodeScope]:
    """Parse Python source code using built-in ast module into CodeScope list."""
    try:
        tree = ast.parse(raw_code)
    except Exception:
        return []

    scopes: List[CodeScope] = []

    def visit_node(node: ast.AST, current_class: Optional[str] = None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                scopes.append(
                    CodeScope(
                        scope_type="class",
                        name=child.name,
                        start_line=child.lineno,
                        end_line=getattr(child, "end_lineno", child.lineno),
                        enclosing_class=current_class,
                    )
                )
                visit_node(child, current_class=child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes.append(
                    CodeScope(
                        scope_type="function",
                        name=child.name,
                        start_line=child.lineno,
                        end_line=getattr(child, "end_lineno", child.lineno),
                        enclosing_class=current_class,
                    )
                )
                # Also visit nested functions inside this function
                visit_node(child, current_class=current_class)
            else:
                visit_node(child, current_class=current_class)

    visit_node(tree)
    return scopes


def _parse_kotlin_scopes(raw_code: str) -> List[CodeScope]:
    """Parse Kotlin source code using tree-sitter-kotlin into CodeScope list."""
    try:
        import tree_sitter_kotlin as tskotlin
        from tree_sitter import Language, Parser
        kotlin_lang = Language(tskotlin.language())
        parser = Parser(kotlin_lang)
        tree = parser.parse(raw_code.encode("utf-8"))
    except Exception:
        return []

    scopes: List[CodeScope] = []

    def get_node_name(node) -> str:
        for child in node.children:
            if child.type in ("identifier", "simple_identifier", "type_identifier"):
                return child.text.decode("utf-8")
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf-8")
        return "anonymous"

    def walk(node, current_class: Optional[str] = None):
        for child in node.children:
            if child.type in ("class_declaration", "object_declaration", "companion_object", "interface_declaration"):
                class_name = get_node_name(child)
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
            elif child.type in ("function_declaration", "secondary_constructor"):
                func_name = get_node_name(child)
                scopes.append(
                    CodeScope(
                        scope_type="function",
                        name=func_name,
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


def _parse_swift_scopes(raw_code: str) -> List[CodeScope]:
    """Parse Swift source code using tree-sitter-swift into CodeScope list."""
    try:
        import tree_sitter_swift as tsswift
        from tree_sitter import Language, Parser
        swift_lang = Language(tsswift.language())
        parser = Parser(swift_lang)
        tree = parser.parse(raw_code.encode("utf-8"))
    except Exception:
        return []

    scopes: List[CodeScope] = []

    def get_node_name(node) -> str:
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "simple_identifier"):
                return child.text.decode("utf-8")
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf-8")
        return "anonymous"

    def walk(node, current_class: Optional[str] = None):
        for child in node.children:
            if child.type in ("class_declaration", "struct_declaration", "enum_declaration", "extension_declaration", "protocol_declaration"):
                decl_name = get_node_name(child)
                scope_type = "class" if "class" in child.type else ("struct" if "struct" in child.type else "enum")
                scopes.append(
                    CodeScope(
                        scope_type=scope_type,
                        name=decl_name,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        enclosing_class=current_class,
                    )
                )
                walk(child, current_class=decl_name)
            elif child.type in ("function_declaration", "init_declaration", "deinit_declaration"):
                func_name = get_node_name(child)
                scopes.append(
                    CodeScope(
                        scope_type="function",
                        name=func_name,
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


def parse_code_scopes(raw_code: str, file_path: str) -> List[CodeScope]:
    """Parse code structure for the given file based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".py":
        return _parse_python_scopes(raw_code)
    elif ext in (".kt", ".kts"):
        return _parse_kotlin_scopes(raw_code)
    elif ext == ".swift":
        return _parse_swift_scopes(raw_code)
    return []


def find_enclosing_scope(scopes: List[CodeScope], target_line: int) -> Optional[CodeScope]:
    """
    Find the smallest useful enclosing scope for a target line.
    Prefers functions/methods over classes/structs.
    """
    matching = [
        s for s in scopes
        if s.start_line <= target_line <= s.end_line
    ]
    if not matching:
        return None

    # Prefer functions over classes
    funcs = [s for s in matching if s.scope_type == "function"]
    if funcs:
        # Return smallest enclosing function
        return min(funcs, key=lambda s: s.end_line - s.start_line)

    # Otherwise smallest enclosing class/struct
    return min(matching, key=lambda s: s.end_line - s.start_line)


def _expand_start_for_comments_and_decorators(
    lines_dict: Dict[int, str],
    start_line: int,
    max_scan_back: int = 15,
) -> int:
    """
    Expand start_line upwards to include immediately preceding comments,
    docstrings, or decorators attached to the scope/function.
    """
    curr = start_line - 1
    expanded_start = start_line
    blank_lines = 0

    while curr >= 1 and (start_line - curr) <= max_scan_back:
        line_str = lines_dict.get(curr, "")
        clean = re.sub(r"^L\d+:\s?", "", line_str).strip()

        if not clean:
            blank_lines += 1
            if blank_lines > 1:
                break
            curr -= 1
            continue

        is_comment = (
            clean.startswith("#")
            or clean.startswith("//")
            or clean.startswith("/*")
            or clean.startswith("*")
            or clean.endswith("*/")
            or clean.startswith('"""')
            or clean.startswith("'''")
            or clean.endswith('"""')
            or clean.endswith("'''")
        )
        is_decorator = clean.startswith("@")

        if is_comment or is_decorator:
            expanded_start = curr
            blank_lines = 0
            curr -= 1
        else:
            break

    return expanded_start


def extract_file_change_context(
    file_path: str,
    changed_line_numbers: List[int],
    lines_dict: Dict[int, str],
    total_lines: int,
    max_context_lines: int = MAX_CONTEXT_LINES,
    fallback_window_lines: int = FALLBACK_CONTEXT_WINDOW_LINES,
) -> List[ChangeContextRegion]:
    """
    Extract relevant context regions for the changed lines of a single file:
    1. Parse AST to find enclosing functions/classes.
    2. Prefer smallest enclosing function/method.
    3. If function exceeds max_context_lines, cap to limited window around changes.
    4. Fall back to limited window if parsing is unavailable.
    5. Deduplicate and merge regions when multiple changed lines share context.
    """
    if not changed_line_numbers:
        return []

    # Reconstruct raw source code without L<no>: prefix for AST parsers
    raw_lines = []
    for i in range(1, total_lines + 1):
        line_str = lines_dict.get(i, "")
        clean_line = re.sub(r"^L\d+:\s?", "", line_str)
        raw_lines.append(clean_line)
    raw_code = "\n".join(raw_lines)

    scopes = parse_code_scopes(raw_code, file_path)

    # Map each changed line to its effective context boundaries and metadata
    candidate_regions: List[Tuple[int, int, CodeScope, int]] = []

    for line_no in sorted(changed_line_numbers):
        # Bound line number to valid file range
        valid_line = max(1, min(total_lines, line_no)) if total_lines > 0 else line_no

        enclosing = find_enclosing_scope(scopes, valid_line) if scopes else None

        if enclosing and enclosing.scope_type == "function":
            expanded_start = _expand_start_for_comments_and_decorators(lines_dict, enclosing.start_line)
            span_size = enclosing.end_line - expanded_start + 1
            if span_size <= max_context_lines:
                r_start = expanded_start
                r_end = enclosing.end_line
                scope_info = enclosing
            else:
                # Function exceeds MAX_CONTEXT_LINES limit -> cap to limited window around change
                r_start = max(enclosing.start_line, valid_line - fallback_window_lines)
                r_end = min(enclosing.end_line, valid_line + fallback_window_lines)
                scope_info = CodeScope(
                    scope_type="function (windowed)",
                    name=enclosing.name,
                    start_line=r_start,
                    end_line=r_end,
                    enclosing_class=enclosing.enclosing_class,
                )
        elif enclosing:
            # Enclosing is class or struct
            expanded_start = _expand_start_for_comments_and_decorators(lines_dict, enclosing.start_line)
            span_size = enclosing.end_line - expanded_start + 1
            if span_size <= max_context_lines:
                r_start = expanded_start
                r_end = enclosing.end_line
                scope_info = enclosing
            else:
                r_start = max(enclosing.start_line, valid_line - fallback_window_lines)
                r_end = min(enclosing.end_line, valid_line + fallback_window_lines)
                scope_info = CodeScope(
                    scope_type="class (windowed)",
                    name=enclosing.name,
                    start_line=r_start,
                    end_line=r_end,
                    enclosing_class=enclosing.enclosing_class,
                )
        else:
            # Fallback context window: lines around changed line
            r_start = max(1, valid_line - fallback_window_lines)
            r_end = min(total_lines, valid_line + fallback_window_lines)
            scope_info = CodeScope(
                scope_type="window",
                name=f"Lines {r_start}..{r_end}",
                start_line=r_start,
                end_line=r_end,
                enclosing_class=None,
            )

        candidate_regions.append((r_start, r_end, scope_info, valid_line))

    # Merge overlapping or identical regions within the file
    merged_regions: List[ChangeContextRegion] = []
    candidate_regions.sort(key=lambda x: (x[0], x[1]))

    for r_start, r_end, scope_info, changed_line in candidate_regions:
        if not merged_regions:
            ctx_lines = [lines_dict[ln] for ln in range(r_start, r_end + 1) if ln in lines_dict]
            ch_lines = [lines_dict[changed_line]] if changed_line in lines_dict else [f"L{changed_line}: (modified)"]
            merged_regions.append(
                ChangeContextRegion(
                    file_path=file_path,
                    scope=scope_info,
                    changed_lines=[changed_line],
                    start_line=r_start,
                    end_line=r_end,
                    context_lines=ctx_lines,
                    changed_lines_code=ch_lines,
                )
            )
        else:
            prev = merged_regions[-1]
            # Only merge if line ranges actually overlap and belong to the same scope or window
            if r_start <= prev.end_line and (prev.scope.name == scope_info.name or prev.scope.scope_type == "window"):
                prev.end_line = max(prev.end_line, r_end)
                if changed_line not in prev.changed_lines:
                    prev.changed_lines.append(changed_line)
                    if changed_line in lines_dict:
                        prev.changed_lines_code.append(lines_dict[changed_line])
                prev.context_lines = [
                    lines_dict[ln]
                    for ln in range(prev.start_line, prev.end_line + 1)
                    if ln in lines_dict
                ]
            else:
                ctx_lines = [lines_dict[ln] for ln in range(r_start, r_end + 1) if ln in lines_dict]
                ch_lines = [lines_dict[changed_line]] if changed_line in lines_dict else [f"L{changed_line}: (modified)"]
                merged_regions.append(
                    ChangeContextRegion(
                        file_path=file_path,
                        scope=scope_info,
                        changed_lines=[changed_line],
                        start_line=r_start,
                        end_line=r_end,
                        context_lines=ctx_lines,
                        changed_lines_code=ch_lines,
                    )
                )

    return merged_regions


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
    """
    parsed_files = parse_full_code_files(full_code)
    parsed_diff = parse_diff_by_file(diff)

    all_changed_blocks: List[str] = []
    all_context_blocks: List[str] = []

    for rel_path in changed_files:
        norm_path = rel_path.strip().replace("\\", "/").lstrip("./")

        file_info = parsed_files.get(norm_path)
        diff_info = parsed_diff.get(norm_path)

        if not file_info:
            for p, info in parsed_files.items():
                if p.endswith(norm_path) or norm_path.endswith(p):
                    file_info = info
                    norm_path = p
                    break

        if not diff_info:
            for p, dinfo in parsed_diff.items():
                if p.endswith(norm_path) or norm_path.endswith(p):
                    diff_info = dinfo
                    break

        if not file_info:
            continue

        lang = file_info["lang"]
        lines_dict = file_info["lines_dict"]
        total_lines = file_info["total_lines"]

        # Collect all changed line numbers from diff hunks
        changed_line_numbers: List[int] = []
        if diff_info and diff_info.get("hunks"):
            for hunk in diff_info["hunks"]:
                if hunk.modified_lines:
                    changed_line_numbers.extend(hunk.modified_lines)
                else:
                    # Pure deletion hunk -> context around new_start
                    changed_line_numbers.append(hunk.new_start)

        # Fallback if no hunks detected but file was touched
        if not changed_line_numbers and total_lines > 0:
            changed_line_numbers = [1]

        regions = extract_file_change_context(
            file_path=rel_path,
            changed_line_numbers=changed_line_numbers,
            lines_dict=lines_dict,
            total_lines=total_lines,
            max_context_lines=max_context_lines,
            fallback_window_lines=fallback_window_lines,
        )

        for reg in regions:
            # 1. Changed code block
            ch_code_inner = "\n".join(reg.changed_lines_code)
            all_changed_blocks.append(
                f"FILE: {rel_path} (Lines {', '.join(str(l) for l in reg.changed_lines)})\n"
                f"```{lang}\n{ch_code_inner}\n```"
            )

            # 2. Relevant context block
            header_parts = [f"FILE: {rel_path}"]
            if reg.scope.enclosing_class:
                header_parts.append(f"[Enclosing Class: {reg.scope.enclosing_class}]")
            header_parts.append(
                f"[Scope: {reg.scope.name} (Lines {reg.start_line}..{reg.end_line})]"
            )
            header = "\n".join(header_parts)
            ctx_code_inner = "\n".join(reg.context_lines)

            all_context_blocks.append(
                f"{header}\n```{lang}\n{ctx_code_inner}\n```"
            )

    changed_code_text = "\n\n".join(all_changed_blocks) if all_changed_blocks else "(no changed code detected)"
    relevant_context_text = "\n\n".join(all_context_blocks) if all_context_blocks else "(no context available)"

    return changed_code_text, relevant_context_text


def extract_file_focused_review_context(
    file_path: str,
    file_diff: str,
    lines_dict: Dict[int, str],
    total_lines: int,
    lang: str = "text",
) -> Tuple[str, str, bool]:
    """
    Extract focused review context for a single changed file:
    1. Complete associated function(s) containing changes without truncation.
    2. Up to 15 lines before the function (excluding the function itself).
    3. Up to 15 lines after the function (excluding the function itself).
    4. Fallback window if exact function boundaries cannot be determined,
       clearly marked to preserve the distinction.

    Returns:
        (changed_code_text, relevant_context_text, is_guaranteed_function)
    """
    if not lines_dict or total_lines <= 0:
        return file_diff or "(no diff available)", "(no source context available)", False

    # 1. Parse changed lines from diff
    parsed_diff = parse_diff_by_file(file_diff)
    changed_line_numbers: List[int] = []
    norm_path = file_path.strip().replace("\\", "/").lstrip("./")

    diff_info = parsed_diff.get(norm_path)
    if not diff_info:
        for p, dinfo in parsed_diff.items():
            if p.endswith(norm_path) or norm_path.endswith(p):
                diff_info = dinfo
                break

    if diff_info and diff_info.get("hunks"):
        for hunk in diff_info["hunks"]:
            if hunk.modified_lines:
                changed_line_numbers.extend(hunk.modified_lines)
            else:
                changed_line_numbers.append(hunk.new_start)

    if not changed_line_numbers:
        matches = re.findall(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", file_diff or "")
        for m_start, m_count in matches:
            st = int(m_start)
            cnt = int(m_count) if m_count else 1
            changed_line_numbers.extend(range(st, st + max(1, cnt)))

    if not changed_line_numbers:
        changed_line_numbers = [1]

    # Reconstruct raw code for AST parsers
    raw_lines = [re.sub(r"^L\d+:\s?", "", lines_dict.get(i, "")) for i in range(1, total_lines + 1)]
    raw_code = "\n".join(raw_lines)
    scopes = parse_code_scopes(raw_code, file_path)

    # Find unique function scopes covering the changes
    function_scopes: List[CodeScope] = []
    seen_scope_keys = set()
    unmatched_lines: List[int] = []

    for line_no in sorted(set(changed_line_numbers)):
        valid_line = max(1, min(total_lines, line_no))
        enclosing = find_enclosing_scope(scopes, valid_line) if scopes else None
        if enclosing and enclosing.scope_type == "function":
            key = (enclosing.name, enclosing.start_line, enclosing.end_line)
            if key not in seen_scope_keys:
                seen_scope_keys.add(key)
                function_scopes.append(enclosing)
        else:
            unmatched_lines.append(valid_line)

    context_sections: List[str] = []
    is_guaranteed = False

    if function_scopes:
        is_guaranteed = True
        function_scopes.sort(key=lambda s: s.start_line)

        for scope in function_scopes:
            func_start = _expand_start_for_comments_and_decorators(lines_dict, scope.start_line)
            func_end = scope.end_line

            # Function itself without truncation
            func_lines = [lines_dict[ln] for ln in range(func_start, func_end + 1) if ln in lines_dict]
            func_body = "\n".join(func_lines)

            # Up to 15 lines before function (excluding the function)
            before_start = max(1, func_start - 15)
            before_end = func_start - 1
            before_lines = [lines_dict[ln] for ln in range(before_start, before_end + 1) if ln in lines_dict]

            # Up to 15 lines after function (excluding the function)
            after_start = func_end + 1
            after_end = min(total_lines, func_end + 15)
            after_lines = [lines_dict[ln] for ln in range(after_start, after_end + 1) if ln in lines_dict]

            scope_title = f"[Complete Associated Function: '{scope.name}' (Lines {func_start}..{func_end})]"
            if scope.enclosing_class:
                scope_title = f"[Enclosing Class: {scope.enclosing_class}]\n{scope_title}"

            parts = [f"FILE: {file_path}", scope_title]

            if before_lines:
                parts.append(
                    f"--- Context Before Function (Up to 15 lines: Lines {before_start}..{before_end}) ---\n"
                    f"```{lang}\n" + "\n".join(before_lines) + "\n```"
                )

            parts.append(
                f"--- Complete Associated Function (Lines {func_start}..{func_end}) ---\n"
                f"```{lang}\n{func_body}\n```"
            )

            if after_lines:
                parts.append(
                    f"--- Context After Function (Up to 15 lines: Lines {after_start}..{after_end}) ---\n"
                    f"```{lang}\n" + "\n".join(after_lines) + "\n```"
                )

            context_sections.append("\n\n".join(parts))

    # If there are changed lines outside any detected function (or no functions detected):
    if not function_scopes:
        target_lines = unmatched_lines if unmatched_lines else changed_line_numbers
        win_start = max(1, min(target_lines) - 15)
        win_end = min(total_lines, max(target_lines) + 15)
        win_lines = [lines_dict[ln] for ln in range(win_start, win_end + 1) if ln in lines_dict]

        fallback_block = (
            f"FILE: {file_path}\n"
            f"[Scope: Lines {win_start}..{win_end} (Fallback Window - Exact Function Boundaries Not Determined)]\n"
            f"```{lang}\n" + "\n".join(win_lines) + "\n```"
        )
        context_sections.append(fallback_block)
        is_guaranteed = False
    elif unmatched_lines:
        win_start = max(1, min(unmatched_lines) - 15)
        win_end = min(total_lines, max(unmatched_lines) + 15)
        win_lines = [lines_dict[ln] for ln in range(win_start, win_end + 1) if ln in lines_dict]

        fallback_block = (
            f"FILE: {file_path}\n"
            f"[Scope: Lines {win_start}..{win_end} (Fallback Window - Additional Changes Outside Functions)]\n"
            f"```{lang}\n" + "\n".join(win_lines) + "\n```"
        )
        context_sections.append(fallback_block)

    # Changed code snippet
    ch_code_lines = [lines_dict[ln] for ln in sorted(set(changed_line_numbers)) if ln in lines_dict]
    if ch_code_lines:
        changed_code_text = f"FILE: {file_path}\n```{lang}\n" + "\n".join(ch_code_lines) + "\n```"
    else:
        changed_code_text = file_diff or "(no diff available)"

    relevant_context_text = "\n\n========================================\n\n".join(context_sections)
    return changed_code_text, relevant_context_text, is_guaranteed
