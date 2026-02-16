import difflib
import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProcessOptions:
    auto_edit: bool = True
    remove_fillers: bool = True
    remove_habits: bool = True


@dataclass
class ProcessResult:
    preprocessed_text: str
    final_text: str
    edits: List[str]
    fallback_reason: str | None = None


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _append_terminal_punctuation(text: str) -> str:
    if not text:
        return text
    if text[-1] in "。．.!！？?":
        return text
    return f"{text}。"


def _remove_fillers(text: str, filler_words: List[str]) -> str:
    if not filler_words:
        return text
    pattern = r"|".join(re.escape(word) for word in filler_words if word)
    if not pattern:
        return text
    result = re.sub(pattern, "", text)
    return _normalize_whitespace(result)


def _remove_habits(text: str, habit_patterns: List[Dict[str, str]]) -> str:
    output = text
    for rule in habit_patterns:
        pattern = rule.get("pattern", "")
        replace = rule.get("replace", "")
        if not pattern:
            continue
        output = re.sub(pattern, replace, output)
    return _normalize_whitespace(output)


def _auto_edit(text: str) -> str:
    text = _normalize_whitespace(text)
    text = _append_terminal_punctuation(text)
    return text


def create_edit_list(before: str, after: str, max_items: int = 8) -> List[str]:
    if before == after:
        return []

    edits: List[str] = []
    matcher = difflib.SequenceMatcher(a=before, b=after)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old = before[i1:i2].strip()
        new = after[j1:j2].strip()
        if tag == "replace" and old and new:
            edits.append(f"{old} -> {new}")
        elif tag == "delete" and old:
            edits.append(f"- {old}")
        elif tag == "insert" and new:
            edits.append(f"+ {new}")

        if len(edits) >= max_items:
            break

    return edits


def process_text(
    raw_text: str,
    rules: Dict[str, List[Dict[str, str]]],
    options: ProcessOptions,
) -> ProcessResult:
    output = raw_text
    if options.remove_fillers:
        output = _remove_fillers(output, rules.get("filler_words", []))
    if options.remove_habits:
        output = _remove_habits(output, rules.get("habit_patterns", []))
    if options.auto_edit:
        output = _auto_edit(output)

    return ProcessResult(
        preprocessed_text=output,
        final_text=output,
        edits=create_edit_list(raw_text, output),
    )


def create_diff_text(before: str, after: str) -> str:
    before_lines = before.splitlines() or [before]
    after_lines = after.splitlines() or [after]
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile="raw_text",
        tofile="final_text",
        lineterm="",
    )
    return "\n".join(diff)
