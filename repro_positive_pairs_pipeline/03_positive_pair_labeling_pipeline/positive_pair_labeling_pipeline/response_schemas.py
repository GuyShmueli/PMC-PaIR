from __future__ import annotations

import re

from .errors import ResponseValidationError


_CITATION_RE = re.compile(
    r"\AReason for Citation:[ \t]*(?P<reason>\S[^\r\n]*)\r?\n"
    r"Possible Similarities:[ \t]*(?P<similarities>\S[^\r\n]*)\r?\n"
    r"Explanation:[ \t]*(?P<explanation>\S[\s\S]*)\Z"
)
_LEVEL_RE = re.compile(r"^Level[ \t]+(?P<level>[123])(?:\b.*)?$", re.IGNORECASE)
_REFUSAL_MARKERS = (
    "as an ai language model",
    "i cannot assist",
    "i can't assist",
    "i am unable to comply",
    "i'm unable to comply",
)


def _reject_refusal_text(text: str, *, stage: str) -> str:
    value = text.strip()
    lowered = value.casefold()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        raise ResponseValidationError(f"{stage} response appears to be a refusal")
    return value


def _level_sections(text: str, *, stage: str) -> dict[int, str]:
    lines = _reject_refusal_text(text, stage=stage).splitlines()
    headings: list[tuple[int, int]] = []
    for position, line in enumerate(lines):
        match = _LEVEL_RE.fullmatch(line.strip())
        if match is not None:
            headings.append((position, int(match.group("level"))))
    if [level for _, level in headings] != [1, 2, 3]:
        raise ResponseValidationError(
            f"{stage} response must contain exactly one ordered Level 1, Level 2, and Level 3 heading"
        )
    if any(line.strip() for line in lines[: headings[0][0]]):
        raise ResponseValidationError(f"{stage} response contains text before Level 1")
    sections: dict[int, str] = {}
    for heading_index, (line_index, level) in enumerate(headings):
        end = headings[heading_index + 1][0] if heading_index + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_index + 1 : end]).strip()
        if not body:
            raise ResponseValidationError(f"{stage} Level {level} section is empty")
        sections[level] = body
    return sections


def validate_intermediate_response(stage: str, text: str) -> None:
    if stage == "citation":
        value = _reject_refusal_text(text, stage=stage)
        if _CITATION_RE.fullmatch(value) is None:
            raise ResponseValidationError(
                "citation response must contain exactly the three requested labeled fields"
            )
        return
    if stage == "question":
        sections = _level_sections(text, stage=stage)
        for level, body in sections.items():
            if "?" not in body:
                raise ResponseValidationError(
                    f"question Level {level} section contains no question mark"
                )
        return
    if stage == "answer":
        _level_sections(text, stage=stage)
        return
    raise ValueError(f"No intermediate response schema exists for stage {stage!r}")


__all__ = ["validate_intermediate_response"]
