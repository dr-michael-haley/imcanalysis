"""Documentation-backed help used by the Napari workflow and Sphinx guide."""

from __future__ import annotations

import re
from pathlib import Path

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _normalise_heading(value: str) -> str:
    cleaned = re.sub(r"[`*_]", "", str(value))
    return " ".join(cleaned.casefold().split())


def extract_help_section(markdown: str, section: str) -> str:
    """Extract one Markdown heading and its children from a help document."""

    requested = _normalise_heading(section)
    lines = str(markdown).splitlines()
    available: list[str] = []
    for start, line in enumerate(lines):
        match = _HEADING_PATTERN.match(line)
        if match is None:
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()
        available.append(heading)
        if _normalise_heading(heading) != requested:
            continue
        end = len(lines)
        for index in range(start + 1, len(lines)):
            following = _HEADING_PATTERN.match(lines[index])
            if following is not None and len(following.group(1)) <= level:
                end = index
                break
        return "\n".join(lines[start:end]).strip()
    raise KeyError(
        f"Help section {section!r} was not found. Available headings: {available}"
    )


def load_help_markdown(topic: str, section: str | None = None) -> str:
    """Load a complete external help document or one named section from it."""

    topic = str(topic).strip()
    if not topic or not re.fullmatch(r"[a-z0-9_]+", topic):
        raise ValueError(f"Invalid help topic: {topic!r}")
    help_path = Path(__file__).with_name(f"{topic}.md")
    if not help_path.is_file():
        raise FileNotFoundError(f"NapariSBT help is missing: {help_path}")
    markdown = help_path.read_text(encoding="utf-8")
    return extract_help_section(markdown, section) if section else markdown


__all__ = ["extract_help_section", "load_help_markdown"]
