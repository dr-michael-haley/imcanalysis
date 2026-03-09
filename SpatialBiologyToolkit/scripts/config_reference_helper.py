"""
Create and validate config_reference.md against config dataclass defaults.

Behavior:
1) If config_reference.md does not exist, create it with all config entries and
   default values plus TODO placeholders for descriptions.
2) If it exists, validate that all config entries are documented and flag:
   - missing entries
   - extra/stale entries
   - default value mismatches

Usage:
  python config_reference_helper.py
  python config_reference_helper.py --path /path/to/config_reference.md
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Import local config utilities from scripts folder.
sys.path.insert(0, str(Path(__file__).parent))
from config_and_utils import generate_default_config_dict  # noqa: E402


SECTION_RE = re.compile(r"^##\s+`?([A-Za-z0-9_.-]+)`?\s*$")
ENTRY_RE = re.compile(r"^###\s+`?([A-Za-z0-9_.-]+)`?\s*$")
DEFAULT_RE = re.compile(r"^-\s*Default:\s*`?(.*?)`?\s*$")


def _strip_wrapping_backticks(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def _escape_table_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def _unescape_table_cell(value: str) -> str:
    return (
        value.replace(r"\|", "|")
        .replace("<br />", "\n")
        .replace("<br/>", "\n")
        .replace("<br>", "\n")
    )


def _default_to_text(value: Any) -> str:
    """Stable textual representation for defaults in markdown."""
    return repr(value)


def _iter_expected_entries(defaults: Dict[str, Any]) -> List[Tuple[str, Any]]:
    entries: List[Tuple[str, Any]] = []
    for section, section_defaults in defaults.items():
        if not isinstance(section_defaults, dict):
            continue
        for field_name, default_value in section_defaults.items():
            entries.append((f"{section}.{field_name}", default_value))
    return entries


def _build_reference_markdown(defaults: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Config Reference",
        "",
        "Auto-generated from config dataclass defaults in `config_and_utils.py`.",
        "Keep section headings and table `Field`/`Default` columns intact for validation.",
        "Fill in `Description`, `Choices`, `Units`, and `Advice`.",
        "",
    ]

    for section, section_defaults in defaults.items():
        if not isinstance(section_defaults, dict):
            continue
        lines.append(f"## `{section}`")
        lines.append("")
        lines.append("| Field | Default | Description | Choices | Units | Advice |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for field_name, default_value in section_defaults.items():
            default_text = _escape_table_cell(_default_to_text(default_value))
            lines.append(
                f"| `{field_name}` | `{default_text}` | TODO | TODO | TODO | TODO |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _parse_reference_entries(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse markdown entries keyed by section.field.

    Supports:
      1) Table format under section headings:
         ## `section`
         | Field | Default | ... |
      2) Legacy heading format:
         ### `section.field`
         - Default: `<repr(default)>`
    """
    parsed: Dict[str, Dict[str, Any]] = {}
    current_section: str | None = None
    current_entry: str | None = None

    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        m_section = SECTION_RE.match(line)
        if m_section:
            current_section = m_section.group(1).strip()
            current_entry = None
            continue

        if current_section is not None and line.startswith("|"):
            cells = [
                cell.strip()
                for cell in re.split(r"(?<!\\)\|", line.strip("|"))
            ]
            if len(cells) >= 2:
                first_cell = _strip_wrapping_backticks(cells[0]).strip().lower()
                if first_cell in {"field", "---", ":---", "---:", ":---:"}:
                    continue
                if set(cells[0].replace(" ", "")) <= {"-", ":"}:
                    continue
                field_name = _strip_wrapping_backticks(cells[0]).strip()
                default_text = _unescape_table_cell(
                    _strip_wrapping_backticks(cells[1]).strip()
                )
                if field_name:
                    entry_name = f"{current_section}.{field_name}"
                    if entry_name not in parsed:
                        parsed[entry_name] = {"line": line_no, "default": default_text}
            continue

        m_entry = ENTRY_RE.match(line)
        if m_entry:
            current_entry = m_entry.group(1).strip()
            if current_entry not in parsed:
                parsed[current_entry] = {"line": line_no, "default": None}
            continue

        if current_entry is None:
            continue

        m_default = DEFAULT_RE.match(line)
        if m_default and parsed[current_entry]["default"] is None:
            parsed[current_entry]["default"] = m_default.group(1).strip()

    return parsed


def _validate_reference(
    reference_path: Path,
    defaults: Dict[str, Any],
) -> int:
    """
    Validate existing markdown reference.

    Returns:
      0 if valid
      1 if mismatches found
    """
    expected_entries = {
        entry_name: _default_to_text(default_value)
        for entry_name, default_value in _iter_expected_entries(defaults)
    }
    parsed_entries = _parse_reference_entries(reference_path)

    expected_keys = set(expected_entries.keys())
    documented_keys = set(parsed_entries.keys())

    missing_entries = sorted(expected_keys - documented_keys)
    extra_entries = sorted(documented_keys - expected_keys)

    default_mismatches: List[Tuple[str, str | None, str]] = []
    for key in sorted(expected_keys & documented_keys):
        current_default = parsed_entries[key].get("default")
        expected_default = expected_entries[key]
        if current_default is None:
            default_mismatches.append((key, None, expected_default))
            continue
        if current_default != expected_default:
            default_mismatches.append((key, str(current_default), expected_default))

    if not missing_entries and not extra_entries and not default_mismatches:
        logging.info("config_reference is up to date: %s", reference_path)
        return 0

    logging.warning("config_reference mismatches detected: %s", reference_path)

    if missing_entries:
        logging.warning("Missing entries in markdown (%d):", len(missing_entries))
        for entry in missing_entries:
            logging.warning("  - %s", entry)

    if extra_entries:
        logging.warning("Extra/stale entries in markdown (%d):", len(extra_entries))
        for entry in extra_entries:
            logging.warning("  - %s", entry)

    if default_mismatches:
        logging.warning("Entries with missing/mismatched defaults (%d):", len(default_mismatches))
        for entry, got_default, expected_default in default_mismatches:
            if got_default is None:
                logging.warning(
                    "  - %s: missing default value in reference (expected `%s`)",
                    entry,
                    expected_default,
                )
            else:
                logging.warning(
                    "  - %s: default mismatch (markdown `%s` vs expected `%s`)",
                    entry,
                    got_default,
                    expected_default,
                )

    logging.warning(
        "Tip: if you want a clean baseline, delete %s and rerun this script to regenerate it.",
        reference_path,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create/validate config_reference.md from config defaults."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(Path(__file__).parent / "config_reference.md"),
        help="Path to config reference markdown (default: scripts/config_reference.md).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    reference_path = Path(args.path)
    defaults = generate_default_config_dict()

    if not reference_path.exists():
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        content = _build_reference_markdown(defaults)
        reference_path.write_text(content, encoding="utf-8")
        logging.info("Created new config reference file: %s", reference_path)
        logging.info("Fill in Description/Choices/Units/Advice sections as needed.")
        return 0

    return _validate_reference(reference_path, defaults)


if __name__ == "__main__":
    raise SystemExit(main())
