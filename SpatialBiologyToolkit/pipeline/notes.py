"""Concurrency-safe project notes editing for lightweight interfaces."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path

from .manifests import utc_now, write_text
from .project import ProjectContext


NOTES_BACKUP_DIRECTORY = Path(".sbt/notes-backups")


class ProjectNotesChangedError(RuntimeError):
    """Raised when project notes changed after an edit session was opened."""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ProjectNotesSession:
    """Explicit-save session for the configured project notes file."""

    path: Path
    source_text: str
    source_hash: str

    @classmethod
    def open(cls, context: ProjectContext) -> "ProjectNotesSession":
        path = (context.root / context.project_metadata.notes_file).resolve(
            strict=False
        )
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        displayed = current or "# Project notes\n"
        return cls(path=path, source_text=displayed, source_hash=_hash(current))

    def save(self, text: str, project_root: str | Path) -> Path:
        current = self.path.read_text(encoding="utf-8") if self.path.is_file() else ""
        if _hash(current) != self.source_hash:
            raise ProjectNotesChangedError(
                "Project notes changed after they were opened. Reload before saving."
            )
        root = Path(project_root).expanduser().resolve(strict=False)
        token = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        backup = root / NOTES_BACKUP_DIRECTORY / f"{token}.md"
        write_text(backup, current)
        write_text(self.path, text)
        self.source_text = text
        self.source_hash = _hash(text)
        return backup


__all__ = [
    "NOTES_BACKUP_DIRECTORY",
    "ProjectNotesChangedError",
    "ProjectNotesSession",
]
