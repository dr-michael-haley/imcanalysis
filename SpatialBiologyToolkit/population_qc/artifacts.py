"""Deterministic artifact export for notebook-based population assessment."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd

from .models import PlotResult


MANIFEST_COLUMNS = (
    "artifact_id",
    "stage",
    "population",
    "kind",
    "name",
    "path",
    "source",
    "created_at",
    "sha256",
    "metadata_json",
)


def _clean_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = token.strip("._-")
    if not token:
        raise ValueError("Artifact names and stage identifiers cannot be empty")
    return token


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PopulationQCArtifactWriter:
    """Save complete tables and figures beside staged assessment notebooks.

    The writer is intended for direct notebook workflows. It stores paths
    relative to ``output_root`` in a stable CSV manifest, updates an existing
    artifact record when the same stable identifier is saved again, and never
    writes to the source AnnData or SpatialData object.
    """

    def __init__(
        self,
        output_root: str | Path,
        *,
        stage: str,
        population: Any | None = None,
        manifest_path: str | Path = "manifests/artifacts.csv",
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.stage = _clean_token(stage)
        self.population = None if population is None else str(population)
        self.manifest_path = self._resolve_relative_path(manifest_path)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def child(
        self,
        *,
        stage: str,
        population: Any | None = None,
    ) -> "PopulationQCArtifactWriter":
        """Return a writer sharing the same root and manifest."""

        return type(self)(
            self.output_root,
            stage=stage,
            population=population,
            manifest_path=self.manifest_path.relative_to(self.output_root),
        )

    def save_table(
        self,
        frame: pd.DataFrame,
        name: str,
        *,
        category: str = "tables",
        index: bool = False,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save a dataframe as UTF-8 CSV and register it in the manifest."""

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        path = self._artifact_path(category, name, ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        frame.to_csv(temporary, index=index, encoding="utf-8")
        temporary.replace(path)
        self._register(
            path,
            kind="table",
            name=name,
            source=source,
            metadata={"rows": len(frame), "columns": list(map(str, frame.columns)), **dict(metadata or {})},
        )
        return path

    def save_figure(
        self,
        figure: Any,
        name: str,
        *,
        category: str = "figures",
        dpi: int = 200,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save a Matplotlib-compatible figure as PNG and register it."""

        if not hasattr(figure, "savefig"):
            raise TypeError("figure must provide a savefig() method")
        if dpi < 1:
            raise ValueError("dpi must be at least 1")
        path = self._artifact_path(category, name, ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        figure.savefig(temporary, dpi=dpi, bbox_inches="tight", format="png")
        temporary.replace(path)
        self._register(
            path,
            kind="figure",
            name=name,
            source=source,
            metadata={"dpi": int(dpi), **dict(metadata or {})},
        )
        return path

    def save_json(
        self,
        value: Mapping[str, Any] | list[Any] | tuple[Any, ...],
        name: str,
        *,
        category: str = "manifests",
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save JSON metadata atomically and register it."""

        path = self._artifact_path(category, name, ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self._register(
            path,
            kind="json",
            name=name,
            source=source,
            metadata=metadata,
        )
        return path

    def save_plot_result(
        self,
        result: PlotResult,
        name: str,
        *,
        category: str = "figures",
        table_category: str = "tables",
        dpi: int = 200,
        source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Save a plot, its full numerical data, and displayed matrix."""

        if not isinstance(result, PlotResult):
            raise TypeError("result must be a PlotResult")
        paths = {
            "figure": self.save_figure(
                result.figure,
                name,
                category=category,
                dpi=dpi,
                source=source,
                metadata=metadata,
            ),
            "data": self.save_table(
                result.data,
                f"{name}_data",
                category=table_category,
                index=True,
                source=source,
                metadata=metadata,
            ),
        }
        if result.display_data is not None:
            paths["display_data"] = self.save_table(
                result.display_data,
                f"{name}_display",
                category=table_category,
                index=True,
                source=source,
                metadata=metadata,
            )
        return paths

    def save_result_tables(
        self,
        result: Any,
        name: str,
        *,
        category: str = "tables",
        source: str = "",
    ) -> dict[str, Path]:
        """Save every dataframe field on a typed QC result without truncation."""

        if is_dataclass(result):
            values = asdict(result)
        elif hasattr(result, "__dict__"):
            values = vars(result)
        else:
            raise TypeError("result must be a dataclass or expose __dict__")
        paths: dict[str, Path] = {}
        metadata: dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, pd.DataFrame):
                paths[key] = self.save_table(
                    value,
                    f"{name}_{key}",
                    category=category,
                    index=True,
                    source=source or type(result).__name__,
                )
            elif key in {"parameters", "warnings"}:
                metadata[key] = value
        if metadata:
            paths["metadata"] = self.save_json(
                metadata,
                f"{name}_metadata",
                category="manifests",
                source=source or type(result).__name__,
            )
        return paths

    def manifest(self) -> pd.DataFrame:
        """Return the current manifest in canonical column order."""

        if not self.manifest_path.exists():
            return pd.DataFrame(columns=MANIFEST_COLUMNS)
        frame = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        for column in MANIFEST_COLUMNS:
            if column not in frame:
                frame[column] = ""
        return frame.loc[:, MANIFEST_COLUMNS]

    def _artifact_path(self, category: str, name: str, suffix: str) -> Path:
        parts = [_clean_token(category), self.stage]
        if self.population is not None:
            parts.append(f"population_{_clean_token(self.population)}")
        directory = self.output_root.joinpath(*parts)
        return directory / f"{_clean_token(name)}{suffix}"

    def _resolve_relative_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        resolved = (
            candidate.expanduser().resolve()
            if candidate.is_absolute()
            else (self.output_root / candidate).resolve()
        )
        try:
            resolved.relative_to(self.output_root)
        except ValueError as error:
            raise ValueError("Artifact paths must stay inside output_root") from error
        return resolved

    def _register(
        self,
        path: Path,
        *,
        kind: str,
        name: str,
        source: str,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        relative = path.relative_to(self.output_root).as_posix()
        population = "" if self.population is None else self.population
        artifact_id = ":".join(
            [
                self.stage,
                _clean_token(population) if population else "all",
                _clean_token(kind),
                _clean_token(name),
            ]
        )
        record = {
            "artifact_id": artifact_id,
            "stage": self.stage,
            "population": population,
            "kind": kind,
            "name": str(name),
            "path": relative,
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sha256": _file_sha256(path),
            "metadata_json": json.dumps(
                dict(metadata or {}),
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            ),
        }
        frame = self.manifest()
        if not frame.empty:
            frame = frame.loc[frame["artifact_id"] != artifact_id]
        frame = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
        frame = frame.sort_values(
            ["stage", "population", "kind", "name"],
            kind="stable",
        ).reset_index(drop=True)
        temporary = self.manifest_path.with_name(
            f".{self.manifest_path.stem}.tmp{self.manifest_path.suffix}"
        )
        frame.to_csv(temporary, index=False, encoding="utf-8")
        temporary.replace(self.manifest_path)


__all__ = ["MANIFEST_COLUMNS", "PopulationQCArtifactWriter"]
