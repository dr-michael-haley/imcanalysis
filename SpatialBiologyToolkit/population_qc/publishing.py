"""End-of-workflow publication of reviewed population labels.

Assessment code keeps evidence and candidate annotations in memory.  This
dedicated publisher completes a population-QC workflow by replacing the chosen
posterior label column in the source H5AD and one SpatialData table element.
It stages the H5AD, performs exactly one SpatialData table-element write,
verifies both persisted targets, and records recovery/audit artifacts for the
slow Zarr operation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping

import pandas as pd


@dataclass(frozen=True)
class PosteriorPublicationConfig:
    """Inputs for the one-write posterior publication transaction."""

    zarr: Path
    table_name: str
    h5ad: Path
    mapping_csv: Path
    source_key: str
    output_key: str
    artifact_root: Path
    overwrite_output_column: bool = True
    heartbeat_seconds: int = 900


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _series_hash(series: pd.Series) -> str:
    values = pd.util.hash_pandas_object(series.astype(str), index=True).values
    return sha256(values.tobytes()).hexdigest()


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    """Return a JSON object when an existing publication receipt is readable."""

    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _publication_receipt_matches(
    receipt: Mapping[str, Any] | None,
    config: "PosteriorPublicationConfig",
    *,
    mapping_sha256: str,
) -> bool:
    """Check whether a receipt describes this exact requested publication."""

    if receipt is None:
        return False
    return (
        receipt.get("mapping_sha256") == mapping_sha256
        and receipt.get("zarr") == str(config.zarr)
        and receipt.get("h5ad") == str(config.h5ad)
        and receipt.get("table_name") == config.table_name
        and receipt.get("source_key") == config.source_key
        and receipt.get("output_key") == config.output_key
    )


def _persisted_labels_match(
    h5ad: Any,
    table: Any,
    *,
    config: "PosteriorPublicationConfig",
    mapping: Mapping[str, str],
    categories: list[str],
    source_hash: str,
) -> bool:
    """Return whether both persisted objects already contain the requested map."""

    if config.output_key not in h5ad.obs or config.output_key not in table.obs:
        return False
    if _series_hash(h5ad.obs[config.source_key]) != source_hash:
        return False
    if _series_hash(table.obs[config.source_key]) != source_hash:
        return False
    expected_h5ad = map_posterior_labels(
        h5ad.obs,
        source_key=config.source_key,
        output_key=config.output_key,
        mapping=mapping,
        categories=categories,
        overwrite_output_column=True,
    )
    expected_table = map_posterior_labels(
        table.obs,
        source_key=config.source_key,
        output_key=config.output_key,
        mapping=mapping,
        categories=categories,
        overwrite_output_column=True,
    )
    return (
        h5ad.obs[config.output_key].astype(str).equals(expected_h5ad.astype(str))
        and table.obs[config.output_key].astype(str).equals(expected_table.astype(str))
    )


class _ProgressReporter:
    """Write an immediate progress state and 15-minute heartbeats by default."""

    def __init__(self, path: Path, interval_seconds: int) -> None:
        if interval_seconds < 60:
            raise ValueError("heartbeat_seconds must be at least 60")
        self.path = path
        self.interval_seconds = interval_seconds
        self.state: dict[str, Any] = {"started_at": _utc_now(), "status": "starting"}
        self._stop = threading.Event()

    def update(self, status: str, **extra: Any) -> None:
        self.state.update({"status": status, "updated_at": _utc_now(), **extra})
        _write_json(self.path, self.state)

    @contextmanager
    def heartbeat(self, status: str) -> Iterator[None]:
        self.update(status)

        def write_heartbeats() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.update(status, heartbeat=True)

        self._stop.clear()
        worker = threading.Thread(target=write_heartbeats, daemon=True)
        worker.start()
        try:
            yield
        finally:
            self._stop.set()
            worker.join(timeout=5)


def _read_mapping(path: Path) -> tuple[dict[str, str], list[str]]:
    mapping = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"source_population", "proposed_label"}
    missing = sorted(required - set(mapping.columns))
    if missing:
        raise ValueError(f"Mapping is missing required columns: {missing}")
    if mapping["source_population"].duplicated().any():
        duplicates = mapping.loc[
            mapping["source_population"].duplicated(), "source_population"
        ].tolist()
        raise ValueError(f"Mapping has duplicate source populations: {duplicates}")
    labels = list(dict.fromkeys(mapping["proposed_label"].tolist()))
    if not labels or any(not value for value in labels):
        raise ValueError("Mapping contains an empty proposed_label")
    return dict(zip(mapping["source_population"], mapping["proposed_label"])), labels


def map_posterior_labels(
    observations: pd.DataFrame,
    *,
    source_key: str,
    output_key: str,
    mapping: Mapping[str, str],
    categories: list[str],
    overwrite_output_column: bool = True,
) -> pd.Series:
    """Create one ordered posterior label series without mutating observations."""

    if source_key not in observations:
        raise KeyError(f"Source population column is absent: {source_key}")
    if output_key == source_key:
        raise ValueError(
            "output_key must differ from source_key so the source population "
            "column remains unchanged"
        )
    if output_key in observations and not overwrite_output_column:
        raise ValueError(
            f"Output column already exists: {output_key}. "
            "Choose a new output key or set overwrite_output_column=False to "
            "reject replacement of the existing output column."
        )
    source = observations[source_key]
    if source.isna().any():
        raise ValueError("Source population column contains missing values")
    values = source.astype(str)
    missing = sorted(set(values) - set(mapping))
    if missing:
        raise ValueError(f"Mapping does not cover source populations: {missing}")
    mapped = values.map(mapping)
    if mapped.isna().any():  # defensive; coverage was checked above
        raise AssertionError("Posterior mapping unexpectedly produced missing labels")
    return pd.Series(
        pd.Categorical(mapped, categories=categories, ordered=True),
        index=observations.index,
        name=output_key,
    )


def publish_posterior_mapping(
    config: PosteriorPublicationConfig,
) -> dict[str, Any]:
    """Publish a reviewed mapping to H5AD and one SpatialData table element.

    This named API is the end-of-workflow publication operation. Existing
    values in ``config.output_key`` are replaced by default; the source
    population column, images, labels, and unrelated tables are never changed.
    """
    config = PosteriorPublicationConfig(
        **{
            key: Path(value).expanduser().resolve()
            if key in {"zarr", "h5ad", "mapping_csv", "artifact_root"}
            else value
            for key, value in asdict(config).items()
        }
    )
    if not config.zarr.exists():
        raise FileNotFoundError(f"SpatialData Zarr does not exist: {config.zarr}")
    if not config.h5ad.exists():
        raise FileNotFoundError(f"Source H5AD does not exist: {config.h5ad}")
    if not config.mapping_csv.exists():
        raise FileNotFoundError(f"Posterior mapping does not exist: {config.mapping_csv}")

    import anndata as ad
    from spatialdata import read_zarr

    config.artifact_root.mkdir(parents=True, exist_ok=True)
    progress = _ProgressReporter(
        config.artifact_root / "manifests" / "finalization_progress.json",
        config.heartbeat_seconds,
    )
    manifest_path = config.artifact_root / "manifests" / "posterior_finalization.json"
    observation_path = config.artifact_root / "tables" / "posterior_observation_labels.csv"
    staged_h5ad = config.h5ad.with_name(
        f".{config.h5ad.stem}.{config.output_key}.partial{config.h5ad.suffix}"
    )
    if staged_h5ad.exists():
        raise FileExistsError(
            f"Staged H5AD already exists; inspect it before retrying: {staged_h5ad}"
        )

    mapping, categories = _read_mapping(config.mapping_csv)
    mapping_sha256 = _file_sha256(config.mapping_csv)
    progress.update("preflight", mapping_sha256=mapping_sha256)
    h5ad = ad.read_h5ad(config.h5ad)
    sdata = read_zarr(config.zarr)
    if config.table_name not in sdata.tables:
        raise KeyError(f"SpatialData table is absent: {config.table_name}")
    table = sdata.tables[config.table_name]
    if not h5ad.obs_names.equals(table.obs_names):
        raise ValueError(
            "H5AD and SpatialData table obs_names must match exactly, including order"
        )
    h5ad_source = h5ad.obs[config.source_key].copy()
    table_source = table.obs[config.source_key].copy()
    prior_receipt = _load_json_mapping(manifest_path)
    source_hash = _series_hash(table_source)
    if _publication_receipt_matches(
        prior_receipt,
        config,
        mapping_sha256=mapping_sha256,
    ) and _persisted_labels_match(
        h5ad,
        table,
        config=config,
        mapping=mapping,
        categories=categories,
        source_hash=source_hash,
    ):
        progress.update("already_published", manifest=str(manifest_path))
        return {
            **prior_receipt,
            "publication_status": "already_published",
            "this_invocation_zarr_write_count": 0,
        }
    h5ad.obs[config.output_key] = map_posterior_labels(
        h5ad.obs,
        source_key=config.source_key,
        output_key=config.output_key,
        mapping=mapping,
        categories=categories,
        overwrite_output_column=config.overwrite_output_column,
    )
    table.obs[config.output_key] = map_posterior_labels(
        table.obs,
        source_key=config.source_key,
        output_key=config.output_key,
        mapping=mapping,
        categories=categories,
        overwrite_output_column=config.overwrite_output_column,
    )
    if not h5ad.obs[config.source_key].equals(h5ad_source):
        raise AssertionError("H5AD source population column changed in memory")
    if not table.obs[config.source_key].equals(table_source):
        raise AssertionError("SpatialData source population column changed in memory")
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "obs_name": h5ad.obs_names.astype(str),
            config.source_key: h5ad.obs[config.source_key].astype(str).to_numpy(),
            config.output_key: h5ad.obs[config.output_key].astype(str).to_numpy(),
        }
    ).to_csv(observation_path, index=False, encoding="utf-8")

    with progress.heartbeat("staging_h5ad"):
        h5ad.write_h5ad(staged_h5ad)
    staged = ad.read_h5ad(staged_h5ad)
    if not staged.obs[config.output_key].astype(str).equals(
        h5ad.obs[config.output_key].astype(str)
    ):
        raise AssertionError("Staged H5AD read-back did not preserve posterior labels")

    # Intentionally the only Zarr write: the selected table element, at the end.
    with progress.heartbeat("writing_spatialdata_table"):
        sdata.write_element(config.table_name, overwrite=True)

    progress.update("validating_zarr_readback")
    written = read_zarr(config.zarr).tables[config.table_name]
    if config.output_key not in written.obs:
        raise AssertionError("Posterior column is absent after SpatialData write")
    if not written.obs[config.source_key].equals(table_source):
        raise AssertionError("Source population column changed during SpatialData write")
    if not written.obs[config.output_key].astype(str).equals(
        table.obs[config.output_key].astype(str)
    ):
        raise AssertionError("SpatialData posterior labels differ after read-back")

    progress.update("committing_h5ad")
    staged_h5ad.replace(config.h5ad)
    manifest = {
        "completed_at": _utc_now(),
        "zarr": str(config.zarr),
        "table_name": config.table_name,
        "h5ad": str(config.h5ad),
        "mapping_csv": str(config.mapping_csv),
        "mapping_sha256": mapping_sha256,
        "source_key": config.source_key,
        "output_key": config.output_key,
        "n_observations": int(written.n_obs),
        "n_posterior_categories": len(categories),
        "source_population_hash": _series_hash(written.obs[config.source_key]),
        "posterior_population_hash": _series_hash(written.obs[config.output_key]),
        "source_population_column_preserved": True,
        "h5ad_readback_verified": True,
        "zarr_readback_verified": True,
        "zarr_write_count": 1,
        "zarr_table_write_count": 1,
        "this_invocation_zarr_write_count": 1,
        "publication_status": "published",
        "observation_export": str(observation_path),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
    }
    _write_json(manifest_path, manifest)
    progress.update("complete", manifest=str(manifest_path))
    return manifest


__all__ = [
    "PosteriorPublicationConfig",
    "map_posterior_labels",
    "publish_posterior_mapping",
]
