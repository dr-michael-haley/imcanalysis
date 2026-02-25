"""
SLURM log organization stage.

This script reads SLURM job metadata recorded in AnnData pipeline logs and:
1. Finds matching `slurm-<job_id>.out` files in the current working directory.
2. Renames local files to `{order}_{stage}_{job_id}.out`.
3. Moves renamed files into `general.slurm_logs_folder`.
4. Marks files in `general.slurm_logs_folder` that do not match AnnData run records
   by appending `_Unverified` to the filename.
"""

from __future__ import annotations

import csv
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config_and_utils import (
    GeneralConfig,
    cleanstring,
    filter_config_for_dataclass,
    load_pipeline_anndata,
    process_config_with_overrides,
    save_pipeline_anndata,
    setup_logging,
)


_RUN_KEY_RE = re.compile(r"^run_(\d+)$")
_UNVERIFIED_RE = re.compile(r"^(?P<base>.+?)_Unverified(?:_\d+)?(?P<ext>\.out)$")


@dataclass
class SlurmRunRecord:
    order: int
    stage: str
    job_id: str
    job_name: str = ""

    @property
    def stage_clean(self) -> str:
        stage_name = cleanstring(self.stage)
        return stage_name if stage_name else "UnknownStage"

    @property
    def expected_filename(self) -> str:
        return f"{self.order:03d}_{self.stage_clean}_{self.job_id}.out"


def _sorted_run_log_items(run_log: Any) -> List[Tuple[str, Any]]:
    if isinstance(run_log, list):
        return [(f"run_{idx:06d}", item) for idx, item in enumerate(run_log, start=1)]

    if not isinstance(run_log, dict):
        return []

    numbered: List[Tuple[int, str, Any]] = []
    unnumbered: List[Tuple[str, Any]] = []
    for key, value in run_log.items():
        key_text = str(key)
        match = _RUN_KEY_RE.match(key_text)
        if match:
            numbered.append((int(match.group(1)), key_text, value))
        else:
            unnumbered.append((key_text, value))

    numbered_sorted = sorted(numbered, key=lambda x: x[0])
    out = [(key, value) for _, key, value in numbered_sorted]
    out.extend(unnumbered)
    return out


def _extract_slurm_run_records(adata: Any, uns_key: str) -> List[SlurmRunRecord]:
    container = adata.uns.get(uns_key)
    if not isinstance(container, dict):
        logging.warning(
            "AnnData.uns['%s'] missing or invalid; cannot parse SLURM run records.", uns_key
        )
        return []

    stage_order = container.get("stage_order")
    if not isinstance(stage_order, list):
        stage_order = []

    run_items = _sorted_run_log_items(container.get("run_log"))
    records: List[SlurmRunRecord] = []

    for index, (_, event) in enumerate(run_items, start=1):
        if not isinstance(event, dict):
            continue

        stage_raw = event.get("stage")
        if not stage_raw and (index - 1) < len(stage_order):
            stage_raw = stage_order[index - 1]
        stage = str(stage_raw) if stage_raw is not None else "UnknownStage"

        slurm = event.get("slurm")
        job_id: Optional[str] = None
        job_name: Optional[str] = None

        if isinstance(slurm, dict):
            if slurm.get("job_id") is not None:
                job_id = str(slurm.get("job_id")).strip()
            if slurm.get("job_name") is not None:
                job_name = str(slurm.get("job_name")).strip()

        if not job_id:
            if event.get("slurm_job_id") is not None:
                job_id = str(event.get("slurm_job_id")).strip()
            elif event.get("job_id") is not None:
                job_id = str(event.get("job_id")).strip()

        if not job_name:
            if event.get("slurm_job_name") is not None:
                job_name = str(event.get("slurm_job_name")).strip()
            elif event.get("job_name") is not None:
                job_name = str(event.get("job_name")).strip()

        if not job_id:
            continue

        records.append(
            SlurmRunRecord(
                order=index,
                stage=stage,
                job_id=job_id,
                job_name=job_name or "",
            )
        )

    return records


def _pick_local_slurm_file(current_dir: Path, record: SlurmRunRecord) -> Optional[Path]:
    expected = current_dir / record.expected_filename
    if expected.exists():
        return expected

    default_name = current_dir / f"slurm-{record.job_id}.out"
    if default_name.exists():
        return default_name

    candidates = sorted(
        p
        for p in current_dir.glob(f"*{record.job_id}*.out")
        if p.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        logging.warning(
            "Multiple local SLURM output candidates found for job id %s: %s",
            record.job_id,
            ", ".join(p.name for p in candidates),
        )
    return None


def _rename_local_file_to_expected(
    source_file: Path,
    current_dir: Path,
    record: SlurmRunRecord,
) -> Path:
    target = current_dir / record.expected_filename
    if source_file == target:
        return target

    if target.exists():
        logging.info(
            "Expected local filename already exists for job id %s: %s",
            record.job_id,
            target.name,
        )
        return target

    source_file.rename(target)
    logging.info("Renamed local SLURM output: %s -> %s", source_file.name, target.name)
    return target


def _move_to_log_folder(source_file: Path, log_dir: Path, target_name: str) -> Path:
    target = log_dir / target_name
    try:
        if source_file.resolve() == target.resolve():
            return target
    except Exception:
        pass
    if target.exists():
        target.unlink()
    shutil.move(str(source_file), str(target))
    return target


def _has_unverified_suffix(filename: str) -> bool:
    return _UNVERIFIED_RE.match(filename) is not None


def _remove_unverified_suffix(filename: str) -> str:
    match = _UNVERIFIED_RE.match(filename)
    if not match:
        return filename
    return f"{match.group('base')}{match.group('ext')}"


def _append_unverified_suffix(path: Path) -> Path:
    if _has_unverified_suffix(path.name):
        return path

    candidate = path.with_name(f"{path.stem}_Unverified{path.suffix}")
    suffix_idx = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_Unverified_{suffix_idx}{path.suffix}")
        suffix_idx += 1
    path.rename(candidate)
    return candidate


def _write_manifest(
    log_dir: Path,
    rows: Sequence[Dict[str, str]],
    filename: str = "slurmlogs_manifest.csv",
) -> Path:
    manifest_path = log_dir / filename
    fieldnames = [
        "order",
        "stage",
        "job_name",
        "job_id",
        "expected_filename",
        "local_status",
        "log_move_status",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return manifest_path


def run_slurm_log_organizer(
    *,
    general_config: GeneralConfig,
    stage_name: str = "SlurmLogs",
) -> Path:
    adata, anndata_path, _, _ = load_pipeline_anndata(
        general_config=general_config,
        stage_name=stage_name,
        stage_config={"slurm_logs_folder": general_config.slurm_logs_folder},
    )
    if adata is None:
        raise ValueError("AnnData could not be loaded for SLURM log organization.")

    run_records = _extract_slurm_run_records(adata, general_config.anndata_uns_log_key)
    if not run_records:
        logging.warning(
            "No SLURM run records found under adata.uns['%s']['run_log'].",
            general_config.anndata_uns_log_key,
        )

    current_dir = Path.cwd()
    log_dir = Path(general_config.slurm_logs_folder)
    log_dir.mkdir(parents=True, exist_ok=True)

    expected_filenames: List[str] = [record.expected_filename for record in run_records]
    manifest_rows: List[Dict[str, str]] = []
    renamed_count = 0
    moved_count = 0
    missing_local_count = 0

    for record in run_records:
        local_status = "missing_local_file"
        move_status = "not_moved"

        source_file = _pick_local_slurm_file(current_dir=current_dir, record=record)
        if source_file is None:
            missing_local_count += 1
            fallback_target = log_dir / record.expected_filename
            if fallback_target.exists():
                local_status = "not_in_cwd_already_in_log_dir"
            else:
                logging.warning(
                    "Could not find local SLURM output for job id %s (expected slurm-%s.out).",
                    record.job_id,
                    record.job_id,
                )
        else:
            renamed_file = _rename_local_file_to_expected(
                source_file=source_file,
                current_dir=current_dir,
                record=record,
            )
            if renamed_file.name == source_file.name:
                if source_file.name == record.expected_filename:
                    local_status = "already_renamed"
                else:
                    local_status = "kept_existing_expected_name"
            else:
                local_status = "renamed_local_file"
                renamed_count += 1

            moved_file = _move_to_log_folder(
                source_file=renamed_file,
                log_dir=log_dir,
                target_name=record.expected_filename,
            )
            if moved_file.exists():
                move_status = "moved_to_log_dir"
                moved_count += 1

        manifest_rows.append(
            {
                "order": str(record.order),
                "stage": record.stage,
                "job_name": record.job_name,
                "job_id": record.job_id,
                "expected_filename": record.expected_filename,
                "local_status": local_status,
                "log_move_status": move_status,
            }
        )

    expected_set = set(expected_filenames)

    restored_verified = 0
    for file_path in sorted(log_dir.glob("*.out")):
        if not _has_unverified_suffix(file_path.name):
            continue
        verified_name = _remove_unverified_suffix(file_path.name)
        if verified_name not in expected_set:
            continue
        verified_target = log_dir / verified_name
        if verified_target.exists():
            continue
        file_path.rename(verified_target)
        restored_verified += 1

    marked_unverified = 0
    for file_path in sorted(log_dir.glob("*.out")):
        if file_path.name in expected_set:
            continue
        if _has_unverified_suffix(file_path.name):
            continue
        new_path = _append_unverified_suffix(file_path)
        if new_path != file_path:
            marked_unverified += 1
            logging.warning(
                "Marked unmatched SLURM log as unverified: %s -> %s",
                file_path.name,
                new_path.name,
            )

    missing_in_log_dir = sorted(
        name for name in expected_set if not (log_dir / name).exists()
    )
    if missing_in_log_dir:
        preview = ", ".join(missing_in_log_dir[:10])
        if len(missing_in_log_dir) > 10:
            preview += ", ..."
        logging.warning(
            "Expected %d SLURM logs are still missing in %s: %s",
            len(missing_in_log_dir),
            log_dir,
            preview,
        )

    manifest_path = _write_manifest(log_dir=log_dir, rows=manifest_rows)

    logging.info(
        "SLURM log organization complete. records=%d, renamed_local=%d, moved=%d, "
        "missing_local=%d, restored_verified=%d, marked_unverified=%d, manifest=%s",
        len(run_records),
        renamed_count,
        moved_count,
        missing_local_count,
        restored_verified,
        marked_unverified,
        manifest_path,
    )

    save_pipeline_anndata(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config={"slurm_logs_folder": general_config.slurm_logs_folder},
        override_path=str(anndata_path),
        extra_details={
            "slurm_logs_folder": str(log_dir),
            "manifest_path": str(manifest_path),
            "records_found": int(len(run_records)),
            "renamed_local": int(renamed_count),
            "moved_to_log_dir": int(moved_count),
            "missing_local": int(missing_local_count),
            "restored_verified": int(restored_verified),
            "marked_unverified": int(marked_unverified),
            "missing_in_log_dir_count": int(len(missing_in_log_dir)),
        },
    )

    return log_dir


if __name__ == "__main__":
    pipeline_stage = "SlurmLogs"
    config = process_config_with_overrides()
    setup_logging(config.get("logging", {}), pipeline_stage)

    general_config = GeneralConfig(
        **filter_config_for_dataclass(config.get("general", {}), GeneralConfig)
    )
    output_dir = run_slurm_log_organizer(
        general_config=general_config,
        stage_name=pipeline_stage,
    )
    logging.info("SLURM log files organized in: %s", output_dir)
