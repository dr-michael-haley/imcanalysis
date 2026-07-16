import os
import yaml
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict, is_dataclass
from collections.abc import MutableMapping

from pydantic import BaseModel

from SpatialBiologyToolkit.config.models import (  # noqa: F401 - legacy re-exports
    BasicProcessConfig,
    BatchIntegrationConfig,
    BioBatchNetConfig,
    CellCharterConfig,
    CreateMasksConfig,
    DEFAULT_CONFIG_CLASSES,
    DenoisingConfig,
    GeneralConfig,
    LoggingConfig,
    NetworkxSpatialConfig,
    NimbusConfig,
    PairwiseSpatialConfig,
    PreprocessConfig,
    RapidsProcessConfig,
    RebuildMetadataConfig,
    RemapObsConfig,
    SegmentationConfig,
    StarlingConfig,
    SubclusteringConfig,
    VisualizationConfig,
)


def generate_default_config_dict() -> Dict[str, Any]:
    """
    Generate defaults from the authoritative Pydantic section models.
    """
    defaults = {}
    for section, cls in DEFAULT_CONFIG_CLASSES.items():
        defaults[section] = cls().model_dump(mode='python')
    return defaults

def filter_config_for_dataclass(config_dict: Dict[str, Any], dataclass_type) -> Dict[str, Any]:
    """
    Filter a config dictionary to fields accepted by a config model.

    The historical function name is retained because pipeline stages import it
    directly. Both Pydantic models and legacy dataclasses are supported.
    Log warnings for any unexpected keys.
    
    Parameters:
    config_dict: Dictionary containing configuration values
    dataclass_type: The dataclass type to filter for
    
    Returns:
    Filtered dictionary with only valid keys for the dataclass
    """
    if hasattr(dataclass_type, 'model_fields'):
        valid_fields = set(dataclass_type.model_fields.keys())
    elif hasattr(dataclass_type, '__dataclass_fields__'):
        valid_fields = set(dataclass_type.__dataclass_fields__.keys())
    else:
        # Fallback: create a temporary instance and get its attributes
        temp_instance = dataclass_type()
        valid_fields = set(temp_instance.__dict__.keys())
    
    filtered_config = {}
    dataclass_name = dataclass_type.__name__
    
    for key, value in config_dict.items():
        if key in valid_fields:
            filtered_config[key] = value
        else:
            logging.warning(f"Ignoring unrecognized config key '{key}' = {value} in {dataclass_name} configuration section. Please check if this key belongs in a different config section.")
    
    return filtered_config

def deep_merge_defaults(config: Dict[str, Any], defaults: Dict[str, Any]) -> bool:
    """
    Recursively merge default values into config. If a key from defaults is not present in config,
    it is added. If a key is present but is a dictionary, we recurse.

    Returns True if changes were made to the config, False otherwise.
    """
    changed = False
    for key, default_value in defaults.items():
        if key not in config:
            # Key missing, add it
            config[key] = default_value
            changed = True
        else:
            # If both are dicts, recurse
            if isinstance(default_value, dict) and isinstance(config[key], dict):
                if deep_merge_defaults(config[key], default_value):
                    changed = True
            # If default_value is not a dict but config[key] is missing keys, this case is handled above
            # If config[key] is already set and not a dict, we do not overwrite existing keys
            # because we assume user config is correct. If we want to always overwrite with defaults
            # if user config is missing fields, we rely on the dictionary recursion above.
    return changed

def load_config(config_file: str = 'config.yaml') -> Dict[str, Any]:
    """
    Legacy dictionary loader using defaults from the Pydantic models.

    If the file does not exist, create it with all default values.
    If fields are missing, add them and update the file.

    New code should use ``SpatialBiologyToolkit.config.load_config`` for typed,
    validating, non-mutating loading.

    Returns the fully populated config dictionary.
    """
    defaults = generate_default_config_dict()

    if not os.path.isfile(config_file):
        # File not found, create it with defaults
        with open(config_file, 'w') as f:
            yaml.safe_dump(defaults, f, default_flow_style=False)
        logging.info(f'Configuration file "{config_file}" not found. Created and saved with default values.')
        return defaults

    # If file exists, load it
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f) or {}

    # Merge defaults into config if any keys missing
    changed = deep_merge_defaults(config, defaults)

    # Save if any changes were made
    if changed:
        with open(config_file, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logging.info(f'Configuration file "{config_file}" updated with default values for missing keys.')

    return config

def setup_logging(logging_config, pipeline_stage):
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper(), logging.INFO)
    log_file = logging_config.get('log_file', 'pipeline.log')
    to_console = logging_config.get('to_console', True)
    console_only = logging_config.get('console_only', False)
    prevent_duplicate = logging_config.get('prevent_duplicate_console', True)
    use_custom_format = logging_config.get('use_custom_format', True)
    
    # Clear any existing handlers to prevent accumulation
    root_logger = logging.getLogger()
    if prevent_duplicate:
        root_logger.handlers.clear()
    
    # Set root logger level
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        f'%(asctime)s [%(levelname)s] [{pipeline_stage}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ) if use_custom_format else logging.Formatter()
    
    if not console_only:
        # Add file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    if to_console:
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate messages if requested
    if prevent_duplicate:
        root_logger.propagate = False


def _normalize_stage_run_mode(mode: Optional[str]) -> str:
    mode_text = str(mode or "intelligent").strip().lower()
    if mode_text not in {"repeat", "skip", "intelligent"}:
        logging.warning(
            "Unknown general.anndata_stage_run_mode='%s'. Falling back to 'intelligent'.",
            mode,
        )
        return "intelligent"
    return mode_text


def _collect_slurm_context_from_env() -> Dict[str, str]:
    """
    Collect SLURM job metadata from environment variables.
    Prefer IMC_* aliases set by job scripts, with SLURM_* as fallback.
    """
    job_id = os.getenv("IMC_SLURM_JOB_ID") or os.getenv("SLURM_JOB_ID")
    job_name = os.getenv("IMC_SLURM_JOB_NAME") or os.getenv("SLURM_JOB_NAME")

    slurm: Dict[str, str] = {}
    if job_id is not None and str(job_id).strip():
        slurm["job_id"] = str(job_id).strip()
    if job_name is not None and str(job_name).strip():
        slurm["job_name"] = str(job_name).strip()
    return slurm


def _sanitize_uns_key(key: Any) -> str:
    """Sanitize dictionary keys for safe storage in AnnData .uns/HDF5."""
    key_text = str(key)
    if "/" in key_text:
        key_text = key_text.replace("/", "__slash__")
    return key_text


def _is_null_like_for_uns(value: Any) -> bool:
    """Return True for null-like values that often break cross-version AnnData I/O."""
    if value is None:
        return True

    # pandas sentinel nulls
    try:
        import pandas as pd  # local import to avoid hard dependency at module import time

        if value is pd.NA or value is pd.NaT:
            return True
    except Exception:
        pass

    # numpy masked sentinel
    try:
        import numpy as np

        if value is np.ma.masked:
            return True
    except Exception:
        pass

    return False


def _contains_null_like_object_array(value: Any) -> bool:
    """Detect object-dtype arrays containing null-like values."""
    try:
        import numpy as np

        if not isinstance(value, np.ndarray) or value.dtype != object:
            return False
        for item in value.ravel():
            if _is_null_like_for_uns(item):
                return True
    except Exception:
        return False
    return False


def _sanitize_uns_payload(
    value: Any,
    *,
    max_depth: int = 50,
    _depth: int = 0,
) -> Tuple[Any, int]:
    """
    Recursively sanitize payloads for adata.uns storage.
    Returns (cleaned_value, removed_item_count).
    """
    if _depth > max_depth:
        return value, 0

    if _is_null_like_for_uns(value):
        return None, 1

    if isinstance(value, Path):
        return str(value), 0

    if isinstance(value, BaseModel):
        value = value.model_dump(mode='python')
    elif is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        removed = 0
        for key, item in value.items():
            cleaned, removed_count = _sanitize_uns_payload(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            removed += removed_count
            if cleaned is None:
                continue
            out[_sanitize_uns_key(key)] = cleaned
        return out, removed

    if isinstance(value, list):
        out_list: List[Any] = []
        removed = 0
        for item in value:
            cleaned, removed_count = _sanitize_uns_payload(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            removed += removed_count
            if cleaned is None:
                continue
            out_list.append(cleaned)
        return out_list, removed

    if isinstance(value, tuple):
        out_tuple: List[Any] = []
        removed = 0
        for item in value:
            cleaned, removed_count = _sanitize_uns_payload(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            removed += removed_count
            if cleaned is None:
                continue
            out_tuple.append(cleaned)
        return out_tuple, removed

    if isinstance(value, set):
        out_set_as_list: List[Any] = []
        removed = 0
        for item in value:
            cleaned, removed_count = _sanitize_uns_payload(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            removed += removed_count
            if cleaned is None:
                continue
            out_set_as_list.append(cleaned)
        return out_set_as_list, removed

    # Object arrays containing null-like values can trigger 'null' encoding on write.
    if _contains_null_like_object_array(value):
        return None, 1

    # Handle NumPy scalars without importing numpy at module import time.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item(), 0
        except Exception:
            pass

    return value, 0


def _sanitize_anndata_uns_inplace(adata: Any, *, max_depth: int = 50) -> int:
    """
    Clean adata.uns in-place to improve backward compatibility across anndata versions.
    Removes null-like values and unsupported nested payloads.
    """
    uns_obj = getattr(adata, "uns", None)
    if not isinstance(uns_obj, MutableMapping):
        return 0

    cleaned_uns, removed = _sanitize_uns_payload(dict(uns_obj), max_depth=max_depth)
    if not isinstance(cleaned_uns, dict):
        cleaned_uns = {}

    try:
        uns_obj.clear()
        uns_obj.update(cleaned_uns)
    except Exception:
        adata.uns = cleaned_uns

    return int(removed)


def _h5_attr_to_text(value: Any) -> str:
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", "ignore")
        if hasattr(value, "item") and callable(getattr(value, "item")):
            item_value = value.item()
            if isinstance(item_value, bytes):
                return item_value.decode("utf-8", "ignore")
            return str(item_value)
        return str(value)
    except Exception:
        return str(value)


def _is_h5_node_null_encoded(node: Any) -> bool:
    for attr_key in ("encoding-type", "encoding_type"):
        try:
            if attr_key in node.attrs:
                attr_val = _h5_attr_to_text(node.attrs[attr_key]).strip().lower()
                if attr_val == "null":
                    return True
        except Exception:
            continue
    return False


def _collect_null_encoded_h5_paths(group: Any, prefix: str) -> List[str]:
    paths: List[str] = []
    try:
        import h5py
    except Exception:
        return paths

    for name in list(group.keys()):
        child = group[name]
        child_path = f"{prefix}/{name}"
        if _is_h5_node_null_encoded(child):
            paths.append(child_path)
            # Whole node will be deleted, so skip recursion into this subtree.
            continue
        if isinstance(child, h5py.Group):
            paths.extend(_collect_null_encoded_h5_paths(child, child_path))
    return paths


def _remove_null_encoded_uns_entries_in_h5ad(anndata_path: Path) -> List[str]:
    """
    In-place repair for files containing 'null' encoded datasets under /uns.
    Returns a list of removed HDF5 paths.
    """
    import h5py

    removed_paths: List[str] = []
    with h5py.File(anndata_path, "r+") as handle:
        if "uns" not in handle:
            return removed_paths

        paths_to_remove = _collect_null_encoded_h5_paths(handle["uns"], "/uns")
        for path in sorted(set(paths_to_remove), key=lambda p: p.count("/"), reverse=True):
            parent_path, leaf = path.rsplit("/", 1)
            parent = handle[parent_path.lstrip("/")] if parent_path and parent_path != "/" else handle
            if leaf in parent:
                del parent[leaf]
                removed_paths.append(path)

    return removed_paths


def _looks_like_null_encoding_read_error(exc: Exception) -> bool:
    msg = str(exc)
    if "encoding_type='null'" in msg or 'encoding_type="null"' in msg:
        return True
    if "No read method registered for IOSpec" in msg and "null" in msg:
        return True
    return False


def _sanitize_for_uns(value: Any) -> Any:
    """Recursively sanitize values for safe storage in adata.uns and drop None entries."""
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, BaseModel):
        value = value.model_dump(mode='python')
    elif is_dataclass(value):
        value = asdict(value)

    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            cleaned = _sanitize_for_uns(item)
            if cleaned is None:
                continue
            out[_sanitize_uns_key(key)] = cleaned
        return out

    if isinstance(value, (list, tuple, set)):
        out_list: List[Any] = []
        iterable = value
        if isinstance(value, set):
            # Keep set handling deterministic for stable stage snapshot comparison.
            iterable = sorted(value, key=lambda x: str(x))
        for item in iterable:
            cleaned = _sanitize_for_uns(item)
            if cleaned is None:
                continue
            out_list.append(cleaned)
        return out_list

    # Handle array-like payloads (e.g., numpy arrays, pandas index/series) by
    # converting to python-native lists recursively.
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _sanitize_for_uns(value.tolist())
        except Exception:
            pass

    # Handle NumPy scalars without importing numpy at module import time.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass

    return value


def build_uns_config_snapshot(config_obj: Any) -> Dict[str, Any]:
    """
    Build a sanitized config snapshot suitable for adata.uns.
    All None/null values are removed recursively.
    """
    cleaned = _sanitize_for_uns(config_obj)
    if cleaned is None:
        return {}
    if isinstance(cleaned, dict):
        return cleaned
    return {"value": cleaned}


def _is_nan_like(value: Any) -> bool:
    try:
        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False


def _safe_snapshot_equal(left: Any, right: Any) -> bool:
    """
    Robust deep equality for stage snapshots.
    Handles nested dict/list payloads and treats NaN values as equal.
    """
    if _is_nan_like(left) and _is_nan_like(right):
        return True

    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_safe_snapshot_equal(left[k], right[k]) for k in left.keys())

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_safe_snapshot_equal(lv, rv) for lv, rv in zip(left, right))

    # Fallback to normalized list representation for remaining array-like objects.
    if hasattr(left, "tolist") and callable(getattr(left, "tolist")):
        try:
            left = left.tolist()
        except Exception:
            pass
    if hasattr(right, "tolist") and callable(getattr(right, "tolist")):
        try:
            right = right.tolist()
        except Exception:
            pass

    try:
        return left == right
    except Exception:
        return str(left) == str(right)


def coalesce_config_text(*values: Any, default: Optional[str] = None) -> Optional[str]:
    """
    Return the first non-empty string-like value from a list of candidates.
    """
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def coalesce_config_list(*values: Any, default: Optional[List[str]] = None) -> Optional[List[str]]:
    """
    Return the first non-null list-like value from a list of candidates as a list of strings.
    """
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        return [str(value)]
    return default


def resolve_anndata_path(
    general_config: GeneralConfig,
    override_path: Optional[str] = None,
) -> Path:
    target = override_path if override_path else general_config.anndata_path
    return Path(str(target))


def _get_stage_log_container(adata: Any, uns_key: str) -> Dict[str, Any]:
    container = adata.uns.get(uns_key)
    if not isinstance(container, dict):
        container = {}
    if not isinstance(container.get("stage_order"), list):
        container["stage_order"] = []
    run_log_raw = container.get("run_log")
    migrated: Dict[str, Any] = {}
    if isinstance(run_log_raw, dict):
        for idx, item in enumerate(run_log_raw.values(), start=1):
            run_key = f"run_{idx:06d}"
            cleaned = _sanitize_for_uns(item)
            if isinstance(cleaned, dict):
                migrated[run_key] = cleaned
            elif cleaned is not None:
                migrated[run_key] = {"value": cleaned}
    elif isinstance(run_log_raw, list):
        # Migrate legacy list-based run logs (can fail HDF5 serialization) to dict form.
        for idx, item in enumerate(run_log_raw, start=1):
            run_key = f"run_{idx:06d}"
            cleaned = _sanitize_for_uns(item)
            if isinstance(cleaned, dict):
                migrated[run_key] = cleaned
            elif cleaned is not None:
                migrated[run_key] = {"value": cleaned}
    container["run_log"] = migrated
    if not isinstance(container.get("stages"), dict):
        container["stages"] = {}
    adata.uns[uns_key] = container
    return container


def get_stage_run_record(
    adata: Any,
    general_config: GeneralConfig,
    stage_name: str,
) -> Optional[Dict[str, Any]]:
    uns_key = str(general_config.anndata_uns_log_key)
    container = _get_stage_log_container(adata, uns_key)
    record = container.get("stages", {}).get(str(stage_name))
    if isinstance(record, dict):
        return record
    return None


def should_run_stage(
    adata: Any,
    general_config: GeneralConfig,
    stage_name: str,
    stage_config: Optional[Any] = None,
) -> Tuple[bool, str]:
    """
    Decide whether a stage should run based on general.anndata_stage_run_mode.
    """
    mode = _normalize_stage_run_mode(getattr(general_config, "anndata_stage_run_mode", "intelligent"))
    record = get_stage_run_record(adata, general_config, stage_name)
    if record is None:
        return True, f"Stage '{stage_name}' has no previous run record."

    if mode == "repeat":
        return True, "general.anndata_stage_run_mode=repeat."

    if mode == "skip":
        return False, f"Stage '{stage_name}' already recorded and mode=skip."

    current_snapshot = build_uns_config_snapshot(stage_config)
    previous_snapshot = build_uns_config_snapshot(record.get("config", {}))
    if _safe_snapshot_equal(current_snapshot, previous_snapshot):
        return (
            False,
            f"Stage '{stage_name}' already recorded with matching config and mode=intelligent.",
        )
    return (
        True,
        f"Stage '{stage_name}' config changed since last run; mode=intelligent so it will run again.",
    )


def load_pipeline_anndata(
    *,
    general_config: GeneralConfig,
    stage_name: str,
    stage_config: Optional[Any] = None,
    override_path: Optional[str] = None,
    allow_missing: bool = False,
) -> Tuple[Optional[Any], Path, bool, str]:
    """
    Standardized AnnData loader with stage-run decision logic.

    Returns
    -------
    tuple
        (adata_or_none, resolved_path, skip_stage, decision_message)
    """
    import anndata as ad

    anndata_path = resolve_anndata_path(general_config, override_path=override_path)
    if not anndata_path.exists():
        if allow_missing:
            msg = f"AnnData not found at {anndata_path}; proceeding because allow_missing=True."
            logging.info(msg)
            return None, anndata_path, False, msg
        raise FileNotFoundError(f"AnnData file not found: {anndata_path}")

    logging.info("Loading AnnData from %s", anndata_path)
    try:
        adata = ad.read_h5ad(anndata_path)
    except Exception as exc:
        if not _looks_like_null_encoding_read_error(exc):
            raise

        logging.warning(
            "AnnData read failed due null-encoded payloads (likely from newer anndata/scanpy). "
            "Attempting in-place repair of /uns in %s.",
            anndata_path,
        )
        removed_paths = _remove_null_encoded_uns_entries_in_h5ad(anndata_path)
        if not removed_paths:
            raise

        preview = ", ".join(removed_paths[:5])
        if len(removed_paths) > 5:
            preview += ", ..."
        logging.warning(
            "Removed %d null-encoded /uns entries from %s: %s",
            len(removed_paths),
            anndata_path,
            preview,
        )
        adata = ad.read_h5ad(anndata_path)

    removed_from_uns = _sanitize_anndata_uns_inplace(adata)
    if removed_from_uns > 0:
        logging.warning(
            "Removed %d null-like entries from adata.uns after load for compatibility.",
            removed_from_uns,
        )
    should_run, reason = should_run_stage(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=stage_config,
    )
    skip_stage = not should_run
    logging.info("Stage decision for '%s': %s", stage_name, reason)
    return adata, anndata_path, skip_stage, reason


def record_stage_run_in_uns(
    *,
    adata: Any,
    general_config: GeneralConfig,
    stage_name: str,
    stage_config: Optional[Any] = None,
    extra_details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record a stage run in adata.uns using the configured pipeline log key.
    """
    uns_key = str(general_config.anndata_uns_log_key)
    container = _get_stage_log_container(adata, uns_key)
    stage_name = str(stage_name)
    timestamp = datetime.now(timezone.utc).isoformat()

    stage_snapshot = build_uns_config_snapshot(stage_config)
    detail_snapshot = build_uns_config_snapshot(extra_details) if extra_details is not None else {}
    slurm_snapshot = build_uns_config_snapshot(_collect_slurm_context_from_env())

    container["stage_order"].append(stage_name)
    run_event: Dict[str, Any] = {"stage": stage_name, "run_utc": timestamp}
    if stage_snapshot:
        run_event["config"] = stage_snapshot
    if detail_snapshot:
        run_event["details"] = detail_snapshot
    if slurm_snapshot:
        run_event["slurm"] = slurm_snapshot
    run_log = container.get("run_log")
    if not isinstance(run_log, dict):
        run_log = {}
        container["run_log"] = run_log
    run_idx = len(run_log) + 1
    run_key = f"run_{run_idx:06d}"
    run_log[run_key] = run_event

    entry: Dict[str, Any] = {"last_run_utc": timestamp}
    if stage_snapshot:
        entry["config"] = stage_snapshot
    if detail_snapshot:
        entry["details"] = detail_snapshot
    if slurm_snapshot:
        entry["slurm"] = slurm_snapshot
    container["stages"][stage_name] = entry
    adata.uns[uns_key] = container


def save_pipeline_anndata(
    *,
    adata: Any,
    general_config: GeneralConfig,
    stage_name: str,
    stage_config: Optional[Any] = None,
    override_path: Optional[str] = None,
    extra_details: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Record stage metadata in adata.uns and save AnnData to the canonical path.
    """
    target_path = resolve_anndata_path(general_config, override_path=override_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    record_stage_run_in_uns(
        adata=adata,
        general_config=general_config,
        stage_name=stage_name,
        stage_config=stage_config,
        extra_details=extra_details,
    )

    removed_from_uns = _sanitize_anndata_uns_inplace(adata)
    if removed_from_uns > 0:
        logging.warning(
            "Removed %d null-like entries from adata.uns before save for backward compatibility.",
            removed_from_uns,
        )

    try:
        adata.write_h5ad(target_path)
    except Exception as exc:
        logging.warning(
            "Initial AnnData write failed (%s). Retrying after additional uns sanitization.",
            exc,
        )
        removed_retry = _sanitize_anndata_uns_inplace(adata, max_depth=100)
        if removed_retry > 0:
            logging.warning(
                "Removed %d additional null-like uns entries before write retry.",
                removed_retry,
            )
        adata.write_h5ad(target_path)
    logging.info("Saved AnnData to %s", target_path)
    return target_path


def get_filename(path: Path, name: str) -> str:
    """
    Retrieves a filename from the specified directory that contains a specific substring.
    """
    files = [x.name for x in path.iterdir() if name in x.name]

    if len(files) == 0:
        raise FileNotFoundError(f"No file {name} found in {path}")
    elif len(files) > 1:
        raise ValueError(f"More than one file or image in {str(path)} matches {name}")
    else:
        return files[0]

def update_config_file(config_file: str, updates: Dict[str, Any]) -> None:
    """
    Update the YAML configuration file with the given updates.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
            config = {}

    config.update(updates)

    with open(config_file, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    logging.info(f'Configuration file "{config_file}" updated with: {updates}')

def apply_override(config: Dict, key_path: str, value: str) -> None:
    keys = key_path.split('.')
    d = config
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]

    if ',' in value:
        value = value.split(',')
    d[keys[-1]] = value

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the pipeline with overrides.")
    default_config = os.environ.get('SBT_CONFIG', 'config.yaml')
    parser.add_argument(
        '--config',
        type=str,
        default=default_config,
        help=(
            'Path to the config (default: SBT_CONFIG when set, otherwise '
            'config.yaml)'
        ),
    )
    parser.add_argument('--override', action='append', help='Overrides in key=value format. Use dot-notation for keys.')
    return parser.parse_args()

def process_config_with_overrides():
    args = parse_arguments()

    # Load config with default merging
    config = load_config(args.config)

    # Apply overrides
    if args.override:
        for ov in args.override:
            if '=' not in ov:
                logging.warning(f"Invalid override (no '=' found): {ov}")
                continue
            key_path, value = ov.split('=', 1)
            apply_override(config, key_path.strip(), value.strip())

        # If overrides potentially added new keys not in defaults, we could re-run
        # deep_merge_defaults if desired. But since we only wanted to ensure old configs
        # get updated, this may not be necessary.

        # Save config after overrides?
        with open(args.config, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logging.info(f'Configuration file "{args.config}" updated with overrides.')

    return config

def create_config(config_class, **overrides):
    """
    Create a configuration object with defaults and optional overrides.
    
    This is useful for programmatically creating config objects when using
    individual functions from the pipeline outside of the main scripts.
    
    Parameters
    ----------
    config_class : type
        The Pydantic configuration model to instantiate (e.g., GeneralConfig,
        VisualizationConfig)
    **overrides : dict
        Keyword arguments to override default values
    
    Returns
    -------
    config object
        Instance of the specified config class with applied overrides
    
    Examples
    --------
    >>> # Create a GeneralConfig with custom masks folder
    >>> general_cfg = create_config(GeneralConfig, masks_folder='custom_masks')
    
    >>> # Create a VisualizationConfig with specific settings
    >>> viz_cfg = create_config(
    ...     VisualizationConfig,
    ...     create_umaps=True,
    ...     create_tissue_overlays=True,
    ...     save_high_res=False
    ... )
    """
    filtered_overrides = filter_config_for_dataclass(overrides, config_class)
    return config_class(**filtered_overrides)


def cleanstring(data: Any) -> str:
    """
    Helper function that returns a clean string with underscores replacing non-word characters.

    Parameters
    ----------
    data : Any
        Input data to be cleaned.

    Returns
    -------
    str
        Cleaned string with underscores instead of special characters.
    """
    import re
    data = str(data)
    # Replace sequences of non-word characters (except underscores) with single underscores
    data = re.sub(r'[^\w]+', '_', data)
    # Remove leading/trailing underscores and collapse multiple underscores
    data = re.sub(r'^_+|_+$', '', data)
    data = re.sub(r'_+', '_', data)
    return data

