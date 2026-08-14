from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from SpatialBiologyToolkit.cellvision import load_normalization_dict
from SpatialBiologyToolkit.config.models import NimbusConfig
from SpatialBiologyToolkit.nimbus_normalization import (
    load_normalization_file,
    normalize_nimbus_image,
    resolve_normalization_input_path,
    resolve_normalization_parameters,
    write_normalization_csv,
)


def test_preferred_csv_round_trip_includes_zero_and_nonzero_lower_thresholds(
    tmp_path: Path,
):
    path = write_normalization_csv(
        tmp_path / "normalization_dict.csv",
        {"CD3": 20, "FOXP3": 5},
        {"FOXP3": 0.8},
    )
    assert path.read_text(encoding="utf-8").splitlines() == [
        "marker,vmax,lower_threshold",
        "CD3,20,0",
        "FOXP3,5,0.8",
    ]
    loaded = load_normalization_file(path)
    assert loaded["CD3"].vmax == 20
    assert loaded["CD3"].lower_threshold == 0
    assert loaded["FOXP3"].lower_threshold == pytest.approx(0.8)


def test_legacy_json_and_two_column_csv_default_lower_threshold_to_zero(
    tmp_path: Path,
):
    json_path = tmp_path / "normalization_dict.json"
    json_path.write_text(json.dumps({"CD3": "12.5"}), encoding="utf-8")
    assert load_normalization_file(json_path)["CD3"].lower_threshold == 0

    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text("Marker,Value\nCD20,8\n", encoding="utf-8")
    loaded = load_normalization_file(csv_path)
    assert loaded["CD20"].vmax == 8
    assert loaded["CD20"].lower_threshold == 0

    scan_path = tmp_path / "scan_parameters.csv"
    scan_path.write_text(
        "marker,baseline,lower_bound\nFOXP3,5,0.8\n",
        encoding="utf-8",
    )
    scan_values = load_normalization_file(scan_path)
    assert scan_values["FOXP3"].vmax == 5
    assert scan_values["FOXP3"].lower_threshold == pytest.approx(0.8)


def test_explicit_normalization_csv_path_takes_precedence_and_requires_csv(
    tmp_path: Path,
):
    output_dir = tmp_path / "nimbus_output"
    output_dir.mkdir()
    saved = write_normalization_csv(
        output_dir / "normalization_dict.csv", {"CD3": 10}
    )
    explicit = write_normalization_csv(tmp_path / "reviewed.csv", {"CD3": 20})

    assert resolve_normalization_input_path(
        output_dir,
        configured_path=explicit,
        reuse_saved=True,
    ) == explicit.resolve()
    assert resolve_normalization_input_path(
        output_dir, reuse_saved=True
    ) == saved
    assert resolve_normalization_input_path(output_dir, reuse_saved=False) is None

    legacy = tmp_path / "normalization_dict.json"
    legacy.write_text('{"CD3": 12}', encoding="utf-8")
    with pytest.raises(ValueError, match="must point to a .csv"):
        resolve_normalization_input_path(output_dir, configured_path=legacy)
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_normalization_input_path(
            output_dir, configured_path=tmp_path / "missing.csv"
        )


def test_nimbus_config_accepts_an_explicit_normalization_csv_path():
    assert NimbusConfig().normalization_dict_path is None
    configured = NimbusConfig(normalization_dict_path=" metadata/reviewed_norm.csv ")
    assert configured.normalization_dict_path == "metadata/reviewed_norm.csv"
    with pytest.raises(ValueError, match="must point to a .csv"):
        NimbusConfig(normalization_dict_path="metadata/legacy.json")


def test_normalization_bounds_are_validated_and_case_insensitively_resolved(
    tmp_path: Path,
):
    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "marker,vmax,lower_threshold\nCD3,2,2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be below vmax"):
        load_normalization_file(invalid)

    valid = tmp_path / "valid.csv"
    valid.write_text(
        "marker,vmax,lower_threshold\nfoxp3,5,0.5\n", encoding="utf-8"
    )
    resolved = resolve_normalization_parameters(
        load_normalization_file(valid), ["FOXP3"]
    )
    assert resolved["FOXP3"].vmax == 5
    with pytest.raises(ValueError, match="missing markers"):
        resolve_normalization_parameters(load_normalization_file(valid), ["CD3"])


def test_two_point_normalization_removes_background_and_zero_preserves_old_formula():
    image = np.asarray([0.0, 0.5, 1.0, 3.0, 5.0, 8.0], dtype=np.float32)
    thresholded = normalize_nimbus_image(
        image, vmax=5.0, lower_threshold=1.0
    )
    assert np.allclose(thresholded, [0, 0, 0, 0.5, 1, 1])

    legacy = np.clip(image / 5.0, 0, 1)
    assert np.array_equal(
        normalize_nimbus_image(image, vmax=5.0, lower_threshold=0.0), legacy
    )


def test_cellvision_can_read_vmax_from_preferred_csv(tmp_path: Path):
    path = tmp_path / "normalization_dict.csv"
    path.write_text(
        "marker,vmax,lower_threshold\nCD3,12.5,0.8\nCD20,8,0\n",
        encoding="utf-8",
    )
    assert load_normalization_dict(
        path,
        channel_names=["CD3", "CD20"],
    ) == {"CD3": 12.5, "CD20": 8.0}
