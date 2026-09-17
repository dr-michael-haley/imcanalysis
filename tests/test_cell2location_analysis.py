from __future__ import annotations

import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from SpatialBiologyToolkit.cell2location_analysis import (
    BARCODE,
    PRIOR_MEAN,
    attach_cell_count_prior,
    align_mapping_genes,
    prepare_reference,
    prepare_visium,
    validate_counts,
    save_result,
    Cell2locationResult,
)
from SpatialBiologyToolkit.cell2location_contract import preflight_errors
from SpatialBiologyToolkit.config import Cell2locationConfig, VisiumInputConfig
from SpatialBiologyToolkit.pipeline.project import (
    initialize_project,
    load_project,
    stage_readiness,
)
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.registry import get_stage, MODES


def sample(library="A", genes=None):
    genes = genes or ["g1", "g2", "g3", "MT-A"]
    x = sparse.csr_matrix(np.array([[1, 2, 3, 1], [4, 3, 2, 1]], dtype=np.float32))
    data = ad.AnnData(
        x,
        obs=pd.DataFrame(
            {"library_id": [library] * 2, "cell_type": ["T", "B"], "imc": [0, 8]},
            index=["bc1", "bc2"],
        ),
        var=pd.DataFrame(index=genes),
    )
    data.obsm["spatial"] = np.array([[10, 20], [20, 30]], dtype=float)
    data.uns["spatial"] = {library: {"scalefactors": {"tissue_hires_scalef": 0.5}}}
    return data


def test_multi_file_equivalent_to_combined_and_preserves_identity():
    settings = Cell2locationConfig(min_shared_genes=2)
    first, second = sample("A"), sample("B")
    multiple = prepare_visium([first, second], settings)
    combined = ad.concat([first, second], uns_merge="unique")
    combined.obs_names = ["bc1", "bc2", "bc1", "bc2"]
    single = prepare_visium([combined], settings)
    np.testing.assert_array_equal(multiple.X.toarray(), single.X.toarray())
    assert multiple.obs_names.tolist() == single.obs_names.tolist()
    assert multiple.obs_names.is_unique
    assert multiple.obs[BARCODE].tolist() == ["bc1", "bc2", "bc1", "bc2"]
    assert set(multiple.uns["spatial"]) == {"A", "B"}
    assert "MT-A" not in multiple.var_names
    assert first.n_vars == 4
    assert not any(k.startswith("_sbt_c2l_") for k in first.obs)


@pytest.mark.parametrize(
    "matrix", [np.array([[1, -1]]), np.array([[np.nan, 0]]), np.array([[1.1, 0]])]
)
def test_invalid_counts_rejected(matrix):
    with pytest.raises(ValueError):
        validate_counts(sparse.csr_matrix(matrix), "test")


def test_raw_layer_and_gene_ids_are_explicit():
    source = sample()
    source.layers["counts"] = source.X.copy()
    source.X = np.log1p(source.X.toarray())
    source.var["ids"] = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
    source.var["symbols"] = ["g1", "g2", "g3", "MT-A"]
    settings = Cell2locationConfig(
        reference_counts_layer="counts",
        reference_gene_id_key="ids",
        gene_symbol_key="symbols",
    )
    result = prepare_reference(source, settings)
    assert result.var_names.tolist() == ["ENSG1", "ENSG2", "ENSG3"]
    with pytest.raises(ValueError, match="raw integer counts"):
        prepare_reference(source, Cell2locationConfig())
    source.var["ids"] = ["same"] * 4
    with pytest.raises(ValueError, match="duplicate gene IDs"):
        prepare_reference(source, settings)


def test_alignment_uses_gene_ids_and_detects_empty_overlap():
    vis = prepare_visium([sample()], Cell2locationConfig())
    signatures = pd.DataFrame({"T": [2, 5], "B": [3, 1]}, index=["g3", "g1"])
    result, aligned = align_mapping_genes(vis, signatures, min_shared_genes=2)
    assert result.var_names.tolist() == ["g1", "g3"]
    assert aligned.index.equals(result.var_names)
    assert aligned.loc["g1", "T"] == 5
    with pytest.raises(ValueError, match="shared expressed genes"):
        align_mapping_genes(vis, signatures, min_shared_genes=3)


def test_prior_alignment_by_library_and_barcode_not_row_order():
    settings = Cell2locationConfig(
        cell_count_prior_scale=2, cell_count_prior_floor=0.25
    )
    vis = prepare_visium([sample("A"), sample("B")], settings)
    counts = pd.DataFrame(
        {
            "library_id": ["B", "A", "B", "A", "X"],
            "barcode": ["bc2", "bc1", "bc1", "bc2", "unused"],
            "n_cells": [9, 0, 4, 5, 20],
        }
    )
    result = attach_cell_count_prior(vis, settings, counts=counts)
    assert result.obs[PRIOR_MEAN].tolist() == [0.25, 10, 8, 18]
    assert result.uns["sbt_cell2location"]["prior_floored_spots"] == 1
    assert result.uns["sbt_cell2location"]["prior_extra_table_rows"] == 1
    with pytest.raises(ValueError, match="missing 1 spots"):
        attach_cell_count_prior(vis, settings, counts=counts.iloc[1:])
    with pytest.raises(ValueError, match="duplicate"):
        attach_cell_count_prior(vis, settings, counts=pd.concat([counts, counts]))
    counts.loc[0, "n_cells"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        attach_cell_count_prior(vis, settings, counts=counts)


def test_prior_obs_and_missing_library_are_not_silently_imputed():
    settings = Cell2locationConfig(cell_count_prior_obs_key="imc")
    second = sample("B")
    second.obs.drop(columns="imc", inplace=True)
    vis = prepare_visium([sample("A"), second], settings)
    with pytest.raises(ValueError, match="finite"):
        attach_cell_count_prior(vis, settings)


def test_duplicate_libraries_and_conflicting_metadata_fail():
    with pytest.raises(ValueError, match="multiple files"):
        prepare_visium([sample(), sample()], Cell2locationConfig())
    with pytest.raises(ValueError, match="disagrees"):
        prepare_visium([sample()], Cell2locationConfig(), library_ids=["Z"])


def test_config_and_preflight_action_boundaries(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        Cell2locationConfig(
            cell_count_prior_csv="a.csv", cell_count_prior_obs_key="imc"
        )
    settings = Cell2locationConfig(action="reference")
    (tmp_path / "reference.h5ad").touch()
    assert not preflight_errors(settings, tmp_path)
    settings = settings.model_copy(
        update={"action": "map", "visium_inputs": [VisiumInputConfig(path="vis.h5ad")]}
    )
    errors = preflight_errors(settings, tmp_path)
    assert len(errors) == 2  # Visium and saved signatures, not scRNA reference
    assert all("reference.h5ad" not in e for e in errors)


def test_planning_is_standalone_and_checks_all_visium_files(tmp_path):
    config = {
        "cell2location": {
            "action": "full",
            "visium_inputs": [
                {"path": "one.h5ad", "library_id": "A"},
                {"path": "two.h5ad", "library_id": "B"},
            ],
        }
    }
    initialize_project(tmp_path)
    original_index = (tmp_path / ".sbt" / "executions.yaml").read_bytes()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    for name in ["reference.h5ad", "one.h5ad"]:
        (tmp_path / name).touch()
    context = load_project(tmp_path)
    plan = build_run_plan(context, ["cell2location"])
    assert not plan.ready
    assert any("two.h5ad" in e for e in plan.errors)
    (tmp_path / "two.h5ad").touch()
    plan = build_run_plan(context, ["cell2location"])
    assert plan.ready, plan.errors
    assert [s.name for s in plan.resolved_stages] == ["cell2location"]
    assert get_stage("cell2location").environment_keys == ["cell2location"]
    assert all("cell2location" not in mode.stages for mode in MODES)
    assert (tmp_path / ".sbt" / "executions.yaml").read_bytes() == original_index
    from typer.testing import CliRunner
    from SpatialBiologyToolkit.cli.main import app

    preview = CliRunner().invoke(
        app, ["run", "cell2location", "--project", str(tmp_path), "--dry-run"]
    )
    assert preview.exit_code == 0, preview.stdout
    assert "job_cell2location.sh" in preview.stdout
    assert (tmp_path / ".sbt" / "executions.yaml").read_bytes() == original_index
    (tmp_path / "cell2location" / "mapping").mkdir(parents=True)
    ready, errors = stage_readiness(
        get_stage("cell2location"),
        resolve_assets(context.config, tmp_path),
        context=context,
    )
    assert not ready and any("already exists" in e for e in errors)


def test_save_never_overwrites(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        save_result(Cell2locationResult(None, sample()), destination)


def test_notebook_prior_cannot_silently_fall_back_to_global():
    data = prepare_visium([sample()], Cell2locationConfig())
    with pytest.raises(ValueError, match="counts DataFrame"):
        attach_cell_count_prior(
            data, Cell2locationConfig(cell_count_prior_csv="counts.csv")
        )
    informed = attach_cell_count_prior(
        data, Cell2locationConfig(cell_count_prior_obs_key="imc")
    )
    global_prior = attach_cell_count_prior(informed, Cell2locationConfig())
    assert PRIOR_MEAN not in global_prior.obs


def test_cli_help_does_not_import_scientific_stack():
    code = "from SpatialBiologyToolkit.cli.main import app; import sys; assert not {'torch','cell2location','scanpy','anndata'}.intersection(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())
