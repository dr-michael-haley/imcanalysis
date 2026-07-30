from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from SpatialBiologyToolkit.config import (
    MaxFuseConfig,
    MaxFuseGeneListConfig,
    PipelineConfig,
)
from SpatialBiologyToolkit.maxfuse_matching import (
    build_transfer_anndata,
    extract_target_unique_matching,
    prepare_maxfuse_inputs,
    read_feature_mapping,
    zscore_float32,
)
from SpatialBiologyToolkit.maxfuse_reports import (
    _population_coverage,
    concordance_tables,
    deterministic_stratified_indices,
    gene_list_enrichment,
    generate_maxfuse_report,
    load_gene_lists,
    plot_ranked_genes,
)
from SpatialBiologyToolkit.pipeline.assets import resolve_assets
from SpatialBiologyToolkit.pipeline.planner import build_run_plan
from SpatialBiologyToolkit.pipeline.project import initialize_project
from SpatialBiologyToolkit.pipeline.registry import MODES, STAGE_REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame, MaxFuseConfig]:
    rng = np.random.default_rng(11)
    reference = ad.AnnData(
        rng.gamma(2.0, 1.0, size=(50, 12)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "ref_label": np.repeat(["A", "B"], 25),
                "ref_state": np.tile(["S1", "S2"], 25),
            },
            index=[f"rna_{index}" for index in range(50)],
        ),
        var=pd.DataFrame(
            {
                "highly_variable": [True] * 8 + [False] * 4,
                "highly_variable_rank": list(range(8)) + [np.nan] * 4,
            },
            index=[f"g{index}" for index in range(12)],
        ),
    )
    reference.obsm["X_umap"] = rng.normal(size=(50, 2)).astype(np.float32)
    target = ad.AnnData(
        rng.gamma(2.0, 1.0, size=(80, 8)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "target_population": np.tile(["0", "1"], 40),
                "sample": np.repeat(["case1", "case2"], 40),
                "roi": np.repeat(["roi1", "roi2", "roi3", "roi4"], 20),
            },
            index=[f"imc_{index}" for index in range(80)],
        ),
        var=pd.DataFrame(index=[f"p{index}" for index in range(8)]),
    )
    target.obsm["X_umap"] = rng.normal(size=(80, 2)).astype(np.float32)
    mapping = pd.DataFrame(
        {
            "IMC": [f"p{index}" for index in range(6)],
            "snRNAseq": [f"g{index}" for index in range(6)],
            "All": [1] * 6,
        }
    )
    settings = MaxFuseConfig(
        reference_name="test_atlas",
        reference_smoothing_obs="ref_label",
        reference_transfer_obs=["ref_label", "ref_state"],
        target_population_obs="target_population",
        target_smoothing_obs="target_population",
        sample_obs="sample",
        roi_obs="roi",
        reference_active_features=8,
        reference_shared_sd_min=0,
        target_shared_sd_min=0,
        min_shared_features=6,
        graph_svd_reference=4,
        graph_svd_target=4,
        initial_svd_reference=4,
        initial_svd_target=4,
        refine_svd_reference=4,
        refine_svd_target=4,
        refine_cca_components=4,
        max_umap_points=1000,
        figure_dpi=80,
        figure_formats=["png"],
        plot_stacked_violin=False,
        deg_top_genes=5,
        deg_min_cells=5,
    )
    return reference, target, mapping, settings


def _matches(reference: ad.AnnData, target: ad.AnnData) -> pd.DataFrame:
    target_indices = np.arange(target.n_obs, dtype=np.int64)
    reference_indices = target_indices % reference.n_obs
    scores = np.linspace(0.4, 0.95, target.n_obs, dtype=np.float32)
    return pd.DataFrame(
        {
            "reference_index": reference_indices,
            "target_index": target_indices,
            "mod1_indx": reference_indices,
            "mod2_indx": target_indices,
            "score": scores,
            "match_source": np.where(target_indices < 30, "pivot", "propagated"),
            "reference_obs_name": reference.obs_names[reference_indices],
            "target_obs_name": target.obs_names[target_indices],
            "rna_originalindex": reference.obs_names[reference_indices],
            "protein_originalindex": target.obs_names[target_indices],
            "target_population": target.obs["target_population"].to_numpy(),
            "reference_ref_label": reference.obs.iloc[reference_indices][
                "ref_label"
            ].to_numpy(),
            "reference_ref_state": reference.obs.iloc[reference_indices][
                "ref_state"
            ].to_numpy(),
        }
    )


def _check_zscore_and_preparation_preserve_historical_feature_defaults():
    reference, target, mapping, settings = _inputs()
    prepared = prepare_maxfuse_inputs(reference, target, mapping, settings)

    assert prepared.reference_active.shape == (50, 8)
    assert prepared.target_active.shape == (80, 8)
    assert prepared.reference_shared.shape == (50, 6)
    assert prepared.target_shared.shape == (80, 6)
    assert prepared.feature_audit["decision"].eq("retain").all()
    assert np.allclose(prepared.reference_active.mean(axis=0), 0, atol=1e-5)
    standardized, keep = zscore_float32(
        np.array([[1, 2], [1, 4], [1, 6]], dtype=np.float32),
        return_keep=True,
    )
    assert keep.tolist() == [False, True]
    assert standardized.shape == (3, 1)


def _check_mapping_filter_and_validation(tmp_path: Path):
    path = tmp_path / "mapping.csv"
    pd.DataFrame(
        {
            "IMC": ["p1", "p2"],
            "snRNAseq": ["g1", "g2"],
            "All": [1, 0],
        }
    ).to_csv(path, index=False)

    result = read_feature_mapping(
        path,
        target_column="IMC",
        reference_column="snRNAseq",
        filter_column="All",
    )

    assert result[["IMC", "snRNAseq"]].to_dict("records") == [
        {"IMC": "p1", "snRNAseq": "g1"}
    ]


def _check_fast_target_unique_extraction_prefers_pivots_and_best_scores():
    fusor = SimpleNamespace(
        active_arr2=np.zeros((5, 2), dtype=np.float32),
        _pivot2_to_pivots1={0: [(1, 0.5), (2, 0.8)], 2: [(3, 0.7)]},
        _propidx2_to_propindices1={
            0: [(4, 0.99)],
            1: [(0, 0.4), (2, 0.6)],
            4: [(1, 0.9)],
        },
    )

    reference, target, scores, sources = extract_target_unique_matching(fusor)

    assert target.tolist() == [0, 1, 2, 4]
    assert reference.tolist() == [2, 2, 3, 1]
    np.testing.assert_allclose(scores, [0.8, 0.6, 0.7, 0.9])
    assert sources.tolist() == ["pivot", "propagated", "pivot", "propagated"]


def _check_transfer_asset_uses_population_qc_column_contract():
    reference, target, mapping, settings = _inputs()
    matches = _matches(reference, target)
    transfer = build_transfer_anndata(
        target,
        matches,
        mapping,
        settings,
        reference_path="reference.h5ad",
    )

    assert transfer.n_obs == target.n_obs
    assert transfer.n_vars == 0
    assert "test_atlas_maxfuse_score" in transfer.obs
    assert "test_atlas_ref_label" in transfer.obs
    source = transfer.uns["maxfuse"]["sources"]["test_atlas"]
    assert source["score_column"] == "test_atlas_maxfuse_score"
    assert source["label_columns"] == [
        "test_atlas_ref_label",
        "test_atlas_ref_state",
    ]
    assert transfer.uns["shared_proteins"] == mapping["IMC"].tolist()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "transfer.h5ad"
        transfer.write_h5ad(path)
        restored = ad.read_h5ad(path)
        assert restored.uns["maxfuse"]["sources"]["test_atlas"][
            "score_column"
        ] == "test_atlas_maxfuse_score"


def _check_concordance_and_sampling_are_auditable():
    reference, target, _mapping, _settings = _inputs()
    matches = _matches(reference, target)
    counts, fractions, score_summary = concordance_tables(
        matches,
        target_column="target_population",
        reference_column="reference_ref_label",
        score_threshold=0.3,
    )

    assert int(counts.to_numpy().sum()) == target.n_obs
    assert np.allclose(fractions.sum(axis=1), 1)
    assert set(score_summary.columns) == {"mean", "median", "count"}
    first = deterministic_stratified_indices(
        target.obs["target_population"],
        max_points=20,
        seed=7,
    )
    second = deterministic_stratified_indices(
        target.obs["target_population"],
        max_points=20,
        seed=7,
    )
    assert np.array_equal(first, second)
    assert len(first) == 20
    category_heavy = deterministic_stratified_indices(
        [f"category_{index}" for index in range(30)],
        max_points=10,
        seed=7,
    )
    assert len(category_heavy) == 10
    assert np.array_equal(
        category_heavy,
        deterministic_stratified_indices(
            [f"category_{index}" for index in range(30)],
            max_points=10,
            seed=7,
        ),
    )


def _check_population_coverage_keeps_unmatched_and_missing_populations(
    tmp_path: Path,
):
    import matplotlib.pyplot as plt

    reference, target, _mapping, settings = _inputs()
    target.obs["target_population"] = target.obs["target_population"].astype(object)
    target.obs.loc[target.obs_names[:5], "target_population"] = "unmatched"
    target.obs.loc[target.obs_names[5:7], "target_population"] = np.nan
    matches = _matches(reference, target).loc[
        lambda frame: ~frame["target_obs_name"].isin(target.obs_names[:7])
    ]
    artifacts = SimpleNamespace(figures=[], tables=[], warnings=[], metrics={})
    _population_coverage(
        matches,
        target,
        settings,
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        artifacts=artifacts,
    )
    coverage = pd.read_csv(
        tmp_path / "tables" / "population_match_coverage.csv",
        index_col=0,
    )
    assert coverage.loc["unmatched", "matched_cells"] == 0
    assert coverage.loc["Missing", "matched_cells"] == 0
    plt.close("all")


def _check_gene_lists_and_full_notebook_derived_report(tmp_path: Path):
    reference, target, mapping, settings = _inputs()
    gene_list_path = tmp_path / "programmes.csv"
    pd.DataFrame({"programme_a": ["g0", "g1"], "programme_b": ["g2", "g3"]}).to_csv(
        gene_list_path,
        index=False,
    )
    settings = settings.model_copy(
        update={
            "gene_lists": [
                MaxFuseGeneListConfig(
                    path=str(gene_list_path),
                    name="programmes",
                    format="wide",
                )
            ],
            "run_degs": False,
        }
    )
    normalized = load_gene_lists(
        settings.gene_lists,
        resolve_path=lambda value: Path(value),
    )
    assert set(normalized["group"]) == {"programme_a", "programme_b"}

    figures = tmp_path / "figures"
    tables = tmp_path / "tables"
    artifacts = generate_maxfuse_report(
        reference,
        target,
        _matches(reference, target),
        mapping,
        settings,
        figures_dir=figures,
        tables_dir=tables,
        resolve_path=lambda value: Path(value),
    )

    figure_names = {path.name for path in artifacts.figures}
    table_names = {path.name for path in artifacts.tables}
    assert "population_concordance_annotated.png" in figure_names
    assert "mean_matching_score_annotated.png" in figure_names
    assert "umap_target_space.png" in figure_names
    assert "umap_reference_space.png" in figure_names
    assert "linked_gene_matrixplot.png" in figure_names
    assert "population_concordance_row_fractions.csv" in table_names
    assert all(path.is_file() for path in artifacts.figures + artifacts.tables)

    degs = pd.DataFrame(
        {
            "group": ["0"] * 4 + ["1"] * 4,
            "rank": [1, 2, 3, 4] * 2,
            "names": ["g0", "g1", "g4", "g5", "g2", "g3", "g6", "g7"],
            "scores": [5, 4, 3, 2, 6, 5, 2, 1],
        }
    )
    enrichment = gene_list_enrichment(
        degs,
        normalized,
        universe=reference.var_names,
        top_n=4,
    )
    assert not enrichment.empty
    assert {"pvalue", "pvalue_adjusted", "overlap_genes"}.issubset(enrichment)
    ranked, legend = plot_ranked_genes(
        degs,
        top_n=4,
        panel_genes=mapping["snRNAseq"],
        gene_lists=normalized,
    )
    assert ranked is not None
    assert legend is not None


def _check_stage_environment_assets_wrapper_and_docs_are_aligned(tmp_path: Path):
    stage = STAGE_REGISTRY["maxfuse"]
    assert stage.catalogue_order == 38
    assert stage.depends_on == []
    assert stage.groups == []
    assert stage.environment_keys == ["maxfuse"]
    assert stage.config_sections == ["general", "maxfuse"]
    assert stage.requires_assets == [
        "maxfuse_reference",
        "maxfuse_target",
        "maxfuse_feature_mapping",
    ]
    assert all("maxfuse" not in mode.stages for mode in MODES)
    assert (REPO_ROOT / stage.documentation_path).is_file()
    wrapper = (REPO_ROOT / stage.slurm_script).read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=8" in wrapper
    assert "#SBATCH --mem=128G" in wrapper
    assert "#@ENV:  imc_maxfuse" in wrapper

    config = PipelineConfig()
    assets = {
        asset.role: asset
        for asset in resolve_assets(config, tmp_path)
    }
    assert assets["maxfuse_reference"].path == (
        tmp_path / "maxfuse" / "reference.h5ad"
    ).resolve()
    assert assets["maxfuse_target"].path == (tmp_path / "anndata.h5ad").resolve()
    assert assets["maxfuse_assets"].path == (
        tmp_path / "maxfuse_results"
    ).resolve()


def _check_temporary_project_maxfuse_plan_is_ready_without_submission(tmp_path: Path):
    context = initialize_project(tmp_path / "project")
    reference = context.root / context.config.maxfuse.reference_adata_path
    mapping = context.root / context.config.maxfuse.feature_mapping_path
    target = context.root / context.config.general.anndata_path
    reference.parent.mkdir(parents=True, exist_ok=True)
    mapping.parent.mkdir(parents=True, exist_ok=True)
    reference.write_bytes(b"placeholder")
    mapping.write_text("IMC,snRNAseq\np,g\n", encoding="utf-8")
    target.write_bytes(b"placeholder")

    plan = build_run_plan(context, ["maxfuse"])

    assert plan.ready, plan.errors
    assert [stage.name for stage in plan.resolved_stages] == ["maxfuse"]


class MaxFusePipelineTests(unittest.TestCase):
    def test_preparation(self):
        _check_zscore_and_preparation_preserve_historical_feature_defaults()

    def test_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_mapping_filter_and_validation(Path(temporary))

    def test_extraction(self):
        _check_fast_target_unique_extraction_prefers_pivots_and_best_scores()

    def test_transfer(self):
        _check_transfer_asset_uses_population_qc_column_contract()

    def test_concordance(self):
        _check_concordance_and_sampling_are_auditable()

    def test_population_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_population_coverage_keeps_unmatched_and_missing_populations(
                Path(temporary)
            )

    def test_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_gene_lists_and_full_notebook_derived_report(Path(temporary))

    def test_control_plane(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_stage_environment_assets_wrapper_and_docs_are_aligned(
                Path(temporary)
            )

    def test_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            _check_temporary_project_maxfuse_plan_is_ready_without_submission(
                Path(temporary)
            )


if __name__ == "__main__":
    unittest.main()
