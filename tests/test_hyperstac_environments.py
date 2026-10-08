import numpy as np
import pandas as pd
import pytest
from SpatialBiologyToolkit.hyperstac.environments import (
    radius_graph,
    neighbour_means,
    aligned_metrics,
    summarize_metrics,
)
from SpatialBiologyToolkit.config.models import (
    HyperstacEnvironmentsConfig,
    PipelineConfig,
)


def test_graph_radius_roi_and_disconnected_gap():
    xy = np.array([[0, 0], [100, 0], [100, 100], [400, 0], [0, 0]])
    roi = ["a", "a", "a", "a", "b"]
    g = radius_graph(xy, roi, 150, 100).toarray()
    assert g[0, 2] == 1 and g[0, 4] == 0 and g[2, 3] == 0
    assert not np.diag(g).any()
    assert radius_graph(xy, roi, 100, 100)[0, 2] == 0
    assert radius_graph(xy, roi, 500, 100)[0, 3] == 0
    assert radius_graph(xy, roi, 0, 100).nnz == 0


def test_reject_duplicate_and_wrong_units():
    with pytest.raises(ValueError, match="Duplicate"):
        radius_graph([[0, 0], [0, 0]], ["a", "a"], 100, 100)
    with pytest.raises(ValueError, match="grid"):
        radius_graph([[0, 0], [1, 0]], ["a", "a"], 100, 100)


def test_means_exclude_self_and_missing_values():
    g = radius_graph([[0, 0], [100, 0], [200, 0], [500, 0]], ["a"] * 4, 100, 100)
    m = neighbour_means(g, [[2, 4], [100, np.nan], [6, 8], [0, 0]])
    np.testing.assert_allclose(m[1], [4, 6])
    assert np.isnan(m[3]).all() and np.isnan(m[0, 1])


def test_metric_alignment(tmp_path):
    import anndata as ad

    a = ad.AnnData(np.array([[2.0], [1.0]]), obs=pd.DataFrame(index=["b", "a"]))
    p = tmp_path / "a.h5ad"
    a.write_h5ad(p)
    assert aligned_metrics(p, ["a", "b"], "m_").iloc[:, 0].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="identity"):
        aligned_metrics(p, ["a", "c"], "m_")


def test_summaries_do_not_duplicate_neighbour_area():
    obs = pd.DataFrame(
        {"roi": ["a", "a", "b"], "case_id": ["c", "c", "d"], "ref": ["1", "2", "2"]}
    )
    values = pd.DataFrame({"marker": [0.0, 2.0, 10.0]})
    summary, strat = summarize_metrics(values, np.array(["0"] * 3), obs, "ref")
    assert summary.iloc[0]["mean"] == 4
    assert summary.iloc[0].roi_balanced_mean == 5.5
    assert strat.n_patches.sum() == 3


def test_config_and_registry():
    from SpatialBiologyToolkit.pipeline.registry import STAGE_REGISTRY

    assert PipelineConfig().hyperstac_environments.cpu_threads == 4
    assert STAGE_REGISTRY["hyperstac-environments"].environment_keys == ["analysis"]
    with pytest.raises(ValueError):
        HyperstacEnvironmentsConfig(radii_um=[100, 100])
    with pytest.raises(ValueError):
        HyperstacEnvironmentsConfig(n_clusters=[1])


def test_managed_command_and_assets(tmp_path):
    from SpatialBiologyToolkit.pipeline.assets import resolve_assets
    from SpatialBiologyToolkit.pipeline.commands import stage_commands

    c = PipelineConfig(hyperstac_environments={"case_mapping_csv": "cases.csv"})
    assets = {a.role: a for a in resolve_assets(c, tmp_path)}
    assert assets["hyperstac_environment_mapping"].kind == "file"
    assert stage_commands("hyperstac-environments")[0].environment_key == "analysis"


def test_small_end_to_end_and_resume(tmp_path, monkeypatch):
    import anndata as ad
    import SpatialBiologyToolkit.hyperstac.environments as mod

    root = tmp_path / "hyperstac"
    root.mkdir()
    (root / "permutation_sensitivity").mkdir()
    monkeypatch.setenv("SBT_PROJECT_ROOT", str(tmp_path))
    rng = np.random.default_rng(8)
    obs = pd.DataFrame(
        [
            {
                "roi": f"r{r}",
                "center_col_um": 50 + 100 * x,
                "center_row_um": 50 + 100 * y,
                "roi_width_um": 300.0,
                "roi_height_um": 300.0,
                "ref": str(x % 2),
            }
            for r in range(6)
            for y in range(3)
            for x in range(3)
        ],
        index=[f"p{i}" for i in range(54)],
    )
    x = rng.normal(size=(54, 5))
    x[:, 0] += obs.ref.astype(int) * 6
    a = ad.AnnData(x, obs=obs)
    source = root / "imc_hyperstac_representations.h5ad"
    a.write_h5ad(source)
    ad.AnnData(
        np.abs(x[:, :2]),
        obs=obs,
        var=pd.DataFrame(index=["mean_intensity_norm_a", "mean_intensity_norm_b"]),
    ).write_h5ad(root / "imc_hyperstac_patch_metrics.h5ad")
    ad.AnnData(
        np.abs(x[:, :2]),
        obs=obs,
        var=pd.DataFrame(index=["zero__a", "shuffle__a__rep01"]),
    ).write_h5ad(root / "permutation_sensitivity/imc_permutation_sensitivity.h5ad")
    pd.DataFrame(
        {"ROI": [f"r{i}" for i in range(6)], "Case": [f"c{i}" for i in range(6)]}
    ).to_csv(tmp_path / "cases.csv", index=False)

    def aggregate(features, obs, graph):
        return np.concatenate([features, mod.neighbour_means(graph, features)], axis=1)

    monkeypatch.setattr(mod, "aggregate_features", aggregate)

    class Reporter:
        def __getattr__(self, name):
            return lambda *a, **k: None

    c = PipelineConfig(
        hyperstac_environments={
            "reference_cluster": "ref",
            "case_mapping_csv": "cases.csv",
            "n_pcs": 2,
            "radii_um": [0, 100],
            "n_clusters": [2],
            "patient_repeats": 1,
            "fit_repeats": 2,
            "cpu_threads": 1,
        }
    )
    before = mod.file_hash(source)
    out = tmp_path / "env"
    mod.run(c, out, Reporter())
    mod.run(c, out, Reporter())
    result = ad.read_h5ad(out / "spatial_environments.h5ad")
    assert result.n_obs == 54 and "ref" in result.obs
    assert "env_r100_k2" in result.obs
    assert mod.file_hash(source) == before
    fractions = pd.read_csv(out / "r100/k2/case_patch_fractions.csv", index_col=0)
    np.testing.assert_allclose(fractions.sum(axis=1), 1)
    c.hyperstac_environments.seed = 9
    with pytest.raises(ValueError, match="different analysis"):
        mod.run(c, out, Reporter())
