"""Small scientific/identity fixtures for paired spot comparisons."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

from SpatialBiologyToolkit import visium_comparison as vc


def test_export_alignment_uses_library_and_barcode(tmp_path):
    qc = pd.DataFrame({"library_id": ["A", "B"], "barcode": ["same", "same"]}, index=["exportA", "exportB"])
    qc.to_csv(tmp_path / "mapping_spot_qc.csv")
    pd.DataFrame({"cell": [3, 9]}, index=qc.index).to_csv(tmp_path / "mapping_mean_abundance.csv.gz")
    obs = pd.DataFrame({"library_id": ["B", "A"], "Barcode": ["same", "same"]}, index=["newB", "newA"])
    abundance, aligned = vc.read_cell2location_tables(tmp_path, obs)
    assert abundance.cell.tolist() == [9, 3]
    assert abundance.index.tolist() == ["newB", "newA"]
    assert aligned.export_spot_id.tolist() == ["exportB", "exportA"]
    obs.loc["newB", "library_id"] = "missing"
    with pytest.raises(ValueError, match="absent"):
        vc.read_cell2location_tables(tmp_path, obs)


def test_mapping_additive_and_complete():
    x = pd.DataFrame([[2, 3, 1]], columns=["a", "b", "c"])
    mapping = pd.DataFrame({"annotation_granular": ["a", "b", "c"], "annotation_coarse": ["AB", "AB", "C"]})
    result = vc.aggregate_populations(x, mapping)
    assert result.iloc[0].tolist() == [5, 1]
    np.testing.assert_allclose(result.sum(axis=1), x.sum(axis=1))
    with pytest.raises(ValueError, match="Unmapped"):
        vc.aggregate_populations(x, mapping.iloc[:2])
    ambiguous = pd.concat([mapping, pd.DataFrame({"annotation_granular": ["a"], "annotation_coarse": ["C"]})])
    with pytest.raises(ValueError, match="exactly one"):
        vc.aggregate_populations(x, ambiguous)


def test_counting_excludes_unassigned_not_missing_labels():
    cells = pd.DataFrame({"spot": ["s1", "s1", "s2", None], "pop": ["a", "b", "a", "b"]})
    counts = vc.spot_population_counts(cells, spot_key="spot", population_key="pop")
    assert counts.index.tolist() == ["s1", "s2"]
    assert counts.loc["s2", "b"] == 0
    cells.loc[0, "pop"] = None
    with pytest.raises(ValueError, match="missing population"):
        vc.spot_population_counts(cells, spot_key="spot", population_key="pop")


def test_filter_uses_argument_and_inclusive_count_boundary():
    x = pd.DataFrame([[4, 1], [3, 3], [0, 0]], columns=["a", "b"])
    assert vc.spot_filter(x, min_cells=5, purity=0.8).tolist() == [True, False, False]
    assert vc.spot_filter(x, min_cells=6).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="zero-total"):
        vc.population_fractions(x)
    np.testing.assert_allclose(vc.population_fractions(x.iloc[:2]).sum(axis=1), 1)


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_correlations_identity_alignment_scipy_and_constant(method):
    x = pd.DataFrame({"a": [1, 3, 2, 5, 9], "constant": [0] * 5}, index=list("abcde"))
    y = pd.DataFrame({"a": [9, 7, 7, 1, 2]}, index=list("abcde")).iloc[::-1]
    result = vc.correlate_spot_features(x, y, method=method)
    expected = getattr(stats, method + "r")(x.a, y.reindex(x.index).a)
    assert result.iloc[0].r == pytest.approx(expected.statistic)
    assert result.iloc[0].p_nominal == pytest.approx(expected.pvalue)
    assert result.iloc[0].q_nominal == pytest.approx(expected.pvalue)
    assert np.isnan(result.iloc[1].r)


def test_within_case_removes_between_case_confounding():
    x = pd.DataFrame({"x": [1, 2, 3, 101, 102, 103]})
    y = pd.DataFrame({"y": [3, 2, 1, 103, 102, 101]})
    groups = pd.Series(["a"] * 3 + ["b"] * 3)
    assert vc.correlate_spot_features(x, y).iloc[0].r > 0.99
    result = vc.correlate_spot_features(x, y, groups=groups)
    assert result.iloc[0].r == pytest.approx(-1)
    assert result.p_nominal.isna().all()
    with pytest.raises(ValueError, match="nonmissing group"):
        vc.correlate_spot_features(x, y, groups=groups.iloc[:4])


def test_ora_upper_tail_and_sparse_parity():
    genes = pd.Index(list("abcde"))
    spots = pd.Index(["s1", "s2"])
    expression = np.array([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]], dtype=float)
    network = pd.DataFrame({"source": ["set"] * 3, "target": ["a", "b", "absent"]})
    scores, p, coverage = vc.gene_set_ora(expression, genes, spots, network, top_fraction=0.4, background=5, min_targets=2)
    expected = stats.hypergeom.sf(1, 5, 2, 2)
    assert p.iloc[0, 0] == pytest.approx(expected)
    assert scores.iloc[0, 0] == pytest.approx(-np.log10(expected))
    assert scores.iloc[1, 0] == 0
    assert coverage.loc["set", "n_measured"] == 2
    sparse_scores, _, _ = vc.gene_set_ora(sparse.csr_matrix(expression), genes, spots, network, top_fraction=0.4, background=5, min_targets=2)
    pd.testing.assert_frame_equal(scores, sparse_scores)


def test_case_scopes_and_leave_one_out():
    x = pd.DataFrame({"a": range(8)})
    cases = pd.Series(["A"] * 4 + ["B"] * 4)
    result = vc.compare_by_case(x, x, cases, leave_one_out=True)
    assert set(result.scope) == {"pooled", "within_case", "case:A", "case:B", "without:A", "without:B"}
    assert result.loc[result.scope.str.startswith("case:"), "n_spots"].eq(4).all()
    assert result.r.eq(1).all()


def test_ora_ties_reproducible_and_input_unchanged():
    x = np.ones((4, 10))
    net = pd.DataFrame({"source": ["set"] * 5, "target": list("abcde")})
    args = (x, pd.Index(list("abcdefghij")), pd.RangeIndex(4), net)
    a, _, _ = vc.gene_set_ora(*args)
    b, _, _ = vc.gene_set_ora(*args)
    pd.testing.assert_frame_equal(a, b)
    assert (x == 1).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_invalid_counts_rejected(bad):
    with pytest.raises(ValueError):
        vc.spot_filter(pd.DataFrame({"a": [bad]}))


def test_plot_smoke_with_undefined_features():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = pd.DataFrame({"a": [1, 2, 3, 4], "empty": [0] * 4})
    result = vc.correlate_spot_features(counts, counts)
    with pytest.warns(UserWarning, match="clustering disabled"):
        fig = vc.plot_correlation_heatmap(result)
    fig.canvas.draw()
    plt.close(fig)
    coords = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 0, 1, 1]})
    fig = vc.plot_spatial_features(counts.iloc[:3], coords)
    fig.canvas.draw()
    plt.close(fig)


def test_clustermap_colours_stars_and_limits_follow_clustered_labels():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    rows, cols = ["r1", "r2", "r3"], ["c1", "c2", "c3"]
    values = pd.DataFrame([[0.8, -0.8, 0.7], [-0.7, 0.7, -0.8],
                           [0.7, -0.7, 0.8]], index=rows, columns=cols)
    result = values.rename_axis(index="left", columns="right").stack().rename("r").reset_index()
    result["q_nominal"] = [0.04999, 0.05, 0.9, 0.8, 0.2, 0.01, 0.4, np.nan, 0.5]
    result["p_nominal"] = 0.00001  # Raw p-values must not determine stars.
    palette = {"r3": "blue", "r1": "red"}  # Shuffled and incomplete.
    col_palette = pd.Series({"c3": "cyan", "c1": "black", "c2": "yellow"})
    grid = vc.plot_correlation_heatmap(result, row_colors=palette, col_colors=col_palette,
                                      vmin=-0.5, vmax=0.9, figsize=(5, 4),
                                      dendrogram_ratio=(0.02, 0.04), return_grid=True)
    grid.fig.canvas.draw()
    assert grid.dendrogram_row is not None and grid.dendrogram_col is not None
    assert list(grid.data2d.index) != rows
    assert grid.ax_heatmap.collections[0].get_clim() == (-0.5, 0.9)
    np.testing.assert_allclose(grid.ax_heatmap.collections[0].get_linewidths(), 0.3)
    np.testing.assert_allclose(grid.ax_heatmap.collections[0].get_edgecolors(), [to_rgba("black")])
    heat = grid.ax_heatmap.get_window_extent()
    assert heat.width / len(cols) == pytest.approx(heat.height / len(rows))
    assert grid.ax_row_dendrogram.get_window_extent().height == pytest.approx(heat.height)
    assert grid.ax_col_dendrogram.get_window_extent().width == pytest.approx(heat.width)
    assert grid.ax_row_colors.get_window_extent().y0 == pytest.approx(heat.y0)
    assert grid.ax_col_colors.get_window_extent().x0 == pytest.approx(heat.x0)
    np.testing.assert_allclose(grid.ax_row_colors.collections[0].get_facecolors(),
                               [to_rgba(palette.get(label, "#dddddd")) for label in grid.data2d.index])
    np.testing.assert_allclose(grid.ax_col_colors.collections[0].get_facecolors(),
                               [to_rgba(col_palette[label]) for label in grid.data2d.columns])
    starred = {(grid.data2d.index[int(text.get_position()[1])],
                grid.data2d.columns[int(text.get_position()[0])])
               for text in grid.ax_heatmap.texts if text.get_text() == "*"}
    assert starred == {("r1", "c1"), ("r2", "c3")}
    plt.close(grid.fig)


def test_clustermap_unavailable_tests_singletons_and_validation():
    import matplotlib.pyplot as plt

    result = pd.DataFrame({"left": ["a"], "right": ["b"], "r": [0.5], "q_nominal": [np.nan]})
    grid = vc.plot_correlation_heatmap(result, return_grid=True)
    assert grid.dendrogram_row is None and grid.dendrogram_col is None
    assert all(text.get_text() != "*" for text in grid.ax_heatmap.texts)
    assert "unavailable" in grid.fig._suptitle.get_text()
    plt.close(grid.fig)
    with pytest.raises(ValueError, match="vmin"):
        vc.plot_correlation_heatmap(result, vmin=1, vmax=0)
    with pytest.raises(ValueError, match="dendrogram_ratio"):
        vc.plot_correlation_heatmap(result, dendrogram_ratio=-0.1)
    with pytest.raises(ValueError, match="Adjusted p-values"):
        vc.plot_correlation_heatmap(result.assign(q_nominal=-0.1))
    with pytest.raises(ValueError, match="unique feature labels"):
        vc.plot_correlation_heatmap(result, row_colors=pd.Series(["red", "blue"], index=["a", "a"]))


def test_paired_maps_align_spots_scale_images_and_keep_missing_coverage_distinct():
    import matplotlib.pyplot as plt

    imc = pd.DataFrame({"a": [1, 2, 3, 4, 100], "empty": [0] * 5}, index=list("abcde"))
    rna = pd.DataFrame({"rna": [2, 4, 6, 8, 200, 999]}, index=list("abcdef"))
    xy = pd.DataFrame({"x": [4, 1, 5, 3, 2], "y": [3, 4, 5, 2, 1]}, index=list("dafcb"))
    fig, diagnostic = vc.plot_paired_population_maps(
        imc.iloc[::-1], rna, xy, {"rna": ["a", "empty"]}, spot_diameter=0.8,
        image=np.ones((12, 12, 3)), image_scale=2, quantile=1,
    )
    assert diagnostic.n_spots.tolist() == [4, 4]
    assert diagnostic.iloc[0].pearson_r == pytest.approx(1)
    assert diagnostic.iloc[0].spearman_rho == pytest.approx(1)
    assert np.isnan(diagnostic.iloc[1].pearson_r)
    # Scale uses the cohort-wide shared reference, including spot e outside this
    # panel, but excluding RNA-only f. Coefficients use only the displayed case.
    assert diagnostic.iloc[0].imc_vmax == 100
    assert diagnostic.iloc[0].rna_vmax == 200
    for ax in fig.axes[:3]:
        np.testing.assert_allclose(ax.collections[0].get_offsets(), xy.to_numpy() * 2)
        assert len(ax.collections[0].get_offsets()) == 5  # Grey coverage layer.
        assert len(ax.collections[1].get_array()) == 4  # Only paired, including true zeros.
        assert ax.get_ylim()[0] > ax.get_ylim()[1]
        assert ax.get_xlim() == fig.axes[0].get_xlim()
    assert (fig.axes[1].collections[1].get_array() == 0).all()
    fig.canvas.draw()
    plt.close(fig)
    enlarged, enlarged_diagnostic = vc.plot_paired_population_maps(
        imc.iloc[::-1], rna, xy, {"rna": ["a", "empty"]}, spot_diameter=0.8,
        image=np.ones((12, 12, 3)), image_scale=2, quantile=1, spot_diameter_scale=1.8,
        annotate_correlations=False,
    )
    # Display enlargement must not alter the analytical result or spot positions.
    pd.testing.assert_frame_equal(enlarged_diagnostic, diagnostic)
    assert not any(text.get_text().startswith("r =") for ax in enlarged.axes for text in ax.texts)
    np.testing.assert_allclose(enlarged.axes[0].collections[0].get_offsets(), xy.to_numpy() * 2)
    plt.close(enlarged)
    case_fig, case_diagnostic = vc.plot_paired_population_maps(
        imc, rna, xy, {"rna": ["a"]}, spot_diameter=0.8,
        quantile=1, scale_scope="case", annotate_correlations=False,
    )
    assert case_diagnostic.iloc[0].imc_vmax == 4
    assert case_diagnostic.iloc[0].rna_vmax == 8
    assert case_diagnostic.iloc[0].pearson_r == pytest.approx(diagnostic.iloc[0].pearson_r)
    assert case_diagnostic.iloc[0].scale_scope == "case"
    assert any("within-case scales" in text.get_text() for text in case_fig.texts)
    plt.close(case_fig)
    fig, _ = vc.plot_paired_population_maps(
        imc, rna, xy, {"rna": ["a"]}, spot_diameter=0.8, matched_footprint=False,
    )
    assert len(fig.axes[0].collections[1].get_array()) == 4
    assert len(fig.axes[1].collections[1].get_array()) == 5
    plt.close(fig)
    with pytest.raises(ValueError, match="Every requested population"):
        vc.plot_paired_population_maps(imc, rna, xy, {"rna": ["missing"]}, spot_diameter=1)
    with pytest.raises(ValueError, match="Positive spot diameter"):
        vc.plot_paired_population_maps(imc, rna, xy, {"rna": ["a"]}, spot_diameter=1, spot_diameter_scale=0)


@pytest.mark.parametrize("method", ["mlm", "ulm"])
def test_regression_sparse_matches_ordinary_least_squares(method):
    pytest.importorskip("decoupler")
    genes = pd.Index([f"g{i}" for i in range(8)])
    spots = pd.Index(["s1", "s2", "s3"])
    values = np.random.default_rng(13).uniform(0, 5, (3, 8))
    matrix = sparse.csr_matrix(values)
    net = pd.DataFrame({"source": ["A"] * 3 + ["B"] * 3,
                        "target": genes[:6], "weight": [1, -1, 0.5] * 2})
    score, p = vc.gene_set_regression(matrix, genes, spots, net, method=method, min_targets=2)
    weights = np.zeros((8, 2))
    weights[:3, 0] = [1, -1, 0.5]
    weights[3:6, 1] = [1, -1, 0.5]
    expected = np.zeros((3, 2))
    for i, row in enumerate(values):
        if method == "mlm":
            design = np.column_stack([np.ones(8), weights])
            coef, _, _, _ = np.linalg.lstsq(design, row, rcond=None)
            variance = np.sum((row - design @ coef) ** 2) / 5
            se = np.sqrt(np.diag(np.linalg.inv(design.T @ design)) * variance)
            expected[i] = (coef / se)[1:]
        else:
            for j in range(2):
                fit = stats.linregress(weights[:, j], row)
                expected[i, j] = fit.slope / fit.stderr
    np.testing.assert_allclose(score[["A", "B"]], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(matrix.toarray(), values)
    assert score.index.equals(spots)
    assert p.shape == score.shape
