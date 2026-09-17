from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.cluster import hierarchy as sch
from scipy.spatial import distance as ssd

from SpatialBiologyToolkit.napari_sbt.variable_ordering import VariableOrderRegistry


def _adata(*variables: str):
    return SimpleNamespace(var_names=pd.Index(variables))


def test_variable_order_defaults_to_anndata_and_keeps_unmatched_tail_order():
    registry = VariableOrderRegistry()
    registry.set_adata(_adata("CD3", "CD20", "CD68"))

    ordered = registry.ordered(["extra B", "CD68", "CD3", "extra A", "CD20"])

    assert ordered == ["CD3", "CD20", "CD68", "extra B", "extra A"]


def test_alphabetical_order_is_case_insensitive_and_central():
    registry = VariableOrderRegistry(mode="alphabetical")
    registry.set_adata(_adata("CD3", "CD20"))

    assert registry.ordered(["zeta", "Alpha", "beta"]) == [
        "Alpha",
        "beta",
        "zeta",
    ]


def test_similarity_order_is_cached_and_can_order_image_aliases():
    calls = []

    def similarity_orderer(_adata, variables):
        calls.append(list(variables))
        return ["CD68", "CD3", "CD20"]

    registry = VariableOrderRegistry(
        mode="similarity", similarity_orderer=similarity_orderer
    )
    registry.set_adata(_adata("CD3", "CD20", "CD68"))
    aliases = {
        "191Ir_CD68 [images]": "CD68",
        "141Pr_CD3 [images]": "CD3",
        "CD20 [images]": "CD20",
    }

    first = registry.ordered(list(reversed(aliases)), canonical_names=aliases)
    second = registry.ordered(list(aliases), canonical_names=aliases)

    assert first == [
        "191Ir_CD68 [images]",
        "141Pr_CD3 [images]",
        "CD20 [images]",
    ]
    assert second == first
    assert calls == [["CD3", "CD20", "CD68"]]


def test_similarity_failure_falls_back_without_breaking_variable_lists():
    def broken_orderer(_adata, _variables):
        raise RuntimeError("non-finite values")

    registry = VariableOrderRegistry(
        mode="similarity", similarity_orderer=broken_orderer
    )
    registry.set_adata(_adata("CD3", "CD20"))

    assert registry.ordered(["CD20", "CD3"]) == ["CD3", "CD20"]
    assert "non-finite values" in str(registry.last_warning)


def test_unknown_variable_order_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown variable-order mode"):
        VariableOrderRegistry(mode="random")  # type: ignore[arg-type]


def test_reorder_vars_by_expression_uses_selected_layer():
    from SpatialBiologyToolkit.utils import reorder_vars_by_expression

    adata = ad.AnnData(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        var=pd.Index(["marker_a", "marker_b", "marker_c"]),
    )
    adata.layers["alt"] = np.array(
        [
            [0.0, 2.0, 2.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    expected_matrix = adata[:, ["marker_a", "marker_b", "marker_c"]].layers["alt"]
    distance_matrix = ssd.pdist(expected_matrix.T, metric="euclidean")
    linkage_matrix = sch.linkage(distance_matrix, method="ward")
    expected_order = (
        adata[:, ["marker_a", "marker_b", "marker_c"]]
        .var_names[sch.dendrogram(linkage_matrix, no_plot=True)["leaves"]]
        .tolist()
    )

    ordered = reorder_vars_by_expression(adata, ["marker_a", "marker_b", "marker_c"], layer="alt")

    assert ordered == expected_order
