from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

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
