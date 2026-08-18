from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from SpatialBiologyToolkit.napari_sbt.population_curation import (
    GraphSubclusterRequest,
    apply_population_draft,
    component_tables_from_assignments,
    create_population_draft,
    import_base_mapping_csv,
    integrate_component_tables,
    list_population_drafts,
    merge_groups,
    population_draft_sync_state,
    read_population_audit,
    save_population_draft,
    synthesize_population_labels,
)

ad = pytest.importorskip("anndata")


def _adata():
    obs = pd.DataFrame(
        {
            "ROI": pd.Categorical(["r1", "r1", "r2", "r2", "r2"]),
            "ObjectNumber": [1, 2, 1, 2, 3],
            "leiden": pd.Categorical(["0", "0", "1", "2", "2"]),
        },
        index=["a", "b", "c", "d", "e"],
    )
    data = ad.AnnData(np.arange(10, dtype=np.float32).reshape(5, 2), obs=obs)
    data.uns["leiden_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    return data


def test_subclustering_defaults_rebuild_neighbors_from_biobatchnet():
    request = GraphSubclusterRequest(
        anndata_path="cells.h5ad",
        source_obs="leiden",
        source_fingerprint="abc",
        selected_values=["2"],
        output_folder="run",
    )

    assert request.neighbor_source == "rebuild_from_rep"
    assert request.representation_key == "X_biobatchnet"
    assert request.n_neighbors == 15
    assert request.adjacency_key is None


def test_draft_renaming_explicit_merge_and_apply(tmp_path):
    data = _adata()
    workspace, paths, draft, base, components, membership = (
        create_population_draft(
            data,
            tmp_path,
            source_obs="leiden",
            name="Manual review",
            derived_obs="population_named",
        )
    )
    base.loc[base["source_value"].isin(["0", "1"]), "proposed_label"] = "immune"
    base.loc[base["source_value"].eq("2"), "proposed_label"] = "tumour"
    assert merge_groups(base, components) == {
        "immune": ["source:0", "source:1"]
    }

    draft = save_population_draft(
        paths,
        draft,
        base,
        components,
        membership,
        adata=data,
        action="save_test_mapping",
    )
    assert population_draft_sync_state(data, draft) == "missing"
    summary = apply_population_draft(
        data,
        draft=draft,
        base_mapping=base,
        components=components,
        membership=membership,
    )
    assert population_draft_sync_state(data, draft) == "synced"

    assert data.obs["population_named"].astype(str).tolist() == [
        "immune",
        "immune",
        "immune",
        "tumour",
        "tumour",
    ]
    assert summary["merge_groups"]["immune"] == ["source:0", "source:1"]
    assert len(data.uns["population_named_colors"]) == 2
    assert data.uns["napari_sbt"]["population_curation"][
        "population_named"
    ]["source_obs"] == "leiden"
    assert [item.draft_id for item in list_population_drafts(paths)] == [
        draft.draft_id
    ]
    assert any(
        event["action"] == "save_test_mapping"
        for event in read_population_audit(paths)
    )

    newer = save_population_draft(
        paths,
        draft,
        base,
        components,
        membership,
        adata=data,
        action="save_newer_revision",
    )
    assert population_draft_sync_state(data, newer) == "stale"


def test_source_labels_with_trailing_whitespace_use_one_canonical_mapping(tmp_path):
    data = _adata()
    data.obs["named_clusters"] = pd.Categorical(
        [
            "Activated myeloid/DC-like ",
            "Activated myeloid/DC-like ",
            "ERTR7+ reticular stromal-like ",
            "ERTR7+ reticular stromal-like ",
            "ERTR7+ reticular stromal-like ",
        ]
    )
    _workspace, paths, draft, base, components, membership = (
        create_population_draft(
            data,
            tmp_path,
            source_obs="named_clusters",
            name="Whitespace-safe mapping",
            derived_obs="named_clusters_reviewed",
        )
    )

    assert base["source_value"].tolist() == [
        "Activated myeloid/DC-like",
        "ERTR7+ reticular stromal-like",
    ]
    assert base["cell_count"].tolist() == [2, 3]
    saved = save_population_draft(
        paths,
        draft,
        base,
        components,
        membership,
        adata=data,
    )
    labels, summary = synthesize_population_labels(
        data,
        source_obs=saved.source_obs,
        base_mapping=base,
        components=components,
        membership=membership,
    )

    assert labels.astype(str).tolist() == [
        "Activated myeloid/DC-like",
        "Activated myeloid/DC-like",
        "ERTR7+ reticular stromal-like",
        "ERTR7+ reticular stromal-like",
        "ERTR7+ reticular stromal-like",
    ]
    assert summary["missing_source_cells"] == 0


def test_source_labels_that_collide_after_trimming_are_rejected(tmp_path):
    data = _adata()
    data.obs["ambiguous"] = pd.Categorical(["A", "A ", "B", "B", "B"])

    with pytest.raises(ValueError, match="become identical after trimming"):
        create_population_draft(
            data,
            tmp_path,
            source_obs="ambiguous",
            name="Ambiguous source",
            derived_obs="ambiguous_reviewed",
        )


def test_cell_level_components_split_and_override_base_mapping(tmp_path):
    data = _adata()
    _workspace, paths, draft, base, components, membership = (
        create_population_draft(
            data,
            tmp_path,
            source_obs="leiden",
            name="With image split",
            derived_obs="population_split",
        )
    )
    base["proposed_label"] = base["source_value"].map(
        {"0": "immune", "1": "stromal", "2": "tumour"}
    )
    assignments = pd.DataFrame(
        {
            "obs_name": ["d", "e"],
            "class_id": ["cycling tumour", "resting tumour"],
        }
    )
    new_components, new_membership, _summary = component_tables_from_assignments(
        data,
        source_obs="leiden",
        assignments=assignments,
        method="napari_image_classifier",
        run_id="image-run-1",
    )
    components, membership, integration = integrate_component_tables(
        components,
        membership,
        new_components,
        new_membership,
    )
    labels, summary = synthesize_population_labels(
        data,
        source_obs="leiden",
        base_mapping=base,
        components=components,
        membership=membership,
    )

    assert labels.astype(str).to_dict() == {
        "a": "immune",
        "b": "immune",
        "c": "stromal",
        "d": "cycling tumour",
        "e": "resting tumour",
    }
    assert summary["split_cell_count"] == 2
    assert integration["new_component_count"] == 2

    replacement = pd.DataFrame(
        {"obs_name": ["e"], "label": ["ambiguous tumour"]}
    )
    replacement_components, replacement_membership, _ = (
        component_tables_from_assignments(
            data,
            source_obs="leiden",
            assignments=replacement,
            method="manual_import",
            run_id="replacement-run",
        )
    )
    components, membership, integration = integrate_component_tables(
        components,
        membership,
        replacement_components,
        replacement_membership,
    )
    assert integration["replaced_cell_count"] == 1
    assert membership["obs_name"].is_unique


def test_preliminary_csv_import_is_partial_and_repeated_names_are_merges(tmp_path):
    data = _adata()
    _workspace, _paths, draft, base, _components, _membership = (
        create_population_draft(
            data,
            tmp_path / "workspace",
            source_obs="leiden",
            name="CSV seed",
            derived_obs="csv_named",
        )
    )
    mapping_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {"leiden": [0, 1], "name": ["myeloid", "myeloid"]}
    ).to_csv(mapping_path, index=False)
    updated, summary = import_base_mapping_csv(
        mapping_path,
        base,
        source_obs=draft.source_obs,
        derived_obs=draft.derived_obs,
    )

    assert updated.set_index("source_value").loc["0", "proposed_label"] == "myeloid"
    assert updated.set_index("source_value").loc["1", "proposed_label"] == "myeloid"
    assert updated.set_index("source_value").loc["2", "proposed_label"] == "2"
    assert summary["unmapped_source_count"] == 1


def test_source_membership_change_requires_new_workspace(tmp_path):
    data = _adata()
    create_population_draft(
        data,
        tmp_path,
        source_obs="leiden",
        name="Stable source",
        derived_obs="stable_named",
    )
    changed = data.copy()
    changed.obs.loc["a", "leiden"] = "1"
    with pytest.raises(ValueError, match="source observation or cell identities changed"):
        create_population_draft(
            changed,
            tmp_path,
            source_obs="leiden",
            name="Changed source",
            derived_obs="changed_named",
        )
