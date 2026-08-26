# ruff: noqa: N999
"""Cohort-first IMC exploration and human-in-the-loop classification."""

from importlib import import_module

_LAZY_EXPORTS = {
    "CellFilterRequest": (".dataset_maintenance", "CellFilterRequest"),
    "ExperimentManifest": (".models", "ExperimentManifest"),
    "FeatureDiscoveryTrial": (".models", "FeatureDiscoveryTrial"),
    "ModelBundle": (".classifier", "ModelBundle"),
    "PopulationDraft": (".population_curation", "PopulationDraft"),
    "PopulationWorkspace": (".population_curation", "PopulationWorkspace"),
    "PublicationExportPreset": (
        ".publication_export",
        "PublicationExportPreset",
    ),
    "PublicationExportState": (
        ".publication_export",
        "PublicationExportState",
    ),
    "apply_population_draft": (
        ".population_curation",
        "apply_population_draft",
    ),
    "apply_cell_filter": (".dataset_maintenance", "apply_cell_filter"),
    "apply_var_rename": (".dataset_maintenance", "apply_var_rename"),
    "atomic_write_anndata": (".dataset_maintenance", "atomic_write_anndata"),
    "build_assignment_table": (".exports", "build_assignment_table"),
    "build_integrated_identity_table": (
        ".exports",
        "build_integrated_identity_table",
    ),
    "build_roi_features": (".features", "build_roi_features"),
    "cohort_mask": (".cohort", "cohort_mask"),
    "export_annotated_anndata": (".exports", "export_annotated_anndata"),
    "high_confidence_queue": (".classifier", "high_confidence_queue"),
    "integrated_identity_crosstab": (
        ".exports",
        "integrated_identity_crosstab",
    ),
    "refine_trial_features": (".feature_refinement", "refine_trial_features"),
    "rebuild_masks_and_object_numbers": (
        ".dataset_maintenance",
        "rebuild_masks_and_object_numbers",
    ),
    "remap_categorical_observation": (
        ".dataset_maintenance",
        "remap_categorical_observation",
    ),
    "remove_anndata_vars": (".dataset_maintenance", "remove_anndata_vars"),
    "resolve_cohort": (".cohort", "resolve_cohort"),
    "score_cohort": (".classifier", "score_cohort"),
    "train_multiclass_classifier": (
        ".classifier",
        "train_multiclass_classifier",
    ),
    "uncertainty_queue": (".classifier", "uncertainty_queue"),
}


def launch(*args, **kwargs):
    """Launch the unified Napari interface without importing Qt at package import."""

    from .app import launch as _launch

    return _launch(*args, **kwargs)


def launch_notebook(*args, **kwargs):
    """Launch from Jupyter with a live AnnData without eager Qt imports."""

    from .app import launch_notebook as _launch_notebook

    return _launch_notebook(*args, **kwargs)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "CellFilterRequest",
    "ExperimentManifest",
    "FeatureDiscoveryTrial",
    "ModelBundle",
    "PopulationDraft",
    "PopulationWorkspace",
    "PublicationExportPreset",
    "PublicationExportState",
    "apply_cell_filter",
    "apply_population_draft",
    "apply_var_rename",
    "atomic_write_anndata",
    "build_assignment_table",
    "build_integrated_identity_table",
    "build_roi_features",
    "cohort_mask",
    "export_annotated_anndata",
    "high_confidence_queue",
    "integrated_identity_crosstab",
    "launch",
    "launch_notebook",
    "rebuild_masks_and_object_numbers",
    "remap_categorical_observation",
    "refine_trial_features",
    "remove_anndata_vars",
    "resolve_cohort",
    "score_cohort",
    "train_multiclass_classifier",
    "uncertainty_queue",
]
