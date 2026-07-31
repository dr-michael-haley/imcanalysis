# ruff: noqa: N999
"""Cohort-first IMC exploration and human-in-the-loop classification."""

from importlib import import_module

_LAZY_EXPORTS = {
    "ExperimentManifest": (".models", "ExperimentManifest"),
    "FeatureDiscoveryTrial": (".models", "FeatureDiscoveryTrial"),
    "ModelBundle": (".classifier", "ModelBundle"),
    "build_assignment_table": (".exports", "build_assignment_table"),
    "build_roi_features": (".features", "build_roi_features"),
    "cohort_mask": (".cohort", "cohort_mask"),
    "export_annotated_anndata": (".exports", "export_annotated_anndata"),
    "high_confidence_queue": (".classifier", "high_confidence_queue"),
    "refine_trial_features": (".feature_refinement", "refine_trial_features"),
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


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "ExperimentManifest",
    "FeatureDiscoveryTrial",
    "ModelBundle",
    "build_assignment_table",
    "build_roi_features",
    "cohort_mask",
    "export_annotated_anndata",
    "high_confidence_queue",
    "launch",
    "refine_trial_features",
    "resolve_cohort",
    "score_cohort",
    "train_multiclass_classifier",
    "uncertainty_queue",
]
