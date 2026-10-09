"""Headless exemplar-guided splitting of AnnData populations.

Imports are lazy so discovery does not load image, scientific or GUI libraries.
"""

from importlib import import_module

_EXPORTS = {
    "AssignmentPolicy": "models",
    "ClassSpec": "models",
    "SplitSpec": "models",
    "MarkerRule": "models",
    "SamplingSpec": "models",
    "MaxFuseSource": "sources",
    "TableSource": "sources",
    "ExemplarSource": "sources",
    "expression_evidence": "sources",
    "rank_reference_markers": "sources",
    "resolve_split_cohort": "sources",
    "SelectionResult": "selection",
    "assess_markers": "selection",
    "select_exemplars": "selection",
    "image_recipe": "features",
    "image_feature_columns": "features",
    "create_feature_experiment": "features",
    "build_feature_table": "features",
    "RefinementResult": "workflow",
    "fit_refinement": "workflow",
    "EvaluationResult": "evaluation",
    "evaluate_refinement": "evaluation",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
