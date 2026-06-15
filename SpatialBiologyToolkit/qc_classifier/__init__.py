"""
Object-level active-learning QC utilities for CellPose IMC masks.

The modules in this package deliberately consume precomputed per-object
features. They do not calculate morphology, intensity, or CellPose-derived
features; that remains the responsibility of the upstream segmentation script.
"""

from .feature_dictionary import FeatureDictionary, load_feature_dictionary
from .feature_table import FeatureTable, load_feature_table
from .labels import RoiLabels, load_roi_labels, save_roi_labels
from .model import ModelBundle, load_model_bundle, save_model_bundle, train_classifier

__all__ = [
    "FeatureDictionary",
    "FeatureTable",
    "ModelBundle",
    "RoiLabels",
    "load_feature_dictionary",
    "load_feature_table",
    "load_model_bundle",
    "load_roi_labels",
    "save_model_bundle",
    "save_roi_labels",
    "train_classifier",
]
