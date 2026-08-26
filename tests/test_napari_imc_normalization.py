from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from SpatialBiologyToolkit._napari_imc_normalization import (
    find_normalization_parameters,
    find_normalization_value,
    load_normalization_mapping,
    load_normalization_parameters,
    normalization_parameters_payload,
    normalize_imc_image,
    normalized_contrast_limits,
    prepare_normalization_dict,
    prepare_normalization_parameters,
)


class NapariIMCNormalizationTests(unittest.TestCase):
    def test_marker_value_csv_is_loaded(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "normalization.csv"
            source.write_text("Marker,Value\nCD3,10.5\nCD20,20\n", encoding="utf-8")

            values = load_normalization_mapping(source)

        self.assertEqual(values, {"CD3": 10.5, "CD20": 20.0})

    def test_preferred_nimbus_csv_is_loaded(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "normalization_dict.csv"
            source.write_text(
                "marker,vmax,lower_threshold\nCD3,10.5,0.8\nCD20,20,0\n",
                encoding="utf-8",
            )

            values = load_normalization_mapping(source)

        self.assertEqual(values, {"CD3": 10.5, "CD20": 20.0})

    def test_full_nimbus_csv_parameters_preserve_lower_thresholds(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "normalization_dict.csv"
            source.write_text(
                "marker,vmax,lower_threshold\nCD3,10.5,0.8\nCD20,20,0\n",
                encoding="utf-8",
            )

            values = load_normalization_parameters(source)

        self.assertEqual(values["CD3"].vmax, 10.5)
        self.assertEqual(values["CD3"].lower_threshold, 0.8)
        self.assertEqual(values["CD20"].lower_threshold, 0.0)

    def test_normalization_csv_requires_marker_and_value_columns(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "normalization.csv"
            source.write_text("Channel,Maximum\nCD3,10.5\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Marker and Value"):
                load_normalization_mapping(source)

    def test_normalization_csv_rejects_duplicate_markers_ignoring_case(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "normalization.csv"
            source.write_text("Marker,Value\nCD3,10.5\ncd3,20\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unique ignoring case"):
                load_normalization_mapping(source)

    def test_nimbus_string_values_are_accepted(self):
        values = prepare_normalization_dict({"CD3": "10.5", "CD20": 20})

        self.assertEqual(values, {"CD3": 10.5, "CD20": 20.0})

    def test_structured_and_legacy_parameters_share_one_validated_format(self):
        values = prepare_normalization_parameters(
            {
                "CD3": {"vmax": "10.5", "lower_threshold": "0.8"},
                "CD20": 20,
            }
        )

        self.assertEqual(values["CD3"].lower_threshold, 0.8)
        self.assertEqual(values["CD20"].lower_threshold, 0.0)
        self.assertEqual(
            normalization_parameters_payload(values),
            {
                "CD3": {"vmax": 10.5, "lower_threshold": 0.8},
                "CD20": {"vmax": 20.0, "lower_threshold": 0.0},
            },
        )

    def test_invalid_normalization_values_are_rejected(self):
        for value in ("not-a-number", 0, -1, np.inf, np.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    prepare_normalization_dict({"CD3": value})

    def test_lookup_allows_case_and_nimbus_punctuation_cleaning(self):
        values = prepare_normalization_dict({"CD45RO": "12"})

        self.assertEqual(find_normalization_value(values, "cd45ro"), 12.0)
        self.assertEqual(find_normalization_value(values, "CD45-RO"), 12.0)
        self.assertIsNone(find_normalization_value(values, "CD3"))

    def test_parameter_lookup_preserves_lower_threshold(self):
        values = prepare_normalization_parameters(
            {"CD45RO": {"vmax": 12, "lower_threshold": 0.75}}
        )

        parameters = find_normalization_parameters(values, "CD45-RO")

        self.assertIsNotNone(parameters)
        self.assertEqual(parameters.vmax, 12.0)
        self.assertEqual(parameters.lower_threshold, 0.75)

    def test_supplied_value_sets_maximum_and_clips_at_one(self):
        image = np.array([[0, 5, 10, 20]], dtype=np.uint16)

        normalized = normalize_imc_image(
            image,
            quantile=0.5,
            minimum_pixel_counts=0.1,
            normalization_value=10,
        )

        np.testing.assert_allclose(normalized, [[0, 0.5, 1, 1]])

    def test_supplied_lower_threshold_uses_two_point_nimbus_normalization(self):
        image = np.array([[0, 1, 2, 5, 10, 20]], dtype=np.float32)

        normalized = normalize_imc_image(
            image,
            quantile=0.5,
            minimum_pixel_counts=0.1,
            normalization_value=10,
            normalization_lower_threshold=2,
        )

        np.testing.assert_allclose(normalized, [[0, 0, 0, 0.375, 1, 1]])

    def test_missing_value_uses_legacy_quantile_normalization(self):
        image = np.array([[0, 5, 10]], dtype=np.uint16)

        normalized = normalize_imc_image(
            image,
            quantile=1.0,
            minimum_pixel_counts=0.1,
        )

        np.testing.assert_allclose(normalized, [[0, 0.5, 1]])

    def test_minimum_pixel_threshold_is_applied_before_normalization(self):
        image = np.array([[0.05, 0.2, 2.0]])

        normalized = normalize_imc_image(
            image,
            quantile=1.0,
            minimum_pixel_counts=0.1,
            normalization_value=2,
        )

        np.testing.assert_allclose(normalized, [[0, 0.1, 1]])

    def test_normalized_contrast_limit_uses_zero_to_one_display_range(self):
        self.assertEqual(normalized_contrast_limits(0.2, 0.8), (0.2, 0.8))
        self.assertEqual(normalized_contrast_limits(0, 1), (0.0, 1.0))

    def test_normalized_contrast_limit_rejects_values_outside_slider_range(self):
        for limits in ((-0.1, 1), (0, 1.1), (np.nan, 1), (0, np.nan)):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    normalized_contrast_limits(*limits)

    def test_normalized_contrast_limit_requires_ordered_bounds(self):
        for limits in ((0.5, 0.5), (0.8, 0.2)):
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    normalized_contrast_limits(*limits)


if __name__ == "__main__":
    unittest.main()
