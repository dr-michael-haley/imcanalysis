import unittest

from SpatialBiologyToolkit.rapids_compat import resolve_harmony_runtime_kwargs


class RapidsHarmonyCompatibilityTests(unittest.TestCase):
    def test_modern_runtime_preserves_requested_flavor(self):
        def harmony_integrate(*, flavor="harmony2", correction_method=None):
            return None

        self.assertEqual(
            resolve_harmony_runtime_kwargs(harmony_integrate, "harmony2"),
            {"flavor": "harmony2"},
        )
        self.assertEqual(
            resolve_harmony_runtime_kwargs(harmony_integrate, "harmony1"),
            {"flavor": "harmony1"},
        )

    def test_legacy_runtime_maps_harmony1_to_original(self):
        def harmony_integrate(*, correction_method="original", **kwargs):
            return None

        self.assertEqual(
            resolve_harmony_runtime_kwargs(harmony_integrate, "harmony1"),
            {"correction_method": "original"},
        )

    def test_legacy_runtime_rejects_harmony2(self):
        def harmony_integrate(*, correction_method="original", **kwargs):
            return None

        with self.assertRaisesRegex(RuntimeError, "cannot run harmony2"):
            resolve_harmony_runtime_kwargs(harmony_integrate, "harmony2")

    def test_unknown_runtime_api_is_rejected(self):
        def harmony_integrate(**kwargs):
            return None

        with self.assertRaisesRegex(RuntimeError, "not recognized"):
            resolve_harmony_runtime_kwargs(harmony_integrate, "harmony1")


if __name__ == "__main__":
    unittest.main()
