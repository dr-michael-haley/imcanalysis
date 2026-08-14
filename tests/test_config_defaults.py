import ast
import tempfile
import unittest
from pathlib import Path

from SpatialBiologyToolkit.config import PipelineConfig, load_config_data
from SpatialBiologyToolkit.scripts.config_and_utils import (
    generate_default_config_dict,
    load_config as load_legacy_config,
)


class ConfigDefaultsTests(unittest.TestCase):
    def test_model_field_annotations_remain_python38_runtime_compatible(self):
        models_path = (
            Path(__file__).resolve().parents[1]
            / "SpatialBiologyToolkit"
            / "config"
            / "models.py"
        )
        tree = ast.parse(models_path.read_text(encoding="utf-8"))
        builtin_generics = {"dict", "frozenset", "list", "set", "tuple"}
        incompatible: list[str] = []

        for model_class in (
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ):
            for statement in model_class.body:
                if not isinstance(statement, ast.AnnAssign):
                    continue
                annotation = statement.annotation
                uses_builtin_generic = (
                    isinstance(annotation, ast.Subscript)
                    and isinstance(annotation.value, ast.Name)
                    and annotation.value.id in builtin_generics
                )
                uses_union_operator = any(
                    isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
                    for node in ast.walk(annotation)
                )
                if uses_builtin_generic or uses_union_operator:
                    incompatible.append(
                        f"{model_class.name}.{getattr(statement.target, 'id', '?')}"
                    )

        self.assertEqual(incompatible, [])

    def test_pipeline_defaults_match_legacy_default_dictionary(self):
        expected = generate_default_config_dict()
        resolved = PipelineConfig().model_dump(mode="python")

        self.assertEqual(resolved, expected)
        self.assertEqual(len(resolved), 28)
        self.assertEqual(resolved["napari_sbt"]["worker_count"], 8)
        self.assertEqual(resolved["population_embedding_qc"]["mode"], "auto")
        self.assertEqual(resolved["maxfuse"]["batching_scheme"], "cyclic")
        self.assertEqual(resolved["maxfuse"]["refine_iterations"], 1)
        self.assertEqual(resolved["maxfuse"]["report_score_threshold"], 0.30)
        self.assertEqual(resolved["spatialdata"]["action"], "plan")
        self.assertEqual(resolved["denoising"]["truncated_max_rate"], 0.9999)
        self.assertFalse(resolved["denoising"]["intelligent_patch_size"])
        self.assertEqual(resolved["denoising"]["patch_step_size"], 70)
        self.assertEqual(resolved["denoising"]["ratio_thresh"], 0.9)
        self.assertEqual(resolved["neighbour_signal"]["max_halo_px"], 8)
        self.assertEqual(resolved["neighbour_signal"]["n_jobs"], "auto")
        self.assertEqual(
            resolved["neighbour_signal"]["source_target_table_path"],
            "neighbour_signal_source_target.parquet",
        )
        self.assertTrue(
            resolved["neighbour_signal"][
                "source_target_qc_exclude_same_population"
            ]
        )

    def test_sparse_data_fills_missing_sections_and_fields(self):
        config = load_config_data(
            {"createmasks": {"cellpose_cell_diameter": 30}}
        )

        self.assertEqual(config.createmasks.cellpose_cell_diameter, 30.0)
        self.assertEqual(config.general.anndata_path, "anndata.h5ad")
        self.assertEqual(config.preprocess.minimum_roi_dimensions, 200)
        self.assertEqual(config.logging.level, "INFO")

    def test_mutable_defaults_are_isolated(self):
        first = PipelineConfig()
        second = PipelineConfig()

        first.denoising.channels.append("CD3")
        self.assertEqual(second.denoising.channels, [])

    def test_complete_default_config_round_trips_through_validation(self):
        defaults = generate_default_config_dict()
        resolved = load_config_data(defaults).model_dump(mode="python")

        self.assertEqual(resolved, defaults)

    def test_legacy_loader_still_creates_complete_dictionary_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            config = load_legacy_config(str(path))

            self.assertTrue(path.is_file())
            self.assertEqual(config, generate_default_config_dict())


if __name__ == "__main__":
    unittest.main()
