from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "docs" / "tools" / "generate_docs.py"
SPEC = importlib.util.spec_from_file_location("imcanalysis_docs_generator", GENERATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load documentation generator: {GENERATOR_PATH}")
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
SPEC.loader.exec_module(GENERATOR)


class DocumentationGenerationTests(unittest.TestCase):
    def test_every_pipeline_alias_has_documentable_wrapper_metadata(self):
        stages = GENERATOR.load_stage_docs(REPO_ROOT)
        aliases = [stage.alias for stage in stages]

        self.assertEqual(len(aliases), len(set(aliases)))
        self.assertIn("prep", aliases)
        self.assertIn("slogs", aliases)
        self.assertIn("rebuildmeta", aliases)
        self.assertTrue(all(stage.description for stage in stages))

    def test_stage_index_contains_registered_aliases_and_navigation(self):
        markdown = GENERATOR.render_stage_index(GENERATOR.load_stage_docs(REPO_ROOT))

        self.assertIn("[`prep`](prep.md)", markdown)
        self.assertIn("[`rebuildmeta`](rebuildmeta.md)", markdown)
        self.assertIn("```{toctree}", markdown)

    def test_generation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first"
            second = Path(temp_dir) / "second"
            GENERATOR.generate_all(REPO_ROOT, first)
            GENERATOR.generate_all(REPO_ROOT, second)

            for relative_path in GENERATOR.GENERATED_FILES:
                self.assertEqual(
                    (first / relative_path).read_bytes(),
                    (second / relative_path).read_bytes(),
                )
            for relative_path in GENERATOR.GENERATED_DIRS:
                first_files = {
                    path.relative_to(first / relative_path): path.read_bytes()
                    for path in (first / relative_path).rglob("*")
                    if path.is_file()
                }
                second_files = {
                    path.relative_to(second / relative_path): path.read_bytes()
                    for path in (second / relative_path).rglob("*")
                    if path.is_file()
                }
                self.assertEqual(first_files, second_files)


if __name__ == "__main__":
    unittest.main()
