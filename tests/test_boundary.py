import unittest
import tempfile
from pathlib import Path

from benchmark.run_benchmark import load_prompts, write_results
from src.backends import MockBackend, create_backend
from src.models import GenerationConfig


class BoundaryTests(unittest.TestCase):
    def test_empty_prompt_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            MockBackend("test", 0).generate("1", "  ", GenerationConfig())

    def test_invalid_max_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            GenerationConfig(max_new_tokens=0).validate()

    def test_unknown_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            create_backend("missing", {})

    def test_empty_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset is empty"):
                load_prompts(path, limit=1, seed=42)

    def test_empty_result_set_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "empty benchmark result"):
                write_results([], Path(directory) / "results.csv")


if __name__ == "__main__":
    unittest.main()
