import unittest

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


if __name__ == "__main__":
    unittest.main()

