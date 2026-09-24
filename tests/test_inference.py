import unittest

from src.backends import MockBackend
from src.models import GenerationConfig


class InferenceTests(unittest.TestCase):
    def test_mock_generation_is_deterministic(self):
        backend = MockBackend("test", latency_ms=0)
        config = GenerationConfig(max_new_tokens=8)
        first = backend.generate("1", "hello", config)
        second = backend.generate("1", "hello", config)
        self.assertEqual(first.output, second.output)
        self.assertGreater(first.output_tokens, 0)


if __name__ == "__main__":
    unittest.main()

