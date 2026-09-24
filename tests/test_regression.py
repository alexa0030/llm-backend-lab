import unittest

from analysis.compare_results import compare, render_markdown
from src.metrics import percentile, text_similarity
from src.models import InferenceResult


class RegressionMetricTests(unittest.TestCase):
    def test_similarity_normalizes_case_and_punctuation(self):
        self.assertEqual(text_similarity("Hello, WORLD!", "hello world"), 1.0)

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([10, 20, 30], 0.95), 29.0)

    def test_report_fails_when_similarity_regresses(self):
        report = {
            "summaries": {
                "base": {"success_rate": 1.0, "avg_latency_ms": 10, "p95_latency_ms": 10,
                         "avg_ttft_ms": 2, "avg_tokens_per_second": 100},
                "candidate": {"success_rate": 1.0, "avg_latency_ms": 9, "p95_latency_ms": 9,
                              "avg_ttft_ms": 2, "avg_tokens_per_second": 110},
            },
            "comparison": {"baseline": "base", "candidate": "candidate",
                           "average_similarity": 0.2, "latency_change_pct": -10,
                           "throughput_change_pct": 10},
        }
        _, passed = render_markdown(report, [], {
            "min_similarity": 0.85, "max_latency_regression_pct": 20,
            "min_success_rate": 1.0,
        })
        self.assertFalse(passed)

    def test_compare_explains_missing_backend(self):
        row = InferenceResult(
            backend="base", prompt_id="1", prompt="hello", output="hello",
            input_tokens=1, output_tokens=1, ttft_ms=1, latency_ms=2,
            tokens_per_second=500, memory_delta_mb=0,
        )
        with self.assertRaisesRegex(ValueError, "missing backend results for candidate"):
            compare([row], "base", "candidate")


if __name__ == "__main__":
    unittest.main()
