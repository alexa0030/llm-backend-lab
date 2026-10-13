import unittest

from analysis.compare_results import compare, render_markdown
from src.metrics import first_token_divergence, percentile, text_similarity
from src.models import InferenceResult


class RegressionMetricTests(unittest.TestCase):
    def test_similarity_normalizes_case_and_punctuation(self):
        self.assertEqual(text_similarity("Hello, WORLD!", "hello world"), 1.0)

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(percentile([10, 20, 30], 0.95), 29.0)

    def test_first_token_divergence(self):
        self.assertEqual(first_token_divergence("a b c", "a x c"), 1)
        self.assertIsNone(first_token_divergence("Hello!", "hello"))

    def test_report_fails_when_similarity_regresses(self):
        report = {
            "summaries": {
                "base": {"success_rate": 1.0, "avg_latency_ms": 10, "p95_latency_ms": 10,
                         "avg_ttft_ms": 2, "p95_ttft_ms": 2, "avg_tpot_ms": 1,
                         "avg_tokens_per_second": 100, "model_load_ms": 50,
                         "max_peak_rss_mb": 100},
                "candidate": {"success_rate": 1.0, "avg_latency_ms": 9, "p95_latency_ms": 9,
                              "avg_ttft_ms": 2, "p95_ttft_ms": 2, "avg_tpot_ms": 1,
                              "avg_tokens_per_second": 110, "model_load_ms": 40,
                              "max_peak_rss_mb": 90},
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
            input_tokens=1, output_tokens=1, ttft_ms=1, tpot_ms=1, latency_ms=2,
            tokens_per_second=500, peak_rss_mb=100, peak_rss_delta_mb=1,
        )
        with self.assertRaisesRegex(ValueError, "missing backend results for candidate"):
            compare([row], "base", "candidate")

    def test_mock_report_is_explicitly_non_publishable(self):
        summary = {"success_rate": 1.0, "avg_latency_ms": 10, "p95_latency_ms": 10,
                   "avg_ttft_ms": 2, "p95_ttft_ms": 2, "avg_tpot_ms": 1,
                   "avg_tokens_per_second": 100, "model_load_ms": 0,
                   "max_peak_rss_mb": 100}
        report = {
            "summaries": {"hf-mock": summary, "openvino-mock": summary},
            "comparison": {"baseline": "hf-mock", "candidate": "openvino-mock",
                           "average_similarity": 1.0, "latency_change_pct": 0,
                           "throughput_change_pct": 0},
        }
        markdown, _ = render_markdown(report, [], {
            "min_similarity": 0.85, "max_latency_regression_pct": 20,
            "min_success_rate": 1.0,
        })
        self.assertIn("MOCK FRAMEWORK SELF-TEST", markdown)
        self.assertIn("must not be cited", markdown)


if __name__ == "__main__":
    unittest.main()
