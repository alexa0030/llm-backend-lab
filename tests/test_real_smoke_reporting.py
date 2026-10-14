from __future__ import annotations

import unittest

from smoke.run_real_smoke import failed_payload, markdown


class RealSmokeReportingTests(unittest.TestCase):
    def test_native_crash_is_fail_and_dependents_are_not_run(self) -> None:
        payload = failed_payload("openvino-genai", 3221225477, "", "")
        self.assertEqual(payload["cases"][0]["status"], "FAIL")
        self.assertIn("0xC0000005", payload["cases"][0]["detail"])
        self.assertTrue(all(case["status"] == "NOT_RUN" for case in payload["cases"][1:]))

    def test_report_does_not_claim_comparison_when_not_ready(self) -> None:
        result = {
            "environment": {"timestamp_utc": "x", "selected_device": "CPU",
                            "openvino_devices": ["CPU"], "platform": "Windows",
                            "versions": {}},
            "backends": [failed_payload("openvino-genai", 1, "error", "")],
            "comparison": {"status": "NOT_RUN", "reason": "smoke failed"},
        }
        report = markdown(result)
        self.assertIn("**NOT RUN.** smoke failed", report)
        self.assertIn("No performance improvement is claimed", report)


if __name__ == "__main__":
    unittest.main()
