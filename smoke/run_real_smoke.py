from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = ["model_load", "normal_generation", "empty_input", "max_generation_length",
              "result_recording", "exception_handling"]


def versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ["torch", "transformers", "openvino", "openvino-genai", "openvino-tokenizers",
                    "optimum-intel", "psutil"]:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def available_openvino_devices() -> list[str]:
    try:
        import openvino as ov
        return list(ov.Core().available_devices)
    except BaseException:
        return []


def failed_payload(backend: str, returncode: int, stderr: str, stdout: str) -> dict:
    detail = f"Worker terminated before producing a result (exit code {returncode})."
    if returncode in {-1073741819, 3221225477}:
        detail += " Windows status 0xC0000005 indicates a native access violation."
    cases = [{"case": "model_load", "status": "FAIL", "detail": detail,
              "evidence": {"stderr_tail": stderr[-4000:], "stdout_tail": stdout[-4000:]}}]
    cases.extend({"case": name, "status": "NOT_RUN",
                  "detail": "Blocked because model loading did not complete.", "evidence": {}}
                 for name in CASE_NAMES[1:])
    return {"backend": backend, "cases": cases}


def run_backend(args: argparse.Namespace, backend: str, work_dir: Path) -> tuple[dict, str]:
    output = work_dir / f"{backend}.json"
    command = [sys.executable, str(ROOT / "smoke" / "real_worker.py"),
               "--backend", backend, "--hf-model", args.hf_model,
               "--openvino-model-dir", args.openvino_model_dir,
               "--device", args.device, "--output", str(output)]
    try:
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                   timeout=args.timeout, encoding="utf-8", errors="replace")
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
        else:
            payload = failed_payload(backend, completed.returncode, completed.stderr, completed.stdout)
        log = (f"[{backend}] exit_code={completed.returncode}\n"
               f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n")
        return payload, log
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        payload = failed_payload(backend, -1, stderr, stdout)
        payload["cases"][0]["detail"] = f"Worker exceeded timeout of {args.timeout} seconds."
        return payload, f"[{backend}] TIMEOUT after {args.timeout}s\n"


def markdown(result: dict) -> str:
    lines = ["# Real Inference Smoke Test", "",
             f"- Run time (UTC): `{result['environment']['timestamp_utc']}`",
             f"- Model: `Qwen2.5-0.5B-Instruct`",
             f"- Selected device: `{result['environment']['selected_device']}`",
             f"- Available OpenVINO devices: `{', '.join(result['environment']['openvino_devices']) or 'unknown'}`",
             f"- OS: `{result['environment']['platform']}`", "",
             "This is a **REAL** inference validation. Mock unit tests and their timings are excluded.", "",
             "## Software versions", ""]
    for name, value in result["environment"]["versions"].items():
        lines.append(f"- {name}: `{value or 'not installed'}`")
    lines += ["", "## Results", "", "| Backend | Case | Status | Detail |",
              "|---|---|---:|---|"]
    for backend in result["backends"]:
        for item in backend["cases"]:
            detail = item["detail"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {backend['backend']} | {item['case']} | **{item['status']}** | {detail} |")
    lines += ["", "## Measured smoke evidence", "",
              "| Backend | Case | Load ms | E2E ms | TTFT ms | Decode tok/s | Peak RSS MB | Peak RSS delta MB |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    evidence_rows = 0
    for backend in result["backends"]:
        for item in backend["cases"]:
            evidence = item.get("evidence", {})
            if "latency_ms" not in evidence:
                continue
            evidence_rows += 1
            value = lambda key: "n/a" if evidence.get(key) is None else f"{evidence[key]:.2f}"
            lines.append(
                f"| {backend['backend']} | {item['case']} | {value('model_load_ms')} | "
                f"{value('latency_ms')} | {value('ttft_ms')} | {value('tokens_per_second')} | "
                f"{value('peak_rss_mb')} | {value('peak_rss_delta_mb')} |"
            )
    if not evidence_rows:
        lines.append("| n/a | No generation completed | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines += ["", "## Comparison eligibility", ""]
    if result["comparison"]["status"] == "READY":
        lines.append("Both target backends passed all smoke cases. A controlled comparison may be run.")
    else:
        lines.append("**NOT RUN.** " + result["comparison"]["reason"])
    lines += ["", "No performance improvement is claimed by this smoke test. Peak memory is process RSS ",
              "sampled on host RAM; it is not GPU VRAM. Full comparison results must use identical prompts, ",
              "generation settings, device and metric definitions.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated real inference smoke tests.")
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--openvino-model-dir", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "selected_device": args.device,
        "openvino_devices": available_openvino_devices(),
        "versions": versions(),
    }
    logs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="llm-backend-smoke-") as temp:
        payloads = []
        for backend in ["huggingface", "openvino-genai"]:
            payload, log = run_backend(args, backend, Path(temp))
            payloads.append(payload)
            logs.append(log)
    all_pass = all(item["status"] == "PASS" for payload in payloads for item in payload["cases"])
    comparison = {"status": "READY" if all_pass else "NOT_RUN",
                  "reason": "One or more real smoke cases failed or were not run; comparing performance would be invalid."}
    result = {"experiment_type": "REAL_SMOKE", "environment": environment,
              "models": {"hf_model": Path(args.hf_model).name,
                         "openvino_model_dir": Path(args.openvino_model_dir).name,
                         "source": "local pre-downloaded artifacts; absolute paths omitted"},
              "backends": payloads, "comparison": comparison}
    (args.report_dir / "real_smoke_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.report_dir / "real_smoke.log").write_text("\n".join(logs), encoding="utf-8")
    (args.report_dir / "real_smoke_report.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
