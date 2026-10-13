from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
from importlib import metadata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backends import create_backend
from src.models import GenerationConfig, InferenceResult


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_prompts(path: Path, limit: int, seed: int) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be at least 1")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"prompt dataset is empty: {path}")
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row.get("id"), str) or not isinstance(row.get("prompt"), str):
            raise ValueError(f"dataset row {line_number} must contain string id and prompt fields")
        if not row["prompt"].strip():
            raise ValueError(f"dataset row {line_number} has an empty prompt")
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def run(config: dict, prompts: list[dict]) -> list[InferenceResult]:
    generation = GenerationConfig(**config["generation"])
    generation.validate()
    results: list[InferenceResult] = []
    for backend_name in config["run"]["backends"]:
        backend = create_backend(backend_name, config["models"])
        for _ in range(config["run"]["warmup_runs"]):
            backend.generate("warmup", "Explain model inference briefly.", generation)
        for prompt in prompts:
            for run_index in range(config["run"]["measured_runs"]):
                try:
                    result = backend.generate(prompt["id"], prompt["prompt"], generation)
                    result.run_index = run_index
                except Exception as exc:
                    result = InferenceResult(
                        backend=backend_name, prompt_id=prompt["id"], prompt=prompt["prompt"],
                        output="", input_tokens=0, output_tokens=0, ttft_ms=None, tpot_ms=None,
                        latency_ms=0, tokens_per_second=0, peak_rss_mb=None,
                        peak_rss_delta_mb=None, model_load_ms=backend.load_time_ms,
                        success=False, error=f"{type(exc).__name__}: {exc}", run_index=run_index,
                    )
                results.append(result)
    return results


def write_results(results: list[InferenceResult], output: Path) -> None:
    if not results:
        raise ValueError("cannot write an empty benchmark result set")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].as_dict()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in results)


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "openvino", "openvino-genai", "optimum-intel"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_metadata(config: dict, prompt_count: int, output: Path) -> None:
    backends = config["run"]["backends"]
    eligibility_issues = []
    if any(name.endswith("-mock") for name in backends):
        eligibility_issues.append("mock backend present")
    if config["run"]["warmup_runs"] < 1:
        eligibility_issues.append("at least 1 warm-up run is required")
    if config["run"]["measured_runs"] < 3:
        eligibility_issues.append("at least 3 measured runs per prompt are required")
    if prompt_count < 5:
        eligibility_issues.append("at least 5 prompts are required")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backends": config["run"]["backends"],
        "experiment_type": "mock" if any(name.endswith("-mock") for name in backends) else "real",
        "publishable_performance": not eligibility_issues,
        "publishability_issues": eligibility_issues,
        "git_revision": _git_revision(),
        "packages": _package_versions(),
        "prompt_count": prompt_count,
        "warmup_runs": config["run"]["warmup_runs"],
        "measured_runs": config["run"]["measured_runs"],
        "generation": config["generation"],
        "models": config["models"],
        "metric_notes": {
            "peak_rss": "Peak host process RSS sampled during generation; not GPU VRAM.",
            "throughput": "Decode throughput after first token when TTFT is available.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark interchangeable LLM inference backends")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/chatalpaca_sample.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/raw_results.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "reports/run_metadata.json")
    parser.add_argument("--backends", nargs="+", help="Override configured backend names")
    parser.add_argument("--limit", type=int, help="Override the number of prompts")
    parser.add_argument("--runs", type=int, help="Override measured runs per prompt")
    parser.add_argument("--warmup-runs", type=int, help="Override warm-up runs per backend")
    parser.add_argument("--openvino-model-dir", help="Override the OpenVINO IR directory")
    parser.add_argument("--hf-model", help="Override the Hugging Face model ID or local directory")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.backends:
        config["run"]["backends"] = args.backends
    if args.limit is not None:
        config["run"]["limit"] = args.limit
    if args.runs is not None:
        if args.runs < 1:
            parser.error("--runs must be at least 1")
        config["run"]["measured_runs"] = args.runs
    if args.warmup_runs is not None:
        if args.warmup_runs < 0:
            parser.error("--warmup-runs must be non-negative")
        config["run"]["warmup_runs"] = args.warmup_runs
    if args.openvino_model_dir:
        config["models"]["openvino_model_dir"] = args.openvino_model_dir
    if args.hf_model:
        config["models"]["hf_model"] = args.hf_model
    prompts = load_prompts(args.data, config["run"]["limit"], config["run"]["seed"])
    results = run(config, prompts)
    write_results(results, args.output)
    write_metadata(config, len(prompts), args.metadata)
    successes = sum(row.success for row in results)
    print(f"Completed {len(results)} requests ({successes} successful). Results: {args.output}")
    return 0 if successes == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
