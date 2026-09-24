from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
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
                        output="", input_tokens=0, output_tokens=0, ttft_ms=None,
                        latency_ms=0, tokens_per_second=0, memory_delta_mb=None,
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


def write_metadata(config: dict, prompt_count: int, output: Path) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backends": config["run"]["backends"],
        "prompt_count": prompt_count,
        "warmup_runs": config["run"]["warmup_runs"],
        "measured_runs": config["run"]["measured_runs"],
        "generation": config["generation"],
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
    prompts = load_prompts(args.data, config["run"]["limit"], config["run"]["seed"])
    results = run(config, prompts)
    write_results(results, args.output)
    write_metadata(config, len(prompts), args.metadata)
    successes = sum(row.success for row in results)
    print(f"Completed {len(results)} requests ({successes} successful). Results: {args.output}")
    return 0 if successes == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
