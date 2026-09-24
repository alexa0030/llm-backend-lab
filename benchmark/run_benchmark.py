from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backends import create_backend
from src.models import GenerationConfig, InferenceResult


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_prompts(path: Path, limit: int, seed: int) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].as_dict()))
        writer.writeheader()
        writer.writerows(row.as_dict() for row in results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark interchangeable LLM inference backends")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "data/chatalpaca_sample.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/raw_results.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    prompts = load_prompts(args.data, config["run"]["limit"], config["run"]["seed"])
    results = run(config, prompts)
    write_results(results, args.output)
    successes = sum(row.success for row in results)
    print(f"Completed {len(results)} requests ({successes} successful). Results: {args.output}")
    return 0 if successes == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

