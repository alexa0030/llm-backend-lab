from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import summarize, text_similarity
from src.models import InferenceResult


def read_results(path: Path) -> list[InferenceResult]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            rows.append(InferenceResult(
                backend=raw["backend"], prompt_id=raw["prompt_id"], prompt=raw["prompt"],
                output=raw["output"], input_tokens=int(raw["input_tokens"]),
                output_tokens=int(raw["output_tokens"]),
                ttft_ms=float(raw["ttft_ms"]) if raw["ttft_ms"] else None,
                latency_ms=float(raw["latency_ms"]), tokens_per_second=float(raw["tokens_per_second"]),
                memory_delta_mb=float(raw["memory_delta_mb"]) if raw["memory_delta_mb"] else None,
                success=raw["success"].lower() == "true", error=raw["error"] or None,
                run_index=int(raw["run_index"]),
            ))
    return rows


def compare(rows: list[InferenceResult], baseline: str, candidate: str) -> tuple[dict, list[dict]]:
    grouped: dict[str, list[InferenceResult]] = defaultdict(list)
    for row in rows:
        grouped[row.backend].append(row)
    missing = [name for name in (baseline, candidate) if name not in grouped]
    if missing:
        available = ", ".join(sorted(grouped)) or "none"
        raise ValueError(f"missing backend results for {', '.join(missing)}; available: {available}")
    summaries = {name: summarize(items) for name, items in grouped.items()}
    base_outputs = {(r.prompt_id, r.run_index): r.output for r in grouped[baseline] if r.success}
    cases = []
    for row in grouped[candidate]:
        key = (row.prompt_id, row.run_index)
        if row.success and key in base_outputs:
            cases.append({"prompt_id": row.prompt_id, "run_index": row.run_index,
                          "similarity": text_similarity(base_outputs[key], row.output)})
    base_latency = summaries[baseline]["avg_latency_ms"]
    cand_latency = summaries[candidate]["avg_latency_ms"]
    delta = ((cand_latency - base_latency) / base_latency * 100) if base_latency else 0.0
    comparison = {
        "baseline": baseline, "candidate": candidate,
        "average_similarity": sum(c["similarity"] for c in cases) / len(cases) if cases else 0.0,
        "latency_change_pct": delta,
        "throughput_change_pct": (
            (summaries[candidate]["avg_tokens_per_second"] / summaries[baseline]["avg_tokens_per_second"] - 1) * 100
            if summaries[baseline]["avg_tokens_per_second"] else 0.0
        ),
    }
    return {"summaries": summaries, "comparison": comparison}, cases


def render_markdown(report: dict, cases: list[dict], thresholds: dict) -> tuple[str, bool]:
    comp = report["comparison"]
    summaries = report["summaries"]
    checks = {
        "output similarity": comp["average_similarity"] >= thresholds["min_similarity"],
        "latency regression": comp["latency_change_pct"] <= thresholds["max_latency_regression_pct"],
        "candidate success rate": summaries[comp["candidate"]]["success_rate"] >= thresholds["min_success_rate"],
    }
    passed = all(checks.values())
    lines = [
        "# LLM Backend Validation Report", "",
        f"**Overall result: {'PASS' if passed else 'FAIL'}**", "",
        "## Backend summary", "",
        "| Backend | Success | Avg latency (ms) | P95 latency (ms) | Avg TTFT (ms) | Tokens/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summaries.items():
        lines.append(f"| {name} | {item['success_rate']:.1%} | {item['avg_latency_ms']:.2f} | {item['p95_latency_ms']:.2f} | {item['avg_ttft_ms']:.2f} | {item['avg_tokens_per_second']:.2f} |")
    lines += [
        "", "## Migration comparison", "",
        f"- Average output similarity: **{comp['average_similarity']:.3f}**",
        f"- Candidate latency change: **{comp['latency_change_pct']:+.1f}%**",
        f"- Candidate throughput change: **{comp['throughput_change_pct']:+.1f}%**",
        "", "## Regression gates", "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {name}")
    lines += ["", "## Per-prompt correctness", "", "| Prompt | Run | Similarity |", "|---|---:|---:|"]
    lines += [f"| {case['prompt_id']} | {case['run_index']} | {case['similarity']:.3f} |" for case in cases]
    lines += ["", "> Mock results validate the harness only; replace mock backends with real model backends before publishing performance claims.", ""]
    return "\n".join(lines), passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare backend benchmark results")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=ROOT / "reports/raw_results.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/benchmark_report.md")
    parser.add_argument("--json", type=Path, default=ROOT / "reports/summary.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_results(args.input)
    report, cases = compare(rows, config["run"]["baseline"], config["run"]["candidate"])
    markdown, passed = render_markdown(report, cases, config["regression"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    args.json.write_text(
        json.dumps({**report, "cases": cases, "passed": passed}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Regression gates: {'PASS' if passed else 'FAIL'}. Report: {args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
