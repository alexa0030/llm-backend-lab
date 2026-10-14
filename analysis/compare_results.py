from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import first_token_divergence, normalize, summarize, text_similarity
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
                tpot_ms=float(raw["tpot_ms"]) if raw["tpot_ms"] else None,
                latency_ms=float(raw["latency_ms"]), tokens_per_second=float(raw["tokens_per_second"]),
                peak_rss_mb=float(raw["peak_rss_mb"]) if raw["peak_rss_mb"] else None,
                peak_rss_delta_mb=(
                    float(raw["peak_rss_delta_mb"]) if raw["peak_rss_delta_mb"] else None
                ),
                model_load_ms=float(raw["model_load_ms"]) if raw["model_load_ms"] else None,
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
            baseline_output = base_outputs[key]
            cases.append({
                "prompt_id": row.prompt_id,
                "run_index": row.run_index,
                "similarity": text_similarity(baseline_output, row.output),
                "exact_match": normalize(baseline_output) == normalize(row.output),
                "first_token_divergence": first_token_divergence(baseline_output, row.output),
                "baseline_output": baseline_output,
                "candidate_output": row.output,
            })
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


def render_markdown(
    report: dict, cases: list[dict], thresholds: dict, experiment: dict | None = None,
) -> tuple[str, bool]:
    comp = report["comparison"]
    summaries = report["summaries"]
    checks = {
        "output similarity": comp["average_similarity"] >= thresholds["min_similarity"],
        "latency regression": comp["latency_change_pct"] <= thresholds["max_latency_regression_pct"],
        "candidate success rate": summaries[comp["candidate"]]["success_rate"] >= thresholds["min_success_rate"],
    }
    passed = all(checks.values())
    is_mock = any(name.endswith("-mock") for name in summaries)
    experiment_label = "MOCK FRAMEWORK SELF-TEST" if is_mock else "REAL MODEL EXPERIMENT"
    lines = [
        "# LLM Backend Validation Report", "",
        f"**Experiment type: {experiment_label}**", "",
        f"**Regression gates: {'PASS' if passed else 'FAIL'}**", "",
    ]
    if is_mock:
        lines += [
            "> [!WARNING]",
            "> This report uses deterministic test doubles with artificial sleep latency. "
            "Its performance deltas are not OpenVINO or model benchmark results and must not be cited.",
            "",
        ]
    elif experiment and not experiment.get("publishable_performance", False):
        reasons = "; ".join(experiment.get("publishability_issues", []))
        lines += [
            "> [!CAUTION]",
            f"> Pilot run only; performance is not publication-ready. Reasons: {reasons}",
            "",
        ]
    lines += [
        "## Backend summary", "",
        "| Backend | Success | Load ms | Avg/P95 latency ms | Avg/P95 TTFT ms | TPOT ms | Decode tok/s | Peak RSS MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summaries.items():
        lines.append(
            f"| {name} | {item['success_rate']:.1%} | {item['model_load_ms']:.2f} | "
            f"{item['avg_latency_ms']:.2f}/{item['p95_latency_ms']:.2f} | "
            f"{item['avg_ttft_ms']:.2f}/{item['p95_ttft_ms']:.2f} | "
            f"{item['avg_tpot_ms']:.2f} | {item['avg_tokens_per_second']:.2f} | "
            f"{item['max_peak_rss_mb']:.2f} |"
        )
    lines += [
        "", "## Migration comparison", "",
        f"- Average output similarity: **{comp['average_similarity']:.3f}**",
        f"- Candidate latency change: **{comp['latency_change_pct']:+.1f}%**",
        f"- Candidate throughput change: **{comp['throughput_change_pct']:+.1f}%**",
        "", "## Regression gates", "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {name}")
    if not passed and not is_mock:
        lines += [
            "", "> [!CAUTION]",
            "> The candidate may be faster, but the migration is **not accepted** because at least one "
            "regression gate failed. Performance deltas must not be presented as a successful migration.",
        ]
    lines += [
        "", "## Per-prompt correctness", "",
        "| Prompt | Run | Similarity | Exact | First token divergence |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {case['prompt_id']} | {case['run_index']} | {case['similarity']:.3f} | "
        f"{case['exact_match']} | {case['first_token_divergence']} |" for case in cases
    ]
    lines += [
        "", "## Metric scope", "",
        "- HF TTFT is measured from generation start until the streamer receives the first generated token ID.",
        "- TPOT and decode throughput exclude TTFT where available.",
        "- Input tokens include each backend's applied chat template; GenAI uses PerfMetrics.get_num_input_tokens().",
        "- Peak RSS is sampled host-process memory; it is neither GPU VRAM nor whole-system memory.",
        "- Warm-up requests are excluded from reported distributions.", "",
    ]
    return "\n".join(lines), passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare backend benchmark results")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--input", type=Path, default=ROOT / "reports/raw_results.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/benchmark_report.md")
    parser.add_argument("--json", type=Path, default=ROOT / "reports/summary.json")
    parser.add_argument("--failures", type=Path, default=ROOT / "reports/failure_cases.json")
    parser.add_argument("--metadata", type=Path, default=ROOT / "reports/run_metadata.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_results(args.input)
    report, cases = compare(rows, config["run"]["baseline"], config["run"]["candidate"])
    experiment = (
        json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata.exists() else None
    )
    markdown, passed = render_markdown(report, cases, config["regression"], experiment)
    failure_cases = [
        case for case in cases if case["similarity"] < config["regression"]["min_similarity"]
    ]
    failure_cases += [
        {"backend": row.backend, "prompt_id": row.prompt_id, "run_index": row.run_index,
         "error": row.error}
        for row in rows if not row.success
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown, encoding="utf-8")
    args.json.write_text(
        json.dumps({**report, "cases": cases, "passed": passed}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.failures.write_text(
        json.dumps(failure_cases, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Regression gates: {'PASS' if passed else 'FAIL'}. Report: {args.report}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
