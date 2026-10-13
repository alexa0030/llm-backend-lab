from __future__ import annotations

import re
from difflib import SequenceMatcher
from statistics import mean, median

from .models import InferenceResult


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def first_token_divergence(left: str, right: str) -> int | None:
    left_tokens = normalize(left).split()
    right_tokens = normalize(right).split()
    for index, (left_token, right_token) in enumerate(zip(left_tokens, right_tokens)):
        if left_token != right_token:
            return index
    if len(left_tokens) != len(right_tokens):
        return min(len(left_tokens), len(right_tokens))
    return None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(rows: list[InferenceResult]) -> dict[str, float]:
    successful = [row for row in rows if row.success]
    latencies = [row.latency_ms for row in successful]
    ttfts = [row.ttft_ms for row in successful if row.ttft_ms is not None]
    tpots = [row.tpot_ms for row in successful if row.tpot_ms is not None]
    throughputs = [row.tokens_per_second for row in successful]
    peak_rss = [row.peak_rss_mb for row in successful if row.peak_rss_mb is not None]
    peak_rss_delta = [
        row.peak_rss_delta_mb for row in successful if row.peak_rss_delta_mb is not None
    ]
    load_times = [row.model_load_ms for row in successful if row.model_load_ms is not None]
    return {
        "requests": float(len(rows)),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "avg_latency_ms": mean(latencies) if latencies else 0.0,
        "p50_latency_ms": median(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "avg_ttft_ms": mean(ttfts) if ttfts else 0.0,
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "avg_tpot_ms": mean(tpots) if tpots else 0.0,
        "avg_tokens_per_second": mean(throughputs) if throughputs else 0.0,
        "max_peak_rss_mb": max(peak_rss) if peak_rss else 0.0,
        "max_peak_rss_delta_mb": max(peak_rss_delta) if peak_rss_delta else 0.0,
        "model_load_ms": load_times[0] if load_times else 0.0,
    }

