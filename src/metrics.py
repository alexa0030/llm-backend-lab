from __future__ import annotations

import re
from difflib import SequenceMatcher
from statistics import mean, median

from .models import InferenceResult


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


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
    throughputs = [row.tokens_per_second for row in successful]
    return {
        "requests": float(len(rows)),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "avg_latency_ms": mean(latencies) if latencies else 0.0,
        "p50_latency_ms": median(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "avg_ttft_ms": mean(ttfts) if ttfts else 0.0,
        "avg_tokens_per_second": mean(throughputs) if throughputs else 0.0,
    }

