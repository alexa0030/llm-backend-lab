# LLM Backend Validation Report

**Experiment type: REAL MODEL EXPERIMENT**

**Regression gates: FAIL**

## Backend summary

| Backend | Success | Load ms | Avg/P95 latency ms | Avg/P95 TTFT ms | TPOT ms | Decode tok/s | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| huggingface | 100.0% | 12012.30 | 5219.79/7627.04 | 521.13/718.22 | 179.71 | 5.65 | 1978.38 |
| openvino-genai | 100.0% | 4262.34 | 839.65/1117.90 | 126.48/217.34 | 28.79 | 34.81 | 3128.20 |

## Migration comparison

- Average output similarity: **0.636**
- Candidate latency change: **-83.9%**
- Candidate throughput change: **+516.0%**

## Regression gates

- FAIL — output similarity
- PASS — latency regression
- PASS — candidate success rate

> [!CAUTION]
> The candidate may be faster, but the migration is **not accepted** because at least one regression gate failed. Performance deltas must not be presented as a successful migration.

## Per-prompt correctness

| Prompt | Run | Similarity | Exact | First token divergence |
|---|---:|---:|---:|---:|
| qa-004 | 0 | 0.268 | False | 1 |
| qa-004 | 1 | 0.268 | False | 1 |
| qa-004 | 2 | 0.268 | False | 1 |
| qa-002 | 0 | 0.907 | False | 4 |
| qa-002 | 1 | 0.907 | False | 4 |
| qa-002 | 2 | 0.907 | False | 4 |
| qa-003 | 0 | 1.000 | True | None |
| qa-003 | 1 | 1.000 | True | None |
| qa-003 | 2 | 1.000 | True | None |
| qa-005 | 0 | 0.286 | False | 7 |
| qa-005 | 1 | 0.286 | False | 7 |
| qa-005 | 2 | 0.286 | False | 7 |
| qa-001 | 0 | 0.720 | False | 2 |
| qa-001 | 1 | 0.720 | False | 2 |
| qa-001 | 2 | 0.720 | False | 2 |

## Metric scope

- HF TTFT is measured from generation start until the streamer receives the first generated token ID.
- TPOT and decode throughput exclude TTFT where available.
- Input tokens include each backend's applied chat template; GenAI uses PerfMetrics.get_num_input_tokens().
- Peak RSS is sampled host-process memory; it is neither GPU VRAM nor whole-system memory.
- Warm-up requests are excluded from reported distributions.
