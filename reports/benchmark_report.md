# Optimum/OpenVINO Fallback Validation Report

**Experiment type: REAL MODEL EXPERIMENT**

**Backend scope: Transformers vs `openvino-optimum`; this is not an OpenVINO GenAI result.**

**Regression gates: FAIL**

## Backend summary

| Backend | Success | Load ms | Avg/P95 latency ms | Avg/P95 TTFT ms | TPOT ms | Decode tok/s | Peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| huggingface | 100.0% | 12653.69 | 3950.81/5264.43 | 361.89/470.45 | 138.26 | 7.29 | 2365.01 |
| openvino-optimum | 100.0% | 5862.08 | 1512.04/2164.84 | 278.81/356.82 | 50.57 | 21.31 | 2596.03 |

## Migration comparison

- Average output similarity: **0.728**
- Candidate latency change: **-61.7%**
- Candidate throughput change: **+192.4%**

## Regression gates

- FAIL — output similarity
- PASS — latency regression
- PASS — candidate success rate

## Per-prompt correctness

| Prompt | Run | Similarity | Exact | First token divergence |
|---|---:|---:|---:|---:|
| qa-004 | 0 | 0.500 | False | 2 |
| qa-004 | 1 | 0.500 | False | 2 |
| qa-004 | 2 | 0.500 | False | 2 |
| qa-002 | 0 | 1.000 | True | None |
| qa-002 | 1 | 1.000 | True | None |
| qa-002 | 2 | 1.000 | True | None |
| qa-003 | 0 | 1.000 | True | None |
| qa-003 | 1 | 1.000 | True | None |
| qa-003 | 2 | 1.000 | True | None |
| qa-005 | 0 | 0.480 | False | 8 |
| qa-005 | 1 | 0.480 | False | 8 |
| qa-005 | 2 | 0.480 | False | 8 |
| qa-001 | 0 | 0.661 | False | 2 |
| qa-001 | 1 | 0.661 | False | 2 |
| qa-001 | 2 | 0.661 | False | 2 |

## Metric scope

- HF TTFT is measured from generation start until the streamer receives the first generated token ID.
- TPOT and decode throughput exclude TTFT where available.
- Peak RSS is sampled host-process memory; it is neither GPU VRAM nor whole-system memory.
- Warm-up requests are excluded from reported distributions.
