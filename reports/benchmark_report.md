# LLM Backend Validation Report

**Overall result: PASS**

## Backend summary

| Backend | Success | Avg latency (ms) | P95 latency (ms) | Avg TTFT (ms) | Tokens/s |
|---|---:|---:|---:|---:|---:|
| hf-mock | 100.0% | 24.26 | 24.46 | 8.49 | 302.98 |
| openvino-mock | 100.0% | 13.43 | 13.49 | 4.70 | 545.74 |

## Migration comparison

- Average output similarity: **0.931**
- Candidate latency change: **-44.6%**
- Candidate throughput change: **+80.1%**

## Regression gates

- PASS — output similarity
- PASS — latency regression
- PASS — candidate success rate

## Per-prompt correctness

| Prompt | Run | Similarity |
|---|---:|---:|
| qa-004 | 0 | 0.913 |
| qa-002 | 0 | 0.954 |
| qa-003 | 0 | 0.925 |

> Mock results validate the harness only; replace mock backends with real model backends before publishing performance claims.
