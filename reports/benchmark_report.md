# LLM Backend Validation Report

**Overall result: PASS**

## Backend summary

| Backend | Success | Avg latency (ms) | P95 latency (ms) | Avg TTFT (ms) | Tokens/s |
|---|---:|---:|---:|---:|---:|
| hf-mock | 100.0% | 24.38 | 24.66 | 8.53 | 311.42 |
| openvino-mock | 100.0% | 13.36 | 13.51 | 4.67 | 569.16 |

## Migration comparison

- Average output similarity: **0.930**
- Candidate latency change: **-45.2%**
- Candidate throughput change: **+82.8%**

## Regression gates

- PASS — output similarity
- PASS — latency regression
- PASS — candidate success rate

## Per-prompt correctness

| Prompt | Run | Similarity |
|---|---:|---:|
| qa-004 | 0 | 0.913 |
| qa-004 | 1 | 0.913 |
| qa-002 | 0 | 0.954 |
| qa-002 | 1 | 0.954 |
| qa-003 | 0 | 0.925 |
| qa-003 | 1 | 0.925 |
| qa-005 | 0 | 0.956 |
| qa-005 | 1 | 0.956 |
| qa-001 | 0 | 0.902 |
| qa-001 | 1 | 0.902 |

> Mock results validate the harness only; replace mock backends with real model backends before publishing performance claims.
