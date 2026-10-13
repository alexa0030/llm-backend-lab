# Experiment methodology

## Objective

The experiment answers two independent questions after an inference-backend migration:

1. Does the candidate preserve acceptable model behaviour?
2. Does it improve latency, decode throughput, load time, or host-memory usage?

A performance improvement is not accepted when a correctness or reliability gate fails.

## Controlled variables

- Same model family and prompt workload.
- Greedy decoding (`do_sample=false`) with the same maximum output length.
- Batch size 1 and sequential requests for the CPU baseline experiment.
- Warm-up requests excluded from measured distributions.
- Same prompt order, seed, and repeat count for each backend.

## Measurements

- **TTFT:** time until the first generated token ID reaches the streamer.
- **TPOT:** elapsed decode time after TTFT divided by the remaining generated tokens.
- **Decode throughput:** generated tokens after the first token divided by decode time.
- **E2E latency:** wall-clock time around one generation request.
- **Peak RSS:** peak host-process resident memory sampled every 10 ms. It is not GPU VRAM.
- **Model load:** backend construction and model-loading wall-clock time.

The report includes averages and P95 values. Raw per-request evidence is retained in CSV, while environment and dependency versions are retained in `run_metadata.json`.

## Correctness policy

The lightweight default gate uses normalized lexical similarity, exact match, and the first token-divergence position. This gate is deliberately strict and is not a semantic-equivalence metric. Open-ended prompts that fail lexical validation require manual review or a separately configured semantic evaluator; their performance numbers must not be presented as an accepted migration result.

## Publication eligibility

A run is marked performance-publishable only when it uses no mock backend and includes at least:

- 1 warm-up request per backend;
- 5 distinct prompts;
- 3 measured repetitions per prompt.

Eligibility does not override regression gates. A publishable experiment can still conclude that a migration is unacceptable.

## Current verified experiment

The first full CPU experiment completed 30 measured requests successfully. Optimum/OpenVINO INT8 improved the observed latency and decode throughput, but the average lexical similarity failed the configured correctness gate. The correct conclusion is therefore **performance improved, migration validation failed**. Exact numbers remain in the generated report and should only be quoted together with this failed correctness result and the recorded environment.

