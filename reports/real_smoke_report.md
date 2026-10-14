# Real Inference Smoke Test

- Run time (UTC): `2026-10-14T05:32:19.153200+00:00`
- Model: `Qwen2.5-0.5B-Instruct`
- Selected device: `CPU`
- Available OpenVINO devices: `CPU, GPU.0, GPU.1, GPU.2`
- OS: `Windows-11-10.0.26200-SP0`

This is a **REAL** inference validation. Mock unit tests and their timings are excluded.

## Software versions

- torch: `2.7.1`
- transformers: `4.55.4`
- openvino: `2025.4.1`
- openvino-genai: `2025.4.1.0`
- openvino-tokenizers: `2025.4.1.0`
- optimum-intel: `1.26.1`
- psutil: `7.2.2`

## Results

| Backend | Case | Status | Detail |
|---|---|---:|---|
| huggingface | model_load | **PASS** | Model adapter constructed successfully. |
| huggingface | normal_generation | **PASS** | Generated non-empty text. |
| huggingface | empty_input | **PASS** | Rejected empty prompt: prompt must not be empty |
| huggingface | max_generation_length | **PASS** | Generation respected max_new_tokens=4. |
| huggingface | result_recording | **PASS** | Result is complete and JSON serializable. |
| huggingface | exception_handling | **PASS** | Invalid config produced a controlled exception: max_new_tokens must be between 1 and 4096 |
| openvino-genai | model_load | **FAIL** | Worker terminated before producing a result (exit code 3221225477). Windows status 0xC0000005 indicates a native access violation. |
| openvino-genai | normal_generation | **NOT_RUN** | Blocked because model loading did not complete. |
| openvino-genai | empty_input | **NOT_RUN** | Blocked because model loading did not complete. |
| openvino-genai | max_generation_length | **NOT_RUN** | Blocked because model loading did not complete. |
| openvino-genai | result_recording | **NOT_RUN** | Blocked because model loading did not complete. |
| openvino-genai | exception_handling | **NOT_RUN** | Blocked because model loading did not complete. |

## Measured smoke evidence

| Backend | Case | Load ms | E2E ms | TTFT ms | Decode tok/s | Peak RSS MB | Peak RSS delta MB |
|---|---|---:|---:|---:|---:|---:|---:|
| huggingface | normal_generation | 15381.54 | 4652.92 | 633.50 | 3.73 | 2334.46 | 17.34 |
| huggingface | max_generation_length | 15381.54 | 1309.29 | 628.38 | 4.41 | 2337.82 | 4.39 |

## Comparison eligibility

**NOT RUN.** One or more real smoke cases failed or were not run; comparing performance would be invalid.

No performance improvement is claimed by this smoke test. Peak memory is process RSS 
sampled on host RAM; it is not GPU VRAM. Full comparison results must use identical prompts, 
generation settings, device and metric definitions.
