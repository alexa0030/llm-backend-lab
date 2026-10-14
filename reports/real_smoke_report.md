# Real Inference Smoke Test

- Run time (UTC): `2026-10-14T05:48:28.777373+00:00`
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
| openvino-genai | model_load | **PASS** | Model adapter constructed successfully. |
| openvino-genai | normal_generation | **PASS** | Generated non-empty text. |
| openvino-genai | empty_input | **PASS** | Rejected empty prompt: prompt must not be empty |
| openvino-genai | max_generation_length | **PASS** | Generation respected max_new_tokens=4. |
| openvino-genai | result_recording | **PASS** | Result is complete and JSON serializable. |
| openvino-genai | exception_handling | **PASS** | Invalid config produced a controlled exception: max_new_tokens must be between 1 and 4096 |

## Measured smoke evidence

| Backend | Case | Load ms | E2E ms | TTFT ms | Decode tok/s | Peak RSS MB | Peak RSS delta MB |
|---|---|---:|---:|---:|---:|---:|---:|
| huggingface | normal_generation | 11095.54 | 2921.67 | 491.37 | 6.17 | 2331.29 | 15.79 |
| huggingface | max_generation_length | 11095.54 | 860.08 | 423.71 | 6.87 | 2335.30 | 4.57 |
| openvino-genai | normal_generation | 3582.31 | 952.74 | 405.03 | 28.67 | 1158.39 | 1005.04 |
| openvino-genai | max_generation_length | 3582.31 | 324.69 | 205.88 | 25.52 | 1159.62 | 1.22 |

## Comparison eligibility

Both target backends passed all smoke cases. A controlled comparison may be run.

No performance improvement is claimed by this smoke test. Peak memory is process RSS 
sampled on host RAM; it is not GPU VRAM. Full comparison results must use identical prompts, 
generation settings, device and metric definitions.
