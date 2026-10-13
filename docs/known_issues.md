# Known issues and compatibility findings

## OpenVINO GenAI `LLMPipeline` native crash on the test machine

### Symptom

Constructing `openvino_genai.LLMPipeline` exits the Python process with Windows status `-1073741819` before inference begins.

### Isolation performed

1. Reproduced with OpenVINO GenAI 2026.4.0 and 2026.4.1.
2. Re-exported the model with the aligned 2025.4.1 Runtime/GenAI/Tokenizers toolchain.
3. Verified that `openvino.Core` can read and compile the same IR on CPU.
4. Verified that Optimum Intel `OVModelForCausalLM` can load the IR and generate text.
5. Reproduced the GenAI crash with the environment and model copied to ASCII-only paths.

This narrows the failure to the GenAI pipeline on the tested host rather than the model conversion or base Runtime.

### Current fallback

The default real candidate is `openvino-optimum`. The `openvino-genai` adapter remains available for reproduction and future version checks. Reports identify the backend explicitly; results from the fallback are not labelled as GenAI results.

## Non-ASCII Windows paths

OpenVINO 2025.4 Python bindings may fail while decoding native error messages when the environment or model resolves to a path containing non-ASCII characters. A junction is insufficient for clients that call `Path.resolve()`. Use an actual ASCII-only repository, virtual environment, and model path for real Windows experiments.

## Quantized-output divergence

The INT8 candidate produced deterministic but lexically different responses on several open-ended prompts. The framework correctly failed the similarity gate even though latency and throughput improved. Follow-up work should separate deterministic task assertions from open-ended semantic evaluation instead of lowering the threshold until the report passes.

