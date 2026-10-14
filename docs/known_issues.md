# Known issues and compatibility findings

## Resolved: OpenVINO GenAI `LLMPipeline` native crash on the test machine

### Symptom

Constructing `openvino_genai.LLMPipeline` exits the Python process with Windows status `0xC0000005` (`3221225477`, signed `-1073741819`) before inference begins.

### Isolation performed

1. Reproduced with OpenVINO GenAI 2026.4.0 and 2026.4.1.
2. Re-exported the model with the aligned 2025.4.1 Runtime/GenAI/Tokenizers toolchain.
3. Verified that `openvino.Core` can read and compile the same IR on CPU.
4. Verified that Optimum Intel `OVModelForCausalLM` can load the IR and generate text.
5. Reproduced the GenAI crash with the environment and model copied to ASCII-only paths.

This initially narrowed the failure to the GenAI pipeline process rather than the model conversion or base Runtime.

### Root cause and verified fix

The failing virtual environment was created from an Anaconda-build Python 3.12 interpreter. `pip check` reported no dependency conflicts, but the inherited native runtime/DLL environment caused the GenAI C++ extension to terminate with `0xC0000005`.

A fresh ASCII-only venv created from the official python.org CPython 3.12 installation, with the same aligned OpenVINO 2025.4.1 packages and the same model IR, successfully constructed `LLMPipeline`, generated text, and passed all six real smoke cases. After the full project dependency set was installed, both Transformers and OpenVINO GenAI passed all twelve backend smoke cases, confirming that the fix remained stable.

### Current status and fallback

The default real candidate is `openvino-genai`, so the intended migration cannot silently fall back. `scripts/run_real.ps1` now rejects Anaconda-derived interpreters before loading native components. `config.optimum.yaml` remains available only for Runtime/IR diagnosis; fallback results are never labelled as GenAI results.

## Non-ASCII Windows paths

OpenVINO 2025.4 Python bindings may fail while decoding native error messages when the environment or model resolves to a path containing non-ASCII characters. A junction is insufficient for clients that call `Path.resolve()`. Use an actual ASCII-only repository, virtual environment, and model path for real Windows experiments.

## Quantized-output divergence

The INT8 candidate produced deterministic but lexically different responses on several open-ended prompts. The framework correctly failed the similarity gate even though latency and throughput improved. Follow-up work should separate deterministic task assertions from open-ended semantic evaluation instead of lowering the threshold until the report passes.

