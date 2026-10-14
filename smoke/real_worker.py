from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backends import create_backend
from src.models import GenerationConfig


def case(name: str, status: str, detail: str, evidence: dict | None = None) -> dict:
    return {"case": name, "status": status, "detail": detail, "evidence": evidence or {}}


def run(backend_name: str, models: dict, output_path: Path) -> int:
    cases: list[dict] = []
    payload = {"backend": backend_name, "cases": cases}

    try:
        backend = create_backend(backend_name, models)
        cases.append(case("model_load", "PASS", "Model adapter constructed successfully.", {
            "model_load_ms": backend.load_time_ms,
        }))
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except BaseException as exc:
        cases.append(case("model_load", "FAIL", f"{type(exc).__name__}: {exc}", {
            "traceback": traceback.format_exc(),
        }))
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1

    normal: object | None = None
    try:
        normal = backend.generate("normal", "用一句话解释什么是机器学习。", GenerationConfig(max_new_tokens=16))
        assert normal.success and normal.output.strip(), "generation returned no text"
        cases.append(case("normal_generation", "PASS", "Generated non-empty text.", normal.as_dict()))
    except BaseException as exc:
        cases.append(case("normal_generation", "FAIL", f"{type(exc).__name__}: {exc}"))

    try:
        backend.generate("empty", "   ", GenerationConfig(max_new_tokens=8))
        cases.append(case("empty_input", "FAIL", "Empty prompt was accepted unexpectedly."))
    except ValueError as exc:
        cases.append(case("empty_input", "PASS", f"Rejected empty prompt: {exc}"))
    except BaseException as exc:
        cases.append(case("empty_input", "FAIL", f"Unexpected {type(exc).__name__}: {exc}"))

    try:
        limited = backend.generate("max-length", "请列出三种常见水果。", GenerationConfig(max_new_tokens=4))
        assert 0 < limited.output_tokens <= 4, f"output_tokens={limited.output_tokens}"
        cases.append(case("max_generation_length", "PASS", "Generation respected max_new_tokens=4.", limited.as_dict()))
    except BaseException as exc:
        cases.append(case("max_generation_length", "FAIL", f"{type(exc).__name__}: {exc}"))

    try:
        if normal is None:
            raise RuntimeError("No successful result available to record")
        record = normal.as_dict()
        required = {"backend", "output", "input_tokens", "output_tokens", "ttft_ms",
                    "latency_ms", "tokens_per_second", "peak_rss_mb", "success"}
        assert required <= record.keys(), f"missing fields: {sorted(required - record.keys())}"
        json.dumps(record, ensure_ascii=False)
        cases.append(case("result_recording", "PASS", "Result is complete and JSON serializable."))
    except BaseException as exc:
        cases.append(case("result_recording", "FAIL", f"{type(exc).__name__}: {exc}"))

    try:
        backend.generate("invalid-config", "test", GenerationConfig(max_new_tokens=0))
        cases.append(case("exception_handling", "FAIL", "Invalid generation config was accepted."))
    except ValueError as exc:
        cases.append(case("exception_handling", "PASS", f"Invalid config produced a controlled exception: {exc}"))
    except BaseException as exc:
        cases.append(case("exception_handling", "FAIL", f"Unexpected {type(exc).__name__}: {exc}"))

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(item["status"] == "PASS" for item in cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["huggingface", "openvino-genai"])
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--openvino-model-dir", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {"hf_model": args.hf_model, "openvino_model_dir": args.openvino_model_dir,
              "device": args.device}
    return run(args.backend, models, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
