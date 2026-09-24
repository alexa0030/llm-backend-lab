from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .models import GenerationConfig, InferenceResult


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return None


class Backend(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt_id: str, prompt: str, config: GenerationConfig) -> InferenceResult:
        raise NotImplementedError


class MockBackend(Backend):
    """Deterministic local backend for testing the complete harness without model downloads."""

    def __init__(self, name: str, latency_ms: float, wording_variant: bool = False) -> None:
        self.name = name
        self.latency_ms = latency_ms
        self.wording_variant = wording_variant

    def generate(self, prompt_id: str, prompt: str, config: GenerationConfig) -> InferenceResult:
        config.validate()
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        before = _rss_mb()
        start = time.perf_counter()
        time.sleep(self.latency_ms / 1000)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        output = f"Validated response [{digest}]: {prompt[:80]}"
        if self.wording_variant:
            output = output.replace("Validated response", "Validated answer")
        elapsed = (time.perf_counter() - start) * 1000
        output_tokens = min(config.max_new_tokens, max(4, len(output.split())))
        after = _rss_mb()
        return InferenceResult(
            backend=self.name,
            prompt_id=prompt_id,
            prompt=prompt,
            output=output,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=output_tokens,
            ttft_ms=elapsed * 0.35,
            latency_ms=elapsed,
            tokens_per_second=output_tokens / (elapsed / 1000),
            memory_delta_mb=None if before is None or after is None else max(0.0, after - before),
        )


class HuggingFaceBackend(Backend):
    name = "huggingface"

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Real Hugging Face backend requires requirements-real.txt") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.device = device

    def generate(self, prompt_id: str, prompt: str, config: GenerationConfig) -> InferenceResult:
        config.validate()
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        before = _rss_mb()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        start = time.perf_counter()
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
        }
        if config.do_sample:
            generation_kwargs["temperature"] = config.temperature
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        new_tokens = generated[0][inputs.input_ids.shape[1]:]
        output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        count = int(new_tokens.shape[0])
        after = _rss_mb()
        return InferenceResult(
            backend=self.name, prompt_id=prompt_id, prompt=prompt, output=output,
            input_tokens=int(inputs.input_ids.shape[1]), output_tokens=count,
            ttft_ms=None, latency_ms=elapsed,
            tokens_per_second=count / (elapsed / 1000) if elapsed else 0.0,
            memory_delta_mb=None if before is None or after is None else max(0.0, after - before),
        )


class OpenVINOBackend(Backend):
    name = "openvino"

    def __init__(self, model_dir: str, device: str = "CPU") -> None:
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"OpenVINO model directory does not exist: {model_dir}")
        try:
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise RuntimeError("Real OpenVINO backend requires requirements-real.txt") from exc
        self.ov_genai = ov_genai
        self.pipe = ov_genai.LLMPipeline(model_dir, device)

    def generate(self, prompt_id: str, prompt: str, config: GenerationConfig) -> InferenceResult:
        config.validate()
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        before = _rss_mb()
        generation = self.ov_genai.GenerationConfig()
        generation.max_new_tokens = config.max_new_tokens
        generation.do_sample = config.do_sample
        if config.do_sample:
            generation.temperature = config.temperature
        start = time.perf_counter()
        tokenized = self.pipe.get_tokenizer().encode([prompt])
        input_tokens = int(tokenized.input_ids.get_shape()[1])
        result = self.pipe.generate([prompt], generation)
        elapsed = (time.perf_counter() - start) * 1000
        perf = result.perf_metrics
        count = int(perf.get_num_generated_tokens())
        after = _rss_mb()
        return InferenceResult(
            backend=self.name, prompt_id=prompt_id, prompt=prompt, output=result.texts[0],
            input_tokens=input_tokens, output_tokens=count,
            ttft_ms=float(perf.get_ttft().mean), latency_ms=elapsed,
            tokens_per_second=float(perf.get_throughput().mean),
            memory_delta_mb=None if before is None or after is None else max(0.0, after - before),
        )


def create_backend(name: str, models: dict) -> Backend:
    if name == "hf-mock":
        return MockBackend(name, latency_ms=24.0)
    if name == "openvino-mock":
        return MockBackend(name, latency_ms=13.0, wording_variant=True)
    if name == "huggingface":
        return HuggingFaceBackend(models["hf_model"])
    if name == "openvino":
        return OpenVINOBackend(models["openvino_model_dir"], models.get("device", "CPU"))
    raise ValueError(f"Unknown backend: {name}")
