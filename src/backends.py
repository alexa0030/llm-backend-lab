from __future__ import annotations

import hashlib
import threading
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


class PeakRssSampler:
    """Sample peak process RSS. This measures host RAM, never GPU VRAM."""

    def __init__(self, interval_seconds: float = 0.01) -> None:
        self.interval_seconds = interval_seconds
        self.start_mb = _rss_mb()
        self.peak_mb = self.start_mb
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            current = _rss_mb()
            if current is not None and (self.peak_mb is None or current > self.peak_mb):
                self.peak_mb = current
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "PeakRssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        current = _rss_mb()
        if current is not None and (self.peak_mb is None or current > self.peak_mb):
            self.peak_mb = current

    @property
    def delta_mb(self) -> float | None:
        if self.start_mb is None or self.peak_mb is None:
            return None
        return max(0.0, self.peak_mb - self.start_mb)


class Backend(ABC):
    name: str
    load_time_ms: float | None = None

    @abstractmethod
    def generate(self, prompt_id: str, prompt: str, config: GenerationConfig) -> InferenceResult:
        raise NotImplementedError


class MockBackend(Backend):
    """Deterministic test double. Never use its timings as model performance data."""

    def __init__(self, name: str, latency_ms: float, wording_variant: bool = False) -> None:
        self.name = name
        self.latency_ms = latency_ms
        self.wording_variant = wording_variant

    def generate(self, prompt_id: str, prompt: str, config: GenerationConfig) -> InferenceResult:
        config.validate()
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        start = time.perf_counter()
        with PeakRssSampler() as memory:
            time.sleep(self.latency_ms / 1000)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        output = f"Validated response [{digest}]: {prompt[:80]}"
        if self.wording_variant:
            output = output.replace("Validated response", "Validated answer")
        elapsed = (time.perf_counter() - start) * 1000
        output_tokens = min(config.max_new_tokens, max(4, len(output.split())))
        ttft = elapsed * 0.35
        decode_ms = elapsed - ttft
        return InferenceResult(
            backend=self.name,
            prompt_id=prompt_id,
            prompt=prompt,
            output=output,
            input_tokens=max(1, len(prompt.split())),
            output_tokens=output_tokens,
            ttft_ms=ttft,
            tpot_ms=decode_ms / max(1, output_tokens - 1),
            latency_ms=elapsed,
            tokens_per_second=(output_tokens - 1) / (decode_ms / 1000),
            peak_rss_mb=memory.peak_mb,
            peak_rss_delta_mb=memory.delta_mb,
            model_load_ms=self.load_time_ms,
        )


class HuggingFaceBackend(Backend):
    name = "huggingface"

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        load_start = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        except ImportError as exc:
            raise RuntimeError("Real Hugging Face backend requires requirements-real.txt") from exc
        self.torch = torch
        class TimedTextIteratorStreamer(TextIteratorStreamer):
            def __init__(streamer_self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                streamer_self.first_token_at: float | None = None

            def put(streamer_self, value: object) -> None:
                is_prompt = streamer_self.skip_prompt and streamer_self.next_tokens_are_prompt
                if not is_prompt and streamer_self.first_token_at is None:
                    streamer_self.first_token_at = time.perf_counter()
                super().put(value)

        self.TextIteratorStreamer = TimedTextIteratorStreamer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device
        self.load_time_ms = (time.perf_counter() - load_start) * 1000

    def generate(self, prompt_id: str, prompt: str, config: GenerationConfig) -> InferenceResult:
        config.validate()
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        streamer = self.TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120.0,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "streamer": streamer,
        }
        if config.do_sample:
            generation_kwargs["temperature"] = config.temperature
        generation_error: list[BaseException] = []

        def worker_target() -> None:
            try:
                with self.torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except BaseException as exc:
                generation_error.append(exc)

        chunks: list[str] = []
        start = time.perf_counter()
        worker = threading.Thread(target=worker_target, daemon=True)
        with PeakRssSampler() as memory:
            worker.start()
            for chunk in streamer:
                chunks.append(chunk)
            worker.join()
        finished = time.perf_counter()
        if generation_error:
            raise RuntimeError("Hugging Face generation failed") from generation_error[0]
        output = "".join(chunks)
        count = len(self.tokenizer.encode(output, add_special_tokens=False))
        elapsed = (finished - start) * 1000
        ttft = (
            (streamer.first_token_at - start) * 1000
            if streamer.first_token_at is not None else None
        )
        decode_ms = elapsed - ttft if ttft is not None else elapsed
        tpot = decode_ms / max(1, count - 1) if count else None
        return InferenceResult(
            backend=self.name, prompt_id=prompt_id, prompt=prompt, output=output,
            input_tokens=int(inputs.input_ids.shape[1]), output_tokens=count,
            ttft_ms=ttft, tpot_ms=tpot, latency_ms=elapsed,
            tokens_per_second=(count - 1) / (decode_ms / 1000)
            if count > 1 and decode_ms > 0 else 0.0,
            peak_rss_mb=memory.peak_mb, peak_rss_delta_mb=memory.delta_mb,
            model_load_ms=self.load_time_ms,
        )


class OpenVINOBackend(Backend):
    name = "openvino-genai"

    def __init__(self, model_dir: str, device: str = "CPU") -> None:
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"OpenVINO model directory does not exist: {model_dir}")
        load_start = time.perf_counter()
        try:
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise RuntimeError("Real OpenVINO backend requires requirements-real.txt") from exc
        self.ov_genai = ov_genai
        self.pipe = ov_genai.LLMPipeline(model_dir, device)
        self.load_time_ms = (time.perf_counter() - load_start) * 1000

    def generate(self, prompt_id: str, prompt: str, config: GenerationConfig) -> InferenceResult:
        config.validate()
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        generation = self.ov_genai.GenerationConfig()
        generation.max_new_tokens = config.max_new_tokens
        generation.do_sample = config.do_sample
        if config.do_sample:
            generation.temperature = config.temperature
        start = time.perf_counter()
        with PeakRssSampler() as memory:
            result = self.pipe.generate([prompt], generation)
        elapsed = (time.perf_counter() - start) * 1000
        perf = result.perf_metrics
        input_tokens = int(perf.get_num_input_tokens())
        count = int(perf.get_num_generated_tokens())
        return InferenceResult(
            backend=self.name, prompt_id=prompt_id, prompt=prompt, output=result.texts[0],
            input_tokens=input_tokens, output_tokens=count,
            ttft_ms=float(perf.get_ttft().mean), tpot_ms=float(perf.get_tpot().mean),
            latency_ms=elapsed,
            tokens_per_second=float(perf.get_throughput().mean),
            peak_rss_mb=memory.peak_mb, peak_rss_delta_mb=memory.delta_mb,
            model_load_ms=self.load_time_ms,
        )


class OpenVINOOptimumBackend(HuggingFaceBackend):
    """OpenVINO Runtime backend through Optimum, used when GenAI is unavailable."""

    name = "openvino-optimum"

    def __init__(self, model_dir: str, device: str = "CPU") -> None:
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"OpenVINO model directory does not exist: {model_dir}")
        load_start = time.perf_counter()
        try:
            import torch
            from optimum.intel.openvino import OVModelForCausalLM
            from transformers import AutoTokenizer, TextIteratorStreamer
        except ImportError as exc:
            raise RuntimeError("OpenVINO Optimum backend requires requirements-real.txt") from exc
        self.torch = torch

        class TimedTextIteratorStreamer(TextIteratorStreamer):
            def __init__(streamer_self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                streamer_self.first_token_at: float | None = None

            def put(streamer_self, value: object) -> None:
                is_prompt = streamer_self.skip_prompt and streamer_self.next_tokens_are_prompt
                if not is_prompt and streamer_self.first_token_at is None:
                    streamer_self.first_token_at = time.perf_counter()
                super().put(value)

        self.TextIteratorStreamer = TimedTextIteratorStreamer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = OVModelForCausalLM.from_pretrained(model_dir, device=device)
        self.device = "cpu"
        self.load_time_ms = (time.perf_counter() - load_start) * 1000


def create_backend(name: str, models: dict) -> Backend:
    if name == "hf-mock":
        return MockBackend(name, latency_ms=24.0)
    if name == "openvino-mock":
        return MockBackend(name, latency_ms=13.0, wording_variant=True)
    if name == "huggingface":
        return HuggingFaceBackend(models["hf_model"])
    if name in {"openvino", "openvino-genai"}:
        return OpenVINOBackend(models["openvino_model_dir"], models.get("device", "CPU"))
    if name == "openvino-optimum":
        return OpenVINOOptimumBackend(models["openvino_model_dir"], models.get("device", "CPU"))
    raise ValueError(f"Unknown backend: {name}")
