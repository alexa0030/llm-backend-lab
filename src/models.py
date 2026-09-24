from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0
    do_sample: bool = False

    def validate(self) -> None:
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not self.do_sample and self.temperature != 0:
            raise ValueError("temperature must be 0 when do_sample is false")


@dataclass
class InferenceResult:
    backend: str
    prompt_id: str
    prompt: str
    output: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    latency_ms: float
    tokens_per_second: float
    memory_delta_mb: float | None
    success: bool = True
    error: str | None = None
    run_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

