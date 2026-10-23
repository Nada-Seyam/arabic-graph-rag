"""
Cost and token tracking for LLM calls made through the centralized Ollama
Cloud adapter.

Costs are estimated locally using configurable per-1K-token rates (Ollama
Cloud's own billing may differ; these are for internal observability and
reporting only). All calls are logged individually and aggregated per run.

`estimate_tokens` is kept in this module (rather than its own file) since it
exists purely to support cost/token accounting when Ollama Cloud doesn't
report exact counts.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List

from config import settings

_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word for Arabic/English mixed text,
    with a floor based on character count for very short strings."""
    if not text:
        return 0
    words = len(_WORD_RE.findall(text))
    char_based = max(1, len(text) // 4)
    return max(words, char_based)


@dataclass
class LLMCallCost:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    purpose: str = "generation"


@dataclass
class CostTracker:
    """Accumulates LLM usage/cost across a single run. Thread-safe."""

    calls: List[LLMCallCost] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(
        self, model: str, prompt_tokens: int, completion_tokens: int, purpose: str = "generation"
    ) -> LLMCallCost:
        cost = (
            prompt_tokens / 1000.0 * settings.ollama.cost_per_1k_prompt
            + completion_tokens / 1000.0 * settings.ollama.cost_per_1k_completion
        )
        entry = LLMCallCost(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 8),
            purpose=purpose,
        )
        with self._lock:
            self.calls.append(entry)
        return entry

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.calls), 8)

    def summary(self) -> Dict:
        return {
            "num_calls": len(self.calls),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost_usd": self.total_cost_usd,
            "by_purpose": self._by_purpose(),
        }

    def _by_purpose(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for c in self.calls:
            bucket = out.setdefault(
                c.purpose, {"num_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
            )
            bucket["num_calls"] += 1
            bucket["prompt_tokens"] += c.prompt_tokens
            bucket["completion_tokens"] += c.completion_tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + c.cost_usd, 8)
        return out

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()


# A process-wide tracker. Individual pipeline runs may also create their own
# `CostTracker()` instance if isolated accounting per run is desired.
global_cost_tracker = CostTracker()
