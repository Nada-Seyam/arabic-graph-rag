"""
Centralized LLM adapter for Ollama Cloud.

This is the ONLY place in the whole project that talks to an LLM. Every
component that needs text generation (routing, query rewriting, HyDE,
self-query, CRAG grading, answer generation, evaluation judging, ...) must go
through `OllamaCloudClient`. This guarantees:

- a single point of configuration (model, host, key, retries, timeouts),
- centralized cost/token tracking,
- centralized structured logging of every LLM call.

No local LLM is ever downloaded or run — all inference happens against the
Ollama Cloud REST API (`{OLLAMA_HOST}/api/chat`).
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from config import settings
from src.utils.cost_tracker import CostTracker, estimate_tokens, global_cost_tracker
from src.utils.logger import pipeline_logger


class OllamaCloudError(RuntimeError):
    """Raised when the Ollama Cloud API cannot fulfil a request after retries."""


# Ollama Cloud enforces an account-level cap on *concurrent* in-flight
# requests that is considerably stricter than typical API rate limits
# (observed: bursts of just a handful of simultaneous requests from the
# concurrent compression/reranking steps reliably trigger
# '429 too many concurrent requests', which can then cascade and fail
# even unrelated subsequent calls). Since every component in this project
# goes through this single client, gating concurrency here — rather than
# in each individual caller — is the one place that can guarantee the
# limit is respected regardless of how many worker threads a caller uses.
# Raise this only if your Ollama Cloud plan supports higher concurrency.
_MAX_CONCURRENT_OLLAMA_REQUESTS = 2


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw: Dict[str, Any]


class OllamaCloudClient:
    """Thin, robust wrapper around the Ollama Cloud chat completion API.

    Parameters are pulled from `config.settings.ollama` and can be overridden
    per-call. No API key or model name is ever hard-coded here.
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None) -> None:
        cfg = settings.ollama
        if not cfg.api_key:
            pipeline_logger.log(
                "warning",
                message=(
                    "OLLAMA_API_KEY is not set. Set it in your .env file before "
                    "making live calls to Ollama Cloud."
                ),
            )
        self.api_key = cfg.api_key
        self.host = cfg.host.rstrip("/")
        self.default_model = cfg.model
        self.fast_model = cfg.get_fast_model()
        self.timeout = cfg.timeout
        self.max_retries = cfg.max_retries
        self.default_temperature = cfg.temperature
        self.default_max_tokens = cfg.max_tokens
        self.cost_tracker = cost_tracker or global_cost_tracker

        # Shared across every call made through this client instance (and
        # every component uses the same shared `llm_client` singleton), so
        # this correctly gates true concurrency account-wide, not just
        # per-caller.
        self._concurrency_semaphore = threading.Semaphore(_MAX_CONCURRENT_OLLAMA_REQUESTS)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        purpose: str = "generation",
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama Cloud and return the text.

        Args:
            messages: list of {"role": "system"|"user"|"assistant", "content": str}
            model: overrides the configured default model for this call.
            temperature: sampling temperature override.
            max_tokens: max output tokens override.
            purpose: label used for cost/log bucketing (e.g. "routing", "hyde").
            json_mode: if True, asks Ollama to constrain output to valid JSON.
        """
        resolved_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": (
                    self.default_temperature if temperature is None else temperature
                ),
                "num_predict": max_tokens or self.default_max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.host}/api/chat"
        last_error: Optional[Exception] = None
        start = time.perf_counter()

        for attempt in range(1, self.max_retries + 1):
            try:
                # Gate the actual network call so at most
                # _MAX_CONCURRENT_OLLAMA_REQUESTS requests are ever
                # in-flight at once across the whole app, regardless of how
                # many worker threads a caller (e.g. contextual compression,
                # which compresses several chunks concurrently) spins up.
                with self._concurrency_semaphore:
                    resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    raise OllamaCloudError(
                        f"Ollama Cloud returned HTTP 429 (rate/concurrency limited): {resp.text[:500]}"
                    )
                if resp.status_code >= 400:
                    raise OllamaCloudError(
                        f"Ollama Cloud returned HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                text = self._extract_text(data)
                prompt_tokens = data.get("prompt_eval_count") or estimate_tokens(
                    _flatten_messages(messages)
                )
                completion_tokens = data.get("eval_count") or estimate_tokens(text)

                self.cost_tracker.record(
                    model=resolved_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    purpose=purpose,
                )
                pipeline_logger.log(
                    "llm_call",
                    purpose=purpose,
                    model=resolved_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    attempt=attempt,
                )
                return LLMResponse(
                    text=text,
                    model=resolved_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    raw=data,
                )
            except Exception as exc:  # noqa: BLE001 - we want to retry broadly then raise
                last_error = exc
                pipeline_logger.log(
                    "error",
                    stage="llm_call",
                    purpose=purpose,
                    model=resolved_model,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < self.max_retries:
                    # 429s mean "you're over your concurrency/rate limit
                    # right now" rather than a transient blip, so back off
                    # noticeably longer than for other errors before
                    # retrying, giving other in-flight requests time to
                    # finish and free up capacity.
                    if "429" in str(exc):
                        time.sleep(min(5 * attempt, 20))
                    else:
                        time.sleep(min(2 ** attempt, 8))

        raise OllamaCloudError(
            f"Ollama Cloud request failed after {self.max_retries} attempts: {last_error}"
        )

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        purpose: str = "generation",
        json_mode: bool = False,
    ) -> str:
        """Convenience wrapper: single-turn prompt -> text response."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose,
            json_mode=json_mode,
        )
        return response.text

    def complete_json(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        purpose: str = "generation",
    ) -> Dict[str, Any]:
        """Ask the model for JSON and parse it robustly, tolerating minor
        formatting issues (code fences, leading/trailing text)."""
        text = self.complete(
            prompt,
            system=system,
            model=model or self.fast_model,
            temperature=temperature if temperature is not None else 0.0,
            purpose=purpose,
            json_mode=True,
        )
        return _safe_json_parse(text)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        message = data.get("message") or {}
        if isinstance(message, dict) and message.get("content"):
            return message["content"].strip()
        # Some Ollama-compatible endpoints stream differently or return
        # OpenAI-style choices; handle both defensively.
        choices = data.get("choices")
        if choices:
            choice = choices[0]
            msg = choice.get("message", {})
            if msg.get("content"):
                return msg["content"].strip()
            if choice.get("text"):
                return choice["text"].strip()
        return str(data.get("response", "")).strip()


def _flatten_messages(messages: List[Dict[str, str]]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Parse JSON robustly from an LLM response.

    Handles:
    - normal JSON
    - ```json ... ``` code fences
    - leading/trailing explanatory text
    - JSON objects embedded in surrounding text
    """

    cleaned = text.strip()

    # Remove Markdown code fences.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    # First attempt: the entire cleaned response.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Second attempt: extract the JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end > start:
        candidate = cleaned[start : end + 1]

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Third attempt: extract a JSON array.
    start_arr = cleaned.find("[")
    end_arr = cleaned.rfind("]")

    if start_arr != -1 and end_arr > start_arr:
        candidate = cleaned[start_arr : end_arr + 1]

        try:
            parsed_array = json.loads(candidate)
            return {"items": parsed_array}
        except json.JSONDecodeError:
            pass

    pipeline_logger.log(
        "error",
        stage="json_parse",
        error="Unable to parse LLM response as JSON",
        raw_text=text[:1000],
    )

    return {}


# Shared singleton instance for convenience — components may still create
# their own `OllamaCloudClient()` (e.g. with a dedicated CostTracker) if
# they need isolated accounting.
llm_client = OllamaCloudClient()