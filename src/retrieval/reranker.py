"""
API-based multilingual reranking.

Supported providers (RERANKER_PROVIDER in .env):
- jina:        Jina AI Rerank API (one request for all candidates)
- huggingface: Hugging Face Inference API (one request per candidate)
- local:       no model; keeps the retrieval order (score-based ordering)

No local reranker model is downloaded.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import requests
from huggingface_hub import InferenceClient

from config import settings
from src.utils.logger import RunTimer, pipeline_logger


class RerankerError(RuntimeError):
    """Raised when the reranker cannot complete."""


def _retrieval_score(item: Dict[str, Any]) -> float:
    """Score used when no reranker model is available."""
    return float(item.get("rrf_score") or item.get("score") or 0.0)


class MultilingualReranker:
    """Reranks query/document pairs using a remote reranker API."""

    _JINA_URL = "https://api.jina.ai/v1/rerank"
    _JINA_DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
    ) -> None:
        cfg = settings.reranker

        self.provider = cfg.provider.strip().lower()
        self._jina_mode = self.provider == "jina"

        self.model_name = model_name or cfg.model_name
        self.top_k = top_k or cfg.top_k
        self.timeout = cfg.hf_timeout
        self.max_retries = cfg.max_retries
        self._client = None
        self._jina_key = ""

        # ---------------------------------------------------------------
        # Jina
        # ---------------------------------------------------------------
        if self._jina_mode:
            self._fallback_mode = False
            self._jina_key = os.getenv("JINA_API_KEY", "").strip()
            if not self._jina_key:
                raise RerankerError(
                    "JINA_API_KEY is missing. Set it in the .env file."
                )
            # If RERANKER_MODEL still points to a non-Jina model, use the
            # default Jina reranker instead of sending a wrong model name.
            if "jina" not in (self.model_name or "").lower():
                self.model_name = self._JINA_DEFAULT_MODEL

            pipeline_logger.log(
                "reranker_api_init",
                provider="jina",
                model=self.model_name,
            )
            return

        # ---------------------------------------------------------------
        # Local (no model) / Hugging Face
        # ---------------------------------------------------------------
        self._fallback_mode = (
            self.provider == "local"
            or not cfg.hf_api_token.strip()
        )

        if self._fallback_mode:
            pipeline_logger.log(
                "reranker_api_init",
                provider="local",
                model=self.model_name,
                fallback="score-based-ordering",
            )
            return

        token = cfg.hf_api_token.strip()

        if not token:
            raise RerankerError(
                "HF_API_TOKEN is missing. Set it in the .env file."
            )

        self._client = InferenceClient(
            provider="hf-inference",
            api_key=token,
            timeout=self.timeout,
        )

        pipeline_logger.log(
            "reranker_api_init",
            provider="huggingface",
            model=self.model_name,
        )

    # ======================================================================
    # Shared helper: keep retrieval order (used as a safe fallback)
    # ======================================================================
    def _order_by_retrieval(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        k: int,
        provider: str,
        degraded: bool = False,
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            candidates,
            key=_retrieval_score,
            reverse=True,
        )[:k]

        out: List[Dict[str, Any]] = []
        for item in ranked:
            updated = dict(item)
            updated["rerank_score"] = _retrieval_score(item)
            if degraded:
                updated["reranker_degraded"] = True
            out.append(updated)

        pipeline_logger.log(
            "rerank",
            provider=provider,
            model=self.model_name,
            query=query,
            num_candidates=len(candidates),
            top_k=k,
            scores=[round(item["rerank_score"], 4) for item in out],
            fallback=True,
            degraded=degraded,
        )
        return out

    # ======================================================================
    # Jina: ONE request for all candidates
    # ======================================================================
    def _call_jina(
        self,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """
        Call the Jina rerank endpoint.

        Fails fast on 4xx errors (bad key, no credits, bad request) and
        retries only 429, 5xx and network errors.
        """

        headers = {
            "Authorization": f"Bearer {self._jina_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self._JINA_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(
                        f"Jina transient error {resp.status_code}: "
                        f"{resp.text[:200]}"
                    )

                if resp.status_code >= 400:
                    # 400/401/402/403/404/422: retrying cannot help.
                    pipeline_logger.log(
                        "error",
                        stage="reranker_api",
                        model=self.model_name,
                        attempt=attempt,
                        error=f"{resp.status_code}: {resp.text[:200]}",
                    )
                    raise RerankerError(
                        f"Jina rerank error {resp.status_code}: "
                        f"{resp.text[:200]}"
                    )

                results = resp.json().get("results")
                if not results:
                    raise RerankerError(
                        "Jina returned an empty rerank response."
                    )
                return results

            except RerankerError:
                raise  # non-retryable
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                pipeline_logger.log(
                    "error",
                    stage="reranker_api",
                    model=self.model_name,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        raise RerankerError(
            f"Jina reranker failed after {self.max_retries} attempts: "
            f"{last_error}"
        )

    def _rerank_jina(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        max_candidates = max(k * 4, 60)
        candidates = candidates[:max_candidates]
        documents = [c["text"] for c in candidates]

        with RunTimer() as timer:
            try:
                results = self._call_jina(
                    query,
                    documents,
                    top_n=min(k, len(documents)),
                )
            except RerankerError as exc:
                # Degrade gracefully: keep retrieval order, flag it.
                pipeline_logger.log(
                    "error",
                    stage="reranker_degraded",
                    model=self.model_name,
                    error=str(exc),
                )
                return self._order_by_retrieval(
                    query, candidates, k, provider="jina", degraded=True
                )

            ranked: List[Dict[str, Any]] = []
            for res in results:
                idx = int(res["index"])
                if idx < 0 or idx >= len(candidates):
                    continue
                updated = dict(candidates[idx])
                updated["rerank_score"] = float(res["relevance_score"])
                ranked.append(updated)

            ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
            ranked = ranked[:k]

        pipeline_logger.log(
            "rerank",
            provider="jina",
            model=self.model_name,
            query=query,
            num_candidates=len(candidates),
            top_k=k,
            scores=[round(item["rerank_score"], 4) for item in ranked],
            latency_ms=timer.elapsed_ms,
        )

        return ranked

    # ======================================================================
    # Hugging Face: one request per pair (unchanged behaviour)
    # ======================================================================
    def _score_pair(self, query: str, document: str) -> float:
        """
        Score one query/document pair using Hugging Face.

        BGE reranker is exposed through HF as a text-classification task.
        The model produces a relevance score for the supplied text.
        """

        last_error: Exception | None = None

        # Keep query/document clearly separated.
        text = f"{query} [SEP] {document}"

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._client.text_classification(
                    text=text,
                    model=self.model_name,
                    top_k=None,
                    function_to_apply="sigmoid",
                )

                if not result:
                    raise RerankerError(
                        "Hugging Face returned an empty reranker response."
                    )

                scores = []

                for item in result:
                    if hasattr(item, "score"):
                        scores.append(float(item.score))
                    elif isinstance(item, dict) and "score" in item:
                        scores.append(float(item["score"]))

                if not scores:
                    raise RerankerError(
                        f"Unexpected Hugging Face response: {result!r}"
                    )

                return max(scores)

            except Exception as exc:  # noqa: BLE001
                last_error = exc

                pipeline_logger.log(
                    "error",
                    stage="reranker_api",
                    model=self.model_name,
                    attempt=attempt,
                    error=str(exc),
                )

                # 4xx other than 429 cannot be fixed by retrying.
                status = getattr(
                    getattr(exc, "response", None), "status_code", None
                )
                if status in (400, 401, 402, 403, 404, 422):
                    break

                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))

        raise RerankerError(
            f"Hugging Face reranker failed: {last_error}"
        )

    def _rerank_hf(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        max_candidates = max(k * 4, 60)
        candidates = candidates[:max_candidates]

        with RunTimer() as timer:
            reranked: List[Dict[str, Any]] = []
            failures = 0

            max_workers = min(10, len(candidates)) or 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_candidate = {
                    executor.submit(self._score_pair, query, c["text"]): c
                    for c in candidates
                }
                for future in as_completed(future_to_candidate):
                    candidate = future_to_candidate[future]
                    try:
                        score = future.result()
                    except Exception as exc:  # noqa: BLE001
                        failures += 1
                        pipeline_logger.log(
                            "error",
                            stage="reranker_pair",
                            model=self.model_name,
                            error=str(exc),
                        )
                        score = 0.0

                    updated = dict(candidate)
                    updated["rerank_score"] = float(score)
                    reranked.append(updated)

            # If every pair failed, scores are meaningless: keep the
            # retrieval order instead of returning arbitrary 0.0 ordering.
            if failures == len(candidates):
                return self._order_by_retrieval(
                    query, candidates, k,
                    provider="huggingface", degraded=True,
                )

            ranked = sorted(
                reranked,
                key=lambda item: item["rerank_score"],
                reverse=True,
            )[:k]

        pipeline_logger.log(
            "rerank",
            provider="huggingface",
            model=self.model_name,
            query=query,
            num_candidates=len(candidates),
            top_k=k,
            scores=[round(item["rerank_score"], 4) for item in ranked],
            latency_ms=timer.elapsed_ms,
        )

        return ranked

    # ======================================================================
    # Public API
    # ======================================================================
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates and return the highest-scoring chunks.
        """

        if not candidates:
            return []

        k = top_k or self.top_k

        if self._jina_mode:
            return self._rerank_jina(query, candidates, k)

        if self._fallback_mode:
            return self._order_by_retrieval(
                query, candidates, k, provider="local"
            )

        return self._rerank_hf(query, candidates, k)