"""
Embedding backends for the Arabic Graph RAG system.

Supported providers:
- local: FastEmbed with a local ONNX model
- huggingface: Hugging Face Inference Providers API
- jina: Jina AI Embeddings API

The Hugging Face and Jina backends do NOT download the embedding model locally.
"""

from __future__ import annotations

import os
import time
from typing import Protocol, Sequence

import numpy as np
import requests

from config import settings
from src.utils.logger import pipeline_logger


class Embedder(Protocol):
    """Common interface implemented by all embedding backends."""

    @property
    def dimension(self) -> int:
        ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


# ============================================================================
# Local FastEmbed backend
# ============================================================================

class ArabicEmbedder:
    """
    Local embedding backend using FastEmbed.

    This backend downloads and caches the configured model locally.
    """

    def __init__(self) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "FastEmbed is not installed. "
                "Install it with: pip install fastembed"
            ) from exc

        model_name = settings.embedding.model_name
        cache_dir = settings.embedding.cache_dir

        pipeline_logger.log(
            "embedding_init",
            provider="local",
            model=model_name,
            cache_dir=str(cache_dir),
        )

        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
        )

        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        if self._dimension is None:
            vector = self.embed_query("اختبار")
            self._dimension = int(vector.shape[0])

        return self._dimension

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Embed documents using the E5 passage prefix."""

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        prepared = [
            f"passage: {text}"
            for text in texts
        ]

        batch_size = max(
            1,
            settings.embedding.batch_size,
        )

        embeddings = list(
            self._model.embed(
                prepared,
                batch_size=batch_size,
            )
        )

        result = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if result.ndim != 2:
            raise RuntimeError(
                f"Unexpected FastEmbed output shape: {result.shape}"
            )

        if self._dimension is None:
            self._dimension = int(result.shape[1])

        return result

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        """Embed a query using the E5 query prefix."""

        prepared = f"query: {text}"

        embeddings = list(
            self._model.embed(
                [prepared],
                batch_size=1,
            )
        )

        if not embeddings:
            raise RuntimeError(
                "FastEmbed returned no embedding for the query."
            )

        vector = np.asarray(
            embeddings[0],
            dtype=np.float32,
        ).reshape(-1)

        if self._dimension is None:
            self._dimension = int(vector.shape[0])

        return vector


# ============================================================================
# Hugging Face API backend
# ============================================================================

class HFAPIEmbedderError(RuntimeError):
    """Raised when Hugging Face embedding inference fails."""


class HFAPIEmbedder:
    """
    Hugging Face embedding backend.

    Uses huggingface_hub.InferenceClient with provider='auto'.

    The embedding model remains remote and is NOT downloaded locally.
    """

    def __init__(self) -> None:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise HFAPIEmbedderError(
                "huggingface_hub is not installed. "
                "Run: pip install -U huggingface_hub"
            ) from exc

        token = settings.embedding.hf_api_token.strip()
        model = settings.embedding.hf_model_id.strip()
        timeout = settings.embedding.hf_timeout

        if not token:
            raise HFAPIEmbedderError(
                "HF_API_TOKEN is missing. "
                "Set it in the .env file."
            )

        if not model:
            raise HFAPIEmbedderError(
                "HF_MODEL_ID is missing. "
                "Set it in the .env file."
            )

        self._model = model
        self._timeout = timeout
        self._dimension: int | None = None

        self._client = InferenceClient(
            model=model,
            token=token,
            provider="auto",
            timeout=timeout,
        )

        pipeline_logger.log(
            "embedding_init",
            provider="huggingface",
            model=model,
        )

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension.

        It is discovered lazily with one small API request.
        """

        if self._dimension is None:
            vector = self.embed_query("اختبار")
            self._dimension = int(vector.shape[0])

        return self._dimension

    @staticmethod
    def _to_embedding_array(result) -> np.ndarray:
        """
        Convert Hugging Face feature-extraction output to float32.

        Supported forms:

        1. [D]
           Single embedding vector.

        2. [N, D]
           N sentence/document embeddings.

        3. [N, T, D]
           Token-level embeddings. Mean pooling is applied over T.
        """

        # PyTorch tensor
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()

        # Other tensor-like objects
        elif (
            hasattr(result, "numpy")
            and not isinstance(result, np.ndarray)
        ):
            result = result.numpy()

        arr = np.asarray(
            result,
            dtype=np.float32,
        )

        if arr.size == 0:
            raise HFAPIEmbedderError(
                "Hugging Face returned an empty embedding."
            )

        # Single vector: [D]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Sentence embeddings: [N, D]
        elif arr.ndim == 2:
            pass

        # Token embeddings: [N, T, D]
        elif arr.ndim == 3:
            arr = arr.mean(axis=1)

        else:
            raise HFAPIEmbedderError(
                "Unexpected Hugging Face embedding shape: "
                f"{arr.shape}"
            )

        return arr.astype(
            np.float32,
            copy=False,
        )

    def _call_api(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Send feature-extraction request through Hugging Face.

        Retries transient failures up to three times.
        """

        if not texts:
            raise HFAPIEmbedderError(
                "Cannot embed an empty list of texts."
            )

        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                result = self._client.feature_extraction(
                    list(texts),
                    model=self._model,
                )

                embeddings = self._to_embedding_array(result)

                pipeline_logger.log(
                    "hf_embedding_call",
                    model=self._model,
                    attempt=attempt,
                    count=len(texts),
                    dimension=int(embeddings.shape[-1]),
                    status="success",
                )

                return embeddings

            except Exception as exc:
                last_error = exc

                pipeline_logger.log(
                    "error",
                    stage="hf_embedding_call",
                    model=self._model,
                    attempt=attempt,
                    error=str(exc),
                )

                if attempt < max_retries:
                    # 1s, 2s between retries
                    time.sleep(2 ** (attempt - 1))

        raise HFAPIEmbedderError(
            "Hugging Face Inference API request failed after "
            f"{max_retries} attempts: {last_error}"
        ) from last_error

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Embed documents using the E5 passage prefix.

        Documents are sent in batches to avoid unnecessarily large
        API requests.
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        prepared = [
            f"passage: {text}"
            for text in texts
        ]

        batch_size = max(
            1,
            settings.embedding.batch_size,
        )

        all_embeddings: list[np.ndarray] = []

        for start in range(
            0,
            len(prepared),
            batch_size,
        ):
            batch = prepared[
                start:start + batch_size
            ]

            embeddings = self._call_api(batch)

            if embeddings.shape[0] != len(batch):
                raise HFAPIEmbedderError(
                    "Hugging Face returned an unexpected number "
                    f"of embeddings. Expected {len(batch)}, "
                    f"got {embeddings.shape[0]}."
                )

            all_embeddings.append(embeddings)

        result = np.vstack(
            all_embeddings
        ).astype(
            np.float32,
            copy=False,
        )

        if result.ndim != 2:
            raise HFAPIEmbedderError(
                f"Unexpected final embedding shape: {result.shape}"
            )

        if self._dimension is None:
            self._dimension = int(result.shape[1])

        return result

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Embed a query using the E5 query prefix.
        """

        prepared = f"query: {text}"

        embeddings = self._call_api(
            [prepared]
        )

        if embeddings.shape[0] != 1:
            raise HFAPIEmbedderError(
                "Expected exactly one query embedding, "
                f"got shape {embeddings.shape}"
            )

        vector = embeddings[0].astype(
            np.float32,
            copy=False,
        )

        if self._dimension is None:
            self._dimension = int(vector.shape[0])

        return vector


# ============================================================================
# Jina AI API backend
# ============================================================================

class JinaEmbedderError(RuntimeError):
    """Raised when Jina embedding inference fails."""


class JinaEmbedder:
    """Remote embedding backend using the Jina AI Embeddings API."""

    _URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self) -> None:
        self._key = os.getenv("JINA_API_KEY", "").strip()
        if not self._key:
            raise JinaEmbedderError("JINA_API_KEY is missing. Set it in .env.")

        self._model = settings.embedding.model_name or "jina-embeddings-v3"
        self._timeout = settings.embedding.hf_timeout
        self._dimension: int | None = None

        pipeline_logger.log(
            "embedding_init", provider="jina", model=self._model
        )

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.embed_query("اختبار").shape[0])
        return self._dimension

    def _call_api(self, texts: Sequence[str], task: str) -> np.ndarray:
        if not texts:
            raise JinaEmbedderError("Cannot embed an empty list of texts.")

        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model, "task": task, "input": list(texts)}
        max_retries = 3
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                r = requests.post(
                    self._URL, headers=headers, json=payload,
                    timeout=self._timeout,
                )
                # Fail fast on errors that retrying cannot fix
                if r.status_code in (400, 401, 402, 403, 404, 422):
                    pipeline_logger.log(
                        "error", stage="jina_embedding_call",
                        model=self._model, attempt=attempt,
                        error=f"{r.status_code}: {r.text[:200]}",
                    )
                    raise JinaEmbedderError(
                        f"Jina API error {r.status_code}: {r.text[:200]}"
                    )
                r.raise_for_status()

                data = sorted(r.json()["data"], key=lambda d: d["index"])
                arr = np.asarray(
                    [d["embedding"] for d in data], dtype=np.float32
                )
                pipeline_logger.log(
                    "jina_embedding_call", model=self._model,
                    attempt=attempt, count=len(texts),
                    dimension=int(arr.shape[-1]), status="success",
                )
                return arr

            except JinaEmbedderError:
                raise  # non-retryable
            except Exception as exc:
                last_error = exc
                pipeline_logger.log(
                    "error", stage="jina_embedding_call",
                    model=self._model, attempt=attempt, error=str(exc),
                )
                if attempt < max_retries:
                    time.sleep(2 ** (attempt - 1))

        raise JinaEmbedderError(
            f"Jina request failed after {max_retries} attempts: {last_error}"
        ) from last_error

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        batch_size = max(1, settings.embedding.batch_size)
        parts = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            emb = self._call_api(batch, "retrieval.passage")
            if emb.shape[0] != len(batch):
                raise JinaEmbedderError(
                    f"Expected {len(batch)} embeddings, got {emb.shape[0]}."
                )
            parts.append(emb)

        result = np.vstack(parts).astype(np.float32, copy=False)
        if self._dimension is None:
            self._dimension = int(result.shape[1])
        return result

    def embed_query(self, text: str) -> np.ndarray:
        emb = self._call_api([text], "retrieval.query")
        vector = emb[0].astype(np.float32, copy=False)
        if self._dimension is None:
            self._dimension = int(vector.shape[0])
        return vector


# ============================================================================
# Factory
# ============================================================================

def get_embedder() -> Embedder:
    """
    Create the embedding backend configured in .env.

    EMBEDDING_PROVIDER=jina
        -> JinaEmbedder

    EMBEDDING_PROVIDER=huggingface
        -> HFAPIEmbedder

    EMBEDDING_PROVIDER=local
        -> ArabicEmbedder
    """

    provider = (
        settings.embedding.provider
        .strip()
        .lower()
    )

    if provider == "jina":
        return JinaEmbedder()

    if provider == "huggingface":
        try:
            return HFAPIEmbedder()
        except Exception as exc:  # pragma: no cover - defensive fallback
            pipeline_logger.log(
                "error",
                stage="embedding_provider_fallback",
                provider="huggingface",
                error=str(exc),
            )
            pipeline_logger.log(
                "embedding_fallback",
                provider="local",
                model=settings.embedding.model_name,
            )
            return ArabicEmbedder()

    if provider == "local":
        return ArabicEmbedder()

    raise ValueError(
        "Unsupported EMBEDDING_PROVIDER: "
        f"{settings.embedding.provider!r}. "
        "Expected 'local', 'huggingface' or 'jina'."
    )