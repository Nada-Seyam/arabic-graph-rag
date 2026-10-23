"""
Centralized configuration for the Arabic Graph RAG system.

All tunable parameters are loaded from environment variables (via a `.env`
file) so that no component ever hard-codes a model name, path, or threshold.
Every other module in this project must import settings from here instead of
reading `os.environ` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load `.env` from the project root (if present). Safe to call multiple times.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


@dataclass(frozen=True)
class OllamaConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OLLAMA_API_KEY", ""))
    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "https://ollama.com"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:latest"))
    fast_model: str = field(default_factory=lambda: os.getenv("OLLAMA_FAST_MODEL", "") or "")
    timeout: int = field(default_factory=lambda: _get_int("OLLAMA_TIMEOUT", 120))
    max_retries: int = field(default_factory=lambda: _get_int("OLLAMA_MAX_RETRIES", 3))
    temperature: float = field(default_factory=lambda: _get_float("OLLAMA_TEMPERATURE", 0.2))
    max_tokens: int = field(default_factory=lambda: _get_int("OLLAMA_MAX_TOKENS", 1024))
    cost_per_1k_prompt: float = field(
        default_factory=lambda: _get_float("COST_PER_1K_PROMPT_TOKENS", 0.0)
    )
    cost_per_1k_completion: float = field(
        default_factory=lambda: _get_float("COST_PER_1K_COMPLETION_TOKENS", 0.0)
    )

    def get_fast_model(self) -> str:
        return self.fast_model if self.fast_model else self.model


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    )
    cache_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("EMBEDDING_CACHE_DIR", "./data/model_cache"))
    )
    batch_size: int = field(default_factory=lambda: _get_int("EMBEDDING_BATCH_SIZE", 32))
    # Which embedding backend to use: "local" (fastembed, default) or
    # "huggingface" (Hugging Face Inference API). See `src/embeddings/embedder.py`.
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))
    # Hugging Face Inference API settings (only used when provider="huggingface").
    hf_api_token: str = field(default_factory=lambda: os.getenv("HF_API_TOKEN", ""))
    hf_model_id: str = field(
        default_factory=lambda: os.getenv("HF_MODEL_ID", "intfloat/multilingual-e5-large")
    )
    hf_timeout: int = field(default_factory=lambda: _get_int("HF_TIMEOUT", 120))


@dataclass(frozen=True)
class RerankerConfig:
    provider: str = field(
        default_factory=lambda: os.getenv("RERANKER_PROVIDER", "local")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3",
        )
    )
    hf_api_token: str = field(
        default_factory=lambda: os.getenv("HF_API_TOKEN", "")
    )
    hf_timeout: int = field(
        default_factory=lambda: _get_int("RERANKER_HF_TIMEOUT", 120)
    )
    max_retries: int = field(
        default_factory=lambda: _get_int("RERANKER_MAX_RETRIES", 3)
    )
    top_k: int = field(
        default_factory=lambda: _get_int("RERANKER_TOP_K", 5)
    )

@dataclass(frozen=True)
class ChromaConfig:
    persist_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "arabic_graph_rag")
    )


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))
    min_chunk_chars: int = field(default_factory=lambda: _get_int("MIN_CHUNK_CHARS", 40))


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_K", 8))
    dense_top_k: int = field(default_factory=lambda: _get_int("DENSE_TOP_K", 15))
    bm25_top_k: int = field(default_factory=lambda: _get_int("BM25_TOP_K", 15))
    rrf_k: int = field(default_factory=lambda: _get_int("RRF_K", 60))
    hybrid_alpha: float = field(default_factory=lambda: _get_float("HYBRID_ALPHA", 0.5))


@dataclass(frozen=True)
class GraphConfig:
    store_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("GRAPH_STORE_DIR", "./data/graph"))
    )
    store_file: str = field(
        default_factory=lambda: os.getenv("GRAPH_STORE_FILE", "knowledge_graph.json")
    )
    expansion_hops: int = field(default_factory=lambda: _get_int("GRAPH_EXPANSION_HOPS", 1))
    max_neighbors: int = field(default_factory=lambda: _get_int("GRAPH_MAX_NEIGHBORS", 8))
    min_entity_len: int = field(default_factory=lambda: _get_int("GRAPH_MIN_ENTITY_LEN", 2))

    @property
    def store_path(self) -> Path:
        return self.store_dir / self.store_file


@dataclass(frozen=True)
class CragConfig:
    relevance_threshold: float = field(
        default_factory=lambda: _get_float("CRAG_RELEVANCE_THRESHOLD", 0.55)
    )
    max_corrective_queries: int = field(
        default_factory=lambda: _get_int("CRAG_MAX_CORRECTIVE_QUERIES", 2)
    )


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("LOG_DIR", "./logs")))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "pipeline.jsonl"))
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    console: bool = field(default_factory=lambda: _get_bool("CONSOLE_LOG", True))

    @property
    def log_path(self) -> Path:
        return self.log_dir / self.log_file


@dataclass(frozen=True)
class EvaluationConfig:
    reports_dir: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("REPORTS_DIR", "./reports"))
    )
    dataset_path: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("EVAL_DATASET", "./data/eval/eval_questions.json")
        )
    )


@dataclass(frozen=True)
class DataConfig:
    pdf_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("PDF_DIR", "./data/pdfs")))


@dataclass(frozen=True)
class Settings:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    crag: CragConfig = field(default_factory=CragConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    project_root: Path = PROJECT_ROOT

    def ensure_directories(self) -> None:
        """Create every directory this project writes to, if missing."""
        for path in (
            self.chroma.persist_dir,
            self.graph.store_dir,
            self.logging.log_dir,
            self.evaluation.reports_dir,
            self.data.pdf_dir,
            self.evaluation.dataset_path.parent,
            self.embedding.cache_dir,
            
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
