"""
Local BM25 keyword retrieval over the same chunk corpus stored in ChromaDB.

Uses a lightweight Arabic-aware tokenizer (word-boundary regex over Arabic +
Latin/alnum characters) so BM25 works well on Arabic text without requiring a
heavyweight NLP stack.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from src.utils.logger import RunTimer, pipeline_logger

_TOKEN_RE = re.compile(r"[\u0621-\u064A\u0660-\u0669A-Za-z0-9]+")

# Short, high-frequency Arabic stopwords/particles that add noise to BM25.
_ARABIC_STOPWORDS = {
    "في", "من", "الى", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "الذي", "التي", "الذين", "و", "ثم", "او", "أو", "ان", "أن", "إن", "كان",
    "كانت", "يكون", "لا", "ما", "لم", "لن", "قد", "له", "لها", "هو", "هي",
    "هم", "كل", "بعض", "بين", "عند", "بعد", "قبل", "حتى", "كما", "أي", "اي",
}


def arabic_tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _ARABIC_STOPWORDS and len(t) > 1]


class Bm25Retriever:
    """In-memory BM25 index built from the current Chroma corpus.

    Rebuilding is cheap enough (seconds for a few thousand chunks) to just
    reconstruct it whenever the corpus changes; callers should call
    `build(chunks)` once after ingestion and reuse the instance thereafter.
    """

    _lock = threading.Lock()

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus: List[Dict[str, Any]] = []

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        with self._lock, RunTimer() as timer:
            self._corpus = chunks
            tokenized = [arabic_tokenize(c["text"]) for c in chunks]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
        pipeline_logger.log("bm25_index_built", num_docs=len(chunks), latency_ms=timer.elapsed_ms)

    def is_ready(self) -> bool:
        return self._bm25 is not None and len(self._corpus) > 0

    def query(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.is_ready():
            return []
        with RunTimer() as timer:
            tokens = arabic_tokenize(query_text)
            scores = self._bm25.get_scores(tokens)
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            hits = []
            max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
            for idx in ranked_idx:
                if scores[idx] <= 0:
                    continue
                doc = self._corpus[idx]
                hits.append(
                    {
                        "chunk_id": doc["chunk_id"],
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {}),
                        "score": float(scores[idx] / max_score),  # normalized 0..1
                        "raw_bm25_score": float(scores[idx]),
                    }
                )
        pipeline_logger.log(
            "bm25_retrieval",
            query=query_text,
            top_k=top_k,
            num_hits=len(hits),
            latency_ms=timer.elapsed_ms,
        )
        return hits
