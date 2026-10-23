"""
Hybrid retrieval: Dense (ChromaDB) + BM25, fused with Reciprocal Rank Fusion
(RRF). GraphRAG expansion is layered on top by the RAG pipeline — this module
focuses purely on chunk-level hybrid retrieval so it stays reusable/testable.
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import settings
from src.retrieval.bm25_retriever import Bm25Retriever
from src.utils.logger import RunTimer, pipeline_logger
from src.vectorstore.chroma_store import ChromaVectorStore


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]], k: int, weights: List[float] | None = None
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists of hit dicts (each containing "chunk_id")
    using weighted Reciprocal Rank Fusion: score = sum(weight / (k + rank))."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused: Dict[str, Dict[str, Any]] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, hit in enumerate(ranked_list, start=1):
            chunk_id = hit["chunk_id"]
            entry = fused.setdefault(
                chunk_id, {"chunk_id": chunk_id, "text": hit["text"], "metadata": hit.get("metadata", {}),
                           "rrf_score": 0.0, "component_scores": {}}
            )
            entry["rrf_score"] += weight / (k + rank)
            entry["component_scores"][hit.get("_source", "unknown")] = hit.get("score", 0.0)

    return sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)


class HybridRetriever:
    """Combines dense vector search and BM25 keyword search via RRF."""

    def __init__(self, vector_store: ChromaVectorStore, bm25: Bm25Retriever) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        cfg = settings.retrieval
        top_k = top_k or cfg.top_k
        dense_top_k = dense_top_k or cfg.dense_top_k
        bm25_top_k = bm25_top_k or cfg.bm25_top_k

        with RunTimer() as timer:
            dense_hits = self.vector_store.query(query, dense_top_k)
            for h in dense_hits:
                h["_source"] = "dense"

            bm25_hits = self.bm25.query(query, bm25_top_k) if self.bm25.is_ready() else []
            for h in bm25_hits:
                h["_source"] = "bm25"

            fused = reciprocal_rank_fusion(
                [dense_hits, bm25_hits],
                k=cfg.rrf_k,
                weights=[cfg.hybrid_alpha, 1.0 - cfg.hybrid_alpha],
            )
            results = fused[:top_k]

        pipeline_logger.log(
            "hybrid_retrieval",
            query=query,
            num_dense=len(dense_hits),
            num_bm25=len(bm25_hits),
            num_fused=len(fused),
            top_k=top_k,
            latency_ms=timer.elapsed_ms,
        )
        return results
