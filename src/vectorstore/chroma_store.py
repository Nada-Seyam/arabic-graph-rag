"""
Persistent local ChromaDB vector store for chunk embeddings.

Embeddings are computed via whichever backend `get_embedder()` resolves to
(local `ArabicEmbedder` or the remote `HFAPIEmbedder`, per `EMBEDDING_PROVIDER`)
and stored/queried in a disk-persisted ChromaDB collection — no external
vector DB service is used, regardless of the embedding backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from src.embeddings.embedder import Embedder, get_embedder
from src.ingestion.chunker import Chunk
from src.utils.logger import RunTimer, pipeline_logger


class ChromaVectorStore:
    """Thin wrapper around a persistent Chroma collection with our own
    embedding function (so embeddings stay swappable between local and
    remote backends via `EMBEDDING_PROVIDER`)."""

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder = embedder or get_embedder()
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        with RunTimer() as timer:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors.tolist(),
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "source": c.source,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "chunk_index": c.chunk_index,
                        **{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))},
                    }
                    for c in chunks
                ],
            )
        pipeline_logger.log(
            "vectorstore_upsert", count=len(chunks), latency_ms=timer.elapsed_ms
        )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def query(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        with RunTimer() as timer:
            query_vector = self.embedder.embed_query(query_text)
            result = self._collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=max(1, top_k),
                include=["documents", "metadatas", "distances"],
            )
        hits: List[Dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
            similarity = 1.0 - dist  # cosine distance -> similarity
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "score": float(similarity),
                }
            )
        pipeline_logger.log(
            "dense_retrieval",
            query=query_text,
            top_k=top_k,
            num_hits=len(hits),
            latency_ms=timer.elapsed_ms,
        )
        return hits

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Fetch every stored chunk (used to (re)build BM25 index and the graph)."""
        result = self._collection.get(include=["documents", "metadatas"])
        out = []
        for chunk_id, doc, meta in zip(
            result.get("ids", []), result.get("documents", []), result.get("metadatas", [])
        ):
            out.append({"chunk_id": chunk_id, "text": doc, "metadata": meta})
        return out

    def reset(self) -> None:
        self._client.delete_collection(settings.chroma.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
