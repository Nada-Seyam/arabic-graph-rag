#!/usr/bin/env python3
"""
CLI: (Re)build the knowledge graph from whatever chunks are already stored in
ChromaDB, without re-running PDF extraction/chunking/embedding. Useful when
you only want to refresh entity/relationship extraction (e.g. after changing
the entity extraction prompt or the LLM model).

Usage:
    python scripts/build_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph.knowledge_graph import KnowledgeGraphBuilder  # noqa: E402
from src.ingestion.chunker import Chunk  # noqa: E402
from src.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402


def main() -> None:
    vector_store = ChromaVectorStore()
    raw_chunks = vector_store.get_all_chunks()
    if not raw_chunks:
        print("No chunks found in ChromaDB. Run scripts/ingest.py first.")
        return

    chunks = [
        Chunk(
            chunk_id=c["chunk_id"],
            text=c["text"],
            source=c["metadata"].get("source", "unknown"),
            page_start=c["metadata"].get("page_start", 0),
            page_end=c["metadata"].get("page_end", 0),
            chunk_index=c["metadata"].get("chunk_index", 0),
        )
        for c in raw_chunks
    ]

    print(f"Building knowledge graph from {len(chunks)} stored chunks...")
    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_chunks(chunks)
    path = builder.save()
    print(f"Graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
