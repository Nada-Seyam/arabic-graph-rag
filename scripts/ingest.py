#!/usr/bin/env python3
"""
CLI: Ingest all PDFs in `PDF_DIR` into the local ChromaDB vector store and
BM25 index, then build (or rebuild) the knowledge graph from the resulting
chunks.

Usage:
    python scripts/ingest.py [--reset] [--skip-graph]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.ingestion.ingestion_pipeline import IngestionPipeline  # noqa: E402
from src.utils.logger import pipeline_logger  # noqa: E402
from src.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Arabic PDFs into the Graph RAG system.")
    parser.add_argument(
        "--reset", action="store_true", help="Wipe the existing ChromaDB collection before ingesting."
    )
    parser.add_argument(
        "--skip-graph", action="store_true", help="Skip knowledge graph construction (chunks only)."
    )
    parser.add_argument(
        "--pdf-dir", type=str, default=None, help="Override the PDF_DIR configured in .env"
    )
    args = parser.parse_args()

    vector_store = ChromaVectorStore()
    if args.reset:
        print("Resetting existing ChromaDB collection...")
        vector_store.reset()

    pipeline = IngestionPipeline(vector_store=vector_store)
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else settings.data.pdf_dir

    print(f"Ingesting PDFs from: {pdf_dir}")
    chunks = pipeline.ingest_directory(pdf_dir)
    print(f"Ingested {len(chunks)} chunks into ChromaDB (total in store: {vector_store.count()}).")

    if not args.skip_graph and chunks:
        print("Building knowledge graph (this calls the LLM for entity/relationship extraction)...")
        graph_path = pipeline.build_knowledge_graph(chunks)
        print(f"Knowledge graph saved to: {graph_path}")
    elif not chunks:
        print("No chunks produced — skipping knowledge graph construction.")
    else:
        print("Skipped knowledge graph construction (--skip-graph).")

    pipeline_logger.log("cli_ingest_complete", num_chunks=len(chunks))
    print("Done.")


if __name__ == "__main__":
    main()
