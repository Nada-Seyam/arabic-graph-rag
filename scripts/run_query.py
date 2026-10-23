#!/usr/bin/env python3
"""
CLI: Ask a single question against the Arabic Graph RAG system, or start an
interactive REPL if no question is provided.

Usage:
    python scripts/run_query.py "ما هي خطوات استلام البريد الوارد؟"
    python scripts/run_query.py            # interactive mode
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import GraphRAGPipeline  # noqa: E402


def print_result(result) -> None:
    print("\n" + "=" * 70)
    print(f"Route: {result.route}")
    print(f"Techniques executed: {', '.join(result.techniques_executed)}")
    print(f"Model used: {result.model_used}")
    if result.crag_info:
        print(f"CRAG: {json.dumps(result.crag_info, ensure_ascii=False)}")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)
    if result.citations:
        print("Citations:")
        for c in result.citations:
            print(f"  - {c['source']} (page {c['page_start']}-{c['page_end']})")
    print(f"Cost summary: {json.dumps(result.cost_summary, ensure_ascii=False)}")
    if result.errors:
        print(f"Errors encountered: {result.errors}")
    print("=" * 70)


def main() -> None:
    pipeline = GraphRAGPipeline()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = pipeline.answer(query)
        print_result(result)
        return

    print("Arabic Graph RAG — interactive mode. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            query = input("\nسؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "خروج"}:
            break
        result = pipeline.answer(query)
        print_result(result)


if __name__ == "__main__":
    main()
