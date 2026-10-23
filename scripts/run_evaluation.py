#!/usr/bin/env python3
"""
CLI: Run the evaluation pipeline over a JSON dataset of questions (optionally
with reference answers), producing a CSV results file and a Markdown summary
report under `REPORTS_DIR`.

Dataset format (see data/eval/eval_questions.json for the shipped example):
[
    {"question": "...", "reference_answer": "..."},
    {"question": "..."}
]

Usage:
    python scripts/run_evaluation.py [--dataset path/to/questions.json] [--label my_run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from src.evaluation.evaluator import RagEvaluator  # noqa: E402
from src.evaluation.report_generator import EvalRow, EvaluationReportGenerator  # noqa: E402
from src.rag.pipeline import GraphRAGPipeline  # noqa: E402
from src.utils.cost_tracker import CostTracker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Arabic Graph RAG pipeline.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation JSON dataset.")
    parser.add_argument("--label", type=str, default=None, help="Label used in the report filenames.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else settings.evaluation.dataset_path
    if not dataset_path.exists():
        print(f"Evaluation dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    pipeline = GraphRAGPipeline()
    evaluator = RagEvaluator(pipeline.llm)
    rows = []

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        reference_answer = item.get("reference_answer")
        print(f"[{i}/{len(questions)}] {question}")

        cost_tracker = CostTracker()
        result = pipeline.answer(question, cost_tracker=cost_tracker)
        scores = evaluator.evaluate(
            question=question,
            context_chunks=result.retrieved_chunks,
            answer=result.answer,
            reference_answer=reference_answer,
        )
        cost_summary = result.cost_summary

        rows.append(
            EvalRow(
                question=question,
                route=result.route,
                techniques_executed=result.techniques_executed,
                answer=result.answer,
                reference_answer=reference_answer,
                context_relevance=scores.context_relevance,
                faithfulness=scores.faithfulness,
                answer_relevance=scores.answer_relevance,
                correctness=scores.correctness,
                average_score=scores.average(),
                total_latency_ms=result.total_latency_ms,
                total_tokens=cost_summary.get("total_tokens", 0),
                total_cost_usd=cost_summary.get("total_cost_usd", 0.0),
                num_context_chunks=len(result.retrieved_chunks),
                errors=result.errors,
            )
        )

    report_gen = EvaluationReportGenerator()
    paths = report_gen.write(rows, run_label=args.label)
    print(f"\nCSV report: {paths['csv']}")
    print(f"Markdown report: {paths['markdown']}")


if __name__ == "__main__":
    main()
