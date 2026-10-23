"""
Generates CSV results and a Markdown summary report from a batch evaluation
run of the Arabic Graph RAG pipeline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from src.utils.logger import pipeline_logger


@dataclass
class EvalRow:
    question: str
    route: str
    techniques_executed: List[str]
    answer: str
    reference_answer: Optional[str]
    context_relevance: Optional[float]
    faithfulness: Optional[float]
    answer_relevance: Optional[float]
    correctness: Optional[float]
    average_score: float
    total_latency_ms: float
    total_tokens: int
    total_cost_usd: float
    num_context_chunks: int
    errors: List[str] = field(default_factory=list)


class EvaluationReportGenerator:
    """Writes evaluation results to CSV + a human-readable Markdown report."""

    def __init__(self, reports_dir: Optional[Path] = None) -> None:
        self.reports_dir = reports_dir or settings.evaluation.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def write(self, rows: List[EvalRow], run_label: Optional[str] = None) -> Dict[str, Path]:
        label = run_label or datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.reports_dir / f"evaluation_{label}.csv"
        md_path = self.reports_dir / f"evaluation_{label}.md"

        self._write_csv(rows, csv_path)
        self._write_markdown(rows, md_path)

        pipeline_logger.log(
            "evaluation_report_written", csv_path=str(csv_path), md_path=str(md_path), num_rows=len(rows)
        )
        return {"csv": csv_path, "markdown": md_path}

    @staticmethod
    def _write_csv(rows: List[EvalRow], path: Path) -> None:
        fieldnames = [
            "question", "route", "techniques_executed", "answer", "reference_answer",
            "context_relevance", "faithfulness", "answer_relevance", "correctness",
            "average_score", "total_latency_ms", "total_tokens", "total_cost_usd",
            "num_context_chunks", "errors",
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "question": row.question,
                        "route": row.route,
                        "techniques_executed": "; ".join(row.techniques_executed),
                        "answer": row.answer,
                        "reference_answer": row.reference_answer or "",
                        "context_relevance": row.context_relevance,
                        "faithfulness": row.faithfulness,
                        "answer_relevance": row.answer_relevance,
                        "correctness": row.correctness,
                        "average_score": row.average_score,
                        "total_latency_ms": row.total_latency_ms,
                        "total_tokens": row.total_tokens,
                        "total_cost_usd": row.total_cost_usd,
                        "num_context_chunks": row.num_context_chunks,
                        "errors": "; ".join(row.errors),
                    }
                )

    @staticmethod
    def _write_markdown(rows: List[EvalRow], path: Path) -> None:
        def _avg(values: List[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        n = len(rows)
        avg_context = _avg([r.context_relevance for r in rows if r.context_relevance is not None])
        avg_faith = _avg([r.faithfulness for r in rows if r.faithfulness is not None])
        avg_answer_rel = _avg([r.answer_relevance for r in rows if r.answer_relevance is not None])
        correctness_vals = [r.correctness for r in rows if r.correctness is not None]
        avg_correctness = _avg(correctness_vals) if correctness_vals else None
        avg_overall = _avg([r.average_score for r in rows])
        total_cost = round(sum(r.total_cost_usd for r in rows), 6)
        total_tokens = sum(r.total_tokens for r in rows)
        avg_latency = _avg([r.total_latency_ms for r in rows])

        route_counts: Dict[str, int] = {}
        for r in rows:
            route_counts[r.route] = route_counts.get(r.route, 0) + 1

        lines = [
            "# Arabic Graph RAG — Evaluation Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"Number of questions evaluated: **{n}**",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Score |",
            "|---|---|",
            f"| Context Relevance | {avg_context} |",
            f"| Faithfulness | {avg_faith} |",
            f"| Answer Relevance | {avg_answer_rel} |",
            f"| Correctness | {avg_correctness if avg_correctness is not None else 'N/A (no references)'} |",
            f"| **Overall Average** | **{avg_overall}** |",
            "",
            "## Cost & Performance",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total tokens used | {total_tokens} |",
            f"| Total estimated cost (USD) | {total_cost} |",
            f"| Average latency per query (ms) | {avg_latency} |",
            "",
            "## Route Distribution",
            "",
            "| Route | Count |",
            "|---|---|",
        ]
        for route, count in route_counts.items():
            lines.append(f"| {route} | {count} |")

        lines += ["", "## Per-Question Results", ""]
        lines.append(
            "| # | Question | Route | Ctx Rel. | Faithfulness | Ans. Rel. | Correctness | Avg |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(rows, start=1):
            question_preview = r.question.replace("|", "/").replace("\n", " ")[:80]
            lines.append(
                f"| {i} | {question_preview} | {r.route} | {r.context_relevance} | "
                f"{r.faithfulness} | {r.answer_relevance} | {r.correctness} | {r.average_score} |"
            )

        path.write_text("\n".join(lines), encoding="utf-8")
