"""
CRAG (Corrective RAG): grades the relevance of retrieved context to the
query using the centralized LLM as a judge.

To keep API/LLM usage efficient, CRAG only grades a limited number of
top-ranked chunks instead of the full graph-expanded candidate set.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config import settings
from src.llm.ollama_client import OllamaCloudClient, llm_client
from src.utils.logger import RunTimer, pipeline_logger


_GRADE_SYSTEM_PROMPT = """أنت مقيّم دقة استرجاع المعلومات (Retrieval Grader).

قيّم مدى صلة كل مقطع بالسؤال على مقياس من 0 إلى 1:
- 0 = غير ذي صلة إطلاقاً
- 1 = ذو صلة تامة ويجيب عن السؤال مباشرة

أعد النتيجة بصيغة JSON فقط:
{"scores": [0.0, 0.0, ...]}

يجب أن يحتوي scores على درجة واحدة لكل مقطع وبنفس الترتيب."""


_CORRECTIVE_QUERY_SYSTEM_PROMPT = """السياق المسترجع لم يكن كافياً للإجابة عن السؤال التالي.
اقترح صياغة بحث بديلة وأوسع للسؤال قد تساعد في إيجاد معلومات أكثر صلة في نفس مجموعة الوثائق.
أعد جملة واحدة فقط بدون شرح."""


@dataclass
class CragResult:
    chunks: List[Dict[str, Any]]
    relevance_scores: List[float]
    average_relevance: float
    corrective_rounds: int
    corrective_queries: List[str] = field(default_factory=list)
    triggered_correction: bool = False


class CorrectiveRag:
    """Grades retrieved chunks and triggers corrective re-retrieval when needed."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client
        self.threshold = settings.crag.relevance_threshold
        self.max_rounds = settings.crag.max_corrective_queries

        # CRAG should evaluate only the strongest candidates.
        self.max_grade_chunks = 8

    def _prepare_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Limit the number of chunks sent to the grading LLM.

        The incoming list may contain hundreds of graph-expanded candidates.
        We only need the strongest candidates for relevance grading.
        """

        if len(chunks) <= self.max_grade_chunks:
            return chunks

        # Preserve the current retrieval order. The pipeline already orders
        # candidates by retrieval/reranking quality before CRAG.
        return chunks[: self.max_grade_chunks]

    def grade(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[float]:
        if not chunks:
            return []

        grade_chunks = self._prepare_chunks(chunks)

        snippets = "\n".join(
            f"{i + 1}. {c['text'][:1400]}"
            for i, c in enumerate(grade_chunks)
        )

        prompt = (
            f"السؤال: {query}\n\n"
            f"المقاطع:\n{snippets}\n\n"
            "أعد درجات الصلة بصيغة JSON."
        )

        try:
            data = self.client.complete_json(
                prompt,
                system=_GRADE_SYSTEM_PROMPT,
                model=self.client.fast_model,
                purpose="crag_grading",
            )

            scores = [
                float(score)
                for score in data.get("scores", [])
            ]

        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log(
                "error",
                stage="crag_grading",
                error=str(exc),
            )
            scores = []

        # Defensive padding/truncation.
        if len(scores) < len(grade_chunks):
            scores += [0.5] * (len(grade_chunks) - len(scores))

        return scores[: len(grade_chunks)]

    def run(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        retrieve_fn: Callable[[str], List[Dict[str, Any]]],
    ) -> CragResult:
        """
        Grade retrieved context and perform corrective retrieval when needed.
        """

        with RunTimer() as timer:
            current_chunks = chunks

            # Only grade the strongest initial candidates.
            grading_chunks = self._prepare_chunks(current_chunks)
            scores = self.grade(query, grading_chunks)

            avg = (
                sum(scores) / len(scores)
                if scores
                else 0.0
            )

            corrective_queries: List[str] = []
            rounds = 0

            while avg < self.threshold and rounds < self.max_rounds:
                rounds += 1

                corrective_query = self._generate_corrective_query(query)
                if (
                    corrective_query in corrective_queries
                    or corrective_query.strip() == query.strip()
                ):
                    # A repeated query would retrieve the same chunks again.
                    break
                corrective_queries.append(corrective_query)

                new_chunks = retrieve_fn(corrective_query)

                existing_ids = {
                    c["chunk_id"]
                    for c in current_chunks
                }

                fresh = [
                    c
                    for c in new_chunks
                    if c["chunk_id"] not in existing_ids
                ]

                # Only the first `max_grade_chunks` chunks are ever graded.
                # Appending new chunks at the END meant they were never
                # graded, so corrective rounds had no effect. Interleave:
                # keep the best old chunks, then bring in the best new ones
                # so they land inside the graded window.
                half = max(1, self.max_grade_chunks // 2)
                merged = (
                    current_chunks[:half]
                    + fresh[:half]
                    + current_chunks[half:]
                    + fresh[half:]
                )

                # Grade only the strongest candidates from the merged set.
                grading_chunks = self._prepare_chunks(merged)
                merged_scores = self.grade(
                    query,
                    grading_chunks,
                )

                new_avg = (
                    sum(merged_scores) / len(merged_scores)
                    if merged_scores
                    else 0.0
                )

                if new_avg > avg:
                    current_chunks = merged
                    scores = merged_scores
                    avg = new_avg

            # Final context should contain only chunks that CRAG actually
            # evaluated and considered sufficiently relevant.
            final_chunks = self._prepare_chunks(current_chunks)

            filtered = [
                chunk
                for chunk, score in zip(final_chunks, scores)
                if score >= self.threshold * 0.5
            ]

            # CRAG's job is to decide whether to re-retrieve. Its LLM scores
            # are noisy, so do not let them cut the context down to one
            # chunk: keep the best-scored chunks, then fill up with the
            # remaining reranked chunks (in order) up to `min_keep`.
            min_keep = 5
            if len(filtered) < min_keep:
                kept_ids = {c.get("chunk_id") for c in filtered}
                for chunk in final_chunks:
                    if len(filtered) >= min_keep:
                        break
                    if chunk.get("chunk_id") not in kept_ids:
                        filtered.append(chunk)
                        kept_ids.add(chunk.get("chunk_id"))

        result = CragResult(
            chunks=filtered,
            relevance_scores=scores[: len(filtered)],
            average_relevance=avg,
            corrective_rounds=rounds,
            corrective_queries=corrective_queries,
            triggered_correction=rounds > 0,
        )

        pipeline_logger.log(
            "crag",
            query=query,
            average_relevance=avg,
            corrective_rounds=rounds,
            corrective_queries=corrective_queries,
            num_final_chunks=len(filtered),
            num_graded_chunks=len(grading_chunks),
            latency_ms=timer.elapsed_ms,
        )

        return result

    def _generate_corrective_query(
        self,
        query: str,
    ) -> str:
        try:
            return self.client.complete(
                f"السؤال: {query}",
                system=_CORRECTIVE_QUERY_SYSTEM_PROMPT,
                model=self.client.fast_model,
                temperature=0.4,
                max_tokens=100,
                purpose="crag_corrective_query",
            ).strip() or query

        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log(
                "error",
                stage="crag_corrective_query",
                error=str(exc),
            )
            return query