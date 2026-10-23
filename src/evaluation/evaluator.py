"""
Evaluation pipeline for the Arabic Graph RAG system.

Implements four LLM-as-judge metrics (all via the centralized Ollama Cloud
adapter, scored 0.0-1.0):

- Context Relevance: how relevant the retrieved context is to the question.
- Faithfulness: whether the generated answer is fully supported by the
  retrieved context (no hallucination).
- Answer Relevance: whether the answer actually addresses the question asked.
- Correctness: whether the answer matches a reference/gold answer, when one
  is provided in the evaluation dataset (skipped otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.ollama_client import OllamaCloudClient, llm_client
from src.utils.logger import RunTimer, pipeline_logger

_CONTEXT_RELEVANCE_SYSTEM = """قيّم مدى صلة "السياق المسترجع" بـ"السؤال" على مقياس من 0 إلى 1.
0 يعني لا صلة إطلاقاً، 1 يعني السياق يحتوي على المعلومات اللازمة للإجابة الكاملة.
أعد JSON فقط: {"score": 0.0, "reason": "سبب مختصر"}"""

_FAITHFULNESS_SYSTEM = """قيّم مدى التزام "الإجابة" بما ورد حصراً في "السياق المسترجع" على مقياس من 0 إلى 1.
0 يعني الإجابة تحتوي معلومات غير موجودة في السياق (اختلاق)، 1 يعني كل ما في الإجابة مدعوم بالسياق تماماً.
أعد JSON فقط: {"score": 0.0, "reason": "سبب مختصر"}"""

_ANSWER_RELEVANCE_SYSTEM = """قيّم مدى إجابة "الإجابة" فعلياً عن "السؤال" المطروح (بغض النظر عن صحتها) على مقياس من 0 إلى 1.
0 يعني الإجابة لا علاقة لها بالسؤال، 1 يعني الإجابة تعالج السؤال مباشرة وبالكامل.
أعد JSON فقط: {"score": 0.0, "reason": "سبب مختصر"}"""

_CORRECTNESS_SYSTEM = """قارن "الإجابة المولدة" بـ"الإجابة المرجعية الصحيحة" وقيّم مدى تطابقهما في المعنى
على مقياس من 0 إلى 1. 0 يعني خاطئة تماماً، 1 يعني مطابقة تماماً في المعنى.
أعد JSON فقط: {"score": 0.0, "reason": "سبب مختصر"}"""


@dataclass
class EvaluationScores:
    context_relevance: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    correctness: Optional[float] = None
    reasons: Dict[str, str] = field(default_factory=dict)

    def average(self) -> float:
        values = [v for v in (self.context_relevance, self.faithfulness, self.answer_relevance,
                               self.correctness) if v is not None]
        return round(sum(values) / len(values), 4) if values else 0.0


class RagEvaluator:
    """LLM-as-judge evaluator for retrieval + generation quality."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client

    def evaluate(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        answer: str,
        reference_answer: Optional[str] = None,
    ) -> EvaluationScores:
        context_text = "\n---\n".join(c.get("text", "") for c in context_chunks)[:6000]

        with RunTimer() as timer:
            context_relevance, cr_reason = self._score(
                _CONTEXT_RELEVANCE_SYSTEM,
                f"السؤال: {question}\n\nالسياق المسترجع:\n{context_text}",
            )
            faithfulness, f_reason = self._score(
                _FAITHFULNESS_SYSTEM,
                f"السياق المسترجع:\n{context_text}\n\nالإجابة:\n{answer}",
            )
            answer_relevance, ar_reason = self._score(
                _ANSWER_RELEVANCE_SYSTEM,
                f"السؤال: {question}\n\nالإجابة:\n{answer}",
            )
            correctness, c_reason = (None, "")
            if reference_answer:
                correctness, c_reason = self._score(
                    _CORRECTNESS_SYSTEM,
                    f"الإجابة المرجعية الصحيحة:\n{reference_answer}\n\nالإجابة المولدة:\n{answer}",
                )

        scores = EvaluationScores(
            context_relevance=context_relevance,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            correctness=correctness,
            reasons={
                "context_relevance": cr_reason,
                "faithfulness": f_reason,
                "answer_relevance": ar_reason,
                "correctness": c_reason,
            },
        )
        pipeline_logger.log(
            "evaluation",
            question=question,
            context_relevance=context_relevance,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            correctness=correctness,
            average=scores.average(),
            latency_ms=timer.elapsed_ms,
        )
        return scores

    def _score(self, system_prompt: str, user_prompt: str) -> tuple[Optional[float], str]:
        try:
            data = self.client.complete_json(
                user_prompt, system=system_prompt, model=self.client.fast_model, purpose="evaluation"
            )
            score = data.get("score")
            score = float(score) if score is not None else None
            if score is not None:
                score = max(0.0, min(1.0, score))
            reason = str(data.get("reason", ""))
            return score, reason
        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log("error", stage="evaluation_scoring", error=str(exc))
            return None, f"scoring_failed: {exc}"
