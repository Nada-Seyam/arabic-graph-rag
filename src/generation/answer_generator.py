"""
Arabic answer generation grounded strictly in retrieved context.

The generator is instructed to answer only from the supplied context chunks
and to explicitly say it does not know when the context is insufficient,
rather than hallucinating. Every generated answer is traceable back to the
chunks (and pages/sources) used to produce it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.ollama_client import OllamaCloudClient, llm_client
from src.utils.logger import RunTimer, pipeline_logger

_GENERATION_SYSTEM_PROMPT = """أنت مساعد ذكي يجيب باللغة العربية الفصحى عن أسئلة موظفين تتعلق بأدلة
إجراءات عمل داخلية (البريد والملفات، إدارة الأصول والمستودعات، الإنذار المركزي).
اتبع هذه القواعد بصرامة:
1) أجب فقط بناءً على المعلومات الواردة في "السياق" أدناه. لا تستخدم أي معرفة خارجية.
2) إن لم يحتوِ السياق على إجابة كافية، قل بوضوح: "لا تتوفر معلومات كافية في الوثائق للإجابة عن هذا السؤال."
3) اذكر مصدر إجابتك (اسم الوثيقة ورقم الصفحة) عند الإمكان بصيغة [المصدر: اسم الملف، صفحة س].
4) كن دقيقاً ومختصراً ومباشراً، واستخدم نقاطاً مرقمة للخطوات الإجرائية إن وجدت."""


@dataclass
class GeneratedAnswer:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    citations: List[Dict[str, Any]] = field(default_factory=list)


class AnswerGenerator:
    """Generates a grounded Arabic answer from a query + retrieved context."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        use_compressed: bool = False,
    ) -> GeneratedAnswer:
        context_block, citations = self._build_context(chunks, use_compressed)
        prompt = (
            f"السياق:\n{context_block}\n\n"
            f"السؤال: {query}\n\n"
            "الإجابة (بالعربية الفصحى، مستندة فقط إلى السياق أعلاه):"
        )
        with RunTimer() as timer:
            response = self.client.chat(
                [
                    {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                purpose="answer_generation",
            )

        pipeline_logger.log(
            "answer_generated",
            query=query,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=timer.elapsed_ms,
            num_context_chunks=len(chunks),
            answer_preview=response.text[:300],
        )
        return GeneratedAnswer(
            answer=response.text,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=timer.elapsed_ms,
            citations=citations,
        )

    @staticmethod
    def _build_context(
        chunks: List[Dict[str, Any]], use_compressed: bool
    ) -> tuple[str, List[Dict[str, Any]]]:
        parts = []
        citations = []
        for i, chunk in enumerate(chunks, start=1):
            text = chunk.get("compressed_text") if use_compressed else None
            text = text or chunk.get("text", "")
            meta = chunk.get("metadata", {}) or {}
            source = meta.get("source", "unknown")
            page_start = meta.get("page_start", "?")
            page_end = meta.get("page_end", page_start)
            page_label = f"{page_start}" if page_start == page_end else f"{page_start}-{page_end}"
            parts.append(f"[{i}] (المصدر: {source}, صفحة {page_label})\n{text}")
            citations.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": source,
                    "page_start": page_start,
                    "page_end": page_end,
                }
            )
        return "\n\n".join(parts), citations

    def answer_directly(self, query: str) -> GeneratedAnswer:
        """Used for SIMPLE-route queries (greetings/meta) where no retrieval
        is needed — still goes through the centralized LLM adapter."""
        with RunTimer() as timer:
            response = self.client.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "أنت مساعد ودود يرد بإيجاز بالعربية الفصحى على التحيات والأسئلة "
                            "العامة المتعلقة باستخدام النظام، دون اختلاق معلومات عن محتوى الوثائق."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                purpose="direct_answer",
                max_tokens=200,
            )
        pipeline_logger.log(
            "answer_generated",
            query=query,
            model=response.model,
            route="simple",
            latency_ms=timer.elapsed_ms,
        )
        return GeneratedAnswer(
            answer=response.text,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=timer.elapsed_ms,
            citations=[],
        )
