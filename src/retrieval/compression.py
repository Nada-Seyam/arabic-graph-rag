"""
Extractive contextual compression for Arabic RAG.

Keeps only text that is directly relevant to the query without rewriting
or inventing information. This preserves the original wording of procedures,
steps, warnings, and requirements.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from src.llm.ollama_client import OllamaCloudClient, llm_client
from src.utils.logger import RunTimer, pipeline_logger


_COMPRESSION_SYSTEM = """
أنت أداة لاستخلاص النصوص من وثائق إجرائية.

مهمتك اختيار الجمل الموجودة حرفياً في المقطع والتي تساعد مباشرة في الإجابة
عن السؤال.

القواعد:
1) لا تعيد صياغة الجمل.
2) لا تلخص.
3) لا تضف أي معلومة.
4) لا تغير ترتيب الكلمات داخل الجملة.
5) احتفظ بالأرقام والخطوات والتحذيرات والشروط كما هي.
6) إذا كان المقطع يحتوي على معلومات مرتبطة بالسؤال، أعد الجمل الأصلية فقط.
7) إذا لم توجد معلومات مرتبطة، أعد النص فارغاً.
8) أعد النص المستخرج فقط بدون أي شرح.
9) بنود القوائم المرقمة أو المرمزة بحروف (مثل: -1، -2، أ.، ب.، ت.) هي
   جمل كاملة بحد ذاتها ويجب معاملتها كجمل عادية، وليس تجاهلها لأنها لا
   تبدأ بصيغة نحوية معتادة. إذا كانت القائمة بأكملها (أو معظم بنودها)
   مرتبطة بالسؤال، أعد كل بنود القائمة مع رموزها كما وردت، سطراً سطراً،
   دون حذف أي بند من بنودها.
"""


class ContextualCompressor:
    """Extracts relevant original sentences from retrieved chunks."""

    def __init__(self, client: OllamaCloudClient | None = None) -> None:
        self.client = client or llm_client

    def compress(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        max_chunk_chars: int = 1600,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        with RunTimer() as timer:
            # Each chunk is compressed via its own independent LLM call.
            # Running these sequentially (as before) meant total latency
            # scaled linearly with the number of reranked chunks (often
            # 5-15+ seconds on its own). They don't depend on each other,
            # so run them concurrently — but Ollama Cloud enforces a strict
            # account-level concurrency cap (its own client-side semaphore
            # already gates this to _MAX_CONCURRENT_OLLAMA_REQUESTS), so a
            # small thread pool here is enough; anything higher just queues
            # behind that shared limit anyway.
            max_workers = min(3, len(chunks)) or 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        lambda item: self._compress_one(query, item[1], max_chunk_chars),
                        enumerate(chunks),
                    )
                )

            # Preserve the original (rank) order of chunks rather than
            # completion order, since downstream generation/citations
            # display chunks in this order.
            compressed: List[Dict[str, Any]] = []
            for chunk, (extracted, call_failed) in zip(chunks, results):
                if call_failed or extracted:
                    new_chunk = dict(chunk)
                    new_chunk["compressed_text"] = extracted
                    compressed.append(new_chunk)
                else:
                    pipeline_logger.log(
                        "contextual_compression_chunk_dropped",
                        chunk_id=chunk.get("chunk_id"),
                    )

        pipeline_logger.log(
            "contextual_compression",
            query=query,
            num_chunks=len(chunks),
            latency_ms=timer.elapsed_ms,
        )

        return compressed

    def _compress_one(
        self,
        query: str,
        chunk: Dict[str, Any],
        max_chunk_chars: int,
    ) -> Tuple[str, bool]:
        """Compresses a single chunk. Returns (extracted_text, call_failed)."""
        original = chunk.get("text", "")[:max_chunk_chars]

        prompt = (
            f"السؤال:\n{query}\n\n"
            f"المقطع:\n{original}\n\n"
            "الجمل الأصلية المرتبطة بالسؤال فقط:"
        )

        try:
            extracted = self.client.complete(
                prompt,
                system=_COMPRESSION_SYSTEM,
                model=self.client.fast_model,
                temperature=0.0,
                max_tokens=700,
                purpose="contextual_compression",
            ).strip()
            return extracted, False

        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log(
                "error",
                stage="contextual_compression",
                error=str(exc),
            )
            # Never lose the original chunk if the LLM call itself failed
            # (network/API error) — fall back to the raw text. This is
            # distinct from the model succeeding and legitimately finding
            # no relevant sentences, which is handled by the caller.
            return original, True