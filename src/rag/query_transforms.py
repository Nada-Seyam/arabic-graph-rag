"""
Query transformation techniques used by the ADVANCED route.

Each class here is a small, single-purpose wrapper around one LLM call
(via the centralized Ollama Cloud adapter) that reshapes the user's query
before retrieval:

- QueryRewriter:      clarifies/expands the raw query into a search-optimized
                      Arabic query, preserving the original meaning.
- MultiQueryGenerator: generates diverse alternative phrasings of a query so
                      retrieval can be run multiple times and results fused.
- QueryDecomposer:     breaks a complex, multi-part question into simpler
                      independently-answerable sub-questions.
- HydeGenerator:       writes a short hypothetical Arabic passage that would
                      answer the question, used as a dense-retrieval query
                      (Hypothetical Document Embeddings).
- SelfQueryParser:     splits a query into a clean semantic search string plus
                      structured metadata filters (e.g. a source manual name).

They are grouped in one module because each is only a few lines around the
shared "prompt the fast model, parse the result, log it" pattern; splitting
them into five near-identical single-class files added navigation overhead
without adding clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.ollama_client import OllamaCloudClient, llm_client
from src.utils.logger import RunTimer, pipeline_logger

_REWRITE_SYSTEM_PROMPT = """أنت أداة إعادة صياغة استعلامات بحث لنظام استرجاع معلومات من وثائق إجراءات عمل عربية.
أعد صياغة السؤال ليكون أوضح وأكثر ملاءمة للبحث، مع الحفاظ التام على المعنى الأصلي،
دون إضافة معلومات غير مذكورة في السؤال. أعد جملة واحدة فقط بدون أي شرح إضافي."""

_MULTI_QUERY_SYSTEM_PROMPT = """أنت مساعد بحث. مهمتك توليد صياغات بديلة ومتنوعة لنفس السؤال
لتحسين نتائج البحث في قاعدة وثائق عربية، مع الحفاظ على نفس المعنى والقصد الأصلي.
أعد النتيجة بصيغة JSON فقط: {"queries": ["صياغة 1", "صياغة 2", "صياغة 3"]}"""

_DECOMPOSITION_SYSTEM_PROMPT = """أنت أداة تفكيك أسئلة معقدة إلى أسئلة فرعية أبسط يمكن الإجابة
عن كل منها بشكل مستقل من وثائق إجراءات عمل عربية. إن كان السؤال بسيطاً بالفعل أعد سؤالاً فرعياً واحداً فقط (السؤال نفسه).
أعد النتيجة بصيغة JSON فقط: {"sub_questions": ["سؤال فرعي 1", "سؤال فرعي 2"]}"""

_HYDE_SYSTEM_PROMPT = """أنت خبير في كتابة وثائق إجراءات عمل داخلية. اكتب فقرة قصيرة (3-5 جمل)
بالعربية الفصحى تجيب بشكل مباشر عن السؤال التالي كما لو كانت مقتطفاً من دليل إجراءات رسمي،
حتى لو كنت غير متأكد من التفاصيل الدقيقة. لا تذكر أنك لا تعرف الإجابة، فقط اكتب الفقرة الافتراضية."""

_SELF_QUERY_SYSTEM_PROMPT = """أنت أداة تحليل استعلامات. حلل السؤال التالي واستخرج:
1) search_query: نص البحث الدلالي النظيف (بدون أي إشارة لاسم ملف أو مرشحات).
2) filters: أي مرشحات صريحة ذكرها المستخدم مثل اسم الوثيقة المصدر أو رقم الصفحة. إن لم توجد أعد كائناً فارغاً.
أعد النتيجة بصيغة JSON فقط:
{"search_query": "...", "filters": {"source": "اسم الملف إن وجد", "page": "رقم الصفحة إن وجد"}}"""


class QueryRewriter:
    """Rewrites a raw query into a retrieval-optimized version."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client

    def rewrite(self, query: str, conversation_context: Optional[List[str]] = None) -> str:
        context_block = ""
        if conversation_context:
            context_block = "سياق المحادثة السابق:\n" + "\n".join(conversation_context) + "\n\n"

        prompt = f"{context_block}السؤال الأصلي: {query}\n\nالسؤال المعاد صياغته:"
        with RunTimer() as timer:
            try:
                rewritten = self.client.complete(
                    prompt,
                    system=_REWRITE_SYSTEM_PROMPT,
                    model=self.client.fast_model,
                    temperature=0.0,
                    max_tokens=150,
                    purpose="query_rewriting",
                ).strip()
            except Exception as exc:  # noqa: BLE001
                pipeline_logger.log("error", stage="query_rewriting", error=str(exc))
                rewritten = query

        rewritten = rewritten.strip('"\u201c\u201d ') or query
        pipeline_logger.log(
            "query_rewritten", original=query, rewritten=rewritten, latency_ms=timer.elapsed_ms
        )
        return rewritten


class MultiQueryGenerator:
    """Generates N alternative phrasings of a query via the centralized LLM."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client

    def generate(self, query: str, n: int = 3) -> List[str]:
        prompt = f"السؤال: {query}\n\nولّد {n} صياغات بديلة بصيغة JSON."
        with RunTimer() as timer:
            try:
                data = self.client.complete_json(
                    prompt,
                    system=_MULTI_QUERY_SYSTEM_PROMPT,
                    model=self.client.fast_model,
                    purpose="multi_query",
                )
                queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
            except Exception as exc:  # noqa: BLE001
                pipeline_logger.log("error", stage="multi_query", error=str(exc))
                queries = []

        if not queries:
            queries = [query]
        # Always include the original query and de-duplicate while preserving order.
        all_queries = [query] + queries
        seen = set()
        unique_queries = []
        for q in all_queries:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_queries.append(q)

        pipeline_logger.log(
            "multi_query_generated",
            original=query,
            queries=unique_queries,
            latency_ms=timer.elapsed_ms,
        )
        return unique_queries[: n + 1]


class QueryDecomposer:
    """Decomposes a complex query into independently retrievable sub-questions."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client

    def decompose(self, query: str, max_sub_questions: int = 4) -> List[str]:
        prompt = f"السؤال: {query}\n\nفكك هذا السؤال إلى أسئلة فرعية بصيغة JSON."
        with RunTimer() as timer:
            try:
                data = self.client.complete_json(
                    prompt,
                    system=_DECOMPOSITION_SYSTEM_PROMPT,
                    model=self.client.fast_model,
                    purpose="query_decomposition",
                )
                sub_questions = [
                    str(q).strip() for q in data.get("sub_questions", []) if str(q).strip()
                ]
            except Exception as exc:  # noqa: BLE001
                pipeline_logger.log("error", stage="query_decomposition", error=str(exc))
                sub_questions = []

        if not sub_questions:
            sub_questions = [query]

        sub_questions = sub_questions[:max_sub_questions]
        pipeline_logger.log(
            "query_decomposed",
            original=query,
            sub_questions=sub_questions,
            latency_ms=timer.elapsed_ms,
        )
        return sub_questions


class HydeGenerator:
    """Generates a hypothetical answer passage to use as a dense-retrieval query."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client

    def generate(self, query: str) -> str:
        with RunTimer() as timer:
            try:
                hypothetical = self.client.complete(
                    f"السؤال: {query}\n\nالفقرة الافتراضية:",
                    system=_HYDE_SYSTEM_PROMPT,
                    model=self.client.fast_model,
                    temperature=0.3,
                    max_tokens=300,
                    purpose="hyde",
                ).strip()
            except Exception as exc:  # noqa: BLE001
                pipeline_logger.log("error", stage="hyde", error=str(exc))
                hypothetical = query

        pipeline_logger.log(
            "hyde_generated",
            query=query,
            hypothetical_length=len(hypothetical),
            latency_ms=timer.elapsed_ms,
        )
        return hypothetical or query


@dataclass
class SelfQueryResult:
    search_query: str
    filters: Dict[str, Any] = field(default_factory=dict)


class SelfQueryParser:
    """Splits a query into a clean search string + structured metadata filters."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client

    def parse(self, query: str) -> SelfQueryResult:
        with RunTimer() as timer:
            try:
                data = self.client.complete_json(
                    f"السؤال: {query}",
                    system=_SELF_QUERY_SYSTEM_PROMPT,
                    model=self.client.fast_model,
                    purpose="self_query",
                )
                search_query = str(data.get("search_query", "")).strip() or query
                filters = data.get("filters", {}) if isinstance(data.get("filters"), dict) else {}
                filters = {k: v for k, v in filters.items() if v}
            except Exception as exc:  # noqa: BLE001
                pipeline_logger.log("error", stage="self_query", error=str(exc))
                search_query, filters = query, {}

        result = SelfQueryResult(search_query=search_query, filters=filters)
        pipeline_logger.log(
            "self_query_parsed",
            query=query,
            search_query=result.search_query,
            filters=result.filters,
            latency_ms=timer.elapsed_ms,
        )
        return result
