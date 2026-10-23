"""
Router: classifies an incoming Arabic query into one of three complexity
tiers so the pipeline can decide which advanced RAG techniques to run.

- SIMPLE:   greetings/small talk/meta questions — answered directly, still
            grounded (or explicitly refused) without a full retrieval pass.
- BASIC:    a direct factual question answerable with a single hybrid
            retrieval + GraphRAG expansion + generation pass.
- ADVANCED: a complex/ambiguous/multi-part/comparative question that
            benefits from query rewriting, multi-query, decomposition,
            HyDE, self-query filtering, and CRAG correction.

The classification itself uses the centralized Ollama Cloud LLM (fast model)
with a strict JSON response, with a keyword-based fallback if the LLM call
fails so the pipeline never breaks on a router error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.llm.ollama_client import OllamaCloudClient, llm_client
from src.utils.logger import RunTimer, pipeline_logger


class Route(str, Enum):
    SIMPLE = "simple"
    BASIC = "basic"
    ADVANCED = "advanced"


@dataclass
class RoutingDecision:
    route: Route
    reason: str


_ROUTER_SYSTEM_PROMPT = """أنت موجّه استعلامات (Router) لنظام أسئلة وأجوبة يعتمد على وثائق إجراءات عمل عربية.
صنّف السؤال التالي إلى واحدة من ثلاث فئات:
- "simple": تحية أو كلام عام أو سؤال لا علاقة له بمحتوى الوثائق ولا يحتاج بحثاً.
- "basic": سؤال واقعي مباشر وبسيط يمكن الإجابة عنه ببحث واحد في الوثائق.
- "advanced": سؤال معقد أو متعدد الأجزاء أو مقارنة أو يحتاج تفكيكاً أو استدلالاً عبر عدة أقسام من الوثائق.
أعد النتيجة بصيغة JSON فقط: {"route": "simple|basic|advanced", "reason": "سبب مختصر بالعربية"}"""

_GREETING_PATTERNS = re.compile(
    r"^\s*(مرحبا|أهلا|اهلا|السلام عليكم|صباح الخير|مساء الخير|شكرا|شكراً|hi|hello|hey)\b",
    re.IGNORECASE,
)
_COMPLEXITY_MARKERS = ("قارن", "مقارنة", "الفرق بين", "و أيضا", "وأيضاً", "خطوات", "لماذا و", "أولاً")


class QueryRouter:
    """Classifies queries into SIMPLE / BASIC / ADVANCED routes."""

    def __init__(self, client: Optional[OllamaCloudClient] = None) -> None:
        self.client = client or llm_client

    def route(self, query: str) -> RoutingDecision:
        with RunTimer() as timer:
            decision = self._route_with_llm(query) or self._route_with_heuristics(query)
        pipeline_logger.log(
            "route_selected",
            query=query,
            route=decision.route.value,
            reason=decision.reason,
            latency_ms=timer.elapsed_ms,
        )
        return decision

    def _route_with_llm(self, query: str) -> Optional[RoutingDecision]:
        try:
            data = self.client.complete_json(
                f"السؤال: {query}",
                system=_ROUTER_SYSTEM_PROMPT,
                model=self.client.fast_model,
                purpose="routing",
            )
            route_str = str(data.get("route", "")).strip().lower()
            if route_str in {r.value for r in Route}:
                return RoutingDecision(route=Route(route_str), reason=str(data.get("reason", "")))
        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log("error", stage="routing_llm", error=str(exc))
        return None

    @staticmethod
    def _route_with_heuristics(query: str) -> RoutingDecision:
        stripped = query.strip()
        if not stripped or _GREETING_PATTERNS.match(stripped):
            return RoutingDecision(route=Route.SIMPLE, reason="heuristic: greeting/empty query")
        if len(stripped) > 160 or any(marker in stripped for marker in _COMPLEXITY_MARKERS):
            return RoutingDecision(route=Route.ADVANCED, reason="heuristic: long/complex query")
        return RoutingDecision(route=Route.BASIC, reason="heuristic: default factual query")
