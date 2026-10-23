from __future__ import annotations

from unittest.mock import MagicMock

from src.rag.router import QueryRouter, Route


def _router_with_llm_response(response_dict):
    client = MagicMock()
    client.complete_json.return_value = response_dict
    return QueryRouter(client)


def test_router_uses_llm_classification():
    router = _router_with_llm_response({"route": "advanced", "reason": "سؤال مركب"})
    decision = router.route("قارن بين إجراءات البريد وإجراءات المستودعات")
    assert decision.route == Route.ADVANCED


def test_router_falls_back_to_heuristics_on_llm_error():
    client = MagicMock()
    client.complete_json.side_effect = RuntimeError("network down")
    router = QueryRouter(client)

    decision = router.route("مرحبا")
    assert decision.route == Route.SIMPLE
    assert "heuristic" in decision.reason


def test_router_falls_back_to_heuristics_on_invalid_route_value():
    router = _router_with_llm_response({"route": "not_a_real_route", "reason": "x"})
    # Avoid heuristic complexity markers (e.g. "خطوات") so this isolates the
    # invalid-LLM-value fallback path rather than the heuristic's own content rules.
    decision = router.route("ما هو رقم النموذج المستخدم للاستلام؟")
    assert decision.route == Route.BASIC


def test_router_heuristic_detects_greeting():
    decision = QueryRouter._route_with_heuristics("السلام عليكم")
    assert decision.route == Route.SIMPLE


def test_router_heuristic_detects_long_query_as_advanced():
    long_query = "سؤال طويل جداً " * 20
    decision = QueryRouter._route_with_heuristics(long_query)
    assert decision.route == Route.ADVANCED


def test_router_heuristic_default_is_basic():
    decision = QueryRouter._route_with_heuristics("ما هو رقم النموذج المستخدم للاستلام؟")
    assert decision.route == Route.BASIC
