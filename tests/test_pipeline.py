from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx

from src.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from src.graph.knowledge_graph import KnowledgeGraphStore
from src.rag.pipeline import GraphRAGPipeline
from src.rag.router import Route, RoutingDecision
from src.retrieval.bm25_retriever import Bm25Retriever

# The reranker loads a real ONNX model from HuggingFace on first use, which
# requires network access unavailable in this sandbox/CI environment. Since
# these tests exercise pipeline *orchestration* (not the reranker model
# itself — that's covered by manual/integration testing with real network
# access), we patch the reranker class so pipeline construction never
# attempts a real download.


def _mock_vector_store(hits):
    vs = MagicMock()
    vs.query.return_value = hits
    vs.get_all_chunks.return_value = hits
    return vs


def _sample_hits():
    return [
        {
            "chunk_id": "1",
            "text": "خطوات استلام البريد الوارد: 1) الفرز 2) التسجيل 3) التوزيع.",
            "metadata": {"source": "mail.pdf", "page_start": 3, "page_end": 3},
            "score": 0.9,
        }
    ]


def _build_pipeline_with_mocks(route: Route):
    hits = _sample_hits()
    vector_store = _mock_vector_store(hits)
    bm25 = Bm25Retriever()  # empty/untrained is fine — is_ready() is False
    graph_store = KnowledgeGraphStore(nx.MultiDiGraph())

    with patch("src.rag.pipeline.MultilingualReranker"):
        pipeline = GraphRAGPipeline(vector_store=vector_store, bm25=bm25, graph_store=graph_store)

    pipeline.router.route = MagicMock(return_value=RoutingDecision(route=route, reason="test"))
    pipeline.reranker.rerank = MagicMock(side_effect=lambda q, c, **kw: c)
    pipeline.compressor.compress = MagicMock(side_effect=lambda q, c: c)

    fake_answer = GeneratedAnswer(
        answer="يتم استلام البريد عبر ثلاث خطوات: الفرز ثم التسجيل ثم التوزيع.",
        model="test-model",
        prompt_tokens=50,
        completion_tokens=20,
        latency_ms=12.3,
        citations=[{"chunk_id": "1", "source": "mail.pdf", "page_start": 3, "page_end": 3}],
    )
    pipeline.generator.generate = MagicMock(return_value=fake_answer)
    pipeline.generator.answer_directly = MagicMock(
        return_value=GeneratedAnswer(
            answer="أهلاً! كيف يمكنني مساعدتك؟", model="test-model",
            prompt_tokens=5, completion_tokens=5, latency_ms=3.0,
        )
    )
    return pipeline


def test_pipeline_simple_route_skips_retrieval():
    pipeline = _build_pipeline_with_mocks(Route.SIMPLE)
    result = pipeline.answer("مرحبا")

    assert result.route == "simple"
    assert result.retrieved_chunks == []
    pipeline.generator.answer_directly.assert_called_once()
    pipeline.generator.generate.assert_not_called()


def test_pipeline_basic_route_runs_hybrid_and_graph_expansion():
    pipeline = _build_pipeline_with_mocks(Route.BASIC)
    result = pipeline.answer("ما هي خطوات استلام البريد الوارد؟")

    assert result.route == "basic"
    assert "hybrid_retrieval" in result.techniques_executed
    assert "graph_expansion" in result.techniques_executed
    assert "reranking" in result.techniques_executed
    assert "contextual_compression" in result.techniques_executed
    assert result.answer.startswith("يتم استلام البريد")
    assert result.citations[0]["source"] == "mail.pdf"


def test_pipeline_advanced_route_runs_full_technique_stack():
    pipeline = _build_pipeline_with_mocks(Route.ADVANCED)

    pipeline.self_query.parse = MagicMock(
        return_value=MagicMock(search_query="خطوات استلام البريد", filters={})
    )
    pipeline.rewriter.rewrite = MagicMock(return_value="ما هي خطوات استلام البريد الوارد بالتفصيل؟")
    pipeline.decomposer.decompose = MagicMock(return_value=["ما هي خطوات استلام البريد الوارد بالتفصيل؟"])
    pipeline.multi_query.generate = MagicMock(return_value=["ما هي خطوات استلام البريد الوارد بالتفصيل؟"])
    pipeline.hyde.generate = MagicMock(return_value="فقرة افتراضية عن استلام البريد.")
    pipeline.crag.run = MagicMock(
        return_value=MagicMock(
            chunks=_sample_hits(), average_relevance=0.9, corrective_rounds=0,
            corrective_queries=[], triggered_correction=False,
        )
    )

    result = pipeline.answer("قارن بين خطوات استلام البريد وخطوات تسجيل الأصول")

    assert result.route == "advanced"
    for technique in ["self_query", "query_rewriting", "query_decomposition", "multi_query", "hyde", "crag"]:
        assert technique in result.techniques_executed
    assert result.crag_info is not None
    assert result.crag_info["average_relevance"] == 0.9


def test_pipeline_never_bypasses_graph_expansion_on_basic_route():
    pipeline = _build_pipeline_with_mocks(Route.BASIC)
    pipeline.graph_expander.expand = MagicMock(
        return_value={"chunks": _sample_hits(), "graph_nodes": ["وحدة البريد"], "graph_edges": []}
    )
    result = pipeline.answer("ما هي خطوات استلام البريد؟")
    pipeline.graph_expander.expand.assert_called_once()
    assert "وحدة البريد" in result.graph_nodes


def test_pipeline_handles_step_failure_gracefully():
    pipeline = _build_pipeline_with_mocks(Route.BASIC)
    pipeline.hybrid_retriever.retrieve = MagicMock(side_effect=RuntimeError("chroma down"))

    result = pipeline.answer("ما هي خطوات استلام البريد؟")
    assert any("hybrid_retrieval" in e for e in result.errors)
    # Pipeline should still produce an answer rather than crashing.
    assert result.answer
