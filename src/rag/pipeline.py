"""
The main RAG pipeline orchestrator.

Wires together: Router -> (query rewriting, multi-query, decomposition,
HyDE, self-query as needed) -> Hybrid Retrieval (Dense + BM25 + RRF) ->
GraphRAG expansion (ALWAYS applied, never bypassed) -> Reranking ->
Contextual Compression -> CRAG correction -> Grounded Arabic generation.

Every step is logged via the centralized JSONL logger, and every LLM call's
tokens/cost are tracked via the centralized cost tracker.
"""

from __future__ import annotations

import os

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from src.graph.graph_expansion import GraphExpander
from src.graph.knowledge_graph import KnowledgeGraphBuilder, KnowledgeGraphStore
from src.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from src.llm.ollama_client import OllamaCloudClient, llm_client
from src.rag.crag import CorrectiveRag
from src.rag.query_transforms import (
    HydeGenerator,
    MultiQueryGenerator,
    QueryDecomposer,
    QueryRewriter,
    SelfQueryParser,
)
from src.rag.router import QueryRouter, Route, RoutingDecision
from src.retrieval.bm25_retriever import Bm25Retriever
from src.retrieval.compression import ContextualCompressor
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.retrieval.reranker import MultilingualReranker
from src.utils.cost_tracker import CostTracker
from src.utils.logger import RunTimer, new_run_id, pipeline_logger
from src.vectorstore.chroma_store import ChromaVectorStore


@dataclass
class PipelineResult:
    run_id: str
    query: str
    route: str
    techniques_executed: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    graph_nodes: List[str]
    graph_edges: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    model_used: str
    total_latency_ms: float
    cost_summary: Dict[str, Any]
    crag_info: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)


class GraphRAGPipeline:
    """The single production entry point for answering an Arabic query."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25: Optional[Bm25Retriever] = None,
        graph_store: Optional[KnowledgeGraphStore] = None,
        llm: Optional[OllamaCloudClient] = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25 = bm25 or self._build_bm25(self.vector_store)
        self.graph_store = graph_store or KnowledgeGraphBuilder.load()
        self.llm = llm or llm_client

        self.router = QueryRouter(self.llm)
        self.rewriter = QueryRewriter(self.llm)
        self.multi_query = MultiQueryGenerator(self.llm)
        self.decomposer = QueryDecomposer(self.llm)
        self.hyde = HydeGenerator(self.llm)
        self.self_query = SelfQueryParser(self.llm)
        self.crag = CorrectiveRag(self.llm)
        self.reranker = MultilingualReranker()
        self.compressor = ContextualCompressor(self.llm)
        self.generator = AnswerGenerator(self.llm)

        self.hybrid_retriever = HybridRetriever(self.vector_store, self.bm25)
        self.graph_expander = GraphExpander(self.graph_store, self.vector_store)

    @staticmethod
    def _build_bm25(vector_store: ChromaVectorStore) -> Bm25Retriever:
        bm25 = Bm25Retriever()
        chunks = vector_store.get_all_chunks()
        if chunks:
            bm25.build(chunks)
        return bm25

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def answer(self, query: str, cost_tracker: Optional[CostTracker] = None) -> PipelineResult:
        run_id = new_run_id()
        techniques: List[str] = []
        errors: List[str] = []
        cost_tracker = cost_tracker or CostTracker()
        self.llm.cost_tracker = cost_tracker  # isolate accounting per run

        pipeline_logger.log("run_start", run_id=run_id, query=query)
        start = time.perf_counter()

        try:
            routing: RoutingDecision = self.router.route(query)
            techniques.append(f"route:{routing.route.value}")

            if routing.route == Route.SIMPLE:
                generated = self.generator.answer_directly(query)
                result = self._finalize(
                    run_id, query, routing, techniques, [], [], [], generated, cost_tracker, errors
                )
                pipeline_logger.log("run_complete", run_id=run_id, route="simple")
                return result

            if routing.route == Route.BASIC:
                chunks, graph_nodes, graph_edges = self._basic_retrieval(query, techniques, errors)
                crag_info = None
            else:
                chunks, graph_nodes, graph_edges, crag_info = self._advanced_retrieval(
                    query, techniques, errors
                )

            reranked = self._safe_step(
                errors, "reranking", lambda: self.reranker.rerank(query, chunks)
            ) or chunks[: settings.reranker.top_k]
            techniques.append("reranking")

            # Contextual compression can drop the one sentence that holds the
            # answer (tables / appendix rows / short fragments). Toggle it
            # with ENABLE_COMPRESSION=false in .env to pass full reranked
            # chunks to the generator instead (it falls back to chunk text).
            if os.getenv("ENABLE_COMPRESSION", "true").strip().lower() in (
                "0", "false", "no", "off"
            ):
                compressed = reranked
            else:
                compressed = self._safe_step(
                    errors,
                    "contextual_compression",
                    lambda: self.compressor.compress(query, reranked),
                ) or reranked
                techniques.append("contextual_compression")

            generated = self.generator.generate(query, compressed, use_compressed=True)
            techniques.append("answer_generation")

            result = self._finalize(
                run_id,
                query,
                routing,
                techniques,
                compressed,
                graph_nodes,
                graph_edges,
                generated,
                cost_tracker,
                errors,
                crag_info,
            )
            pipeline_logger.log(
                "run_complete",
                run_id=run_id,
                route=routing.route.value,
                total_latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )
            return result
        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log("error", stage="pipeline_run", run_id=run_id, error=str(exc))
            errors.append(str(exc))
            fallback_answer = GeneratedAnswer(
                answer="حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى.",
                model=self.llm.default_model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
            )
            return self._finalize(
                run_id, query, RoutingDecision(Route.SIMPLE, "error"), techniques, [], [], [],
                fallback_answer, cost_tracker, errors,
            )

    # ------------------------------------------------------------------ #
    # Route-specific retrieval strategies
    # ------------------------------------------------------------------ #
    def _basic_retrieval(self, query: str, techniques: List[str], errors: List[str]):
        chunks = self._safe_step(
            errors,
            "hybrid_retrieval",
            lambda: self.hybrid_retriever.retrieve(
                query,
                top_k=settings.retrieval.top_k,
                dense_top_k=settings.retrieval.dense_top_k,
                bm25_top_k=settings.retrieval.bm25_top_k,

            ),
        ) or []
        techniques.append("hybrid_retrieval")

        expansion = self._safe_step(
            errors, "graph_expansion", lambda: self.graph_expander.expand(query, chunks)
        ) or {"chunks": chunks, "graph_nodes": [], "graph_edges": []}
        techniques.append("graph_expansion")

        return expansion["chunks"], expansion["graph_nodes"], expansion["graph_edges"]

    def _advanced_retrieval(self, query, techniques, errors):
        # Self-query: split intent from explicit filters.
        self_query_result = self._safe_step(
            errors, "self_query", lambda: self.self_query.parse(query)
        )
        techniques.append("self_query")
        base_query = self_query_result.search_query if self_query_result else query

        # Query rewriting for clarity.
        rewritten = self._safe_step(
            errors, "query_rewriting", lambda: self.rewriter.rewrite(base_query)
        ) or base_query
        techniques.append("query_rewriting")

        # Decomposition into sub-questions.
        sub_questions = self._safe_step(
            errors, "query_decomposition", lambda: self.decomposer.decompose(rewritten)
        ) or [rewritten]
        techniques.append("query_decomposition")

        # Multi-query expansion of each sub-question (bounded to keep latency sane).
        all_queries: List[str] = []
        for sq in sub_questions[:3]:
            variants = self._safe_step(
                errors, "multi_query", lambda sq=sq: self.multi_query.generate(sq, n=2)
            ) or [sq]
            all_queries.extend(variants)
        techniques.append("multi_query")

        # HyDE hypothetical passage, used as one more retrieval query.
        hyde_passage = self._safe_step(
            errors, "hyde", lambda: self.hyde.generate(rewritten)
        )
        if hyde_passage:
            all_queries.append(hyde_passage)
        techniques.append("hyde")

        # Retrieve for every query variant and fuse everything with RRF.
        ranked_lists = []
        for q in all_queries:
            hits = self._safe_step(errors, "hybrid_retrieval", lambda q=q: self.hybrid_retriever.retrieve(q))
            if hits:
                for h in hits:
                    h["_source"] = h.get("_source", "hybrid")
                ranked_lists.append(hits)
        techniques.append("hybrid_retrieval_multi")

        fused = reciprocal_rank_fusion(ranked_lists, k=settings.retrieval.rrf_k) if ranked_lists else []
        fused = fused[: settings.retrieval.dense_top_k]

        # GraphRAG expansion — always applied, regardless of route.
        expansion = self._safe_step(
            errors, "graph_expansion", lambda: self.graph_expander.expand(query, fused)
        ) or {"chunks": fused, "graph_nodes": [], "graph_edges": []}
        techniques.append("graph_expansion")

        # Rerank the whole pool BEFORE CRAG. CRAG only grades the first few
        # chunks, so without this it graded the first chunks in fusion order
        # (not the best ones) and then discarded the rest before the
        # reranker ever saw them.
        pool = expansion["chunks"]
        pool = self._safe_step(
            errors, "pre_crag_rerank", lambda: self.reranker.rerank(query, pool)
        ) or pool
        techniques.append("pre_crag_rerank")

        def _retrieve_and_rerank(q: str):
            candidates = self.graph_expander.expand(
                q, self.hybrid_retriever.retrieve(q)
            )["chunks"]
            try:
                return self.reranker.rerank(query, candidates)
            except Exception as exc:  # noqa: BLE001
                pipeline_logger.log(
                    "error", stage="crag_retrieve_rerank", error=str(exc)
                )
                return candidates

        # CRAG: grade + correct if retrieval quality is too low.
        crag_result = self._safe_step(
            errors,
            "crag",
            lambda: self.crag.run(
                query,
                pool,
                retrieve_fn=_retrieve_and_rerank,
            ),
        )
        techniques.append("crag")

        final_chunks = crag_result.chunks if crag_result else pool
        crag_info = (
            {
                "average_relevance": crag_result.average_relevance,
                "corrective_rounds": crag_result.corrective_rounds,
                "corrective_queries": crag_result.corrective_queries,
                "triggered_correction": crag_result.triggered_correction,
            }
            if crag_result
            else None
        )
        return final_chunks, expansion["graph_nodes"], expansion["graph_edges"], crag_info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_step(errors: List[str], stage: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log("error", stage=stage, error=str(exc))
            errors.append(f"{stage}: {exc}")
            return None

    def _finalize(
        self,
        run_id: str,
        query: str,
        routing: RoutingDecision,
        techniques: List[str],
        chunks: List[Dict[str, Any]],
        graph_nodes: List[str],
        graph_edges: List[Dict[str, Any]],
        generated: GeneratedAnswer,
        cost_tracker: CostTracker,
        errors: List[str],
        crag_info: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        return PipelineResult(
            run_id=run_id,
            query=query,
            route=routing.route.value,
            techniques_executed=techniques,
            retrieved_chunks=chunks,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            answer=generated.answer,
            citations=generated.citations,
            model_used=generated.model,
            total_latency_ms=generated.latency_ms,
            cost_summary=cost_tracker.summary(),
            crag_info=crag_info,
            errors=errors,
        )