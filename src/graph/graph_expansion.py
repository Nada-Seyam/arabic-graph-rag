"""
Controlled GraphRAG expansion.

Keeps the original hybrid results and adds a limited number of
graph-connected chunks so graph expansion does not flood the reranker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import settings
from src.graph.knowledge_graph import KnowledgeGraphStore
from src.utils.logger import RunTimer, pipeline_logger
from src.vectorstore.chroma_store import ChromaVectorStore


class GraphExpander:
    """Expands hybrid retrieval results using a controlled graph traversal."""

    def __init__(
        self,
        graph_store: KnowledgeGraphStore,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self._chunk_lookup: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_chunk_lookup(self) -> Dict[str, Dict[str, Any]]:
        if self._chunk_lookup is None:
            all_chunks = self.vector_store.get_all_chunks()
            self._chunk_lookup = {
                c["chunk_id"]: c for c in all_chunks
            }
        return self._chunk_lookup

    def expand(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        hops: Optional[int] = None,
        max_neighbors: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Keep all hybrid chunks and add at most 30 graph-expanded chunks.
        """

        hops = hops if hops is not None else settings.graph.expansion_hops
        max_neighbors = (
            max_neighbors
            if max_neighbors is not None
            else settings.graph.max_neighbors
        )

        if self.graph_store.is_empty():
            pipeline_logger.log(
                "graph_expansion_skipped",
                reason="empty_graph",
            )
            return {
                "chunks": retrieved_chunks,
                "graph_nodes": [],
                "graph_edges": [],
                "seed_entities": [],
            }

        # Maximum number of NEW chunks added by GraphRAG.
        max_new_chunks = 30

        with RunTimer() as timer:

            # ---------------------------------------------------------
            # 1. Find graph entities from the query first.
            # ---------------------------------------------------------
            query_entities = (
                self.graph_store.find_entities_by_substring(query)
            )

            # ---------------------------------------------------------
            # 2. Find entities from retrieved chunks.
            #    Limit the amount of text used for graph matching.
            #    This prevents generic entities from hundreds of chunks
            #    becoming graph seeds.
            # ---------------------------------------------------------
            chunk_text = " \n".join(
                c.get("text", "")
                for c in retrieved_chunks[:10]
            )

            chunk_entities = (
                self.graph_store.find_entities_by_substring(chunk_text)
            )

            # Query entities get priority.
            seed_entities: List[str] = []

            for entity in query_entities + chunk_entities:
                if entity not in seed_entities:
                    seed_entities.append(entity)

            # Keep graph expansion focused.
            seed_entities = seed_entities[:20]

            # ---------------------------------------------------------
            # 3. Expand only from the limited seed set.
            # ---------------------------------------------------------
            all_neighbor_nodes: set[str] = set()
            all_edges: List[Dict[str, Any]] = []

            for entity in seed_entities:
                expansion = self.graph_store.neighbors(
                    entity,
                    hops=hops,
                    max_neighbors=max_neighbors,
                )

                all_neighbor_nodes.update(expansion["nodes"])
                all_edges.extend(expansion["edges"])

            # ---------------------------------------------------------
            # 4. Get chunks connected to graph entities.
            # ---------------------------------------------------------
            graph_nodes = set(seed_entities) | all_neighbor_nodes

            expansion_chunk_ids = (
                self.graph_store.chunk_ids_for_nodes(graph_nodes)
            )

            existing_ids = {
                c["chunk_id"] for c in retrieved_chunks
            }

            lookup = self._get_chunk_lookup()

            new_chunks: List[Dict[str, Any]] = []

        # Score graph-expanded chunks by how strongly they match the query.
        # This keeps the most useful graph chunks instead of relying on
        # arbitrary graph traversal order.

        query_terms = set(query.lower().split())

        scored_graph_chunks = []

        for cid in expansion_chunk_ids:
            if cid in existing_ids:
                continue

            chunk = lookup.get(cid)
            if not chunk:
                continue

            text = chunk.get("text", "").lower()

            # Simple lexical relevance score.
            matches = sum(1 for term in query_terms if term in text)

            scored_graph_chunks.append((matches, cid, chunk))

        # Highest lexical overlap first.
        scored_graph_chunks.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for _, cid, chunk in scored_graph_chunks[:max_new_chunks]:
            enriched = dict(chunk)
            enriched["score"] = 0.0
            enriched["_source"] = "graph_expansion"

            new_chunks.append(enriched)
            existing_ids.add(cid)

        combined = retrieved_chunks + new_chunks

        pipeline_logger.log(
            "graph_expansion",
            query=query,
            seed_entities=seed_entities,
            num_neighbor_nodes=len(all_neighbor_nodes),
            num_new_chunks=len(new_chunks),
            total_chunks=len(combined),
            latency_ms=timer.elapsed_ms,
        )

        return {
            "chunks": combined,
            "graph_nodes": list(all_neighbor_nodes),
            "graph_edges": all_edges,
            "seed_entities": seed_entities,
        }