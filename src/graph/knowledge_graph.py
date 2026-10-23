"""
Knowledge graph construction, persistence, and lookup.

Builds a `networkx.MultiDiGraph` where:
  - nodes are normalized entity names (with `type` and `chunk_ids` attributes)
  - edges represent extracted relationships (with `relation` and `chunk_id`)

The graph is persisted to disk as JSON (node-link format) so it survives
between runs without needing a graph database server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx

from config import settings
from src.graph.extraction import Entity, EntityExtractor, Relationship, RelationshipExtractor
from src.ingestion.chunker import Chunk
from src.utils.logger import RunTimer, pipeline_logger


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


class KnowledgeGraphBuilder:
    """Builds and persists the GraphRAG knowledge graph from document chunks."""

    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        relationship_extractor: Optional[RelationshipExtractor] = None,
    ) -> None:
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.relationship_extractor = relationship_extractor or RelationshipExtractor()
        self.graph = nx.MultiDiGraph()

    def build_from_chunks(self, chunks: List[Chunk]) -> nx.MultiDiGraph:
        """Extracts entities + relationships from every chunk and merges them
        into a single knowledge graph."""
        with RunTimer() as timer:
            for chunk in chunks:
                try:
                    self._process_chunk(chunk)
                except Exception as exc:  # noqa: BLE001
                    pipeline_logger.log(
                        "error", stage="graph_build_chunk", chunk_id=chunk.chunk_id, error=str(exc)
                    )
        pipeline_logger.log(
            "graph_built",
            num_nodes=self.graph.number_of_nodes(),
            num_edges=self.graph.number_of_edges(),
            latency_ms=timer.elapsed_ms,
        )
        return self.graph

    def _process_chunk(self, chunk: Chunk) -> None:
        entities = self.entity_extractor.extract(chunk.text)
        if not entities:
            return
        for entity in entities:
            self._add_entity(entity, chunk.chunk_id)

        relationships = self.relationship_extractor.extract(chunk.text, entities)
        for rel in relationships:
            self._add_relationship(rel, chunk.chunk_id)

    def _add_entity(self, entity: Entity, chunk_id: str) -> None:
        name = _normalize_name(entity.name)
        if not name or len(name) < settings.graph.min_entity_len:
            return
        if self.graph.has_node(name):
            node = self.graph.nodes[name]
            chunk_ids = set(node.get("chunk_ids", []))
            chunk_ids.add(chunk_id)
            node["chunk_ids"] = list(chunk_ids)
        else:
            self.graph.add_node(name, type=entity.type, chunk_ids=[chunk_id])

    def _add_relationship(self, rel: Relationship, chunk_id: str) -> None:
        source = _normalize_name(rel.source)
        target = _normalize_name(rel.target)
        if not (self.graph.has_node(source) and self.graph.has_node(target)):
            return
        self.graph.add_edge(source, target, relation=rel.relation, chunk_id=chunk_id)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Optional[Path] = None) -> Path:
        path = path or settings.graph.store_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph, edges="edges")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        pipeline_logger.log("graph_saved", path=str(path), num_nodes=self.graph.number_of_nodes())
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "KnowledgeGraphStore":
        path = path or settings.graph.store_path
        if not path.exists():
            return KnowledgeGraphStore(nx.MultiDiGraph())
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data, directed=True, multigraph=True, edges="edges")
        pipeline_logger.log("graph_loaded", path=str(path), num_nodes=graph.number_of_nodes())
        return KnowledgeGraphStore(graph)


class KnowledgeGraphStore:
    """Read-side convenience wrapper around a loaded knowledge graph."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def is_empty(self) -> bool:
        return self.graph.number_of_nodes() == 0

    def find_entities_by_substring(self, text: str) -> List[str]:
        """Finds graph node names that appear as a substring of `text`
        (case/whitespace-insensitive) — used to link a user query or a
        retrieved chunk to graph nodes without another LLM call."""
        text_norm = " ".join(text.strip().split()).lower()
        matches = []
        for node in self.graph.nodes:
            node_norm = node.lower()
            if node_norm and node_norm in text_norm:
                matches.append(node)
        return matches

    def neighbors(self, node: str, hops: int, max_neighbors: int) -> Dict[str, Any]:
        """Returns neighboring nodes/edges within `hops` hops of `node`."""
        if node not in self.graph:
            return {"nodes": [], "edges": []}

        visited = {node}
        frontier = {node}
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                for neighbor in list(self.graph.successors(n)) + list(self.graph.predecessors(n)):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            visited.update(next_frontier)
            frontier = next_frontier
            if len(visited) >= max_neighbors + 1:
                break

        visited.discard(node)
        visited = set(list(visited)[:max_neighbors])

        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u == node and v in visited or v == node and u in visited:
                edges.append({"source": u, "target": v, "relation": data.get("relation", ""),
                              "chunk_id": data.get("chunk_id")})
        return {"nodes": list(visited), "edges": edges}

    def chunk_ids_for_nodes(self, nodes: Iterable[str]) -> List[str]:
        chunk_ids: List[str] = []
        for node in nodes:
            if node in self.graph:
                chunk_ids.extend(self.graph.nodes[node].get("chunk_ids", []))
        # de-dup while preserving order
        seen = set()
        out = []
        for cid in chunk_ids:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out
