from __future__ import annotations

from unittest.mock import MagicMock

import networkx as nx

from src.graph.extraction import Entity, EntityExtractor, Relationship, RelationshipExtractor
from src.graph.knowledge_graph import KnowledgeGraphBuilder, KnowledgeGraphStore
from src.ingestion.chunker import Chunk


def _chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source="test.pdf", page_start=1, page_end=1, chunk_index=0)


def test_build_from_chunks_adds_entities_and_relationships():
    entity_extractor = MagicMock(spec=EntityExtractor)
    entity_extractor.extract.return_value = [
        Entity(name="وحدة البريد", type="DEPARTMENT"),
        Entity(name="وحدة الأرشيف", type="DEPARTMENT"),
    ]
    relationship_extractor = MagicMock(spec=RelationshipExtractor)
    relationship_extractor.extract.return_value = [
        Relationship(source="وحدة البريد", target="وحدة الأرشيف", relation="ترسل الملفات الى")
    ]

    builder = KnowledgeGraphBuilder(entity_extractor, relationship_extractor)
    graph = builder.build_from_chunks([_chunk("نص تجريبي عن البريد والأرشيف")])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge("وحدة البريد", "وحدة الأرشيف")


def test_build_from_chunks_skips_chunk_with_no_entities():
    entity_extractor = MagicMock(spec=EntityExtractor)
    entity_extractor.extract.return_value = []
    relationship_extractor = MagicMock(spec=RelationshipExtractor)

    builder = KnowledgeGraphBuilder(entity_extractor, relationship_extractor)
    graph = builder.build_from_chunks([_chunk("نص لا يحتوي كيانات")])

    assert graph.number_of_nodes() == 0
    relationship_extractor.extract.assert_not_called()


def test_save_and_load_round_trip(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_node("قسم البريد", type="DEPARTMENT", chunk_ids=["c1"])
    graph.add_node("قسم الأرشيف", type="DEPARTMENT", chunk_ids=["c2"])
    graph.add_edge("قسم البريد", "قسم الأرشيف", relation="يرسل الى", chunk_id="c1")

    builder = KnowledgeGraphBuilder.__new__(KnowledgeGraphBuilder)
    builder.graph = graph
    path = tmp_path / "graph.json"
    builder.save(path)

    store = KnowledgeGraphBuilder.load(path)
    assert isinstance(store, KnowledgeGraphStore)
    assert store.graph.number_of_nodes() == 2
    assert store.graph.number_of_edges() == 1


def test_knowledge_graph_store_find_entities_by_substring():
    graph = nx.MultiDiGraph()
    graph.add_node("قسم البريد", type="DEPARTMENT", chunk_ids=["c1"])
    store = KnowledgeGraphStore(graph)

    matches = store.find_entities_by_substring("تحدثنا اليوم عن قسم البريد وإجراءاته")
    assert "قسم البريد" in matches


def test_knowledge_graph_store_neighbors_within_hops():
    graph = nx.MultiDiGraph()
    graph.add_node("A", type="X", chunk_ids=[])
    graph.add_node("B", type="X", chunk_ids=[])
    graph.add_node("C", type="X", chunk_ids=[])
    graph.add_edge("A", "B", relation="r1", chunk_id="c1")
    graph.add_edge("B", "C", relation="r2", chunk_id="c2")
    store = KnowledgeGraphStore(graph)

    one_hop = store.neighbors("A", hops=1, max_neighbors=10)
    assert "B" in one_hop["nodes"]
    assert "C" not in one_hop["nodes"]

    two_hop = store.neighbors("A", hops=2, max_neighbors=10)
    assert "C" in two_hop["nodes"]


def test_knowledge_graph_store_is_empty():
    store = KnowledgeGraphStore(nx.MultiDiGraph())
    assert store.is_empty()
