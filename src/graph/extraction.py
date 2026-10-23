"""
Entity and relationship extraction for GraphRAG.

Both extractors use the centralized Ollama Cloud LLM adapter to turn an
Arabic text chunk into structured graph data, and are always used together
by `KnowledgeGraphBuilder` (relationships are only extracted for the
entities already found in the same chunk). They are kept in one module since
each is a small, single-purpose LLM-prompt wrapper.

- EntityExtractor: extracts named entities (departments, systems, documents,
  roles, equipment, processes, locations) from a chunk of text.
- RelationshipExtractor: given a chunk and its already-extracted entities,
  identifies directed relationships between those entities as stated in the
  text (never inventing relationships that aren't present).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.llm.ollama_client import OllamaCloudClient, llm_client
from src.utils.logger import RunTimer, pipeline_logger

_ENTITY_SYSTEM_PROMPT = """أنت نظام استخراج كيانات لبناء رسم بياني معرفي (Knowledge Graph) من وثائق إجراءات عمل عربية.
استخرج الكيانات المهمة فقط من النص التالي: أسماء الأقسام/الوحدات، الأنظمة، المستندات والنماذج،
الأدوار الوظيفية، المعدات، الإجراءات/العمليات، الأماكن.
أعد النتيجة بصيغة JSON فقط دون أي شرح، بالشكل التالي:
{"entities": [{"name": "...", "type": "DEPARTMENT|SYSTEM|DOCUMENT|ROLE|EQUIPMENT|PROCESS|LOCATION|OTHER"}]}
لا تكرر نفس الكيان مرتين. تجاهل الكيانات العامة جداً أو غير المفيدة."""

_RELATION_SYSTEM_PROMPT = """أنت نظام استخراج علاقات لبناء رسم بياني معرفي من وثائق إجراءات عمل عربية.
لديك قائمة كيانات مستخرجة من نص معين. مهمتك تحديد العلاقات المباشرة بين هذه الكيانات
كما وردت في النص فقط (لا تخترع علاقات غير مذكورة).
أعد النتيجة بصيغة JSON فقط بالشكل التالي:
{"relationships": [{"source": "اسم الكيان الأول", "target": "اسم الكيان الثاني", "relation": "وصف مختصر للعلاقة بالعربية"}]}
إذا لم توجد علاقات واضحة أعد قائمة فارغة."""


@dataclass
class Entity:
    name: str
    type: str


@dataclass
class Relationship:
    source: str
    target: str
    relation: str


class EntityExtractor:
    """LLM-driven entity extractor operating on chunk text."""

    def __init__(self, client: OllamaCloudClient | None = None) -> None:
        self.client = client or llm_client

    def extract(self, chunk_text: str) -> List[Entity]:
        prompt = f"النص:\n{chunk_text}\n\nأعد كائن JSON فقط."
        with RunTimer() as timer:
            try:
                data: Dict[str, Any] = self.client.complete_json(
                    prompt,
                    system=_ENTITY_SYSTEM_PROMPT,
                    purpose="entity_extraction",
                )
            except Exception as exc:  # noqa: BLE001
                pipeline_logger.log("error", stage="entity_extraction", error=str(exc))
                data = {}

        raw_entities = data.get("entities", []) if isinstance(data, dict) else []
        entities: List[Entity] = []
        seen = set()
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            etype = str(item.get("type", "OTHER")).strip().upper() or "OTHER"
            if not name or len(name) < 2:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(Entity(name=name, type=etype))

        pipeline_logger.log(
            "entity_extraction",
            num_entities=len(entities),
            latency_ms=timer.elapsed_ms,
        )
        return entities


class RelationshipExtractor:
    """LLM-driven relationship extractor operating on (chunk_text, entities)."""

    def __init__(self, client: OllamaCloudClient | None = None) -> None:
        self.client = client or llm_client

    def extract(self, chunk_text: str, entities: List[Entity]) -> List[Relationship]:
        if len(entities) < 2:
            return []

        entity_list = "\n".join(f"- {e.name} ({e.type})" for e in entities)
        prompt = (
            f"النص:\n{chunk_text}\n\nالكيانات المستخرجة:\n{entity_list}\n\n"
            "أعد كائن JSON فقط يحتوي على العلاقات بين هذه الكيانات كما وردت في النص."
        )

        with RunTimer() as timer:
            try:
                data: Dict[str, Any] = self.client.complete_json(
                    prompt,
                    system=_RELATION_SYSTEM_PROMPT,
                    purpose="relationship_extraction",
                )
            except Exception as exc:  # noqa: BLE001
                pipeline_logger.log("error", stage="relationship_extraction", error=str(exc))
                data = {}

        raw_relations = data.get("relationships", []) if isinstance(data, dict) else []
        entity_names = {e.name.lower() for e in entities}
        relationships: List[Relationship] = []
        for item in raw_relations:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation = str(item.get("relation", "")).strip() or "مرتبط بـ"
            if not source or not target or source.lower() == target.lower():
                continue
            # Keep only relations between entities we actually extracted, to
            # avoid the LLM inventing new nodes outside the entity list.
            if source.lower() not in entity_names or target.lower() not in entity_names:
                continue
            relationships.append(Relationship(source=source, target=target, relation=relation))

        pipeline_logger.log(
            "relationship_extraction",
            num_relationships=len(relationships),
            latency_ms=timer.elapsed_ms,
        )
        return relationships
