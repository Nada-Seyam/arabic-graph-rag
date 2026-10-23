from __future__ import annotations

from src.retrieval.bm25_retriever import Bm25Retriever, arabic_tokenize
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion


def test_arabic_tokenize_filters_stopwords_and_short_tokens():
    tokens = arabic_tokenize("هذا هو النص في الوثيقة من أجل الاختبار")
    assert "في" not in tokens
    assert "من" not in tokens
    assert "الوثيقة" in tokens or "وثيقة" in tokens or any("وثيق" in t for t in tokens)


def _corpus():
    return [
        {"chunk_id": "1", "text": "إجراءات استلام البريد الوارد في الوحدة المركزية", "metadata": {}},
        {"chunk_id": "2", "text": "خطوات جرد الأصول في المستودع سنوياً", "metadata": {}},
        {"chunk_id": "3", "text": "تفعيل جهاز الإنذار المركزي عند وجود طارئ", "metadata": {}},
    ]


def test_bm25_retrieves_relevant_chunk_first():
    bm25 = Bm25Retriever()
    bm25.build(_corpus())
    hits = bm25.query("ما هي خطوات استلام البريد الوارد؟", top_k=3)
    assert hits
    assert hits[0]["chunk_id"] == "1"


def test_bm25_not_ready_before_build():
    bm25 = Bm25Retriever()
    assert not bm25.is_ready()
    assert bm25.query("أي سؤال", top_k=3) == []


def test_bm25_scores_are_normalized_0_to_1():
    bm25 = Bm25Retriever()
    bm25.build(_corpus())
    hits = bm25.query("جرد الأصول في المستودع", top_k=3)
    for h in hits:
        assert 0.0 <= h["score"] <= 1.0


def test_reciprocal_rank_fusion_combines_and_ranks():
    dense = [
        {"chunk_id": "a", "text": "t_a", "metadata": {}, "score": 0.9, "_source": "dense"},
        {"chunk_id": "b", "text": "t_b", "metadata": {}, "score": 0.5, "_source": "dense"},
    ]
    bm25 = [
        {"chunk_id": "b", "text": "t_b", "metadata": {}, "score": 0.8, "_source": "bm25"},
        {"chunk_id": "c", "text": "t_c", "metadata": {}, "score": 0.4, "_source": "bm25"},
    ]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    ids = [f["chunk_id"] for f in fused]
    # "b" appears near the top of both lists, so it should be ranked first overall.
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"}


def test_reciprocal_rank_fusion_empty_lists():
    assert reciprocal_rank_fusion([[], []], k=60) == []
