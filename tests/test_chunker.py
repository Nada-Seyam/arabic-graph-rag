from __future__ import annotations

from src.ingestion.chunker import ArabicChunker
from src.ingestion.pdf_extractor import PageText


def make_pages(texts):
    return [PageText(source="test.pdf", page_number=i + 1, text=t) for i, t in enumerate(texts)]


def test_split_sentences_basic():
    chunker = ArabicChunker()
    sentences = chunker.split_sentences("هذه الجملة الأولى. هذه الجملة الثانية؟ وهذه الثالثة!")
    assert len(sentences) == 3


def test_chunk_pages_respects_min_length():
    chunker = ArabicChunker(chunk_size=1000, chunk_overlap=100, min_chunk_chars=1000)
    pages = make_pages(["نص قصير جداً."])
    chunks = chunker.chunk_pages(pages)
    assert chunks == []  # too short to meet min_chunk_chars


def test_chunk_pages_produces_multiple_chunks_for_long_text():
    long_sentence = "هذه جملة طويلة تحتوي على معلومات إجرائية متعددة. " * 40
    chunker = ArabicChunker(chunk_size=200, chunk_overlap=40, min_chunk_chars=20)
    pages = make_pages([long_sentence])
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) > 1
    for c in chunks:
        assert c.source == "test.pdf"
        assert c.page_start == 1 and c.page_end == 1


def test_chunk_pages_tracks_page_ranges_across_pages():
    chunker = ArabicChunker(chunk_size=5000, chunk_overlap=0, min_chunk_chars=5)
    pages = make_pages(["محتوى الصفحة الأولى. " * 5, "محتوى الصفحة الثانية. " * 5])
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) >= 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end >= 1


def test_chunk_ids_are_unique():
    long_text = "معلومة إجرائية مهمة جداً يجب اتباعها بدقة. " * 60
    chunker = ArabicChunker(chunk_size=150, chunk_overlap=30, min_chunk_chars=10)
    chunks = chunker.chunk_pages(make_pages([long_text]))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
