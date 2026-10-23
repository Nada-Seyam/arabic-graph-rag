"""
Arabic-aware chunking.

Splits cleaned document text into overlapping chunks that respect Arabic
sentence boundaries (., !, ?, Arabic full stop ۔, Arabic question mark ؟,
Arabic semicolon ؛, and newlines) instead of cutting mid-word/mid-sentence.
Chunk size and overlap are measured in characters (robust across languages
without requiring a tokenizer) and are fully configurable.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

from config import settings
from src.ingestion.pdf_extractor import PageText

_SENTENCE_BOUNDARY = re.compile(r"(?<=[\.\!\?\؟\؛۔:])\s+")

# Detects Arabic procedural list markers: numbered items like "-1", "1.",
# "-16", and lettered sub-items like "أ.", "ب.", "ت." (commonly used for
# a., b., c. style sub-steps within a numbered procedure). Sentences
# starting with one of these are joined onto their own line instead of
# being run together with a plain space, so downstream compression/
# generation LLMs can actually see "this is a numbered list" instead of
# receiving one long undifferentiated paragraph.
_LIST_MARKER_RE = re.compile(r"^(?:[-–—]\s*\d+\b|\d+[.\-]\s|[\u0621-\u064A]\.\s)")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page_start: int
    page_end: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class ArabicChunker:
    """Sentence-aware sliding-window chunker for Arabic (and mixed) text."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_chars: int | None = None,
    ) -> None:
        cfg = settings.chunking
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.min_chunk_chars = min_chunk_chars or cfg.min_chunk_chars

    def split_sentences(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        sentences = _SENTENCE_BOUNDARY.split(text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _join_sentences(sentences: List[str]) -> str:
        """Joins sentences back into chunk text, putting a newline before
        any sentence that looks like a numbered/lettered list item so the
        list structure survives into the final chunk instead of being
        flattened into one run-on paragraph."""
        parts: List[str] = []
        for i, s in enumerate(sentences):
            if i == 0:
                parts.append(s)
            elif _LIST_MARKER_RE.match(s):
                parts.append("\n" + s)
            else:
                parts.append(" " + s)
        return "".join(parts).strip()

    def chunk_pages(self, pages: List[PageText]) -> List[Chunk]:
        """Chunk a whole document's pages, tracking which page range each
        chunk spans so citations can point back to page numbers."""
        # Build one continuous stream of (sentence, page_number) tuples.
        sentence_stream: List[tuple] = []
        for page in pages:
            for sentence in self.split_sentences(page.text):
                sentence_stream.append((sentence, page.page_number))

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_pages: List[int] = []
        current_len = 0
        chunk_index = 0

        def flush():
            nonlocal current_sentences, current_pages, current_len, chunk_index
            if not current_sentences:
                return
            text = self._join_sentences(current_sentences)
            if len(text) >= self.min_chunk_chars:
                source = pages[0].source if pages else "unknown"
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=text,
                        source=source,
                        page_start=min(current_pages),
                        page_end=max(current_pages),
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        for sentence, page_number in sentence_stream:
            candidate_len = current_len + len(sentence) + 1
            if candidate_len > self.chunk_size and current_sentences:
                flush()
                # Build overlap: keep trailing sentences whose combined length
                # is <= chunk_overlap for continuity between chunks.
                overlap_sentences: List[str] = []
                overlap_pages: List[int] = []
                overlap_len = 0
                for s, p in zip(reversed(current_sentences), reversed(current_pages)):
                    if overlap_len + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_pages.insert(0, p)
                    overlap_len += len(s) + 1
                current_sentences = overlap_sentences
                current_pages = overlap_pages
                current_len = overlap_len

            current_sentences.append(sentence)
            current_pages.append(page_number)
            current_len += len(sentence) + 1

        flush()
        return chunks