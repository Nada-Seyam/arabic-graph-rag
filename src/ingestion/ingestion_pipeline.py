"""
End-to-end ingestion pipeline: PDF -> clean -> chunk -> embed/store ->
knowledge graph. Reusable from both `scripts/ingest.py` and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from config import settings
from src.graph.knowledge_graph import KnowledgeGraphBuilder
from src.ingestion.arabic_cleaner import ArabicTextCleaner
from src.ingestion.chunker import ArabicChunker, Chunk
from src.ingestion.pdf_extractor import ArabicPdfExtractor, PageText
from src.retrieval.bm25_retriever import Bm25Retriever
from src.utils.logger import RunTimer, pipeline_logger
from src.vectorstore.chroma_store import ChromaVectorStore


class IngestionPipeline:
    """Runs the full local ingestion pipeline over a directory of PDFs."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25: Optional[Bm25Retriever] = None,
        extractor: Optional[ArabicPdfExtractor] = None,
        cleaner: Optional[ArabicTextCleaner] = None,
        chunker: Optional[ArabicChunker] = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25 = bm25 or Bm25Retriever()
        self.extractor = extractor or ArabicPdfExtractor()
        self.cleaner = cleaner or ArabicTextCleaner()
        self.chunker = chunker or ArabicChunker()

    def ingest_directory(self, pdf_dir: Optional[Path] = None) -> List[Chunk]:
        pdf_dir = pdf_dir or settings.data.pdf_dir
        pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
        if not pdf_files:
            pipeline_logger.log("warning", stage="ingestion", message=f"No PDFs found in {pdf_dir}")
            return []

        all_chunks: List[Chunk] = []
        with RunTimer() as timer:
            for pdf_path in pdf_files:
                all_chunks.extend(self.ingest_file(pdf_path))
            self.bm25.build(self.vector_store.get_all_chunks())

        pipeline_logger.log(
            "ingestion_complete",
            num_files=len(pdf_files),
            num_chunks=len(all_chunks),
            latency_ms=timer.elapsed_ms,
        )
        return all_chunks

    def ingest_file(self, pdf_path: Path) -> List[Chunk]:
        pages: List[PageText] = self.extractor.extract(pdf_path)
        cleaned_pages = [
            PageText(source=p.source, page_number=p.page_number, text=self.cleaner.clean(p.text))
            for p in pages
        ]
        chunks = self.chunker.chunk_pages(cleaned_pages)
        self.vector_store.add_chunks(chunks)
        pipeline_logger.log(
            "file_ingested", file=pdf_path.name, num_pages=len(pages), num_chunks=len(chunks)
        )
        return chunks

    def build_knowledge_graph(self, chunks: List[Chunk]) -> Path:
        builder = KnowledgeGraphBuilder()
        builder.build_from_chunks(chunks)
        return builder.save()
