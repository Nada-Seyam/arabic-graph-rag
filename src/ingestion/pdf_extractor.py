"""
Arabic PDF ingestion.

Handles two input shapes transparently:

1. Standard PDF files — parsed with `pdfplumber`, with a PyMuPDF fallback for
   documents pdfplumber cannot open, and an OCR fallback (pytesseract, if
   installed) for scanned pages with no extractable text layer.
2. "Paginated archive" PDFs — some document export pipelines (including the
   sample manuals shipped with this project) save each page as a JPEG image
   plus a pre-OCR'd `.txt` sidecar, packaged inside a ZIP container that is
   given a `.pdf` extension. This extractor detects that shape via
   `zipfile.is_zipfile` and reads the per-page text directly, honoring
   `manifest.json` page ordering when present.

Either way, the output is a uniform list of `PageText` records so downstream
cleaning/chunking never needs to know which path was taken.
"""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.utils.logger import pipeline_logger

# Some real (non-OCR) Arabic PDFs store each RTL line's text in on-screen
# *visual* order in the content stream rather than logical reading order —
# pdfplumber (and occasionally PyMuPDF) then extracts it mirrored, e.g.
# "رقم النسخة 1" comes back as "1 ةخسنلا مقر". Embedded left-to-right runs
# (dates, numbers, Latin text) are typically NOT mirrored, so a blind
# per-line reversal would turn "08/2024" into "4202/80". We only reverse
# the non-LTR segments, and only for documents that are actually affected
# (detected via a common-word heuristic below) so correctly-extracted PDFs
# are left untouched.
_LTR_RUN = re.compile(r"[0-9A-Za-z/.\-:%]+")
_ARABIC_WORD_RE = re.compile(r"[\u0621-\u064A]+")

# Very common short Arabic function/procedural words. A correctly-ordered
# Arabic document will contain many of these as whole words; a mirrored one
# will contain almost none (e.g. "من" mirrored becomes "نم", which won't
# match as a whole word even though it might appear as a coincidental
# substring inside garbled text — hence matching whole words, not substrings).
_COMMON_ARABIC_WORDS = {
    "من", "في", "على", "إلى", "الى", "عن", "مع", "هذا", "هذه", "التي",
    "الذي", "أو", "او", "أن", "ان", "كان", "إجراءات", "اجراءات", "تاريخ",
    "يتم", "قبل", "بعد", "كل", "عند", "حال", "التالية", "رقم",
}


def _fix_bidi_line(line: str) -> str:
    """Reverses only the RTL (Arabic) segments of a mirrored-order line,
    leaving embedded LTR runs (numbers/dates/Latin) in their original order."""
    if not line:
        return line
    segments: list[tuple[str, str]] = []
    last_end = 0
    for m in _LTR_RUN.finditer(line):
        if m.start() > last_end:
            segments.append(("rtl", line[last_end:m.start()]))
        segments.append(("ltr", m.group()))
        last_end = m.end()
    if last_end < len(line):
        segments.append(("rtl", line[last_end:]))

    segments.reverse()
    return "".join(seg[::-1] if kind == "rtl" else seg for kind, seg in segments)


def _looks_bidi_mirrored(sample_text: str) -> bool:
    """Heuristic: does this sample look like mirrored-order Arabic text?

    Compares how many common Arabic function words appear as whole words
    versus how many appear as whole words once every line is reversed.
    Word-boundary matching (not substring counting) avoids false positives
    from short words coincidentally appearing inside garbled text.
    """
    if not sample_text.strip():
        return False

    reversed_sample = "\n".join(_fix_bidi_line(l) for l in sample_text.split("\n"))

    def word_hits(text: str) -> int:
        words = _ARABIC_WORD_RE.findall(text)
        return sum(1 for w in words if w in _COMMON_ARABIC_WORDS)

    original_hits = word_hits(sample_text)
    reversed_hits = word_hits(reversed_sample)

    # Require the reversed version to clearly recognize real words while
    # the original essentially doesn't.
    return reversed_hits >= 5 and reversed_hits > original_hits * 2


@dataclass
class PageText:
    """Raw extracted text for a single page of a source document."""

    source: str
    page_number: int
    text: str


class PdfExtractionError(RuntimeError):
    """Raised when a document cannot be parsed by any available strategy."""


class ArabicPdfExtractor:
    """Extracts per-page Arabic text from a PDF (or PDF-shaped archive)."""

    def extract(self, pdf_path: Path) -> List[PageText]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PdfExtractionError(f"File not found: {pdf_path}")

        try:
            if zipfile.is_zipfile(pdf_path):
                pages = self._extract_from_zip_archive(pdf_path)
            else:
                pages = self._extract_from_real_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log("error", stage="pdf_extraction", file=str(pdf_path), error=str(exc))
            raise PdfExtractionError(f"Failed to extract {pdf_path.name}: {exc}") from exc

        pipeline_logger.log(
            "pdf_extracted",
            file=pdf_path.name,
            num_pages=len(pages),
            total_chars=sum(len(p.text) for p in pages),
        )
        return pages

    # ------------------------------------------------------------------ #
    # Strategy 1: ZIP-packaged "page image + OCR text" archives
    # ------------------------------------------------------------------ #
    def _extract_from_zip_archive(self, pdf_path: Path) -> List[PageText]:
        pages: List[PageText] = []
        with zipfile.ZipFile(pdf_path) as zf:
            names = set(zf.namelist())
            manifest = None
            if "manifest.json" in names:
                with zf.open("manifest.json") as f:
                    try:
                        manifest = json.load(f)
                    except json.JSONDecodeError:
                        manifest = None

            if manifest and "pages" in manifest:
                page_entries = sorted(manifest["pages"], key=lambda p: p.get("page_number", 0))
                for entry in page_entries:
                    page_number = entry.get("page_number", len(pages) + 1)
                    text_path = (entry.get("text") or {}).get("path")
                    text = self._read_zip_text(zf, text_path) if text_path else ""
                    pages.append(PageText(source=pdf_path.name, page_number=page_number, text=text))
            else:
                # No manifest: fall back to sorting all *.txt members numerically.
                txt_members = sorted(
                    (n for n in names if n.lower().endswith(".txt")),
                    key=_numeric_sort_key,
                )
                for i, member in enumerate(txt_members, start=1):
                    text = self._read_zip_text(zf, member)
                    pages.append(PageText(source=pdf_path.name, page_number=i, text=text))

        if not pages:
            raise PdfExtractionError("No text sidecars found inside archive.")
        return pages

    @staticmethod
    def _read_zip_text(zf: zipfile.ZipFile, member: Optional[str]) -> str:
        if not member or member not in zf.namelist():
            return ""
        with zf.open(member) as f:
            raw = f.read()
        for encoding in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")

    # ------------------------------------------------------------------ #
    # Strategy 2: real PDF files
    # ------------------------------------------------------------------ #
    def _extract_from_real_pdf(self, pdf_path: Path) -> List[PageText]:
        try:
            pages = self._extract_with_pdfplumber(pdf_path)
        except Exception as plumber_exc:  # noqa: BLE001
            pipeline_logger.log(
                "warning",
                stage="pdf_extraction",
                message=f"pdfplumber failed ({plumber_exc}); trying PyMuPDF",
                file=pdf_path.name,
            )
            pages = self._extract_with_pymupdf(pdf_path)

        return self._maybe_fix_bidi_mirroring(pdf_path, self._normalize_unicode(pages))

    @staticmethod
    def _normalize_unicode(pages: List[PageText]) -> List[PageText]:
        """Applies NFKC (compatibility) Unicode normalization to every page.

        Some PDF generators encode Arabic text using Presentation Form
        glyphs (a separate, shaping-specific Unicode range) instead of the
        standard Arabic letters — e.g. "ﺪﻳ" instead of "دي". These render
        correctly in a PDF viewer but are different codepoints as raw text,
        which silently breaks exact-match search, BM25 tokenization (whose
        regex only covers the standard Arabic block), and — critically —
        the bidi-mirroring detection below, since it can't recognize any
        real words in a codepoint range it isn't looking at. NFKC folds
        presentation forms back to their standard-letter equivalents, so
        this must run BEFORE mirror detection, not after.
        """
        return [
            PageText(
                source=p.source,
                page_number=p.page_number,
                text=unicodedata.normalize("NFKC", p.text),
            )
            for p in pages
        ]

    @staticmethod
    def _maybe_fix_bidi_mirroring(pdf_path: Path, pages: List[PageText]) -> List[PageText]:
        """Detects whether this document's extracted Arabic text came out
        mirrored (visual order instead of logical order) and, if so,
        corrects every page. Left as a no-op for documents that already
        extract correctly."""
        sample = "\n".join(p.text for p in pages[:5])
        if not _looks_bidi_mirrored(sample):
            return pages

        pipeline_logger.log(
            "warning",
            stage="pdf_extraction",
            message="Detected mirrored/visual-order Arabic text; correcting line order.",
            file=pdf_path.name,
        )
        return [
            PageText(
                source=p.source,
                page_number=p.page_number,
                text="\n".join(_fix_bidi_line(line) for line in p.text.split("\n")),
            )
            for p in pages
        ]

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: Path) -> List[PageText]:
        import pdfplumber

        pages: List[PageText] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    text = ArabicPdfExtractor._ocr_page_fallback(pdf_path, i)
                pages.append(PageText(source=pdf_path.name, page_number=i, text=text))
        return pages

    @staticmethod
    def _extract_with_pymupdf(pdf_path: Path) -> List[PageText]:
        import fitz  # PyMuPDF

        pages: List[PageText] = []
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if not text.strip():
                    text = ArabicPdfExtractor._ocr_page_fallback(pdf_path, i)
                pages.append(PageText(source=pdf_path.name, page_number=i, text=text))
        return pages

    @staticmethod
    def _ocr_page_fallback(pdf_path: Path, page_number: int) -> str:
        """Best-effort OCR for scanned pages. Silently returns "" if OCR
        dependencies (pytesseract + Arabic language data) are unavailable —
        this keeps the pipeline fully functional even on minimal installs."""
        try:
            import pytesseract
            from pdf2image import convert_from_path

            images = convert_from_path(
                str(pdf_path), first_page=page_number, last_page=page_number
            )
            if not images:
                return ""
            return pytesseract.image_to_string(images[0], lang="ara+eng")
        except Exception as exc:  # noqa: BLE001
            pipeline_logger.log(
                "warning",
                stage="ocr_fallback",
                file=pdf_path.name,
                page=page_number,
                message=f"OCR unavailable/failed: {exc}",
            )
            return ""


def _numeric_sort_key(name: str) -> tuple:
    stem = Path(name).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)