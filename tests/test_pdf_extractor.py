from __future__ import annotations

import json
import zipfile
from pathlib import Path

from src.ingestion.pdf_extractor import ArabicPdfExtractor


def _make_zip_pdf(tmp_path: Path, pages: dict[int, str], with_manifest: bool = True) -> Path:
    zip_path = tmp_path / "fake_manual.pdf"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for page_num, text in pages.items():
            zf.writestr(f"{page_num}.txt", text)
        if with_manifest:
            manifest = {
                "num_pages": len(pages),
                "pages": [
                    {"page_number": p, "text": {"path": f"{p}.txt"}} for p in sorted(pages)
                ],
            }
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
    return zip_path


def test_extracts_pages_from_zip_with_manifest(tmp_path):
    pages = {1: "محتوى الصفحة الأولى", 2: "محتوى الصفحة الثانية", 3: "محتوى الصفحة الثالثة"}
    pdf_path = _make_zip_pdf(tmp_path, pages, with_manifest=True)

    extractor = ArabicPdfExtractor()
    result = extractor.extract(pdf_path)

    assert len(result) == 3
    assert [p.page_number for p in result] == [1, 2, 3]
    assert result[0].text == "محتوى الصفحة الأولى"


def test_extracts_pages_from_zip_without_manifest(tmp_path):
    pages = {1: "أولاً", 2: "ثانياً", 10: "عاشراً"}
    pdf_path = _make_zip_pdf(tmp_path, pages, with_manifest=False)

    extractor = ArabicPdfExtractor()
    result = extractor.extract(pdf_path)

    # Numeric sort should place page 10 last, not lexicographically after "1".
    assert [p.text for p in result] == ["أولاً", "ثانياً", "عاشراً"]


def test_extract_raises_for_missing_file(tmp_path):
    from src.ingestion.pdf_extractor import PdfExtractionError

    extractor = ArabicPdfExtractor()
    missing = tmp_path / "does_not_exist.pdf"
    try:
        extractor.extract(missing)
        assert False, "expected PdfExtractionError"
    except PdfExtractionError:
        pass
