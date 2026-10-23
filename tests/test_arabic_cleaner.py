from __future__ import annotations

from src.ingestion.arabic_cleaner import ArabicCleaningOptions, ArabicTextCleaner


def test_removes_diacritics():
    cleaner = ArabicTextCleaner()
    result = cleaner.clean("الْعَرَبِيَّة")
    assert "\u064e" not in result  # fatha removed
    assert "ة" in result


def test_normalizes_alef_variants():
    cleaner = ArabicTextCleaner()
    result = cleaner.clean("أحمد إبراهيم آدم")
    assert "أ" not in result
    assert "إ" not in result
    assert "آ" not in result
    assert "احمد ابراهيم ادم" in result


def test_normalizes_arabic_indic_digits():
    cleaner = ArabicTextCleaner()
    result = cleaner.clean("الصفحة ١٢٣")
    assert "123" in result
    assert "١" not in result


def test_strips_page_number_lines():
    cleaner = ArabicTextCleaner()
    text = "هذا محتوى مهم\n42\nمزيد من المحتوى"
    result = cleaner.clean(text)
    assert "42" not in result.splitlines()


def test_collapses_whitespace():
    cleaner = ArabicTextCleaner()
    result = cleaner.clean("كلمة   بمسافات    كثيرة")
    assert "  " not in result


def test_is_mostly_arabic():
    assert ArabicTextCleaner.is_mostly_arabic("هذا نص عربي بالكامل تقريبا")
    assert not ArabicTextCleaner.is_mostly_arabic("This is fully English text 12345")


def test_taa_marbuta_preserved_by_default():
    cleaner = ArabicTextCleaner(ArabicCleaningOptions(normalize_taa_marbuta=False))
    result = cleaner.clean("مدرسة")
    assert result.endswith("ة")


def test_empty_input_returns_empty():
    cleaner = ArabicTextCleaner()
    assert cleaner.clean("") == ""
    assert cleaner.clean(None) == ""  # type: ignore[arg-type]
