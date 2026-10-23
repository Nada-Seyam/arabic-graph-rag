"""
Arabic text cleaning and normalization.

Handles the quirks typical of OCR'd / extracted Arabic PDF text:
- Removing diacritics (tashkeel) and tatweel (kashida)
- Normalizing alef/yaa/taa-marbuta variants
- Fixing common OCR artifacts (broken ligatures, stray control chars)
- Normalizing Arabic-Indic digits to Western digits (configurable)
- Collapsing whitespace while preserving paragraph structure
- Removing page-furniture noise (page numbers, repeated headers/footers)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Arabic diacritics (tashkeel) + Quranic annotation marks
_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u08D4-\u08E1\u08E3-\u08FF]"
)
_TATWEEL = re.compile(r"\u0640+")
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_CONTROL_CHARS = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_SPACE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_ARABIC_LETTER = "\u0621-\u064A\u0660-\u0669\u066E-\u06D3\u06D5-\u06FF"
_LATIN_ALNUM = "A-Za-z0-9"

# Letter normalization map (alef variants -> bare alef, yaa variants, taa marbuta kept distinct)
_NORMALIZATION_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
    "ى": "ي",
    "ئ": "ي",
    "ؤ": "و",
    "ة": "ة",  # keep taa marbuta as-is by default (configurable via normalize_taa_marbuta)
}

_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*(/\s*\d{1,4})?\s*$")


@dataclass
class ArabicCleaningOptions:
    remove_diacritics: bool = True
    normalize_alef_yaa_waw: bool = True
    normalize_digits: bool = True
    normalize_taa_marbuta: bool = False
    strip_page_number_lines: bool = True
    collapse_whitespace: bool = True


class ArabicTextCleaner:
    """Applies a configurable, deterministic Arabic normalization pipeline."""

    def __init__(self, options: ArabicCleaningOptions | None = None) -> None:
        self.options = options or ArabicCleaningOptions()

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # NFKC (not just NFC) so that Arabic Presentation Form glyphs some
        # PDF generators use (a different Unicode range than standard
        # Arabic letters, but visually identical) get folded back to
        # standard letters. Without this, exact-match search, BM25
        # tokenization, and word-boundary regexes elsewhere in the
        # pipeline silently fail on documents using this encoding, even
        # though the text displays correctly. Ideally this has already
        # happened in pdf_extractor.py right after extraction (needed
        # there so bidi-mirroring detection can recognize real words at
        # all); normalizing again here is a harmless no-op in that case,
        # and a safety net for any extraction path that doesn't.
        text = unicodedata.normalize("NFKC", text)
        text = _CONTROL_CHARS.sub("", text)

        if self.options.strip_page_number_lines:
            text = "\n".join(
                line for line in text.splitlines() if not _PAGE_NUMBER_LINE.match(line)
            )

        if self.options.remove_diacritics:
            text = _ARABIC_DIACRITICS.sub("", text)
            text = _TATWEEL.sub("", text)

        if self.options.normalize_alef_yaa_waw:
            text = self._normalize_letters(text)

        if self.options.normalize_digits:
            text = text.translate(_ARABIC_INDIC_DIGITS).translate(_PERSIAN_DIGITS)

        # Fix common hyphenation/line-break artifacts from PDF extraction:
        # a Latin/Arabic word broken across lines with a trailing hyphen.
        text = re.sub(r"-\n(?=\w)", "", text)
        # Join lines that were hard-wrapped mid-sentence (no punctuation before \n)
        text = re.sub(r"(?<![\.\!\?\:\؛\،])\n(?=[^\n])", " ", text)

        if self.options.collapse_whitespace:
            text = _MULTI_SPACE.sub(" ", text)
            text = _MULTI_NEWLINE.sub("\n\n", text)
            text = "\n".join(line.strip() for line in text.splitlines())
            text = text.strip()

        return text

    def _normalize_letters(self, text: str) -> str:
        chars = []
        for ch in text:
            if ch == "ة" and not self.options.normalize_taa_marbuta:
                chars.append(ch)
                continue
            chars.append(_NORMALIZATION_MAP.get(ch, ch))
        return "".join(chars)

    @staticmethod
    def is_mostly_arabic(text: str, threshold: float = 0.3) -> bool:
        """Heuristic used to detect whether a chunk/page has meaningful Arabic
        content (helps filter out noise pages like blank scans)."""
        letters = re.findall(f"[{_ARABIC_LETTER}{_LATIN_ALNUM}]", text)
        if not letters:
            return False
        arabic = re.findall(f"[{_ARABIC_LETTER}]", text)
        return (len(arabic) / max(1, len(letters))) >= threshold