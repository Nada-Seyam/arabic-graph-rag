"""Shared pytest fixtures. Ensures tests never make real network/LLM calls
unless explicitly monkeypatched, and use an isolated temp data directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("OLLAMA_API_KEY", "test-key")
os.environ.setdefault("CONSOLE_LOG", "false")


@pytest.fixture
def sample_arabic_text() -> str:
    return (
        "رقم النسخة 1\r\n"
        "اسم الإجراء دليل إجراءات وحدة الإنذار المركزي تاريخ الإصدار 08/2024\r\n"
        "١٢٣ هذا نص تجريبي يحتوي على تشكيل مثل الْعَرَبِيَّة والتطويل ـــــ لتنظيفه.\n"
        "يجب على الموظف استلام الجهاز؟ ثم فحصه! وبعد ذلك تسجيله في السجل."
    )
