"""
Centralized JSONL logging for the whole pipeline.

Every pipeline step (routing, retrieval, graph expansion, reranking,
generation, evaluation, errors, ...) is written as a single JSON line to a
shared log file so the whole run can be reconstructed and audited later.
A human-readable console logger runs alongside it.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings


class JsonlLogger:
    """Thread-safe append-only JSONL event logger."""

    _lock = threading.Lock()

    def __init__(self, log_path: Optional[Path] = None, console: Optional[bool] = None) -> None:
        self.log_path = log_path or settings.logging.log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.console = settings.logging.console if console is None else console

        self._console_logger = logging.getLogger("arabic_graph_rag")
        if not self._console_logger.handlers:
            self._console_logger.setLevel(settings.logging.level.upper())
            stream = sys.stdout
            if hasattr(stream, "buffer"):
                stream = io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace")
            handler = logging.StreamHandler(stream)
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self._console_logger.addHandler(handler)

    def log(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        """Write one structured event both to JSONL and (optionally) console."""
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **fields,
        }
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if self.console:
            level = logging.ERROR if event_type == "error" else logging.INFO
            self._console_logger.log(level, "[%s] %s", event_type, _shorten(fields))
        return record


def _shorten(fields: Dict[str, Any], max_len: int = 240) -> str:
    text = json.dumps(fields, ensure_ascii=False, default=str)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


# Single shared instance used across the whole project.
pipeline_logger = JsonlLogger()


class RunTimer:
    """Small context manager used to measure step latency in milliseconds."""

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "RunTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
