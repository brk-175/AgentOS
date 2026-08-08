"""Structured (JSON) logging with secret redaction.

Production-style logging: one JSON object per line, UTC timestamps, and a
``with_context`` helper that attaches request/run ids. Cloud-friendly
(no ANSI, machine-parsable) while staying human-readable via ``pretty``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_SENSITIVE_KEYS = {"token", "secret", "password", "key", "authorization"}


def _redact(key: str, value: Any) -> Any:
    if any(s in key.lower() for s in _SENSITIVE_KEYS) and value is not None:
        return "[REDACTED]"
    return value


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            for key, value in extra.items():
                entry[key] = _redact(key, value)
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger emitting structured JSON to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def log_extra(**fields: Any) -> dict[str, Any]:
    """Attach structured fields to the next log call: ``logger.info("msg", extra=log_extra(run_id=...))``."""
    return {"extra_fields": fields}
