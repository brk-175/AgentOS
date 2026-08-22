"""Tests for the embeddings client factory and text chunking."""

from __future__ import annotations

import pytest

from agentos.core.config import get_settings
from agentos.services.embeddings import chunk_text, create_embeddings_client


def test_client_is_openrouter_bound() -> None:
    settings = get_settings()
    client = create_embeddings_client()
    assert client.model == settings.openai_embeddings_model
    assert client.dimensions == settings.embeddings_dimensions
    assert "openai.com" in client.openai_api_base


def test_chunk_empty_and_tiny_texts() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text("hello") == ["hello"]


def test_chunk_splits_long_text() -> None:
    pieces = chunk_text("a" * 4000, chunk_size=1500)
    assert len(pieces) == 3
    assert all(len(piece) <= 1500 for piece in pieces)


def test_chunk_overlap_keeps_context() -> None:
    text = "x" * 1400 + "|MARKER|" + "y" * 1000
    pieces = chunk_text(text, chunk_size=1500, overlap=150)
    assert len(pieces) == 2
    assert "|MARKER|" in pieces[0] and "|MARKER|" in pieces[1]


def test_chunk_rejects_too_large_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)
