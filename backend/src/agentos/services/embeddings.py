"""Embeddings: OpenRouter-bound client factory + text chunking for RAG.

OpenRouter serves OpenAI-compatible embeddings behind ``base_url``, so a
``langchain_openai.OpenAIEmbeddings`` bound to it works out of the box (the
same trick as ``create_agent_llm``).
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from agentos.core.config import get_settings

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 150


def create_embeddings_client() -> OpenAIEmbeddings:
    """OpenRouter-bound embeddings model (dimensions come from settings)."""
    settings = get_settings()
    return OpenAIEmbeddings(
        base_url=settings.openai_base_url,
        api_key=SecretStr(settings.openai_api_key),
        model=settings.openai_embeddings_model,
        dimensions=settings.embeddings_dimensions,
    )


def chunk_text(
    text: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """Split ``text`` into overlapping chunks by character count.

    Chunks are trimmed of leading/trailing whitespace; empty text yields no
    chunks. ``overlap`` must stay below ``chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text.strip():
        return []
    step = chunk_size - overlap
    return [text[index : index + chunk_size].strip() for index in range(0, len(text), step)]
