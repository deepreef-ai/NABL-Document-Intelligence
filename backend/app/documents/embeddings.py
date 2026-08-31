"""Lazy-loaded sentence-embedding model for documents/retrieval.py.

A small, CPU-friendly model on purpose: this process runs on the same
machine as Ollama's model, and RAM here is genuinely tight. all-MiniLM-L6-v2
is ~90MB and produces 384-dim embeddings — plenty to tell "this page is
about equipment" from "this page is about staff" without competing
meaningfully with Ollama for memory. Deferred import (sentence-transformers
pulls in PyTorch) so it only costs anything on the completed_application_form
path that actually needs it.
"""
from functools import lru_cache

from app.config import get_settings


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _model().encode(texts, convert_to_numpy=True).tolist()
