"""Embedded/on-disk Qdrant — no server or Docker process. One collection per
document, so re-extracting or uploading a second document never mixes its
chunks into another document's retrieval results.

A single shared client (guarded by a lock) rather than one per call: Qdrant's
local mode holds a file lock on its storage directory, and the upload
endpoint's blocking work already runs in a thread pool (see
routers/documents.py), so concurrent requests are a real possibility, not a
theoretical one.
"""
import threading
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import get_settings

# all-MiniLM-L6-v2's output size (see embeddings.py) — fixed for this app,
# not derived at runtime, so collection creation never needs to embed first.
EMBEDDING_DIM = 384

_lock = threading.Lock()
_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        path = Path(get_settings().qdrant_storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(path))
    return _client


def _collection_name(document_id: str) -> str:
    return f"doc_{document_id}"


def index_chunks(document_id: str, chunk_ids: list[int], texts: list[str], vectors: list[list[float]]) -> None:
    with _lock:
        client = _get_client()
        name = _collection_name(document_id)
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(name, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        client.upsert(
            name,
            points=[
                PointStruct(id=chunk_id, vector=vector, payload={"chunk_id": chunk_id, "text": text})
                for chunk_id, text, vector in zip(chunk_ids, texts, vectors)
            ],
        )


def query_top_k(document_id: str, query_vector: list[float], top_k: int) -> list[int]:
    """Returns matching chunk_ids (== page numbers, see retrieval.py), best match first."""
    with _lock:
        client = _get_client()
        name = _collection_name(document_id)
        if not client.collection_exists(name):
            return []
        result = client.query_points(name, query=query_vector, limit=top_k)
        return [point.payload["chunk_id"] for point in result.points]


def drop_collection(document_id: str) -> None:
    with _lock:
        client = _get_client()
        name = _collection_name(document_id)
        if client.collection_exists(name):
            client.delete_collection(name)
