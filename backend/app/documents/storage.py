import uuid
from pathlib import Path

from app.config import get_settings


class LocalFileStorage:
    """Filesystem-backed storage with an S3-shaped interface (key in, key out)
    so an S3-backed implementation can be swapped in later without touching
    the pipeline code that calls it."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or get_settings().storage_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, filename: str) -> str:
        ext = Path(filename).suffix
        key = f"{uuid.uuid4().hex}{ext}"
        (self.base_dir / key).write_bytes(data)
        return key

    def get(self, key: str) -> bytes:
        return (self.base_dir / key).read_bytes()

    def path(self, key: str) -> Path:
        return self.base_dir / key

    def delete(self, key: str) -> None:
        """Remove a stored object. Missing is not an error.

        A caller deleting a document has already decided the document is
        going; failing here because the bytes were swept up by something else
        would leave a row nobody can remove.
        """
        (self.base_dir / key).unlink(missing_ok=True)
