"""Upload guard: size cap, extension allow-list and a cheap magic-bytes check.

Errors: 413 too large, 415 unsupported type, 400 empty or content that does
not match its extension. Limits come from app.config (MAX_UPLOAD_MB,
ALLOWED_UPLOAD_EXTENSIONS).
"""
import os

from fastapi import HTTPException, UploadFile

from app.config import get_settings

_CHUNK = 1024 * 1024

_MAGIC: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
    ".bmp": (b"BM",),
    ".gif": (b"GIF87a", b"GIF89a"),
    ".webp": (b"RIFF",),
    ".docx": (b"PK\x03\x04",),
}

# Files with no extension are judged by content type, as the pipeline did.
_TYPE_TO_EXT = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def check_extension(filename: str, content_type: str, allowed: set[str]) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if not ext:
        ext = _TYPE_TO_EXT.get((content_type or "").split(";")[0].strip().lower(), "")
    if ext not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported file type {ext or '(none)'!r}; supported: {sorted(allowed)}",
        )
    return ext


def check_magic(ext: str, data: bytes) -> None:
    sigs = _MAGIC.get(ext)
    bad = bool(sigs) and not data.startswith(sigs)
    if ext == ".webp" and data[8:12] != b"WEBP":
        bad = True
    if bad:
        raise HTTPException(
            status_code=400, detail=f"file content does not look like a {ext} file"
        )


async def read_validated_upload(file: UploadFile) -> bytes:
    """Read the upload (bounded) and validate it; raises HTTPException."""
    settings = get_settings()
    allowed = {e.strip().lower() for e in settings.allowed_upload_extensions.split(",") if e.strip()}
    ext = check_extension(file.filename or "", file.content_type or "", allowed)

    limit = settings.max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_CHUNK):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"file exceeds the {settings.max_upload_mb} MB upload limit",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status_code=400, detail="uploaded file is empty")
    check_magic(ext, data)
    return data
