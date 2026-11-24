"""Content-addressed cache of deepreef-local OCR readings, one JSON file per page.

WHY: OCR is the main latency cost of a scanned report. MEASURED 2026-11-24 on
this laptop: ~4.3 s per scanned page, 111 s for a 29-page scanned report — and
re-uploading the same report (testing, resubmissions) re-OCR'd every page. The
engine is deterministic for a given image, model and settings, so a page that
renders to identical pixels can safely reuse the earlier reading.

Key = sha256 of everything that can change the reading:
  - the rendered PNG bytes (exactly what extract_page would be given);
  - the sha256 of the devanagari_rec.onnx file actually in the model dir, so
    swapping in a stand-in or a new pinned model never serves old readings;
  - the settings that change OCR output (render-to-target, clean-scan profile,
    script) and CACHE_VERSION, bumped whenever the stored shape or the OCR
    post-processing changes.

PRIVACY: the values are OCR text of patient lab reports. So entries expire
(ocr_page_cache_ttl_hours, default 7 days) — an expired entry is a miss and is
deleted when seen, and an hourly sweep removes expired files that are never
looked up again. The directory gets a `.gitignore` of `*` so it cannot be
committed by accident. Logs carry hit/miss counts and key prefixes only, never
text.

ROBUSTNESS: every function here swallows its own errors (disk full, corrupt
JSON, permissions) and reports "no cache" — a cache problem must never cost a
page its reading; the caller just runs OCR as if the cache did not exist.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time

log = logging.getLogger(__name__)

#: Bump when the stored fields or how they are produced changes; old entries
#: then simply stop matching (and age out via the TTL).
CACHE_VERSION = 1
_MODEL_FILE = "devanagari_rec.onnx"
_SWEEP_EVERY_S = 3600.0

_lock = threading.Lock()
_stats = {"hit": 0, "miss": 0, "store": 0, "error": 0}
#: {path: ((size, mtime_ns), sha256)} — hashing the ~MB model on every page
#: would cost more than it saves; re-hash only when the file changes.
_model_hash_memo: dict[str, tuple[tuple[int, int], str]] = {}
_last_sweep: dict[str, float] = {}


def _settings():
    from app.graph.config import get_graph_settings

    return get_graph_settings()


def stats() -> dict:
    with _lock:
        return dict(_stats)


def _count(kind: str) -> dict:
    with _lock:
        _stats[kind] += 1
        return dict(_stats)


def cache_dir() -> str | None:
    """The cache directory, or None when caching is off.

    Blank setting = `ocr_cache` next to the graph SQLite db (graph_storage/).
    The test suite switches the cache off globally (tests/conftest.py), so a
    faked engine never writes into, or reads from, the developer's real cache;
    tests that exercise the cache turn it on with a tmp_path directory.
    """
    s = _settings()
    if not getattr(s, "ocr_page_cache_enabled", False):
        return None
    configured = (getattr(s, "ocr_page_cache_dir", "") or "").strip()
    if configured:
        return configured
    return os.path.join(os.path.dirname(s.graph_sqlite_path) or ".", "ocr_cache")


def _model_identity() -> str | None:
    """sha256 of the recogniser file in use; None if it cannot be read (then
    the cache is skipped — extract_page will report the missing model)."""
    from app.documents import deepreef_local

    path = os.path.join(deepreef_local._model_dir(), _MODEL_FILE)
    try:
        st = os.stat(path)
    except OSError:
        return None
    stamp = (st.st_size, st.st_mtime_ns)
    with _lock:
        memo = _model_hash_memo.get(path)
    if memo and memo[0] == stamp:
        return memo[1]
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _lock:
        _model_hash_memo[path] = (stamp, value)
    return value


def page_key(png_bytes: bytes) -> str | None:
    """Cache key for a rendered page, or None when the cache should not be used."""
    try:
        if cache_dir() is None:
            return None
        model = _model_identity()
        if model is None:
            return None
        from app.documents import deepreef_local

        s = _settings()
        h = hashlib.sha256()
        h.update(png_bytes)
        h.update(json.dumps({
            "v": CACHE_VERSION,
            "model": model,
            "script": deepreef_local.SCRIPT,
            "render_to_target": bool(s.ocr_render_to_target),
            "clean_scan_profile": bool(s.ocr_clean_scan_profile),
        }, sort_keys=True).encode())
        return h.hexdigest()
    except Exception as exc:  # noqa: BLE001 — a key failure just means "no cache"
        log.warning("ocr-cache: key failed (%s) — reading without cache", type(exc).__name__)
        return None


def _path(directory: str, key: str) -> str:
    return os.path.join(directory, key[:2], f"{key}.json")


def _ttl_seconds() -> float:
    return max(0.0, float(_settings().ocr_page_cache_ttl_hours)) * 3600.0


def get(key: str):
    """The cached PageOcr for `key`, or None (miss, expired, corrupt, off)."""
    directory = cache_dir()
    if directory is None or not key:
        return None
    path = _path(directory, key)
    try:
        _maybe_sweep(directory)
        try:
            with open(path, encoding="utf-8") as fh:
                entry = json.load(fh)
        except FileNotFoundError:
            counts = _count("miss")
            log.info("ocr-cache: miss key=%s hits=%d misses=%d", key[:12], counts["hit"], counts["miss"])
            return None
        if (not isinstance(entry, dict) or entry.get("v") != CACHE_VERSION
                or time.time() - float(entry["created_at"]) > _ttl_seconds()):
            _remove(path)
            counts = _count("miss")
            log.info("ocr-cache: expired/stale key=%s removed hits=%d misses=%d",
                     key[:12], counts["hit"], counts["miss"])
            return None
        from app.documents.deepreef_local import PageOcr

        page = PageOcr(
            text=str(entry["text"]),
            lines=[str(ln) for ln in entry["lines"]],
            confidence=None if entry["confidence"] is None else float(entry["confidence"]),
            model_used=str(entry["model_used"]),
            ocr_ms=0.0,
            region_count=int(entry["region_count"]),
        )
    except Exception as exc:  # noqa: BLE001 — corrupt/unreadable entry: drop it, OCR again
        _remove(path)
        counts = _count("error")
        log.warning("ocr-cache: unreadable entry key=%s (%s) removed errors=%d",
                    key[:12], type(exc).__name__, counts["error"])
        return None
    counts = _count("hit")
    log.info("ocr-cache: hit key=%s hits=%d misses=%d", key[:12], counts["hit"], counts["miss"])
    return page


def put(key: str, page) -> None:
    """Store a reading. Atomic (temp file + os.replace), so a concurrent reader
    or a crash mid-write never sees half a file. Never raises."""
    directory = cache_dir()
    if directory is None or not key:
        return
    tmp = None
    try:
        _ensure_dir(directory)
        target = _path(directory, key)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        payload = json.dumps({
            "v": CACHE_VERSION,
            "created_at": time.time(),
            "text": page.text,
            "lines": list(page.lines),
            "confidence": page.confidence,
            "model_used": page.model_used,
            "region_count": page.region_count,
        }, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), prefix=".tmp-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, target)
        tmp = None
        _count("store")
    except Exception as exc:  # noqa: BLE001 — a failed store only loses the speed-up
        counts = _count("error")
        log.warning("ocr-cache: store failed key=%s (%s) errors=%d", key[:12], type(exc).__name__,
                    counts["error"])
    finally:
        if tmp:
            _remove(tmp)


def _ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    ignore = os.path.join(directory, ".gitignore")
    if not os.path.exists(ignore):
        with open(ignore, "w", encoding="utf-8") as fh:
            fh.write("# OCR text of patient reports - never commit\n*\n")


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _maybe_sweep(directory: str) -> None:
    """Delete expired entries at most once an hour per directory, so pages that
    are never uploaded again do not keep patient text past the TTL."""
    now = time.time()
    with _lock:
        if now - _last_sweep.get(directory, 0.0) < _SWEEP_EVERY_S:
            return
        _last_sweep[directory] = now
    ttl = _ttl_seconds()
    removed = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            try:
                if now - os.path.getmtime(path) > ttl:
                    os.remove(path)
                    removed += 1
            except OSError:
                continue
    if removed:
        log.info("ocr-cache: sweep removed %d expired entries", removed)
