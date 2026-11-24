"""Client for the remote OCR service (ocr_service/app.py).

WHY (2026-11-24): OCR is the CPU-heavy step; running it on a stronger host
that keeps the models loaded frees the pipeline machine. Not wired into the
pipeline yet — callers opt in by passing the service's base_url.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor

import httpx

from app.documents.deepreef_local import PageOcr


def extract_page_remote(
    image_bytes: bytes,
    *,
    rendered_pdf_page: bool,
    base_url: str,
    timeout: float = 60.0,
) -> PageOcr:
    """OCR one page on the remote service; same result type as extract_page."""
    url = base_url.rstrip("/") + "/ocr/page"
    payload = {
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "rendered_pdf_page": rendered_pdf_page,
    }
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"OCR service timed out after {timeout}s ({url})") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"OCR service unreachable ({url}): {type(exc).__name__}") from exc
    if resp.status_code != 200:
        # WHY: the server's detail is a short error name, never page text, so
        # it is safe to surface in the exception message.
        detail = resp.text[:200]
        raise RuntimeError(f"OCR service returned HTTP {resp.status_code}: {detail}")
    data = resp.json()
    return PageOcr(
        text=data["text"],
        lines=list(data.get("lines") or []),
        confidence=data.get("confidence"),
        model_used=data.get("model_used", ""),
        region_count=int(data.get("region_count") or 0),
    )


def extract_pages_remote(
    images: dict[int, bytes],
    *,
    base_url: str,
    workers: int = 8,
    rendered_pdf_page: bool = True,
    timeout: float = 60.0,
) -> dict[int, PageOcr | Exception]:
    """OCR many pages in parallel; a failed page maps to its exception.

    WHY threads: each call just waits on the network, so threads (not
    processes) are enough to keep the server's workers busy. Per-page errors
    are returned rather than raised so one bad page does not lose the rest.
    """
    if not images:
        return {}

    def one(page_no: int) -> PageOcr | Exception:
        try:
            return extract_page_remote(
                images[page_no],
                rendered_pdf_page=rendered_pdf_page,
                base_url=base_url,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            return exc

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(images)))) as pool:
        results = list(pool.map(one, images))
    return dict(zip(images, results))
