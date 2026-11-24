"""Configuration for the agentic extraction graph.

Separate from app/config.py on purpose: the legacy pipeline's settings encode
a per-document call *ceiling* tuned for short certificates, and this graph
deliberately does not work that way — it scales calls with document size and
bounds cost with a token budget plus a hard wall-clock timeout instead. Mixing
the two vocabularies in one settings object made it very easy to read the
wrong limit.

Every value is overridable by environment variable so a deployment can tune
without a code change. See deploy/.env.example for the full list.
"""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, ".env")


class GraphSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_env_path(), extra="ignore", env_prefix="")

    # ---- LLM providers ---------------------------------------------------
    # Nova is primary per spec. The chain exists because a single provider is
    # a single point of failure for the whole workflow, and this graph makes
    # far more calls than the legacy pipeline did.
    # Ordered fallback. Nova is primary per spec; Gemini then Groq behind it,
    # because this workflow makes one call per chunk and a single provider is a
    # single point of failure for a whole document. MEASURED: a Gemini daily
    # quota exhaustion took out two real uploads that Groq would have served.
    #
    # Groq is text-only (its provider has no `images` parameter), which is fine
    # here — the graph's extraction calls are text-only. An image-bearing call
    # simply skips it and falls to whichever link does accept images.
    graph_provider_order: str = "nova,gemini,groq"
    nova_model: str = "us.amazon.nova-2-lite-v1:0"
    nova_region: str = "us-east-1"
    nova_max_tokens: int = 8192
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_max_tokens: int = 8192
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    groq_max_tokens: int = 8192
    llm_timeout_seconds: float = 180.0
    llm_temperature: float = 0.0

    # ---- retry / rate limiting -------------------------------------------
    max_attempts_per_chunk: int = 3
    max_extraction_rounds: int = 3
    max_reanalysis_rounds: int = 2
    backoff_base_seconds: float = 2.0
    backoff_max_seconds: float = 60.0
    #: How long a call will sit and wait for a provider cooldown to clear.
    #: Separate from backoff_max_seconds, which bounds the delay between
    #: this layer's own attempts. Real provider cooldowns after a 429 run
    #: well past a minute (MEASURED: 78s and 158s), so a 60s ceiling meant
    #: giving up on a chunk that would have succeeded a moment later.
    #:
    #: MUST stay >= llm.chain._BACKOFF_CAP_SECONDS (300s), which is the
    #: largest cooldown the chain can impose on itself. At 240s there was a
    #: dead zone: a provider backed off to 287s/297s (MEASURED, graph run
    #: 137) — under the 300s cap it sets, but over the 240s this layer would
    #: wait — so the call gave up instantly instead of waiting out a cooldown
    #: that was about to clear. Both chunks of a 17-page PDF were abandoned
    #: that way, with all 17 pages read: the data loss was here, not in page
    #: coverage. Waiting is also the only safe response to throttling, since
    #: the alternative — more concurrency — makes the throttling worse.
    max_cooldown_wait_seconds: float = 310.0
    max_requests_per_minute: int = 60
    max_concurrent_llm_calls: int = 4

    # ---- chunking --------------------------------------------------------
    # Chars, not tokens: PyMuPDF gives us characters and the ratio is stable
    # enough (~4 chars/token) that converting adds precision we do not have.
    max_chunk_chars: int = 24000
    chunk_overlap_pages: int = 1
    min_page_text_chars: int = 20
    #: Hard cap on the JSON a single extraction call may be asked to produce.
    #: The binding limit in practice is OUTPUT tokens, not input context — a
    #: chunk small enough to read is not necessarily small enough to answer.
    max_fields_per_chunk: int = 120
    #: Model "thinking" for the extraction node only: minimal/low/medium/high,
    #: blank = provider default (unchanged behaviour). Honoured by Gemini,
    #: ignored by other providers. MEASURED 2026-11-23 on gemini-3.6-flash:
    #: "minimal" removed ~1k thinking tokens per call (7.8s -> 5.6s on a
    #: 14-row report) with the same 14/14 rows found. Validate on the gold
    #: set before making it the default — see llm/thinking.py.
    extraction_thinking_level: str = ""
    #: Which pipeline shape a run uses when the caller does not say.
    #:   "full"      — every node as before (NABL application-form filling).
    #:   "test_rows" — for lab-report readers such as NABL e-Attestation, which
    #:                 need the results table, not the NABL 151 form. Skips the
    #:                 four LLM calls that only serve form filling: structure
    #:                 analysis, semantic field merging, conflict adjudication
    #:                 (conflicts are escalated to a person instead) and form
    #:                 mapping. Extraction and the deterministic evidence checks
    #:                 are unchanged. MEASURED 2026-11-23, hp-lab-report.pdf:
    #:                 those four calls were ~70s of a 146s run.
    extraction_profile: str = "full"
    #: test_rows profile only: numbered lines in, positional arrays out, with
    #: page/evidence derived in code and every row verified against its line
    #: (graph/nodes/compact_rows.py). False falls back to the general
    #: extraction prompt — kept for A/B accuracy comparison.
    compact_test_rows: bool = True

    # ---- OCR -------------------------------------------------------------
    ocr_enabled: bool = True
    #: Engine for English scanned pages:
    #:   "rapidocr"       — RapidOCR's bundled model, one page at a time (previous behaviour)
    #:   "deepreef_local" — the team's deepreef-ocr engine in-process, English
    #:                      model, pages read in parallel (documents/deepreef_local.py)
    #: Indic/Arabic scripts are unaffected: they keep the deepreef-ocr Lambda.
    ocr_engine: str = "rapidocr"
    #: Where deepreef-ocr's model files live; blank = <DEEPREEF_OCR_PATH>/deepreef_ocr/models.
    deepreef_model_dir: str = ""
    #: deepreef_local only: pages OCR'd at the same time, and ONNX threads per
    #: page. Keep workers x threads near the core count (M1: 4 x 2 = 8).
    ocr_page_workers: int = 4
    #: OCR_ENGINE=deepreef_service: base URL of the OCR worker service
    #: (backend/ocr_service/app.py, run on a host with more cores/RAM).
    ocr_service_url: str = "http://127.0.0.1:8100"
    ocr_threads_per_page: int = 2
    #: deepreef_local only: "thread" (default) or "process" for the page pool.
    #: WHY: threads gave no speed-up because the Python pre/post-processing
    #: around each box holds the GIL. "process" gives real parallelism on hosts
    #: with spare cores and RAM, but each process loads its own models
    #: (~200-300 MB); MEASURED no gain on the 8 GB M1 laptop, which swaps.
    #: Any process-pool failure falls back to sequential OCR.
    ocr_worker_mode: str = "thread"
    #: deepreef_local only. Render PDF pages straight at the OCR engine's size
    #: (longest side 1536 px) instead of at ocr_dpi and then shrinking.
    ocr_render_to_target: bool = True
    #: deepreef_local only. Skip the photo-oriented clean-up (illumination,
    #: contrast, sharpen, JPEG re-compression) for rendered PDF pages; camera
    #: images keep the full set. Both switches are accuracy-gated by
    #: scripts/ocr_benchmark.py — turn off if a document set reads worse.
    ocr_clean_scan_profile: bool = True
    #: deepreef_local only. Reuse the reading of a page whose rendered pixels,
    #: model file and OCR settings are identical (documents/ocr_page_cache.py).
    #: MEASURED 2026-11-24: OCR is ~4.3 s per scanned page, 111 s for a 29-page
    #: scan, and re-uploading the same report re-OCR'd every page.
    ocr_page_cache_enabled: bool = True
    #: Blank = `ocr_cache` next to graph_sqlite_path (graph_storage/ocr_cache).
    ocr_page_cache_dir: str = ""
    #: The cache holds OCR text of patient reports, so it must not live forever:
    #: older entries are misses and are deleted. Default 7 days.
    ocr_page_cache_ttl_hours: float = 168.0

    # --- Docling: document structure only (app/graph/docling_adapter.py) ---
    #: Off by default. Docling is an optional dependency pulling 45 packages
    #: including torch, so the flag must be the thing that turns it on — an
    #: install alone should not change how documents are processed.
    #:
    #: When off, nothing about the pipeline changes and Docling is never
    #: imported. When on, RapidOCR still does every bit of OCR; Docling only
    #: supplies layout, reading order and table structure.
    use_docling: bool = False
    #: A structure pass runs models over every page and can hang. The pipeline
    #: gives up at this point and keeps the RapidOCR + PyMuPDF result.
    #:
    #: Two env names on purpose. The field is `docling_timeout_seconds`, so
    #: pydantic-settings would accept only DOCLING_TIMEOUT_SECONDS — and
    #: DOCLING_TIMEOUT, the name the deployment docs and operators use, was
    #: silently ignored: MEASURED, setting it left the timeout at 120. A knob
    #: that looks set and does nothing is worse than one that does not exist.
    docling_timeout_seconds: float = Field(
        default=120.0,
        validation_alias=AliasChoices("DOCLING_TIMEOUT", "DOCLING_TIMEOUT_SECONDS",
                                      "docling_timeout_seconds"),
    )
    #: Skip the structure pass on documents longer than this, where the cost
    #: is worst and the timeout most likely. 0 disables the limit.
    docling_max_pages: int = 50
    ocr_dpi: int = 200
    ocr_low_confidence_threshold: float = 0.60
    #: OCR a page that ALREADY has a good text layer, when part of it is a
    #: raster image. MEASURED on a food-testing report: the whole letterhead —
    #: lab name, address, phone, email, website — is one image, so the text
    #: layer is rich and complete-looking and the eight fields a reader sees
    #: first are simply absent. lab_info appears in 50 of the 53 labelled
    #: records, which makes a systematically empty lab_info the single biggest
    #: gap in the output.
    ocr_augments_text_layer: bool = True
    #: How much of a page must be image before it is worth the extra OCR pass.
    #: A signature scribble or a small logo is not worth a second read; a
    #: letterhead band across the top is.
    ocr_image_area_threshold: float = 0.03
    #: Augment by OCR-ing only the image REGIONS (letterhead, logo, stamp), not
    #: the whole page — the text layer already has the rest. MEASURED
    #: 2026-11-24: whole-page augmentation was ~10 s/page and 50 of
    #: hp-lab-report.pdf's 60 s. False restores whole-page OCR.
    ocr_augment_regions_only: bool = True

    # ---- thresholds ------------------------------------------------------
    verified_confidence_threshold: float = 0.75
    evidence_similarity_threshold: float = 0.88
    mapping_confidence_threshold: float = 0.70
    low_confidence_ratio_limit: float = 0.30

    # ---- limits ----------------------------------------------------------
    max_file_size_mb: int = 200
    max_pages: int = 2000
    workflow_timeout_seconds: float = 3600.0

    # ---- persistence -----------------------------------------------------
    graph_database_url: str = ""      # blank => sqlite file below
    graph_sqlite_path: str = "graph_storage/graph.db"
    checkpoint_db_path: str = "graph_storage/checkpoints.db"

    # ---- behaviour flags -------------------------------------------------
    #: The graph is the extraction path. This flag exists as a rollback lever
    #: only — see documents/pipeline.py.
    graph_pipeline_enabled: bool = True
    human_review_enabled: bool = True
    #: When true the graph interrupts and waits for a person on any unresolved
    #: conflict. When false it records the item and carries on to a
    #: MANUAL_REVIEW_REQUIRED verdict without blocking.
    interrupt_for_human_review: bool = False

    allowed_extensions: str = ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.docx"

    @property
    def provider_order(self) -> list[str]:
        return [p.strip() for p in self.graph_provider_order.split(",") if p.strip()]

    @property
    def extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_graph_settings() -> GraphSettings:
    return GraphSettings()
