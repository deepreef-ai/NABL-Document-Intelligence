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

from pydantic import Field
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
    max_cooldown_wait_seconds: float = 240.0
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

    # ---- OCR -------------------------------------------------------------
    ocr_enabled: bool = True
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
