from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/.env. Anchored to this file rather than
# left relative to the process CWD: scripts/ put backend/ on sys.path but
# never chdir into it, so running them from the repo root silently loaded no
# .env at all and reported "No LLM provider configured" with a valid one set.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or a .env file."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, extra="ignore")

    # LLM — powers the wizard chat, classification, and extraction. Amazon
    # Nova on AWS Bedrock (llm/providers.py's NovaProvider) is the primary
    # provider: no API key setting for it — it authenticates via the ambient
    # AWS credential chain, the same one documents/ocr_client.py's Lambda
    # invoke already relies on. Blank model = provider skipped/not configured.
    # `nova_model` must be a region-prefixed inference-profile ID (see
    # NovaProvider's docstring for why the bare model ID is rejected).
    nova_model: str = ""
    nova_region: str = "us-east-1"
    # 8192 was too small for a big results table and cost us whole
    # documents. MEASURED 2026-09-05: high-protein-paneer.pdf's 272 test
    # rows serialize to ~11,100 output tokens, so the reply was cut off
    # mid-JSON and the document scored zero. A reasoning model also spends
    # THINKING tokens out of this same budget, which is why hp-lab-report
    # truncated at ~3,500 tokens of actual content. The cap is a ceiling,
    # not a reservation - a short document still returns a short reply and
    # costs nothing extra - so it is set well clear of the largest document
    # in the set rather than just above it.
    nova_max_tokens: int = 32768

    # Google Gemini (llm/providers.py's GeminiProvider) — a KEYED fallback so
    # extraction and the accuracy benchmark survive a Bedrock outage (see that
    # class's docstring for the 2026-09 account-wide block that motivated it).
    # Blank key = provider skipped, exactly like a blank nova_model, so leaving
    # this unset keeps behaviour identical to before it existed.
    # COMMA-SEPARATED for multiple keys. Gemini's free-tier quota is
    # per-PROJECT-per-model (MEASURED 2026-09-04:
    # GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20 requests/day),
    # so a key from a second project is a second allowance — each key
    # becomes its own link in the fallback chain (gemini, gemini-2, ...) and
    # the chain moves to the next one when a key is exhausted.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_max_tokens: int = 32768

    # Groq (llm/providers.py's GroqProvider) — third fallback, a separate
    # free-tier allowance again. MUST be a VISION model: the extraction
    # pipeline sends the page image with the OCR text, and Groq's text-only
    # models ignore image parts silently rather than erroring.
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    # NOT raised alongside the other two: Groq's binding limit is a rate
    # limit (MEASURED: 7,000 input and 1,000 output tokens/minute on the
    # free tier, which rejected high-protein-paneer with a 413 on INPUT
    # size alone), not the per-reply cap. A bigger cap cannot buy headroom
    # the tier does not sell.
    groq_max_tokens: int = 8192

    @property
    def gemini_api_key_list(self) -> list[str]:
        return [k.strip() for k in self.gemini_api_key.split(",") if k.strip()]

    @property
    def groq_api_key_list(self) -> list[str]:
        return [k.strip() for k in self.groq_api_key.split(",") if k.strip()]

    # Comma-separated, tried in order, first success wins (llm/chain.py). Put
    # "nova,gemini" here to fall back automatically; "gemini" alone to force it.
    llm_provider_order: str = "nova"
    # 30s was tuned for short chat-style calls and is too short for extraction:
    # one document is a page image plus its OCR text plus a ~1.5k-token system
    # prompt, and a large results table takes far longer than 30s just to
    # generate and transfer. scripts/generate_predictions_and_score.py already
    # overrode this to 180s for exactly that reason; the app needed the same.
    # A timeout also costs more than the one call — llm/chain.py backs the
    # provider off 15s and DOUBLES that per consecutive timeout.
    llm_timeout_seconds: float = 120.0

    # documents/pipeline.py's completed_application_form (whole-form,
    # per-section) extraction uses this order instead of llm_provider_order —
    # kept as its own setting in case a many-section whole-form run ever
    # needs a different provider order than everything else, though today
    # both resolve to the same single provider.
    chunked_extraction_provider_order: str = "nova"

    # deepreef-ocr: invoked directly as a Lambda function via the AWS SDK —
    # the same mechanism the pcsapaiv2 production service uses (it has no
    # public Function URL). Needs AWS credentials in the environment (a
    # profile, role, or access keys) with lambda:InvokeFunction on this
    # function.
    # Which engine reads ENGLISH/Latin pages:
    #   dev   -> documents/local_ocr.py, RapidOCR in-process. No cloud call,
    #            no rate limit, no AWS credentials needed.
    #   prod  -> the deepreef-ocr Lambda below, same engine every other
    #            script already uses.
    # Non-Latin scripts ignore this and always use the Lambda: RapidOCR's
    # bundled English model garbles Devanagari digits and table layout badly
    # enough to misattribute whole rows, so dev must not route them locally.
    app_env: str = "dev"

    # The `script` value sent to the Lambda for English. The function picks its
    # recognition model from this string; we have only ever invoked it with the
    # non-Latin codes, so the Latin one is configurable rather than assumed.
    # If prod English OCR returns "no recognition model for script", this is
    # the setting to change.
    ocr_lambda_english_script: str = "english"

    ocr_lambda_function_name: str = "akash-ocr"
    ocr_lambda_region: str = "ap-south-1"
    ocr_timeout_seconds: float = 30.0

    # English/Latin OCR (documents/local_ocr.py) imports deepreef-ocr's own
    # engine.py/preprocessor.py directly from that sibling repo's checkout —
    # not a copy kept inside this project, so there's exactly one place its
    # code lives. Path is relative to this backend/ directory; override via
    # DEEPREEF_OCR_PATH if that repo is checked out somewhere else on this
    # machine. This only matters for English — Devanagari/Arabic/Tamil/
    # Telugu/Kannada still go through the real deployed Lambda (ocr_client.py).
    deepreef_ocr_path: str = "../../deepreef-ocr/deepreef-ocr"

    # Semantic retrieval for the completed_application_form pipeline (see
    # documents/retrieval.py) — only relevant pages are sent to the LLM per
    # section, instead of every page. Qdrant runs embedded/on-disk (no
    # server/Docker process) to keep this machine's RAM budget for Ollama.
    embedding_model: str = "all-MiniLM-L6-v2"
    qdrant_storage_dir: str = "./qdrant_storage"
    retrieval_top_k: int = 4

    # DPI for the page images documents/unified_extraction.py sends alongside
    # each page's text. MEASURED 2026-09-03: at 150 DPI one page of this
    # dataset renders to ~762 KB, so the 17-page document alone is ~12.7 MB
    # of images before base64 inflation — close enough to Bedrock's request
    # ceiling to be the thing that breaks first on a longer document. 110
    # roughly halves that and keeps form text legible.
    page_image_dpi: int = 110
    # Pages read from a SCANNED PDF (documents/pipeline.py). Unlike the
    # born-digital path — where PyMuPDF text is free and all 40 pages are read
    # then chunked by character count — every scanned page costs a rasterize,
    # an OCR pass and its own extraction call, so this is a real budget lever.
    # It was hardcoded at 5, which silently truncated any longer report while
    # still returning a confident-looking result: a 12-page scan reported 5
    # pages' worth of fields with no indication the rest existed. Raised well
    # clear of a typical report; lower it if quota is tighter than coverage.
    max_scanned_pages: int = 25

    # Per-document LLM call ceiling (documents/call_budget.py). Before this
    # nothing counted calls and nothing capped them: MEASURED on this branch,
    # a 17-page scan cost 68 and a 50-page PDF up to 74, against a free-tier
    # allowance of ~40 calls/DAY — so one document could exhaust a day.
    #
    # The cost driver is chunks and pages, which vary hugely between documents
    # of the same page count, so a page cap cannot express it. This can: the
    # pipeline stops asking for more calls once the ceiling is reached and
    # returns what it already has, with the reason recorded on the document
    # (never raising — stopping early is an operating condition, not a
    # failure).
    max_classification_calls: int = 1
    max_initial_extraction_calls: int = 4
    max_recovery_calls: int = 2
    max_total_llm_calls: int = 6

    # documents/lab_report.py's extract_letterhead — a second, focused call for
    # the masthead/footer block. MEASURED 2026-09-04 on a 48-document run: ~180
    # of 363 never-extracted fields were exactly these (lab_phone, lab_email,
    # laboratory_accreditation_no, cin, udyam_no, footer, ...), because one
    # call asked to transcribe a 60-row results table loses the masthead. It
    # DOUBLES the calls per document though, so turn it off against a tight
    # free-tier quota.
    lab_report_letterhead_pass: bool = True

    # Retry ONE chunk per document whose extraction returned almost nothing
    # despite the chunk carrying real text. MEASURED: three identical uploads
    # of one scanned report returned 56, 1 and 60 values — about a third of
    # attempts collapse silently, with no error to detect them by. Charged to
    # the RECOVERY budget, capped at one retry so the letterhead pass keeps
    # its call.
    extraction_empty_retry: bool = True

    # Whether the letterhead pass also receives a raster of page one. A lab's
    # masthead is often a GRAPHIC, so lab_email / lab_address / cin are absent
    # from a born-digital PDF's text layer entirely (MEASURED: 5 of 9 golden
    # letterhead fields on one report were not in the 2,679 characters being
    # sent). Turning this off is also how the benchmark isolates the effect of
    # the image from model-to-model variance — same code, one flag.
    letterhead_vision: bool = True

    # documents/classifier.py's classify_locally_scored() answers with a
    # confidence; at or above this it is trusted and NO classification LLM
    # call is made at all. Below it, the real LLM classifier is asked. 0.70
    # means "several independent signals agree" — see that function's
    # docstring; the score is a bounded sum of signals, not a probability.
    local_classification_min_confidence: float = 0.70

    database_url: str = "sqlite:///./nabl.db"
    storage_dir: str = "./storage"

    # Fields at or above this score are shown as confirmed; below it they are
    # flagged yellow/red for mandatory human review, per the HITL spec.
    confidence_threshold: float = 0.85

    # Both spellings of the dev host. A browser treats http://localhost:5173
    # and http://127.0.0.1:5173 as different origins, so allowing only one
    # makes the UI silently fail to reach the API depending on how the dev
    # server happened to be started.
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


    @field_validator("app_env")
    @classmethod
    def _known_env(cls, value: str) -> str:
        """Fail at startup, not at the first scanned page. A typo like
        APP_ENV=production would otherwise read as "not prod" and quietly send
        production traffic to the local engine."""
        normalized = value.strip().lower()
        if normalized not in {"dev", "prod"}:
            raise ValueError(f"app_env must be 'dev' or 'prod', got {value!r}")
        return normalized

    @field_validator("storage_dir", "qdrant_storage_dir")
    @classmethod
    def _anchor_dir(cls, value: str) -> str:
        """Resolve a relative directory against backend/, not the process CWD.

        These defaults are written as "./storage" and "./qdrant_storage", which
        silently meant a DIFFERENT directory depending on where you happened to
        launch from: running a script from backend/scripts/ created a second,
        empty storage tree there while the server used backend/'s."""
        path = Path(value)
        return str(path if path.is_absolute() else (_BACKEND_DIR / path).resolve())

    @field_validator("database_url")
    @classmethod
    def _anchor_sqlite(cls, value: str) -> str:
        """Same for a RELATIVE sqlite path.

        MEASURED: running scripts/benchmark_api.py from backend/scripts/ pointed
        at backend/scripts/nabl.db — a brand new, empty file — and every query
        failed with "no such table: applications", while the server was happily
        using backend/nabl.db. An absolute URL (a test's tmp_path, a real
        server) is left exactly as given.
        """
        prefix = "sqlite:///"
        if not value.startswith(prefix):
            return value
        raw = value[len(prefix):]
        if not raw or raw.startswith("/") or Path(raw).is_absolute():
            return value  # in-memory, or already absolute
        return prefix + str((_BACKEND_DIR / raw).resolve()).replace("\\", "/")


@lru_cache
def get_settings() -> Settings:
    return Settings()
