from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — powers the wizard chat, classification, and extraction. Amazon
    # Nova on AWS Bedrock (llm/providers.py's NovaProvider) is the primary
    # provider: no API key setting for it — it authenticates via the ambient
    # AWS credential chain, the same one documents/ocr_client.py's Lambda
    # invoke already relies on. Blank model = provider skipped/not configured.
    # `nova_model` must be a region-prefixed inference-profile ID (see
    # NovaProvider's docstring for why the bare model ID is rejected).
    nova_model: str = ""
    nova_region: str = "us-east-1"
    nova_max_tokens: int = 8192

    # Google Gemini (llm/providers.py's GeminiProvider) — a KEYED fallback so
    # extraction and the accuracy benchmark survive a Bedrock outage (see that
    # class's docstring for the 2026-09 account-wide block that motivated it).
    # Blank key = provider skipped, exactly like a blank nova_model, so leaving
    # this unset keeps behaviour identical to before it existed.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_max_tokens: int = 8192

    # Comma-separated, tried in order, first success wins (llm/chain.py). Put
    # "nova,gemini" here to fall back automatically; "gemini" alone to force it.
    llm_provider_order: str = "nova"
    llm_timeout_seconds: float = 30.0

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

    database_url: str = "sqlite:///./nabl.db"
    storage_dir: str = "./storage"

    # Fields at or above this score are shown as confirmed; below it they are
    # flagged yellow/red for mandatory human review, per the HITL spec.
    confidence_threshold: float = 0.85

    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
