from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — powers the wizard chat, classification, and extraction. Tried in
    # `llm_provider_order` order, falling through to the next on failure (rate
    # limit, outage, etc.) — set an API key for any subset of these; only
    # Gemini currently gets used for the image-input (vision) calls.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    huggingface_api_key: str = ""
    huggingface_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # Cerebras and OpenRouter are both OpenAI-chat-completions-compatible —
    # see llm/providers.py's OpenAiCompatibleProvider. Cerebras runs Qwen/Llama
    # on its own inference hardware, which is dramatically faster than either
    # a CPU-only local Ollama or Groq/Gemini's shared free-tier queues; kept
    # optional (empty key = provider skipped) like every other link here.
    cerebras_api_key: str = ""
    cerebras_model: str = "gpt-oss-120b"
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-4-31b-it:free"
    # Amazon Nova on AWS Bedrock (llm/providers.py's NovaProvider) — no API
    # key setting: authenticates via the ambient AWS credential chain, the
    # same one documents/ocr_client.py's Lambda invoke already relies on.
    # Blank model = provider skipped, same convention as every key above.
    # `nova_model` must be a region-prefixed inference-profile ID (see
    # NovaProvider's docstring for why the bare model ID is rejected).
    nova_model: str = ""
    nova_region: str = "us-east-1"
    nova_max_tokens: int = 8192

    # Extra free-tier OpenAI-chat-completions-compatible providers — same
    # OpenAiCompatibleProvider class, just another base URL/model. Each is
    # independently optional (empty key = skipped, same as every provider
    # above); adding one only widens the fallback chain, it doesn't replace
    # anything. Model names/free-tier terms for all of these change often —
    # verify against each provider's own docs before relying on a default here.
    sambanova_api_key: str = ""
    sambanova_model: str = "Meta-Llama-3.3-70B-Instruct"
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"
    nvidia_api_key: str = ""
    nvidia_model: str = "meta/llama-3.1-8b-instruct"
    # GitHub Models (https://docs.github.com/en/github-models) — auth is a
    # plain GitHub personal access token (needs the "models: read" permission
    # scope), not a provider-specific API key.
    github_token: str = ""
    github_model: str = "gpt-4o-mini"

    # Key rotation: a second/third key for a provider you're already using is
    # NOT a different provider — it's the same account family hitting a
    # different rate-limit bucket. Each configured key becomes its own named
    # entry in llm_provider_order (e.g. "gemini_2"), so llm/chain.py's
    # per-provider cooldown tracks it independently — a 429 on key 1 doesn't
    # cool down key 2, since they're genuinely separate quotas. Leave unset
    # (empty) to skip; only makes sense once the base key above is also set.
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    groq_api_key_2: str = ""

    llm_provider_order: str = "gemini,groq,huggingface,ollama"
    llm_timeout_seconds: float = 30.0

    # Local model via Ollama — no API key, no rate limit, no hard payload-size
    # cap (unlike Groq/HF's gateways), which is exactly what a 15+ page
    # document extraction needs. Bounded instead by this machine's own RAM and
    # the model's context window (see llm/providers.py's OllamaProvider for
    # why `num_ctx` is set explicitly). No install required for the rest of
    # the app to run — a missing/unreachable Ollama just fails over to the
    # next provider in whichever order below includes it.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    # Local CPU inference is much slower than a cloud API round-trip, and if
    # nothing's currently resident in RAM the model has to cold-load from disk
    # first — measured ~45s just for that on this machine before a single
    # token of a trivial prompt comes back, before a real (longer) extraction
    # prompt even starts generating.
    #
    # MEASURED 2026-08-30 on a 12th-gen i5 (qwen2.5:3b, Q4): a 24-field whole-
    # form section (organisation.*) took 297s end to end (cold load + decode)
    # asking the model to spell out a null entry for every absent field —
    # right at the old 300s ceiling, which is what produced real timeouts.
    # Instructing it to omit absent fields instead (see extractor.py's
    # _FULL_FORM_SYSTEM / extract_section_fields) cut the same section to
    # 115s warm. 210s balances "long enough for a cold-started, moderately
    # full section to actually finish" against "short enough that a
    # genuinely stuck/overloaded call falls through to the next provider in
    # a bounded time" — a densely-populated worst-case section can still
    # legitimately exceed this and fall back to Gemini/etc., which is the
    # intended behavior (Ollama is the fast-and-free primary, not the only
    # option), not a bug.
    ollama_timeout_seconds: float = 210.0
    # llama.cpp/Ollama sometimes under-uses the efficiency cores on a hybrid
    # Intel chip when left to its own default thread count; 0 here means "ask
    # the OS how many logical processors exist and use all of them" (see
    # llm/providers.py's OllamaProvider) rather than hardcoding a number that
    # wouldn't fit every machine this runs on.
    ollama_num_thread: int = 0
    # A generous ceiling, not a normal-case constraint: comfortably above
    # what even a fully-populated 24-field section needs (measured well
    # under 1024 tokens), so it only ever bites as a safety net against a
    # runaway/repeating generation rather than truncating a real answer.
    ollama_num_predict: int = 1024

    # documents/pipeline.py's completed_application_form (whole-form,
    # per-section) extraction uses this order instead of llm_provider_order.
    # Ollama is PRIMARY here — local, free, no rate limit, no per-request
    # payload cap, which is exactly what a many-section whole-form run wants
    # — with every cloud provider kept as fallback for whenever it times out
    # or is unreachable. llm/chain.py's per-provider cooldown means one
    # timeout backs Ollama off (briefly, growing if it keeps happening) for
    # THAT and the next few sections rather than disabling it outright, so
    # it's retried on a later section instead of being benched for the rest
    # of the run — see OllamaProvider's timeout/thread/num_predict tuning
    # above for what makes it fast enough to actually carry this role.
    chunked_extraction_provider_order: str = "ollama,gemini,cerebras,openrouter,groq,huggingface"

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

    # DPI for the page images documents/unified_extraction.py sends alongside
    # each page's text. MEASURED 2026-09-03: at 150 DPI one page of this
    # dataset renders to ~762 KB, so the 17-page document alone is ~12.7 MB
    # of images before base64 inflation — close enough to Bedrock's request
    # ceiling to be the thing that breaks first on a longer document. 110
    # roughly halves that and keeps form text legible.
    page_image_dpi: int = 110

    # --- LLM call budget (documents/call_budget.py) -----------------------
    # Every Nova call is metered against these. MEASURED 2026-09-03 before
    # this existed: a 1-page certificate cost 3 calls (classify + schema +
    # unified) and a long completed_application_form cost 11-21 (one call per
    # schema section plus one retry per section plus the unified pass), with
    # nothing counting or capping them. The pipeline stops retrying once
    # max_total_llm_calls is reached rather than looping.
    max_classification_calls: int = 1
    max_initial_extraction_calls: int = 4
    max_recovery_calls: int = 2
    max_total_llm_calls: int = 6

    # Local (LLM-free) classification is trusted at or above this confidence;
    # below it, and only if the budget allows, Nova is asked instead.
    local_classification_min_confidence: float = 0.70

    # Dense-retrieval similarity below this is treated as "this page is not
    # evidence for that field", instead of always taking the top K whatever
    # their scores (which is what retrieval.py did before).
    retrieval_min_score: float = 0.25

    # Chunk budget for adaptive grouping (documents/adaptive_chunking.py).
    # Chars, not tokens: everything else in this codebase measures text in
    # chars, and ~4 chars/token is close enough to keep one estimate rather
    # than adding a tokenizer dependency.
    max_chunk_chars: int = 12000

    # OCR below this confidence marks a page as *possibly* needing visual
    # verification. It does NOT by itself trigger a Nova vision call — see
    # documents/recovery.py for the escalation policy.
    ocr_low_confidence_threshold: float = 0.75

    # Hard cap on page images per extraction call. Every provider except
    # Nova rejects more than one image per request, so a low cap keeps the
    # fallback chain usable instead of making Nova a hard dependency.
    max_images_per_call: int = 2

    # Rollback switch: False restores the pre-optimisation extraction path.
    optimized_extraction_enabled: bool = True

    database_url: str = "sqlite:///./nabl.db"
    storage_dir: str = "./storage"

    # Fields at or above this score are shown as confirmed; below it they are
    # flagged yellow/red for mandatory human review, per the HITL spec.
    confidence_threshold: float = 0.85

    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
