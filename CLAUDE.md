# NABL Document Intelligence — context for Claude

Reads lab reports, certificates and filled NABL application forms; extracts structured
fields with OCR + LLMs; fills the NABL accreditation forms (151, 152, 153/153A, 154, 155,
157, 158, 159) for a human to review. Next goal: be the report reader behind the NABL
e-Attestation PoC (`../../nabl-attestation`). README.md and SETUP.md are the human docs;
this file is the working context. Live git/PR state is injected at session start by a hook
(`../../.claude/hooks/session-context.sh`), so it is not repeated here.

## Keeping this file current (instructions to Claude)

- This file is a living document. When work changes something a future session needs to
  know — a merged PR, a new decision, a fixed or newly found problem, a changed command —
  edit the relevant section before finishing. Prefer editing over appending; delete
  anything that has become false.
- "Current state" holds what is true *now*. Date every entry (YYYY-MM-DD). Move finished
  items out; don't keep a changelog here (git history is the changelog).
- A Stop hook blocks once when code or git history is newer than this file. If nothing
  in here is affected, make the smallest honest update (e.g. bump "Last reviewed").
- Never write secret values, API keys, or patient details into this file.

## Layout

```text
backend/app/
  main.py              FastAPI app, startup (DB init, Docling warm-up, orphan reaper)
  config.py            Settings (pydantic-settings, reads backend/.env)
  graph/               LIVE extraction path: LangGraph workflow (~18 nodes, SQLite checkpoints)
    graph.py           wiring + routing; diagram in its docstring
    nodes/             one module per node (preprocess, chunking, extraction, compact_rows,
                       evidence, conflicts, completeness, mapping…)
    adapter.py         graph output -> legacy PipelineResult (so the UI didn't need a rewrite)
    config.py          graph settings (profile, OCR engine, concurrency, chunking…)
  documents/pipeline.py  LEGACY path; runs only if the graph raises or graph_pipeline_enabled=false
  documents/           OCR: local_ocr (RapidOCR), deepreef_local (deepreef-ocr in-process),
                       ocr_client (deepreef-ocr Lambda, Indic scripts); classifier, extractor,
                       compiler (fields -> form), value_matching, RAG (chunking/embeddings/qdrant)
  llm/                 provider chain (Nova/Bedrock -> Gemini -> Groq), JSON salvage,
                       thinking.py (per-call Gemini thinking level via contextvar)
  schemas/             Pydantic models of the 9 real NABL forms
  benchmark/           scoring vs gold dataset; dataset_normalization/ text repair
  routers/             documents, review, chat, wizard (legacy UI API) + graph_documents
backend/tests/         pytest, LLM/OCR mocked; tests/conftest.py isolates tests from backend/.env
backend/scripts/       benchmark, ground truth, seeding, UI screenshots
frontend/src/          React 19 + Vite + TS. Routes: / and /applications/:id/upload -> UploadPage
                       (per-document timer), /applications/:id/review -> ReviewPage.
scripts/anonymization/ standalone PII redaction tool for lab reports
```

## Commands

```bash
# backend — durable venv is backend/.venv (created 2026-11-24)
cd backend
.venv/bin/uvicorn app.main:app --port 8000      # no --reload: restart after code/.env changes
.venv/bin/python -m pytest tests -q             # expect 623+ passed, 3 known stale failures

# rebuilding the venv: requirements.txt as pinned does NOT resolve (httpx 0.27.2 and
# python-docx 1.1.2 clash with docling). Install from a copy with httpx==0.28.1 and
# python-docx==1.2.0, plus CPU torch and OpenCV (deepreef-ocr needs it):
#   pip install --extra-index-url https://download.pytorch.org/whl/cpu -r <patched copy> \
#       opencv-python-headless onnx

# frontend — must be port 5173 (CORS allowlist)
cd frontend && npm install && npm run dev       # also: npm run build, npm run lint (oxlint)
```

## Conventions and gotchas

- Every change goes on its own branch + PR against `feat/demo22` (not main). Don't merge,
  deploy, or touch AWS (Lambda/Bedrock config) without explicit permission.
- Values must be compared strictly: use `strict_value_key` / `normalise_for_match`
  (graph/nodes/helpers.py) and documents/value_matching.py — never loose substring checks
  (1.5 must not match 0.15; mg/m2 == mg/m²).
- A field the graph marks CONFLICTED/unsupported is surfaced as a warning, never placed in
  a form slot (see adapter.py docstring).
- Confidence threshold 0.85: below goes to human review.
- Runtime data (nabl.db, storage/, graph_storage/*.db, uploads, qdrant_storage/) is not
  source — don't commit it. Some uploads may be real patient data.
- Graph settings are read from backend/.env, and tests used to inherit them. tests/conftest.py
  now pins profile/thinking/compact/OCR-engine defaults; a test that needs another value
  sets it with monkeypatch. Don't rely on .env inside tests.
- Never keep anything you need under /private/tmp: macOS temp cleanup wiped the old venv
  there on 2026-11-24 (every .py file deleted).
- Code comments in this repo are long and explain *why* with MEASURED dates; keep that style
  when touching nearby code.

## Current state

Last reviewed: 2026-11-24

### Branch and environment

- Work is on branch `feat/15-fast-extraction` (from `feat/demo22` @ c950cf0), all
  pushed; PR #14 to main is open for review (2026-11-24; do not merge). Next: the e-Attestation integration plan at ../../INTEGRATION_PLAN.md awaits the user's review and decisions D1-D5 -- no integration code until then. Start the backend with the
  command above.
- LLM: Bedrock/Nova is denied for IAM user `dr-dev` (and so is lambda:InvokeFunction and
  ECR), so the graph uses `GRAPH_PROVIDER_ORDER=gemini,groq` on free tiers (Gemini ~20
  requests/day/key, 3 keys; Groq rate-limits bursts).
- backend/.env (local, untracked; backup of the pre-session file at
  backend/.env.backup.before-parallel) sets: MAX_CONCURRENT_LLM_CALLS=8,
  MAX_REQUESTS_PER_MINUTE=30, EXTRACTION_THINKING_LEVEL=minimal,
  EXTRACTION_PROFILE=test_rows, OCR_ENGINE=deepreef_local, OCR_PAGE_WORKERS=4,
  OCR_THREADS_PER_PAGE=2, DEEPREEF_OCR_PATH pointing at ../../deepreef-ocr.

### What the branch changes (2026-11-23/24)

- `test_rows` profile (EXTRACTION_PROFILE or GraphState.profile): for e-Attestation, which
  needs only test rows. Skips structure analysis, semantic merge, conflict adjudication
  (conflicts go to a person) and form mapping; completeness does not re-analyse pages
  without a results table.
- Compact extraction (graph/nodes/compact_rows.py, COMPACT_TEST_ROWS default true under
  test_rows): numbered lines in, positional arrays out; page and evidence come from the
  cited line; a row whose result is not on its line (±1/+2) or whose name words are not
  within ±3 lines is dropped with a HALLUCINATED_VALUE error. In the general path test rows
  are NOT evidence-checked anywhere.
- Tolerant schemas: numbers/nulls in text columns are coerced instead of failing a chunk.
- Gemini thinking level for the extraction call only (measured: small gain, 153s -> 146s).
- Chunking repeats a split page's table header in continuation chunks.
- OCR: app/documents/deepreef_local.py runs the team's deepreef-ocr engine in-process,
  English (`en`), max_words=0, row-grouped `text_lines`, one engine per thread;
  preprocess._prefetch_scanned_pages OCRs scanned pages in parallel. RapidOCR is the
  fallback. Model files are checked against deepreef-ocr's models.sha256.
- Upload page shows a per-document timer (file pick -> done).

### Measured results

- hp-lab-report.pdf (7 pages, digital): 300s (original) -> 61s (test_rows).
- high-protein-paneer.pdf (11 pages, digital): 113s -> 21s, 272 test rows = gold count,
  100% of rows on their cited page (was 48%, with 35% duplicates).
- Ingredient_Lab_Reports (30 pages, 29 scanned; answer key of 281 rows transcribed by
  hand, scorer in the session scratchpad): RapidOCR 148s, recall 94.3%, precision 81.5%
  -> deepreef-local + fixes 150s, recall 100% (281/281), precision 99.3% (only 2
  "Description" rows extra). OCR is 129s of the 150s.
- Gold labels for the 50-doc benchmark live on `G:\Shared drives\...\labelled_dataset`,
  not on this Mac — true precision/recall needs that folder.

### OCR latency analysis (2026-11-24, page 17 of the scan, 143 text boxes)

- Per page ≈ render 0.2s + detection 0.5-1s + boxes x ~50ms (recognition is ~78%).
  The model itself is ~28ms/box; the rest is Python pre/post-processing (CTC decode over
  the stand-in's 6,625-char dictionary).
- On this 8 GB M1 parallel pages do not help: 4.6 GB swap in use, CPU idle ~2/3 of the
  time. More threads (8) is 44% slower; batch 24 is 51% slower AND changes 13/143
  readings; CoreML is 2.4x slower; DPI is irrelevant (pages are resized to 1187x1536).
- Small safe wins measured: INT8 (MatMul/Gemm only) 13% faster with identical text on
  one page (validate on all pages first); 13% of boxes are pixel-identical repeats.
- The decisive lever is where OCR runs: same engine on an 8-core/16 GB host ~30-40s for
  the 30-page scan; deepreef-ocr Lambda fan-out ~3-5s (user chose local for now).
- Proposed next: overlap OCR with AI extraction, page/box caching, INT8 after validation.
- 2026-11-24, 2-hour OCR sprint agreed with the user: A1 benchmark, A2 genuine model, then
  B1 (Docling warm-up only when USE_DOCLING), B2 (no table detection on scanned pages,
  ~10s/30 pages), B3 (render at OCR size), B4 (clean-scan preprocessing profile; deepreef
  clean-up is 233ms/page default vs 57ms without illumination/contrast/sharpen/compression),
  each gated on unchanged accuracy.
- A1 done: `scripts/ocr_benchmark.py` — OCR-only (no LLM), per-stage timing + accuracy vs
  an answer key; `--out` / `--compare` for before/after. Answer key and results live outside
  git in `../../benchmarks/` (ingredient_lab_reports.key.json, baseline_standin.json).
  A2 done: genuine devanagari_rec.onnx + dict downloaded from deepreef-ocr's documented
  public sources (ModelScope RapidAI/RapidOCR v3.9.2 PP-OCRv5 + PaddleOCR dict; sha256
  match models.sha256) into ../../deepreef-ocr/deepreef_ocr/models. Genuine vs stand-in:
  111.1s vs 132.5s for 29 pages (-16%, 39 vs 48 ms/box), 265/268 vs 267/268 rows (read
  "LOQ" as "L0Q" with default clean-up). Results: ../../benchmarks/baseline_genuine.json.
  B2 done: table detection skipped on scanned pages (was 289 ms/page and said "table" on
  all 29 pages); has_table for OCR pages now comes from chunking._last_table_header.
  B3+B4 done (settings OCR_RENDER_TO_TARGET, OCR_CLEAN_SCAN_PROFILE, default on;
  deepreef_local.render_for_ocr / prepare_image): alternating A/B on 6 pages: -0.25 s/page,
  OCR time unchanged; full run 267/268 rows (L0Q fixed). B1 measured: Docling warm-up is
  +262 MB / 7 s (not 1-2 GB as first estimated).
  B1 done (agent, reviewed): main._warm_docling returns early unless USE_DOCLING;
  tests/test_docling_warmup_gate.py.
  C1 done (agent, reviewed): app/documents/ocr_page_cache.py — key = sha256(rendered PNG,
  model file sha256, render/clean-scan settings, CACHE_VERSION); JSON files under
  graph_storage/ocr_cache/ (own `*` .gitignore, holds patient text, plain at rest), TTL
  OCR_PAGE_CACHE_TTL_HOURS=168, never fails a page; empty readings not cached. Bump
  CACHE_VERSION if OCR processing changes in a way the key doesn't cover. Test suite
  disables it globally in tests/conftest.py (agent's PYTEST_CURRENT_TEST check in app code
  was removed). Suite: 634 passed, 3 known stale failures.
  B5 ENABLED at the user's request (2026-11-24): INT8 (MatMul/Gemm only) build of the
  pinned model, 3-6% faster, identical rows (267/268), 2/1004 lines differ (logo only).
  File: ../../ocr-models/deepreef-int8/ (outside git; sha256 b07bbad3...7601, dict =
  pinned dict). Approved via deepreef_local.APPROVED_DERIVED_MODELS (second approved
  checksum, records how it was made + validation); model_report() -> variant "INT8",
  readings labelled "(INT8, approved)". backend/.env DEEPREEF_MODEL_DIR points at it;
  delete that line to return to the fp32 pinned model. Report: ../../benchmarks/int8/.
  2026-11-24 (parallel agents, reviewed): D1 OCR_WORKER_MODE=thread|process
  (preprocess._make_ocr_process_pool, spawn; pool failure -> sequential; unmeasured on a
  real multi-core host). E1 OCR worker service: backend/ocr_service/app.py (GET /health,
  POST /ocr/page; run `.venv/bin/uvicorn ocr_service.app:app --port 8100`) + client
  app/documents/ocr_service_client.py; OCR_ENGINE=deepreef_service + OCR_SERVICE_URL.
  E2 Lambda fan-out: app/documents/deepreef_lambda.py (one invoke/page in parallel,
  script en, max_words 0); OCR_ENGINE=deepreef_lambda — mock-tested only (dr-dev lacks
  lambda:InvokeFunction). Both remote engines run via preprocess._prefetch_remote;
  failed pages fall back to local RapidOCR. Skipped for time/risk: C2 box cache, D2
  OCR/AI overlap. Suite: 650 passed, 3 known stale.
  LATEST 30-page measurement (2026-11-24, INT8 + B2-B4 + cache, 8 GB M1): first upload
  131s (OCR 118s, AI 13s); OCR_WORKER_MODE=process 344s (swap 6.6/7.2 GB — never use
  process mode on this laptop); re-upload from cache 18s. All three: recall 100%
  (281/281), precision 98.9%. Results: `../../benchmarks/latest_*.json`.
  BATCH RUN (2026-11-24): 54-document dataset (../../datasets/lab_reports_2026-11-24, from
  "drive all pdf .zip"; deleted/ dupes, mp4, bak excluded) run with
  ../../benchmarks/batch/run_batch.py (2 shards; cache off), Excel via build_report.py ->
  ../../benchmarks/batch/Lab_Reports_Latency_Accuracy_Report.xlsx. Proxy accuracy only
  (no gold here): rows verified on the page + overlap with old pipeline outputs
  (benchmark_output/predictions_gold) + their Sept gold-scored metrics.
  BUG FOUND + FIXED: image uploads (PNG/JPG) went through RapidOCR `.text` = whole page
  as ONE line, so compact extraction refused every row (006_Lab-report.png: 0 rows,
  status still VALIDATED). preprocess image branch now uses deepreef_local
  (rendered_pdf_page=False, full clean-up) with row lines, RapidOCR lines as fallback;
  006 -> 20 rows. tests/test_image_ocr_path.py. PNG results from the running shards
  predate the fix and were re-run (pre-fix copies in results_2026-11-24/png_before_fix/).
  RESULTS (54 docs, 164 pages, 15 scanned/image docs): latency median 18.3s, mean 31.2s,
  p90 62.1s, max 176.9s (quota back-offs inflate a few AI times, e.g. 109s); scanned median
  9.8s, digital median 22.7s. 1153 rows, 100% verified on their page, 1% duplicates; old
  pipeline rows also found by new: 508/539 = 94.2%; 305 new-only rows. Status: VALIDATED 44,
  INCOMPLETE 6 (image docs with rows but no header fields — decision counts fields, not test
  rows; honest-status fix), CONFLICTED 4. paneer row has a bad prep/ai split (audit timestamp
  glitch; wall time is right).
  FIXED after the batch (2026-11-24): digital pages were OCR'd in FULL by the old RapidOCR
  route when images covered >= 3% (a logo) — ~10s/page, 50 of hp-lab-report's 60s. Now
  OCR_AUGMENT_REGIONS_ONLY (default true): only the image regions are OCR'd
  (preprocess._image_regions/_ocr_image_regions, deepreef engine), and under test_rows only
  on page 1 (letterhead repeats; e-Attestation identifies the lab by login). Region OCR also
  reads the letterhead correctly where whole-page OCR added garbled duplicates.
  hp-lab-report 60s -> 10s, CASTOR-OIL 15s -> 7s, paneer -> 27s; same rows.
  HONEST STATUS fixed (decision.py): unreadable pages -> INCOMPLETE naming the pages;
  test_rows is judged by test rows (none -> INCOMPLETE; rows without header fields no longer
  INCOMPLETE). tests/test_honest_status_and_regions.py. Batch re-run with both fixes:
  ../../benchmarks/batch/results_2026-11-24_v2/ (Excel: Lab_Reports_Latency_Accuracy_Report_v2.xlsx).
  v2 vs v1 (54 docs): median 18.3s -> 10.7s, mean 31.2 -> 15.3s, p90 61 -> 27s, max 177 ->
  64s; digital median 22.7 -> 10.4s; scanned 9.8 -> 11.9s (AI-time noise). 1139 rows, 99.9%
  on their cited page; old-pipeline rows found 94.2% -> 95.0%. Status: VALIDATED 47,
  CONFLICTED 3, INCOMPLETE 2 (zero test rows: 008 png, VU3 cytology), LOW_CONFIDENCE 2.
  a516f…pdf 141 -> 129 rows: checked, the later-page images are letterheads/footers only;
  v1's extra rows came from whole-page OCR re-reading the text layer.
  LOCKED SUITE (2026-11-24): backend/tests/locked/ — 9 tests / 12 items guarding the
  product invariants (strict numbers, rows not on the page refused, test_rows: only
  extraction calls the model + conflicts go to a person, OCR fallback never loses a page,
  model approval labels, page cache correctness, real OCR on pages 7/17 if the PDF is
  present, lock integrity). Files are sha256-locked in tests/locked/SUITE.lock; any edit
  fails test_lock_integrity. Change only deliberately:
  `.venv/bin/python scripts/lock_tests.py --update --reason "<why>"` (logs to
  tests/locked/LOCK_HISTORY.md). Run: `.venv/bin/python -m pytest tests/locked -q` (~15 s).
  Note: compact_rows accepts a result within -1/+2 lines of the cited line (by design,
  for wrapped rows); the locked test covers results far from the cited line.
  END-TO-END after B1-B4 + C1 (30-page scan, full pipeline, test_rows, genuine model):
  cold 125s (OCR+preprocess 112s, AI 13s) vs 150s before; re-upload (all 29 pages from
  the page cache) 18s. Accuracy both runs: recall 100% (281/281), precision 98.9% (3
  "Description" rows). Results: ../../benchmarks/final_cold.json, final_warm.json.

### Open problems

- ocr_client (deepreef-ocr Lambda for Indic scripts) never sends max_words, so those pages
  are capped at the engine's default 80 words.
- "Description = Brown Colour Powder"-style descriptive rows are extracted as test rows.
- Under test_rows, image regions on digital pages are OCR'd on page 1 only. A results table
  pasted as an image on page 2+ of a digital PDF would be missed (not seen in the 54-doc
  set). Possible fix: also OCR a later page's images when its text layer has no results table.

### Repo-wide (from before this session)

- `feat/demo22` @ c950cf0 includes Fix 5/6/7 (strict number matching, PRs #4 #5 #6) and
  Fix 13 units/symbols (#9). Open PRs: #13 Anthropic provider · #12 optional API key (keep
  API_KEY unset) · #11 eval harness · #10 honest status · #8 upload limits · #7 untrack
  runtime data · #3 / #1 OCR engine · #2 demo22 -> main.
- Known failing: 3 stale tests in tests/test_pipeline.py (duplicated in
  scripts/integration_snapshots/); fixing them needs the user's OK.
- Pending decisions (../../V2_MORNING_REPORT.md §8): stale-test repair; requirements pins;
  purge uploads from git history; is the "Hematuria" upload real patient data; double LLM
  cost when the graph fails after its calls; workflow_timeout_seconds not enforced.
- Hygiene (cleaned 2026-11-24): the stray key files (`env (3) (1)`, ../.env.backup.1449)
  were moved out of the workspace to ~/.nabl-secrets (owner-only); pipeline.py.*.bak,
  docid.tmp, newapp.txt removed from git; agent worktrees removed (their branches remain);
  OCR page cache emptied. Still open: README/SETUP say Nova-only but config uses a provider
  chain; requirements.txt pins (httpx, python-docx) need the user's approval; low disk
  (~3 GB) and heavy swap on this 8 GB Mac.
