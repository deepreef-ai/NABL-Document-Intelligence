# NABL Document Intelligence

A 3-phase assistant for NABL lab accreditation and recognition-scheme applications — all 9 real
NABL application forms are supported: **151, 152, 153, 153A, 154, 155, 157, 158, 159** (NABL 100B
is a procedure document, not an application form, and isn't modeled):

1. **Eligibility wizard** — a conversational chatbot walks the applicant through mandatory
   prerequisites (standard implemented ≥3-6mo, internal audit + MRM done, valid PT/ILC, dedicated
   Quality Manager + trained staff) before unlocking document upload.
2. **Document ingestion pipeline** — multi-file upload (PDF/DOCX/JPG/PNG) → text/OCR extraction →
   LLM classification → LLM field extraction into a Pydantic schema for the chosen NABL form.
3. **HITL review & auto-fill** — split-pane UI: source document with highlighted regions on the
   left, pre-filled form on the right, low-confidence fields flagged for manual confirmation, and
   a chat drawer to ask questions or trigger re-extraction.

## Architecture

```
backend/   FastAPI + SQLite + SQLAlchemy + a free-tier LLM fallback chain
frontend/  React (Vite) + TypeScript
```

Key modules (`backend/app/`):

- `llm/` — every LLM call (wizard chat, classification, extraction) goes through
  `llm.factory.get_llm_chain()`, which tries each configured provider in `LLM_PROVIDER_ORDER` and
  falls through to the next on any failure (network error, rate limit, unparseable reply). See
  "LLM providers" below.
- `wizard/` — deterministic prerequisite state machine (`prerequisites.py`, `engine.py`); the LLM
  only parses the user's free-text reply into a satisfied/not-satisfied verdict, it never decides
  gating on its own. Two question sets: the 4 usual ISO-readiness questions for full-accreditation
  forms, and a form-specific 2-question eligibility gate for the lightweight recognition schemes
  (155/157/159) — see `schemas/forms.py`'s `FormCategory`.
- `documents/pipeline.py` — routes each upload by document type:
  - **born-digital PDF** (has a text layer) → PyMuPDF extracts text *and* true bounding boxes.
  - **DOCX** → `python-docx` text extraction (no pixel bounding boxes — not a rasterized document).
  - **scanned/photo page, supported script** (devanagari/arabic/ta/te/ka) → invokes the real
    `deepreef-ocr` Lambda directly (boto3 `lambda:InvokeFunction`, no HTTP endpoint — see
    `documents/ocr_client.py`).
  - **scanned/photo English page** → local RapidOCR, in-process, no AWS/network call (see
    `documents/local_ocr.py`) — the same engine `deepreef-ocr` runs, using its own bundled
    default English/Latin model. Falls back to an LLM vision call only if that itself fails
    (not installed, corrupt image, etc. — see "Known limitation" below).
- `documents/compiler.py` — merges each document's extracted fields into the NABL form shape
  (`schemas/forms.py`).

## LLM providers

`backend/app/llm/` implements a provider chain, currently configured with one link: **Amazon Nova**
on AWS Bedrock (`llm/providers.py`'s `NovaProvider`), used for the wizard chat, classification,
extraction, and the English-scan vision fallback. It authenticates via the ambient AWS credential
chain (the same one `documents/ocr_client.py`'s Lambda invoke relies on) rather than an API key —
set `NOVA_MODEL` (a region-prefixed inference-profile ID, e.g. `us.amazon.nova-2-lite-v1:0`) and
`NOVA_REGION` in `backend/.env`. `LLM_PROVIDER_ORDER` (default `nova`) is still an ordered,
comma-separated list — the chain design supports adding a second provider later — but with only
one link configured, a Nova failure surfaces directly as a `503` naming the failure, rather than
falling through to anything else.

## Known limitation: vision-LLM OCR fallback has no bounding boxes

`deepreef-ocr`'s deployed Lambda still only ships recognition models for **devanagari, arabic, ta,
te, ka** — no English/Latin model (its `dev` branch/local checkout now supports English too, via
RapidOCR's own bundled default model, but that hasn't been redeployed). NABL doesn't need it there
either way: English scans are handled locally by RapidOCR (`documents/local_ocr.py`, same engine,
in-process, no AWS call) rather than routed through the Lambda.

The remaining gap is narrower than "no English OCR": if RapidOCR itself can't process an image
(not installed, corrupt file, etc.) or a script is neither `english` nor one of `deepreef-ocr`'s
supported scripts, extraction falls back to the LLM chain's vision input (Nova, see above). That
path has no reliable per-field bounding box (the review UI shows a page-level highlight, not a
tight box) — a real limitation, but one that's now rarely hit rather than the primary English path.

Born-digital PDFs (most GST/CIN certs, SOPs, PT/ILC reports) and DOCX files skip OCR entirely and
are unaffected by this either way.

## NABL schema — all 9 forms, two shapes

`backend/app/schemas/common.py` and `forms.py` are modeled directly on the official NABL form
templates (read in full — not sampled — on 2026-08-29), not a guess. There are two distinct
schema bases, because the real forms split into two genuinely different shapes:

- **`BaseNablForm`** (151, 152, 153, 153A, 154, 158) — full ISO/IEC 17025 or ISO 15189
  accreditation applications: organisation/legal info, senior management, internal audit/review,
  application fees, equipment, reference materials, staff, authorized signatories, scope, PT/ILC.
  Notably: separate `gst_number`/`pan_number`/`tan_number` (the real forms ask all three as one
  line, not a `cin_number`, which isn't a real field anywhere); no `least_count` on equipment (not
  a real column); `ScopeStatement`/`PTILCRecord` keep testing/medical's "Measurement
  Uncertainty"/Z-score separate from calibration's "CMC"/En-value, since NABL 100B treats them as
  distinct concepts; `StaffRecord` (general staff) and `AuthorizedSignatoryRecord` ("proposed
  personnel... authorized for results") are two separate entities, not one table with a role flag,
  because the real forms carry them as two tables with different columns.
  - `Nabl153Form` also covers **153A** (Operational/Supporting Entities arrangement) via the
    optional `associated_entity` field on personnel/equipment/reference-material records — 153A is
    the same form plus that one attribute, not a separate schema.
  - `Nabl154Form` adds the regulatory-recognition overlay (FSSAI/APEDA/EIC/etc.) fields.
  - `Nabl158Form` adds a `products` list and the shareholder/director ownership-disclosure
    annexure.
- **`BaseRecognitionForm`** (155, 157, 159) — lightweight PT-performance-based recognition
  schemes, not full accreditation: a flat `lab_details` section (no senior management/internal
  audit/fees), narrower scope/equipment/PT tables. `Nabl159Form`'s project fields
  (`project_name`/`project_size`/`project_duration`/`project_reference`) live on `lab_details`;
  `Nabl157Form` is the only one of the three with its own reference-materials table.

**Not auto-extracted (schema-only for now):** 154's Mobile Laboratory/Complaints/FSSAI-scope/
Sampling-scope tables and 158's Affiliations/Family-members/Related-labs/Contracts tables are
modeled for fidelity to the real forms but have no wired document `doc_type` yet — same treatment
`disciplines: list[str]` already gets. They're real, present in the compiled-form JSON, and
editable via the review UI's field-patch endpoint; just not populated from an uploaded document
automatically. See `documents/classifier.py`'s `DOC_TYPES` and `documents/compiler.py`'s
`LIST_TARGET`/`FLAT_MERGE_TARGETS` for exactly which doc types are wired to which entity.

## Uploading a filled copy of the form itself

Every other doc_type above assumes a narrow, single-purpose supporting certificate (one GST proof,
one equipment cert, one staff CV). If instead the upload IS an applicant's own filled copy of the
NABL form — or a draft covering many of its sections at once (organisation details, senior
management, a whole equipment table, a whole staff table, ...) — classification returns
`completed_application_form`, which routes to a different extraction path
(`documents/extractor.py`'s `extract_full_form_fields`): it introspects the actual Pydantic schema
for the application's form_type and asks for every section at once, using an `attr[i].subfield`
convention for repeating tables so `documents/compiler.py` can build multiple records (and multiple
different lists) from one document — every other doc_type only ever targets one list or one flat
object. Born-digital-PDF text extraction reads up to 40 pages for this reason (a real filled form
runs 15-30 pages, and the first several are NABL's own boilerplate amendment sheet/table of
contents, not applicant data — 5 pages would never reach the real content).

This path needs a large context window — a real filled form's text can run to 100K+ characters —
which is why `extract_full_form_fields_chunked` (one LLM call per page/chunk, see
`documents/extractor.py`) is the real entry point for a genuine 15-30 page upload rather than
`extract_full_form_fields`'s single-shot path. With only Nova configured, a rate limit or outage on
it fails the upload outright with a clear "every provider failed" error rather than silently
returning nothing — there's no second provider to fall through to.

NABL 100B is a pure accreditation-procedure document with no applicant fields at all — useful only
as terminology reference (MU vs. CMC, Z-score vs. En-value), not modeled as a form.

## Running locally

### 1. deepreef-ocr (optional — only needed for devanagari/arabic/ta/te/ka scans)

The `akash-ocr` Lambda is invoked directly via boto3 (`lambda:InvokeFunction`) — same mechanism
`pcsapaiv2` production uses. You need AWS credentials in the environment (profile or access keys)
with `lambda:InvokeFunction` on that function; set `OCR_LAMBDA_FUNCTION_NAME`/`OCR_LAMBDA_REGION`
in `backend/.env` if it differs from the defaults. English documents never need this — see "Known
limitation" below.

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # set NOVA_MODEL and make sure AWS credentials are available
uvicorn app.main:app --reload --port 8000
```

`GET http://localhost:8000/health` should return `{"status": "ok"}`.

Run the test suite (no API key or live OCR service needed — the LLM/OCR calls are mocked):

```bash
pytest
```

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:5173`. Pick a form type → answer the eligibility questions → upload
documents → review & confirm the auto-filled form.

## What happens without any LLM key set

The wizard chat, classification, and extraction all need at least one provider configured (see
"LLM providers" above). Without one: a wizard/chat request returns `503` with a clear message; a
document upload completes but the document is marked `status: "failed"` with a clear error rather
than crashing the request. Set at least one key in `backend/.env` to see the full pipeline run.
