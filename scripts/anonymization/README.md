# Lab Report Anonymiser

A standalone utility that produces de-identified copies of laboratory reports from a folder
(e.g. a Google Drive for Desktop mount), leaving the originals untouched.

> **Not part of the NABL Document Intelligence application.** This tool does not import the
> extraction graph, OCR pipeline, Docling adapter, classifier or form-filling code, and none
> of them import this. It can be installed and run on a machine where the rest of the system
> does not exist.

---

## Quick start

```bash
pip install -r scripts/anonymization/requirements.txt

# See what would be detected — writes nothing
python scripts/anonymization/anonymize_lab_reports.py \
    -i "G:/Shared drives/.../lab_reports" \
    -o "D:/anonymised/lab_reports" \
    --dry-run

# Real run
python scripts/anonymization/anonymize_lab_reports.py \
    -i "G:/Shared drives/.../lab_reports" \
    -o "D:/anonymised/lab_reports" \
    --mode label
```

---

## What it guarantees

| Guarantee | How it is enforced |
|---|---|
| Originals are never modified | Inputs opened read-only; output paths are always in a separate tree; verified by checksum in the test suite |
| Output cannot overwrite input | `ensure_safe_roots()` refuses overlapping, nested or identical roots before any file is opened |
| Redaction removes, not covers | Text pages use `apply_redactions()` (deletes glyphs); scanned pages are rebuilt from a painted raster |
| A scanned original never leaks | Scanned pages are emitted as *new* pages; the original image object is never copied into the output |
| Reports don't leak | Reports record kind, page and box — never the value, unless `--include-pii-samples` |
| Nothing fails silently | Every document gets a status: `OK`, `FAILED_REVIEW`, `ERROR` or `SKIPPED` |

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
  input folder ───► │ runner.py — walk, safety checks, per-file │
  (read-only)       └───────────────────┬──────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │  1. TEXT SOURCE            textsource.py           │
              │     PDF text layer ──► Word(text, bbox, page)      │
              │     OCR (rapidocr) ──► Word(text, bbox, page)      │
              └─────────────────────────┬──────────────────────────┘
                                        │  one Word type, either source
              ┌─────────────────────────▼──────────────────────────┐
              │  2. DETECT                 detect.py + patterns.py │
              │     pattern detectors  (Aadhaar+Verhoeff, PAN,     │
              │                         email, phone, IFSC …)      │
              │     label detectors    ("Patient Name:" → value)   │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │  3. PROTECT                protect.py              │
              │     discard anything that is lab vocabulary, a     │
              │     reference range, or a measured value           │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │  4. REDACT                 redact.py               │
              │     text page  → redaction annots + apply()        │
              │     scan page  → rasterise, paint, rebuild page    │
              │     image file → paint pixels                      │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │  5. VERIFY                 verify.py               │
              │     re-open the OUTPUT, re-extract text, re-OCR,   │
              │     re-run detectors → OK or FAILED_REVIEW         │
              └─────────────────────────┬──────────────────────────┘
                                        │
              output folder + per-document reports (report.py)
```

### Why it is built this way

**One `Word` type from both sources.** A born-digital PDF gives exact text with coordinates;
a scan gives OCR lines. Normalising both into `Word(text, bbox, page, source)` means the
detection and protection logic is written once and tested once. The `source` field rides
along to every detection so a report can say whether a finding came from exact text or a
transcription.

**Two detector families, failing in opposite directions.** Pattern detectors prove
themselves — Aadhaar by Verhoeff checksum, PAN and IFSC by structure — so they need no
context and are trustworthy enough to fail a document during verification. Label detectors
("Patient Name:") are the only way to find a name, and they are where nearly all false
positives come from, which is why every candidate must clear the protection layer.

**Protection beats detection, always.** The expensive failure is not a missed identifier —
verification catches those loudly. It is a redacted haemoglobin value: silent, permanent in
the copy, and it makes the anonymised corpus useless for the purpose it was created for. So
`protect.py` holds analytes, units, report furniture and reference-range shapes, and a
detection that collides with any of them is dropped.

**Redaction that actually removes.** Drawing a black rectangle over text leaves the text
selectable underneath — the classic anonymiser failure. Text pages therefore use PyMuPDF's
redaction annotations, which delete glyphs from the content stream. Scanned pages are
rebuilt from a painted raster into a *brand-new* page, so the original scan image cannot
survive as an unreferenced object, which is exactly what happens if you blank a page's
content stream and drop a new image on top.

**Verification assumes the anonymiser is wrong.** It re-opens the file that will actually be
handed over and reads it back through two independent channels — text layer and fresh OCR —
then re-runs the detectors. Only self-validating identifiers can fail a document, because a
verification step that fails everything teaches its operator to ignore it.

### Two bugs this design caught during development

Both were found by the verification pass on real documents, not by the unit tests — which is
the argument for having it:

1. **Scanned-page redactions landed ~2.8× off position.** OCR boxes are in pixels; the page
   rebuild scaled them by the output zoom a second time. Boxes were painted, the count
   looked correct, and the identifier was still plainly legible. Fixed by naming the
   coordinate unit explicitly (`PageRedactions.source_dpi`) instead of a bare scale factor.

2. **PII inside letterhead images was never detected.** A page can have 1,600 characters of
   perfectly good text *and* carry the lab's email inside a raster letterhead. Detection read
   only the text layer and missed it entirely. Fixed by `ocr.augment_text_pages`: pages whose
   embedded images cover more than `image_area_threshold` are also OCR'd, and words falling
   outside the text layer are folded into detection.

---

## What is detected

| Kind | Rule | Default |
|---|---|---|
| `AADHAAR` | 12 digits, not starting 0/1, **Verhoeff checksum** | on |
| `PAN` | `AAAAA9999A` | on |
| `EMAIL` | RFC-ish shape | on |
| `PHONE` | Indian mobile, starts 6–9, optional `+91` | on |
| `PASSPORT`, `VOTER_ID`, `DRIVING_LICENCE`, `IFSC` | structural | on |
| `GSTIN` | structural | **off** — identifies the *lab*, not a patient |
| `PATIENT_NAME` | label + value | on |
| `DOCTOR_NAME` | "Referred By", "Consultant" … | on |
| `ADDRESS`, `DOB`, `PATIENT_ID` (UHID/MRN/ABHA…) | label + value | on |
| `AGE_SEX` | label + value | **off** — usually clinically relevant |

**The Verhoeff checksum matters.** A lab report is full of 12-digit specimen and accession
numbers. Without the checksum every one of them looks like an Aadhaar number, and the tool
becomes a data-destruction tool. `987654321011` is rejected; `234123456783` is accepted.

**"Test Name" is not a patient name.** A bare `Name:` label is ignored when preceded by
`test`, `method`, `sample`, `product`, `instrument` and similar — the single most important
over-redaction guard, since those are column headings on nearly every lab report.

---

## CLI

```
-i, --input PATH          Folder to scan recursively. Never modified.
-o, --output PATH         Folder for anonymised copies. Must be outside --input.
-c, --config PATH         YAML or JSON config file.
    --mode MODE           black_box | label (default) | pseudonym
    --reports PATH        Report location. Default: <output>/_anonymisation_reports
    --no-ocr              Skip OCR. Scanned pages are reported unreadable, never
                          passed through unredacted.
    --no-verify           Skip verification (not recommended).
    --overwrite           Replace existing output files.
    --limit N             Process only the first N files.
    --dry-run             Detect and report; write no anonymised files.
    --include-pii-samples Put truncated samples in reports. Makes reports sensitive.
    --pseudonym-salt S    Secret that makes pseudonyms stable across runs.
    --dpi N               Raster DPI for scanned pages.
    --log-file PATH       Also write logs to a file.
-q, --quiet / -v, --verbose
```

**Exit codes:** `0` all clean · `1` something is `FAILED_REVIEW` or `ERROR` · `2` could not start.

### Redaction modes

| Mode | Result |
|---|---|
| `black_box` | Filled rectangle, text deleted. Most secure-looking, least readable. |
| `label` *(default)* | Text deleted, `[PATIENT_NAME]` stamped in the gap. Keeps the report readable and shows *what* was removed. |
| `pseudonym` | Text deleted, a stable synthetic value stamped (`ALFA CARTER`). With a fixed `--pseudonym-salt` the same person maps to the same pseudonym across runs, preserving longitudinal structure — which also makes the mapping reversible by anyone holding the salt. Treat it as a secret. |

---

## Output

```
output/
├── report.pdf                          anonymised copy, original structure preserved
├── nested/scan.pdf
└── _anonymisation_reports/
    ├── run_report.json                 totals, config, per-document status
    ├── run_report.csv                  same, spreadsheet-friendly
    ├── report.report.json              per-document detail
    └── nested/scan.report.json
```

| Status | Meaning |
|---|---|
| `OK` | Anonymised; verification found no disqualifying PII in the output |
| `FAILED_REVIEW` | **Verification found possible PII in the output.** Review by hand before release |
| `ERROR` | Processing failed; no output written (any partial file is deleted) |
| `SKIPPED` | Output already existed; pass `--overwrite` |

---

## Configuration

See [`anonymizer.example.yaml`](anonymizer.example.yaml). All keys are optional.

The two most useful site-specific settings:

```yaml
extra_protected_terms:     # never redact these — your analytes, method codes
  - "25-OH Vitamin D"
  - "IFSH/SOP/C/TE/008"

extra_patient_labels:      # your templates' own way of introducing a name
  - "name of the subject"
```

---

## Tests

```bash
cd scripts/anonymization
python -m pytest tests -q
```

90 tests, no network and no real patient data — fixtures are generated in memory. Coverage is
weighted toward the two failure directions that matter:

- **Over-redaction** (silent, permanent): `Test Name:` is not a patient name; reference ranges,
  units and analytes survive; a lab result line produces zero detections.
- **Under-redaction** (loud, caught by verification): identifiers are absent from the output's
  *extracted text*, not merely covered.

Plus: originals byte-identical after a run, folder structure preserved, corrupt files reported
rather than raised, partial outputs deleted, and reports containing no PII by default.

---

## Limitations

- **Names without a label are not found.** A name appearing in free text with no
  "Patient Name:" style introduction will be missed. Verification does not catch this either,
  since names are not self-validating. This is the main residual risk.
- **OCR quality bounds everything on scanned pages.** A misread identifier is neither detected
  nor redacted. Pages below `ocr.low_confidence` are flagged in the report — treat those as
  needing a human look.
- **Handwriting is effectively not covered.** RapidOCR targets printed text.
- **`FAILED_REVIEW` means look at it**, not "the tool is broken". It is working as designed:
  it read its own output back and was not satisfied.
- **Verification only fails on self-validating identifiers.** A surviving name or address will
  not flip a document to `FAILED_REVIEW`.

---

## Performance

Roughly, on a typical laptop: born-digital pages are near-instant; OCR costs a few seconds per
page and dominates everything else. The first scanned page also pays a one-off model load.
Verification with `check_ocr: true` roughly doubles OCR cost, since the output is read back the
same way — use `--no-ocr` for a fast pass over a known born-digital corpus.
