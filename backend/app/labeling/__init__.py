"""Offline labeling stage (Step 5) — for every normalized document (Step 2's
output, ALL of them, not a sample), classifies its domain and extracts
structured fields/tests in one LLM call, guided by Step 4's master schema.
Writes one label JSON per document plus a dataset index. Never invents a
value: a hinted document-level field with no match in the text is recorded
as null rather than omitted or guessed, and every extracted value is
preserved exactly as written (no rounding, unit conversion, or reformatting).

Each label starts with `"annotation_status": "pending"` — this stage never
marks anything "approved" itself; that requires human verification (see
app/quality_control, Step 6, for the gate that gets enforced before any
label is allowed to reach "approved").

Deliberately independent of the live FastAPI app, same as the earlier
offline stages — a standalone batch tool over dataset_normalization's and
master_schema's output, run via scripts/create_labels.py. Reuses
app/llm/factory.py's LlmChain and schema_discovery/sampling.py's normalized-
document loader exactly as they already exist.
"""
