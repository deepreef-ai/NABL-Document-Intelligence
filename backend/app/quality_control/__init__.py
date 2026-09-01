"""Offline quality-control stage (Step 6) — checks every label Step 5
produced against its source normalized document and the dataset as a whole:
valid document_id, readable source file, correct page count, valid domain,
valid schema shape, missing required fields, invalid values, incorrect
table (test-row) mappings, duplicate document ids, duplicate documents (by
source content hash), OCR failures, and extraction failures.

Never corrects a value — a label's `fields`/`tests` content is read-only to
this stage. The only things it ever writes back into a label are its
`annotation_status` and a `qc` block recording what was checked. The
transition rule is a pure gate, not a promoter: a label that PASSES keeps
whatever status it already had (so "pending" stays "pending" — a human
still has to review and approve it; an already-"approved" label that still
passes stays "approved"), and a label that FAILS is always forced to
"rejected", even overriding a prior "approved" — this is what makes "do not
allow an invalid annotation to become approved" an enforced invariant rather
than a one-time check.

Deliberately independent of the live FastAPI app, same as the earlier
offline stages — a standalone batch tool over labeling's output, run via
scripts/run_quality_control.py.
"""
