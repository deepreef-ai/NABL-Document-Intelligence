"""Read-only final validation of the COMPILED dataset artifact
(final_dataset/final_labeled_dataset.jsonl) — an independent integrity
check on the output file itself, distinct from Step 6's per-label QC gate
(which runs before compilation, on labeled_dataset/labels/*.json, and can
change a label's annotation_status). This stage never writes back into the
dataset, a label, or anything upstream — it only ever produces
final_dataset_qc.json. "Do not modify valid annotations automatically"
applies literally: there is no write path here at all.

Checks: unique document_id, valid domain, valid source_format, every
annotation actually approved, keys recognized against Step 4's master
schema (unrecognized ones counted, not auto-rejected — see validate.py's
docstring for why), no duplicate JSON keys, missing values represented as
null rather than empty/placeholder text, test rows keep
test_name/result/unit/reference_range together, multi-page documents stay
one JSONL record, and every record's source file is traceable back to it
via Step 2's id_registry.json content-hash mapping.

Run via scripts/validate_final_dataset.py.
"""
