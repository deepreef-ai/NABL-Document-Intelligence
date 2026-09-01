"""Offline dataset-finalization stage (Step 7) — compiles ONE combined
labeled dataset (every domain together, no train/validation/test split)
from ONLY the labels Step 6 left (or a human explicitly moved to)
`"annotation_status": "approved"`. Uses the canonical field names Step 5
already wrote (guided by Step 4's master schema) as-is — this stage does no
re-extraction and no re-classification beyond a deterministic, filename-based
document_type derivation; it never invents a value.

Writes final_labeled_dataset.jsonl (full-fidelity, one JSON record per
document), final_labeled_dataset.csv (the same records, tabular — see
pipeline.py's docstring for how nested `tests` is represented in CSV), and
dataset_summary.json. Every assembled record is validated before being
written; a record that fails is excluded from the output rather than
written anyway (see validation.py).

Deliberately independent of the live FastAPI app, same as the earlier
offline stages — a standalone batch tool over labeling's and quality
control's output, run via scripts/finalize_dataset.py.
"""
