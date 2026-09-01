"""Phase 1 (expensive, resumable): runs the real production pipeline once
per ground-truth document and caches each raw prediction to disk
immediately — so a crash partway through only costs the documents not yet
predicted, never the whole run. Phase 2 (pipeline.py's score()) is cheap
and reads only this cache, so it can be re-run freely without repeating any
OCR/LLM call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.benchmark.run_pipeline import run_production_pipeline
from app.labeling.schema_hints import build_domain_hints, load_master_schema


@dataclass
class PredictStats:
    total: int = 0
    predicted: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_predictions(
    final_dataset_dir: Path, predictions_dir: Path, master_schema_dir: Path, force: bool = False,
) -> PredictStats:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(final_dataset_dir / "final_labeled_dataset.jsonl")
    hints = build_domain_hints(load_master_schema(master_schema_dir))

    stats = PredictStats(total=len(records))
    for record in records:
        document_id = record["document_id"]
        out_path = predictions_dir / f"{document_id}.json"
        if out_path.exists() and not force:
            stats.skipped += 1
            continue

        source_path = Path(record.get("source_path", ""))
        prediction = run_production_pipeline(document_id, source_path, hints)
        out_path.write_text(json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8")

        if prediction["pipeline_error"]:
            stats.failed += 1
            stats.failures.append((document_id, prediction["pipeline_error"]))
        else:
            stats.predicted += 1
    return stats


def print_predict_summary(stats: PredictStats) -> None:
    print()
    print("=== Benchmark Prediction Summary ===")
    print(f"Total documents:  {stats.total}")
    print(f"Predicted (new):  {stats.predicted}")
    print(f"Skipped (cached): {stats.skipped}")
    print(f"Failed:           {stats.failed}")
    if stats.failures:
        print("--- Failures ---")
        for document_id, reason in stats.failures:
            print(f"  {document_id}: {reason}")
