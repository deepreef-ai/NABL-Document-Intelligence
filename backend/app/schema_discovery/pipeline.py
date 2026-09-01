"""Orchestrates the schema-discovery stage: load Step 2's normalized
documents -> pick a representative sample spread across (heuristic) domain
buckets -> one LLM call per sampled document to discover candidate_domain +
keys -> write one JSON per sampled document -> aggregate into a
domain -> keys summary, which is this stage's actual deliverable.

Deliberately does NOT call the LLM for every document in the dataset — see
sampling.py's docstring for why a representative sample is enough for
schema discovery (as opposed to value extraction or ground truth, which
would need full coverage).

Resumable the same way as dataset_normalization/pipeline.py: an existing
samples/<document_id>.json is reused unless --force, and one failed LLM
call is recorded and skipped rather than aborting the whole batch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.schema_discovery.llm_discovery import discover_document_keys
from app.schema_discovery.domains import heuristic_domain
from app.schema_discovery.models import DiscoveryIndexEntry
from app.schema_discovery.sampling import load_normalized_documents, select_representative_sample


@dataclass
class RunStats:
    total_documents: int = 0
    heuristic_bucket_sizes: dict[str, int] = field(default_factory=dict)
    sampled: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)  # (document_id, reason)


def _sample_path(output_dir: Path, document_id: str) -> Path:
    return output_dir / "samples" / f"{document_id}.json"


def run(input_dir: Path, output_dir: Path, max_per_domain: int = 5, force: bool = False) -> RunStats:
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)

    records = load_normalized_documents(input_dir)
    stats = RunStats(total_documents=len(records))

    buckets = select_representative_sample(records, max_per_domain)
    stats.heuristic_bucket_sizes = {domain: len(bucket) for domain, bucket in buckets.items()}
    sampled_records = [record for bucket in buckets.values() for record in bucket]
    stats.sampled = len(sampled_records)

    index_entries: list[DiscoveryIndexEntry] = []

    for record in sampled_records:
        sample_path = _sample_path(output_dir, record.document_id)
        h_domain = heuristic_domain(record.text)

        if sample_path.exists() and not force:
            cached = json.loads(sample_path.read_text(encoding="utf-8"))
            stats.skipped += 1
            index_entries.append(DiscoveryIndexEntry(
                document_id=record.document_id, original_filename=record.original_filename,
                heuristic_domain=h_domain, candidate_domain=cached.get("candidate_domain"),
                status="skipped", key_count=len(cached.get("keys", [])),
            ))
            continue

        try:
            result = discover_document_keys(record.document_id, record.text)
        except Exception as exc:  # noqa: BLE001 — one bad LLM call must never stop the batch
            stats.failed += 1
            stats.failures.append((record.document_id, f"{type(exc).__name__}: {exc}"))
            index_entries.append(DiscoveryIndexEntry(
                document_id=record.document_id, original_filename=record.original_filename,
                heuristic_domain=h_domain, candidate_domain=None,
                status="failed", key_count=0, error=str(exc),
            ))
            continue

        sample_path.write_text(json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        stats.processed += 1
        index_entries.append(DiscoveryIndexEntry(
            document_id=record.document_id, original_filename=record.original_filename,
            heuristic_domain=h_domain, candidate_domain=result.candidate_domain,
            status="processed", key_count=len(result.keys),
        ))

    with (output_dir / "discovery_index.jsonl").open("w", encoding="utf-8") as index_file:
        for entry in index_entries:
            index_file.write(json.dumps(entry.to_json_dict(), ensure_ascii=False) + "\n")

    _write_domain_keys_summary(output_dir)
    return stats


def _write_domain_keys_summary(output_dir: Path) -> None:
    """Aggregates every samples/*.json on disk (this run's fresh results
    plus any reused/cached ones from earlier runs) into one keys-by-domain
    view, grouped by the LLM's own candidate_domain — the actual deliverable
    of this stage."""
    domain_keys: dict[str, dict] = {}
    for sample_path in sorted((output_dir / "samples").glob("*.json")):
        data = json.loads(sample_path.read_text(encoding="utf-8"))
        domain = data.get("candidate_domain") or "other"
        bucket = domain_keys.setdefault(domain, {"sample_document_ids": [], "key_frequency": {}})
        bucket["sample_document_ids"].append(data["document_id"])
        for key in data.get("keys", []):
            bucket["key_frequency"][key] = bucket["key_frequency"].get(key, 0) + 1

    for bucket in domain_keys.values():
        bucket["sample_count"] = len(bucket["sample_document_ids"])
        bucket["keys"] = sorted(bucket["key_frequency"], key=lambda k: (-bucket["key_frequency"][k], k))

    (output_dir / "domain_keys.json").write_text(
        json.dumps(domain_keys, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def print_summary(stats: RunStats) -> None:
    print()
    print("=== Schema Discovery Summary ===")
    print(f"Normalized documents available:  {stats.total_documents}")
    print(f"Representative sample size:      {stats.sampled} (upper bound on LLM calls this run)")
    print(f"Heuristic bucket sizes sampled from: {stats.heuristic_bucket_sizes}")
    print(f"Processed (new LLM calls):       {stats.processed}")
    print(f"Skipped (already cached):        {stats.skipped}")
    print(f"Failed:                          {stats.failed}")
    if stats.failures:
        print()
        print("--- Failures ---")
        for document_id, reason in stats.failures:
            print(f"  {document_id}: {reason}")
