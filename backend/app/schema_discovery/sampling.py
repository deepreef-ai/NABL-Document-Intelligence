"""Loads Step 2's normalized documents and picks a small, domain-diverse
sample to send to the LLM for schema discovery — the whole point of this
stage is to minimize LLM calls, so it deliberately does NOT process every
document in the dataset.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.schema_discovery.domains import heuristic_domain


@dataclass
class NormalizedDocRecord:
    document_id: str
    original_filename: str
    text: str
    char_count: int


def load_normalized_documents(input_dir: Path) -> list[NormalizedDocRecord]:
    """Only documents Step 2 marked `status == "processed"` are eligible —
    a failed normalization has no usable text to discover keys from."""
    records = []
    for doc_json_path in sorted((input_dir / "normalized").glob("*/document.json")):
        data = json.loads(doc_json_path.read_text(encoding="utf-8"))
        if data.get("status") != "processed":
            continue
        text = "\n".join(page.get("text", "") for page in data.get("pages", []))
        records.append(NormalizedDocRecord(
            document_id=data["document_id"],
            original_filename=data["original_filename"],
            text=text,
            char_count=len(text),
        ))
    return records


def select_representative_sample(
    records: list[NormalizedDocRecord], max_per_domain: int
) -> dict[str, list[NormalizedDocRecord]]:
    """Buckets records by a cheap keyword heuristic (domains.heuristic_domain
    — not the LLM's own, authoritative candidate_domain, just enough to make
    sure the sample spans every domain instead of, say, drawing 40 milk
    reports in a row), then takes up to `max_per_domain` per bucket, spread
    evenly across that bucket's text-length distribution — the shortest, the
    longest, and points in between — so the sample also spans different
    report layouts within a domain, not just different domains."""
    buckets: dict[str, list[NormalizedDocRecord]] = defaultdict(list)
    for record in records:
        buckets[heuristic_domain(record.text)].append(record)

    sample: dict[str, list[NormalizedDocRecord]] = {}
    for domain, bucket in buckets.items():
        bucket_by_length = sorted(bucket, key=lambda r: (r.char_count, r.document_id))
        if max_per_domain <= 1:
            sample[domain] = [bucket_by_length[len(bucket_by_length) // 2]] if bucket_by_length else []
            continue
        if len(bucket_by_length) <= max_per_domain:
            sample[domain] = bucket_by_length
            continue
        last_index = len(bucket_by_length) - 1
        indices = sorted({round(i * last_index / (max_per_domain - 1)) for i in range(max_per_domain)})
        sample[domain] = [bucket_by_length[i] for i in indices]
    return sample
