#!/usr/bin/env python3
"""Seed three completed runs so a demo does not depend on live LLM quota.

Everything except the model is real: real PDFs, real PyMuPDF extraction, real
chunking, real evidence validation against the page text, real conflict
detection, real form filling and quality control. Only the model's replies are
scripted, so the runs are reproducible and cost nothing.

The three runs are chosen to tell a complete story:

  1. clean milk report      -> VALIDATED    (headings and their tests, all evidenced)
  2. report with a clash    -> CONFLICTED   (both values kept, review queue populated)
  3. report with a bad read -> the evidence gate catching an unsupported value

Run from backend/ so the SQLite path resolves the same way the server's does:

    python scripts/seed_demo_runs.py
"""
from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402

from app.graph.llm import LlmOutcome  # noqa: E402
from app.graph.runner import run_document  # noqa: E402
from app.graph.schemas import LlmCallMetric  # noqa: E402

SAMPLES = os.path.abspath(os.path.join("graph_storage", "demo"))

TARGET_FORM = {
    "form_id": "DEMO_MILK",
    "source": "provided",
    "fields": [
        {"name": "report.number", "data_type": "string", "required": True,
         "description": "report identifier", "enum_values": [], "repeating": False, "parent": None},
        {"name": "report.collected_on", "data_type": "date", "required": True,
         "description": "collection date", "enum_values": [], "repeating": False, "parent": None},
        {"name": "analysis.fat_percent", "data_type": "string", "required": False,
         "description": "fat as a percentage", "enum_values": [], "repeating": False, "parent": None},
        {"name": "analysis.protein", "data_type": "string", "required": False,
         "description": "protein content", "enum_values": [], "repeating": False, "parent": None},
        {"name": "analysis.moisture", "data_type": "string", "required": False,
         "description": "moisture as a percentage", "enum_values": [], "repeating": False, "parent": None},
    ],
}

DOCS: dict[str, list[list[str]]] = {
    "clean_milk_report": [
        ["MILK ANALYSIS REPORT",
         "Report No: LR-2024-0195",
         "Sample Name: Cow Milk",
         "Collected On: 12/03/2024",
         "Lab: Deepreef Analytical",
         "Customer: Anand Dairy Co-operative"],
        ["RESULTS",
         "Fat: 4.2 %",
         "Protein: 3.4 g/100ml",
         "SNF: 8.6 %",
         "Total Solids: 12.8 %",
         "Analyst: R. Menon"],
    ],
    "conflicting_report": [
        ["MILK ANALYSIS REPORT",
         "Report No: LR-2024-0311",
         "Sample Name: Buffalo Milk",
         "Collected On: 04/05/2024"],
        ["RESULTS",
         "Fat: 6.1 %",
         "Protein: 4.0 g/100ml"],
        ["AMENDED SUMMARY",
         "Report No: LR-2024-0312",
         "Fat: 6.4 %"],
    ],
    "unsupported_read": [
        ["WATER TEST CERTIFICATE",
         "Certificate No: WT-88-2024",
         "Source: Borewell 3",
         "pH: 7.4"],
    ],
}

SECTION_FOR_PAGE = {
    "clean_milk_report": {1: "Header and Sample Information", 2: "Results"},
    "conflicting_report": {1: "Header", 2: "Results", 3: "Amended Summary"},
    "unsupported_read": {1: "Certificate Details"},
}


def write_pdf(name: str, pages: list[list[str]]) -> str:
    """One insert_text per line so the text layer keeps real line breaks."""
    os.makedirs(SAMPLES, exist_ok=True)
    path = os.path.join(SAMPLES, f"{name}.pdf")
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 90
        for line in lines:
            page.insert_text((70, y), line, fontsize=11)
            y += 24
    doc.save(path)
    doc.close()
    return path


def make_stub(doc_name: str):
    sections = SECTION_FOR_PAGE[doc_name]

    def field(label: str, value: str, page: int, chunk_id: str, evidence: str, *, occ: int = 0):
        return {
            "field_name": label,
            "normalized_field_name": label.lower().replace(" ", "_").replace(".", ""),
            "value": value, "normalized_value": value,
            "data_type": "date" if "collected" in label.lower() else "string",
            "page_number": page, "chunk_id": chunk_id,
            "section_name": sections.get(page, ""), "table_context": "",
            "exact_source_evidence": evidence, "confidence_score": 0.95,
            "extraction_status": "VERIFIED", "occurrence_index": occ, "notes": "",
        }

    def stub(*, node, system, user_text, output_model, images=None,
             image_media_type=None, max_attempts=None, on_metric=None):
        if on_metric:
            on_metric(LlmCallMetric(node=node, model="us.amazon.nova-2-lite-v1:0",
                                    provider="nova", latency_seconds=1.1,
                                    input_tokens=1650, output_tokens=290))
        chunk_id = next((l.split(":", 1)[1].strip() for l in user_text.splitlines()
                         if l.startswith("chunk_id:")), "")

        if node == "document_analysis":
            data = {
                "document_type": "milk analysis report" if "milk" in doc_name else "water test certificate",
                "sections": [
                    {"name": name, "start_page": page, "end_page": page,
                     "kind": "table" if "result" in name.lower() else "section", "description": ""}
                    for page, name in sections.items()
                ],
                "repeated_entities": [], "candidate_fields": [],
                "cross_page_references": [], "suspected_duplicates": [], "notes": "",
            }
        elif node == "dynamic_extraction":
            fields = []
            page_now = None
            for raw in user_text.splitlines():
                line = raw.strip()
                if line.startswith("--- Page "):
                    try:
                        page_now = int(line.split()[2])
                    except (IndexError, ValueError):
                        page_now = None
                    continue
                if ":" not in line or page_now is None:
                    continue
                label, _, value = line.partition(":")
                label, value = label.strip(), value.strip()
                if not value or label.isupper() and not value:
                    continue
                if label.upper() == label and len(label.split()) > 2:
                    continue  # a heading, not a field
                fields.append(field(label, value, page_now, chunk_id, line))

            # One deliberately unsupported value, so the demo shows the
            # evidence gate rejecting something rather than only succeeding.
            if doc_name == "unsupported_read":
                fields.append({
                    "field_name": "Total Coliform", "normalized_field_name": "total_coliform",
                    "value": "0 CFU/100ml", "normalized_value": "0 CFU/100ml",
                    "data_type": "string", "page_number": 1, "chunk_id": chunk_id,
                    "section_name": "Certificate Details", "table_context": "",
                    "exact_source_evidence": "Total Coliform: 0 CFU/100ml (absent)",
                    "confidence_score": 0.88, "extraction_status": "VERIFIED",
                    "occurrence_index": 0, "notes": "",
                })
            data = {"chunk_id": chunk_id, "fields": fields, "tables_seen": [], "notes": ""}

        elif node == "field_normalization":
            data = {"groups": [], "notes": ""}
        elif node == "conflict_resolution":
            data = {"resolutions": [{
                "normalized_field_name": "fat",
                "chosen_field_uid": None,
                "resolution": "MANUAL_REVIEW_REQUIRED",
                "reason": "the amended summary states a different figure and the document "
                          "does not say which supersedes the other",
                "criteria_used": ["source_clarity", "recency"],
            }, {
                "normalized_field_name": "report_no",
                "chosen_field_uid": None,
                "resolution": "MANUAL_REVIEW_REQUIRED",
                "reason": "two report numbers appear; the amendment may be a separate report",
                "criteria_used": ["context"],
            }]}
        elif node == "form_mapping":
            data = {"mappings": [
                {"source_field": "report_no", "target_field": "report.number",
                 "source_field_uid": "", "mapping_confidence": 0.95,
                 "mapping_reason": "'Report No' is the document's identifier",
                 "mapping_status": "MAPPED"},
                {"source_field": "collected_on", "target_field": "report.collected_on",
                 "source_field_uid": "", "mapping_confidence": 0.93,
                 "mapping_reason": "collection date matches the target's date field",
                 "mapping_status": "MAPPED"},
                {"source_field": "fat", "target_field": "analysis.fat_percent",
                 "source_field_uid": "", "mapping_confidence": 0.92,
                 "mapping_reason": "'Fat' is reported as a percentage",
                 "mapping_status": "MAPPED"},
                {"source_field": "protein", "target_field": "analysis.protein",
                 "source_field_uid": "", "mapping_confidence": 0.9,
                 "mapping_reason": "direct correspondence",
                 "mapping_status": "MAPPED"},
                {"source_field": "", "target_field": "analysis.moisture",
                 "source_field_uid": "", "mapping_confidence": 0.0,
                 "mapping_reason": "the document reports no moisture value",
                 "mapping_status": "NOT_FOUND"},
            ]}
        else:
            data = {}

        try:
            return LlmOutcome(ok=True, parsed=output_model.model_validate(data), raw=data)
        except Exception as exc:  # noqa: BLE001
            return LlmOutcome(ok=False, error_message=str(exc))

    return stub


MODULES = ("app.graph.llm", "app.graph.nodes.analysis", "app.graph.nodes.extraction",
           "app.graph.nodes.normalize", "app.graph.nodes.conflicts", "app.graph.nodes.mapping")


def install(stub) -> None:
    for name in MODULES:
        module = importlib.import_module(name)
        if hasattr(module, "call_structured"):
            setattr(module, "call_structured", stub)


def main() -> int:
    for i, (name, pages) in enumerate(DOCS.items(), start=1):
        path = write_pdf(name, pages)
        install(make_stub(name))
        result = run_document(path, document_id=f"demo-{i:02d}-{name}",
                              target_form_schema=TARGET_FORM)
        print(f"{name:22} {result.overall_status.value:24} "
              f"{len(result.extracted_fields):2} field(s), "
              f"{len(result.conflicts)} conflict(s), "
              f"{len(result.manual_review_items)} review item(s)")
    print(f"\nSeeded {len(DOCS)} run(s). They appear under 'Recent' in the UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
