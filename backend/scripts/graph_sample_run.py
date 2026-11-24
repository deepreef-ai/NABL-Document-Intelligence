#!/usr/bin/env python3
"""Generate the documented sample input/output pair for the graph workflow.

Runs the real graph end to end against a small generated PDF, with a scripted
model standing in for the LLM so the output is byte-reproducible and needs no
provider credentials. Everything except the model is genuine: real PyMuPDF
text extraction, real chunking, real evidence validation against the page,
real mapping and fill rules, real quality control and decision logic.

    python scripts/graph_sample_run.py

Writes docs/samples/sample_milk_report.pdf and docs/samples/sample_output.json.
"""
from __future__ import annotations

import importlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from app.graph.llm import LlmOutcome  # noqa: E402
from app.graph.runner import run_document  # noqa: E402
from app.graph.schemas import LlmCallMetric  # noqa: E402

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "docs")
SAMPLES = os.path.abspath(os.path.join(DOCS, "samples"))

PAGES = [
    ["MILK ANALYSIS REPORT",
     "Report No: LR-2024-0195",
     "Sample Name: Cow Milk",
     "Collected On: 12/03/2024",
     "Lab: Deepreef Analytical"],
    ["RESULTS",
     "Fat: 4.2 %",
     "Protein: 3.4 g/100ml",
     "SNF: 8.6 %",
     "Analyst: R. Menon"],
]

TARGET_FORM = {
    "form_id": "SAMPLE_MILK",
    "source": "provided",
    "fields": [
        {"name": "report.number", "data_type": "string", "required": True,
         "description": "report identifier", "enum_values": [], "repeating": False, "parent": None},
        {"name": "analysis.fat_percent", "data_type": "string", "required": False,
         "description": "fat as a percentage", "enum_values": [], "repeating": False, "parent": None},
        {"name": "analysis.moisture", "data_type": "string", "required": False,
         "description": "moisture as a percentage", "enum_values": [], "repeating": False, "parent": None},
    ],
}

# Which labels the scripted model "reads", and which page each is on.
LABELS = {
    "Report No": 1, "Sample Name": 1, "Collected On": 1,
    "Fat": 2, "Protein": 2, "SNF": 2, "Analyst": 2,
}


def write_sample_pdf(path: str) -> None:
    """One insert_text per line so the text layer has real line breaks.

    Writing a multi-line string in a single insert_text call produces a page
    whose extracted text has no line structure, which is not representative of
    a real document and would make the sample misleading.
    """
    doc = fitz.open()
    for lines in PAGES:
        page = doc.new_page()
        y = 100
        for line in lines:
            page.insert_text((72, y), line, fontsize=11)
            y += 22
    doc.save(path)
    doc.close()


def scripted_llm(*, node, system, user_text, output_model, images=None,
                 image_media_type=None, max_attempts=None, on_metric=None):
    if on_metric:
        on_metric(LlmCallMetric(
            node=node, model="us.amazon.nova-2-lite-v1:0", provider="nova",
            latency_seconds=1.2, input_tokens=1800, output_tokens=320,
        ))

    chunk_id = next(
        (l.split(":", 1)[1].strip() for l in user_text.splitlines()
         if l.startswith("chunk_id:")), "",
    )

    if node == "document_analysis":
        data = {
            "document_type": "milk analysis report",
            "sections": [{"name": "Results", "start_page": 2, "end_page": 2,
                          "kind": "table", "description": "analyte results"}],
            "repeated_entities": [], "candidate_fields": ["report_no", "fat", "protein"],
            "cross_page_references": [], "suspected_duplicates": [], "notes": "",
        }
    elif node == "dynamic_extraction":
        fields = []
        for raw in user_text.splitlines():
            line = raw.strip()
            for label, page in LABELS.items():
                if line.startswith(label + ":"):
                    value = line.split(":", 1)[1].strip()
                    fields.append({
                        "field_name": label,
                        "normalized_field_name": label.lower().replace(" ", "_"),
                        "value": value, "normalized_value": value,
                        "data_type": "date" if label == "Collected On" else "string",
                        "page_number": page, "chunk_id": chunk_id,
                        "section_name": "Results" if page == 2 else "Header",
                        "table_context": "", "exact_source_evidence": line,
                        "confidence_score": 0.96, "extraction_status": "VERIFIED",
                        "occurrence_index": 0, "notes": "",
                    })
        data = {"chunk_id": chunk_id, "fields": fields,
                "tables_seen": ["analyte results"], "notes": ""}
    elif node == "field_normalization":
        data = {"groups": [], "notes": ""}
    elif node == "conflict_resolution":
        data = {"resolutions": []}
    elif node == "form_mapping":
        data = {"mappings": [
            {"source_field": "report_no", "target_field": "report.number",
             "source_field_uid": "", "mapping_confidence": 0.94,
             "mapping_reason": "'Report No' is this document's identifier, which is what "
                               "report.number asks for",
             "mapping_status": "MAPPED"},
            {"source_field": "fat", "target_field": "analysis.fat_percent",
             "source_field_uid": "", "mapping_confidence": 0.93,
             "mapping_reason": "'Fat' is reported as a percentage, matching fat_percent",
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


def install_stub() -> None:
    for name in ("app.graph.llm", "app.graph.nodes.analysis", "app.graph.nodes.extraction",
                 "app.graph.nodes.normalize", "app.graph.nodes.conflicts",
                 "app.graph.nodes.mapping"):
        module = importlib.import_module(name)
        if hasattr(module, "call_structured"):
            setattr(module, "call_structured", scripted_llm)


def main() -> int:
    os.makedirs(SAMPLES, exist_ok=True)
    pdf_path = os.path.join(SAMPLES, "sample_milk_report.pdf")
    write_sample_pdf(pdf_path)
    install_stub()

    result = run_document(
        pdf_path, document_id="sample-001",
        target_form_schema=TARGET_FORM, checkpointer=MemorySaver(),
    )
    payload = result.model_dump(mode="json")

    total_audit = len(payload["audit_log"])
    payload["audit_log"] = payload["audit_log"][:10] + [
        {"note": f"... {total_audit - 10} further entries omitted for readability"}
    ]

    out_path = os.path.join(SAMPLES, "sample_output.json")
    io.open(out_path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, default=str))

    print(f"status            : {payload['overall_status']}")
    print(f"pages             : {result.document_summary.page_count}")
    print(f"fields extracted  : {len(result.extracted_fields)}")
    print(f"evidence complete : {result.quality_control.all_values_have_evidence}")
    print(f"filled form       : {json.dumps(result.filled_form)}")
    print(f"missing fields    : {result.missing_fields}")
    print()
    for f in result.extracted_fields:
        print(f"  {f.normalized_field_name:16} = {str(f.value)[:28]:30} "
              f"p{f.page_number}  {f.evidence_status.value if f.evidence_status else '-'}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
