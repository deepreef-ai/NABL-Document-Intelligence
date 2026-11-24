"""LOCKED integration: the test_rows (results-table / e-Attestation) profile,
run end to end through app.graph.runner.run_document with a scripted model."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.graph.runner import run_document
from app.graph.schemas import ConflictResolution


def _reply(user_text: str) -> dict:
    """A compact reply: one genuine row, one invented row, and two different
    values for the same header key, each printed on the line it cites."""
    rows, header = [], []
    for line in user_text.splitlines():
        if "|" not in line or line.startswith("ctx|"):
            continue
        n, text = line.split("|", 1)
        n = int(n)
        if "Protein:" in text:
            rows.append([n, "Protein", "3.4", "g/100ml", "", "", ""])
        if "SNF:" in text:
            rows.append([n, "SNF", "9.9", "%", "", "", ""])            # page says 8.6
        if "Report No:" in text:
            header.append([n, "sample_id", "LR-2024-0195"])
        if "Sample Name:" in text:
            header.append([n, "sample_id", "Cow Milk"])
    return {"panels": [], "rows": rows, "header": header}


def test_only_extraction_calls_the_model_and_conflicts_go_to_a_person(simple_pdf, scripted):
    """Why this must never break: in the attestation profile the model only
    READS the page. It must never be asked to pick a winner between two values
    the document states (a person confirms with the lab), never be called by
    form-filling nodes, and an invented row must not survive into the output."""
    stub = scripted({"dynamic_extraction": _reply})
    result = run_document(simple_pdf, document_id="locked-rows", checkpointer=MemorySaver(),
                          profile="test_rows")

    assert {c["node"] for c in stub.calls} == {"dynamic_extraction"}

    rows = {(t.test_name, t.result) for t in result.tests}
    assert ("Protein", "3.4") in rows
    assert not any(name == "SNF" for name, _ in rows), "a value the page does not show"

    assert result.conflicts, "two different stated values must register as a conflict"
    for c in result.conflicts:
        assert c.resolution == ConflictResolution.MANUAL_REVIEW_REQUIRED
        assert not c.chosen_field_uid
        assert len(c.candidates) >= 2, "every candidate is preserved for the reviewer"
