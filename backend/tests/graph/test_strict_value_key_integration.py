"""The strict key is wired into conflict detection and result de-duplication."""
from app.graph.nodes.conflicts import detect_and_resolve_conflicts
from app.graph.schemas import ExtractedField, ExtractionStatus
from app.graph.state import GraphState
from app.graph.schemas import TestRow
from app.graph.structured import _dedupe_tests


def _field(uid, value, page):
    return ExtractedField(
        field_name="Creatinine", normalized_field_name="creatinine", value=value,
        page_number=page, chunk_id=f"c{uid}", exact_source_evidence=str(value),
        confidence_score=0.9, extraction_status=ExtractionStatus.VERIFIED, field_uid=uid,
    )


def _conflicts(*values):
    fields = [_field(f"u{i}", v, 1) for i, v in enumerate(values)]
    out = detect_and_resolve_conflicts(GraphState(document_id="d", normalized_fields=fields))
    return out.get("conflicts", [])


def test_1_5_vs_0_15_is_a_conflict():
    assert _conflicts("1.5", "0.15")


def test_1_5_vs_15_is_a_conflict():
    assert _conflicts("1.5", "15")


def test_5_vs_minus_5_is_a_conflict():
    assert _conflicts("5", "-5")


def test_same_number_different_format_is_not_a_conflict():
    assert not _conflicts("1,00,000", "100000")


def test_results_1_5_and_0_15_are_both_kept_but_true_repeat_is_dropped():
    rows = [
        TestRow(test_name="Creatinine", result="1.5", unit="mg/dL"),
        TestRow(test_name="Creatinine", result="0.15", unit="mg/dL"),
        TestRow(test_name="Creatinine", result="1.50", unit="mg/dL"),  # same as the first
    ]
    assert [r.result for r in _dedupe_tests(rows)] == ["1.5", "0.15"]
