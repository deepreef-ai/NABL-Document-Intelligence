from types import SimpleNamespace

from app.documents.compiler import detect_conflicts


def _field(field_path, value, source="llm"):
    return SimpleNamespace(field_path=field_path, value=value, source=source)


def test_same_value_from_multiple_sources_is_not_a_conflict():
    fields = [
        _field("organisation.gst_number", "27AAECS4821L1ZP", "rule_based"),
        _field("organisation.gst_number", "27AAECS4821L1ZP", "llm"),
    ]

    assert detect_conflicts(fields) == []


def test_different_values_for_the_same_field_path_is_a_conflict():
    fields = [
        _field("organisation.gst_number", "27AAECS4821L1ZP", "rule_based"),
        _field("organisation.gst_number", "27WRONGXXXXL1ZP", "llm"),
    ]

    conflicts = detect_conflicts(fields)

    assert len(conflicts) == 1
    assert conflicts[0].field_path == "organisation.gst_number"
    assert set(conflicts[0].values) == {"27AAECS4821L1ZP", "27WRONGXXXXL1ZP"}
    assert set(conflicts[0].sources) == {"rule_based", "llm"}


def test_sentinel_and_null_values_are_ignored_not_treated_as_conflicting():
    fields = [
        _field("organisation.gst_number", "27AAECS4821L1ZP"),
        _field("organisation.gst_number", None),
        _field("organisation.gst_number", "n/a"),
    ]

    assert detect_conflicts(fields) == []


def test_unrelated_field_paths_never_conflict_with_each_other():
    fields = [_field("organisation.gst_number", "A"), _field("organisation.pan_number", "B")]

    assert detect_conflicts(fields) == []


def test_three_way_conflict_lists_every_distinct_value_and_source():
    fields = [
        _field("equipment[0].name", "Digital Multimeter", "rule_based"),
        _field("equipment[0].name", "Analog Multimeter", "llm"),
        _field("equipment[0].name", "Digital Multi-Meter", "verification"),
    ]

    conflicts = detect_conflicts(fields)

    assert len(conflicts) == 1
    assert len(conflicts[0].values) == 3
    assert set(conflicts[0].sources) == {"rule_based", "llm", "verification"}
