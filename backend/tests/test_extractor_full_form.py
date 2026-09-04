from app.documents import extractor
from app.documents.extractor import (
    _fill_missing_flat_fields,
    _filter_to_requested_indexed_fields,
    _flatten_schema,
    normalize_llm_fields,
    _renumber_chunk_fields,
    extract_full_form_fields_chunked,
    extract_section_fields,
)
from app.schemas.forms import Nabl151Form, Nabl159Form


def testnormalize_llm_fields_defaults_confidence_when_a_local_model_omits_it():
    """This is the actual root cause of a real crash seen with Ollama
    (qwen2.5:3b): under Ollama's generic `format: json` grammar (valid JSON,
    not a specific schema), the model sometimes omits "confidence" entirely
    on some entries, and pipeline.py indexes it unconditionally as f["confidence"]."""
    raw = [
        {"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5"},  # no confidence at all
        {"field": "organisation.pan_number", "value": "AAECP1234L", "confidence": 0.9},
        {"field": "organisation.tan_number", "value": "DELP12345E", "confidence": "high"},  # wrong type
        {"value": "orphaned, no field name"},  # must be dropped entirely
        "not even a dict",  # must be dropped entirely
    ]

    normalized = normalize_llm_fields(raw)

    by_field = {f["field"]: f for f in normalized}
    assert len(normalized) == 3
    assert by_field["organisation.gst_number"]["confidence"] == 0.5
    assert by_field["organisation.pan_number"]["confidence"] == 0.9
    assert by_field["organisation.tan_number"]["confidence"] == 0.5


def test_flatten_schema_covers_flat_and_nested_and_repeating_fields():
    paths = _flatten_schema(Nabl151Form)

    # flat, one level deep into a nested object (OrgLegalInfo)
    assert "organisation.gst_number" in paths
    # two levels deep (SeniorManagement -> ContactInfo)
    assert "senior_management.head_of_laboratory.name" in paths
    # a repeating entity's own fields, templated with [i]
    assert "equipment[i].serial_number" in paths
    assert "staff[i].qualification" in paths
    assert "pt_ilc[i].performance_metric" in paths


def test_flatten_schema_skips_plain_string_lists():
    paths = _flatten_schema(Nabl151Form)
    # `disciplines: list[str]` isn't a repeating entity — nothing structured to ask for.
    assert not any(p.startswith("disciplines") for p in paths)


def test_flatten_schema_handles_a_different_form_shape_too():
    paths = _flatten_schema(Nabl159Form)
    assert "lab_details.project_name" in paths
    assert "technical_staff[i].name" in paths


def test_filter_to_requested_indexed_fields_accepts_any_index():
    templates = ["equipment[i].name", "organisation.gst_number"]
    returned = [
        {"field": "equipment[0].name", "value": "Multimeter", "confidence": 0.9},
        {"field": "equipment[7].name", "value": "Balance", "confidence": 0.8},
        {"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.95},
    ]

    kept = _filter_to_requested_indexed_fields(templates, returned)

    assert len(kept) == 3


def test_filter_to_requested_indexed_fields_drops_out_of_schema_fields():
    templates = ["equipment[i].name"]
    returned = [{"field": "equipment[0].made_up_field", "value": "x", "confidence": 0.5}]

    kept = _filter_to_requested_indexed_fields(templates, returned)

    assert kept == []


def test_renumber_chunk_fields_offsets_each_chunk_against_a_running_total():
    offsets: dict[str, int] = {}

    # Chunk 1 finds two equipment rows, locally indexed 0 and 1.
    chunk1 = [
        {"field": "equipment[0].name", "value": "Multimeter", "confidence": 0.9},
        {"field": "equipment[1].name", "value": "Balance", "confidence": 0.9},
        {"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.95},
    ]
    result1 = _renumber_chunk_fields(chunk1, offsets)
    assert [f["field"] for f in result1] == ["equipment[0].name", "equipment[1].name", "organisation.gst_number"]
    assert offsets == {"equipment": 2}

    # Chunk 2 independently starts counting from 0 again — must be remapped
    # to continue globally from where chunk 1 left off (index 2), not collide.
    chunk2 = [{"field": "equipment[0].name", "value": "Centrifuge", "confidence": 0.85}]
    result2 = _renumber_chunk_fields(chunk2, offsets)
    assert result2 == [{"field": "equipment[2].name", "value": "Centrifuge", "confidence": 0.85}]
    assert offsets == {"equipment": 3}


def test_extract_full_form_fields_chunked_merges_and_renumbers_across_chunks(monkeypatch):
    calls = []

    def fake_chunk_extract(field_templates, text):
        calls.append(text)
        if text == "page one":
            return [
                {"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9},
                {"field": "equipment[0].name", "value": "Multimeter", "confidence": 0.9},
            ]
        return [{"field": "equipment[0].name", "value": "Centrifuge", "confidence": 0.85}]

    monkeypatch.setattr(extractor, "_extract_full_form_chunk", fake_chunk_extract)

    fields, warnings = extract_full_form_fields_chunked("NABL_151", ["page one", "page two", "   "])

    assert calls == ["page one", "page two"]  # the blank third chunk is skipped entirely
    assert warnings == []
    assert {f["field"] for f in fields} == {
        "organisation.gst_number",
        "equipment[0].name",
        "equipment[1].name",
    }
    by_field = {f["field"]: f["value"] for f in fields}
    assert by_field["equipment[0].name"] == "Multimeter"
    assert by_field["equipment[1].name"] == "Centrifuge"


def test_fill_missing_flat_fields_backfills_nulls_but_leaves_indexed_fields_alone():
    templates = ["organisation.gst_number", "organisation.pan_number", "equipment[i].name"]
    returned = [{"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9},
                {"field": "equipment[0].name", "value": "Multimeter", "confidence": 0.8}]

    filled = _fill_missing_flat_fields(templates, returned)

    by_field = {f["field"]: f for f in filled}
    assert by_field["organisation.gst_number"]["value"] == "27ABCDE1234F1Z5"  # untouched
    assert by_field["organisation.pan_number"] == {"field": "organisation.pan_number", "value": None, "confidence": 0.0}
    assert "equipment[i].name" not in by_field  # the template itself is never a real field
    assert by_field["equipment[0].name"]["value"] == "Multimeter"  # indexed output passed through as-is
    assert len(filled) == 3  # no invented equipment[1] or similar


def test_extract_section_fields_asks_the_llm_to_omit_absent_fields_then_backfills_them(monkeypatch):
    """This is the actual fix for the Ollama timeout: the LLM only has to
    generate entries for fields it found, which is what makes a slow local
    model's response short — the caller (pipeline.py's verification retry
    pass) still sees the same one-entry-per-field contract as before,
    reconstructed here in Python at zero LLM cost."""
    captured = {}

    def fake_chunk_extract(field_templates, text):
        captured["templates"] = field_templates
        captured["text"] = text
        # Simulates the LLM omitting everything it didn't find, per the new instruction.
        return [{"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9}]

    monkeypatch.setattr(extractor, "_extract_full_form_chunk", fake_chunk_extract)

    templates = ["organisation.gst_number", "organisation.pan_number", "organisation.tan_number"]
    fields = extract_section_fields(templates, ["page one text", "page two text"])

    assert captured["text"] == "page one text\n\n--- next page ---\n\npage two text"
    by_field = {f["field"]: f["value"] for f in fields}
    assert by_field == {
        "organisation.gst_number": "27ABCDE1234F1Z5",
        "organisation.pan_number": None,
        "organisation.tan_number": None,
    }


def test_extract_full_form_fields_chunked_keeps_earlier_chunks_when_a_later_one_fails(monkeypatch):
    """One chunk's LLM chain being exhausted (every provider rate-limited)
    must not discard chunks that already succeeded — see pipeline.py's
    equivalent fix for the PDF/RAG path."""
    from app.llm.base import LlmProviderError

    def fake_chunk_extract(field_templates, text):
        if text == "page one":
            return [{"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9}]
        raise LlmProviderError("every configured LLM provider failed: groq: 429 | ollama: timeout")

    monkeypatch.setattr(extractor, "_extract_full_form_chunk", fake_chunk_extract)

    fields, warnings = extract_full_form_fields_chunked("NABL_151", ["page one", "page two"])

    assert fields == [{"field": "organisation.gst_number", "value": "27ABCDE1234F1Z5", "confidence": 0.9}]
    assert len(warnings) == 1
    assert "chunk 1" in warnings[0]


def test_extra_fields_is_not_offered_as_an_extractable_field():
    """extra_fields exists to HOLD open-ended output (documents/compiler.py),
    not to be extracted. Left in, retrieval.group_templates_by_section makes
    it its own section, costing one whole extra LLM call per whole-form
    document to ask the model for a field literally named "extra_fields"."""
    from app.documents import retrieval

    templates = extractor.form_field_templates("NABL_151")
    assert not [t for t in templates if "extra_fields" in t]
    assert "extra_fields" not in retrieval.group_templates_by_section(templates)
