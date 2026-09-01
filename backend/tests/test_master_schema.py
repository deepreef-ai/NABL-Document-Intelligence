import json

from app.master_schema.clustering import cluster_domain_keys
from app.master_schema.field_roles import classify_field_role
from app.master_schema.pipeline import build_master_schema
from app.master_schema.synonyms import are_likely_synonyms, normalize_key


# --------------------------------------------------------------------------- synonyms

def test_normalize_key_expands_known_abbreviations():
    assert normalize_key("hb") == "hemoglobin"
    assert normalize_key("haemoglobin") == "hemoglobin"
    assert normalize_key("snf") == "solids_not_fat"
    assert normalize_key("unrelated_key") == "unrelated_key"


def test_are_likely_synonyms_catches_token_overlap_and_spelling_variants():
    assert are_likely_synonyms("patient_name", "name_of_patient")
    assert are_likely_synonyms("ammonium_sulfate", "ammonium_sulphate")
    assert are_likely_synonyms("unit", "units")


def test_are_likely_synonyms_does_not_merge_semantically_related_but_distinct_fields():
    # Real, related concepts that must NOT be silently merged.
    assert not are_likely_synonyms("sample_received_on", "date_of_receipt")
    # Different analytes sharing only a boilerplate suffix must never look
    # similar because of that suffix alone.
    assert not are_likely_synonyms("sgot_conventional", "sgpt_conventional")
    assert not are_likely_synonyms("alkaline_phosphatase_reference_range", "bilirubin_total_reference_range")


# --------------------------------------------------------------------------- clustering

def test_cluster_domain_keys_groups_patient_name_variants_and_flags_for_review():
    clusters = cluster_domain_keys({"patient_name": 2, "name_of_patient": 1})
    by_canonical = {c.canonical_key: c for c in clusters}
    assert "patient_name" in by_canonical
    cluster = by_canonical["patient_name"]
    assert set(cluster.aliases) == {"patient_name", "name_of_patient"}
    assert cluster.status == "review"


def test_cluster_domain_keys_groups_hemoglobin_abbreviation_variants():
    clusters = cluster_domain_keys({"hb": 1, "haemoglobin": 2, "hemoglobin": 3})
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.canonical_key == "hemoglobin"
    assert set(cluster.aliases) == {"hb", "haemoglobin", "hemoglobin"}
    assert cluster.status == "review"
    assert cluster.total_frequency == 6


def test_cluster_domain_keys_leaves_a_single_key_approved():
    clusters = cluster_domain_keys({"sample_id": 5})
    assert len(clusters) == 1
    assert clusters[0].canonical_key == "sample_id"
    assert clusters[0].aliases == ["sample_id"]
    assert clusters[0].status == "approved"


def test_cluster_domain_keys_does_not_merge_unrelated_keys():
    clusters = cluster_domain_keys({"age": 3, "gender": 2, "fat_percent": 1})
    assert {c.canonical_key for c in clusters} == {"age", "gender", "fat_percent"}
    assert all(c.status == "approved" for c in clusters)


def test_cluster_domain_keys_handles_empty_input():
    assert cluster_domain_keys({}) == []


def test_cluster_domain_keys_does_not_conflate_bilirubin_fractions():
    """Regression test for a second real bug: stripping the generic "total"
    token left "bilirubin_total" as the single token {"bilirubin"}, which
    then half-matched both "bilirubin_direct" and "bilirubin_indirect" —
    clinically distinct values, not spelling variants of the same one."""
    clusters = cluster_domain_keys({"bilirubin_total": 2, "bilirubin_direct": 2, "bilirubin_indirect": 2})
    canonical_by_alias = {alias: c.canonical_key for c in clusters for alias in c.aliases}
    assert canonical_by_alias["bilirubin_total"] != canonical_by_alias["bilirubin_direct"]
    assert canonical_by_alias["bilirubin_total"] != canonical_by_alias["bilirubin_indirect"]
    assert canonical_by_alias["bilirubin_direct"] != canonical_by_alias["bilirubin_indirect"]


def test_cluster_domain_keys_does_not_bridge_unrelated_analytes_through_a_generic_hub():
    """Regression test for a real bug found against the actual dataset: a
    bare "reference_range" key plus several "<analyte>_reference_range"
    variants transitively merged completely unrelated lab parameters
    (alkaline_phosphatase, bilirubin_total, total_protein) into one cluster,
    purely because they all share the boilerplate "reference_range" suffix."""
    clusters = cluster_domain_keys({
        "reference_range": 1,
        "alkaline_phosphatase_reference_range": 1,
        "bilirubin_total_reference_range": 1,
        "total_protein_reference_range": 1,
        "alkaline_phosphatase": 2,
        "bilirubin_total": 2,
        "total_protein": 2,
    })
    canonical_by_alias = {alias: c.canonical_key for c in clusters for alias in c.aliases}
    assert canonical_by_alias["alkaline_phosphatase_reference_range"] != canonical_by_alias["bilirubin_total_reference_range"]
    assert canonical_by_alias["alkaline_phosphatase_reference_range"] != canonical_by_alias["total_protein_reference_range"]
    assert canonical_by_alias["bilirubin_total_reference_range"] != canonical_by_alias["total_protein_reference_range"]


# --------------------------------------------------------------------------- field_roles

def test_classify_field_role_document_field_vs_table_column_vs_parameter():
    assert classify_field_role("patient_name") == "document_field"
    assert classify_field_role("sample_id") == "document_field"
    assert classify_field_role("reference_range") == "table_column"
    assert classify_field_role("unit") == "table_column"
    assert classify_field_role("hemoglobin") == "parameter"
    assert classify_field_role("fat_percent") == "parameter"


# --------------------------------------------------------------------------- pipeline: build_master_schema

def _write_domain_keys(tmp_path, data):
    schema_dir = tmp_path / "schema_discovery"
    schema_dir.mkdir()
    (schema_dir / "domain_keys.json").write_text(json.dumps(data), encoding="utf-8")
    return schema_dir


def test_build_master_schema_groups_by_domain_and_covers_all_seven_domains(tmp_path):
    schema_dir = _write_domain_keys(tmp_path, {
        "medical": {"sample_count": 1, "key_frequency": {"patient_name": 1, "hb": 1, "haemoglobin": 1}},
    })
    master_schema, mapping_entries = build_master_schema(schema_dir)

    assert set(master_schema["domains"].keys()) == {
        "medical", "milk", "food", "water", "soil", "chemical", "other",
    }
    assert master_schema["domains"]["medical"]["keys"] == ["hemoglobin", "patient_name"]
    assert master_schema["domains"]["water"]["keys"] == []
    assert all(e.domain == "medical" for e in mapping_entries)


def test_build_master_schema_identifies_common_and_domain_specific_keys(tmp_path):
    schema_dir = _write_domain_keys(tmp_path, {
        "medical": {"sample_count": 1, "key_frequency": {"sample_id": 1, "patient_name": 1}},
        "milk": {"sample_count": 1, "key_frequency": {"sample_id": 1, "fat_percent": 1}},
        "food": {"sample_count": 1, "key_frequency": {"sample_id": 1, "moisture": 1}},
    })
    master_schema, _ = build_master_schema(schema_dir)

    assert master_schema["common_keys"] == ["sample_id"]
    assert master_schema["common_keys_by_domain"]["sample_id"] == ["food", "medical", "milk"]
    assert master_schema["domain_specific_keys"]["medical"] == ["patient_name"]
    assert master_schema["domain_specific_keys"]["milk"] == ["fat_percent"]
    assert master_schema["domain_specific_keys"]["food"] == ["moisture"]


def test_build_master_schema_key_mapping_entry_matches_requested_shape(tmp_path):
    schema_dir = _write_domain_keys(tmp_path, {
        "medical": {"sample_count": 2, "key_frequency": {"patient_name": 2, "name_of_patient": 1}},
    })
    _, mapping_entries = build_master_schema(schema_dir)

    entry = next(e for e in mapping_entries if e.canonical_key == "patient_name")
    entry_dict = entry.to_json_dict()
    assert entry_dict["canonical_key"] == "patient_name"
    assert set(entry_dict["aliases"]) == {"patient_name", "name_of_patient"}
    assert entry_dict["domain"] == "medical"
    assert entry_dict["status"] == "review"


def test_build_master_schema_handles_missing_domain_keys_file(tmp_path):
    schema_dir = tmp_path / "schema_discovery"
    schema_dir.mkdir()
    master_schema, mapping_entries = build_master_schema(schema_dir)
    assert mapping_entries == []
    assert all(info["keys"] == [] for info in master_schema["domains"].values())
