"""Small, hand-curated, deliberately non-exhaustive helpers for detecting
likely duplicate/synonymous discovered keys (Step 4, requirement 4). These
only ever produce CANDIDATES — see clustering.py for how a cluster's status
ends up "review" unless it's a single, unambiguous key.
"""
from __future__ import annotations

from difflib import SequenceMatcher

# Exact-key -> normalized-form lookups for well-known lab-report
# abbreviations/synonyms actually seen in this project's dataset. Extend as
# new ones turn up during review — intentionally a living, incomplete list,
# not a claim of completeness.
ABBREVIATION_EXPANSIONS: dict[str, str] = {
    "no": "number",
    "qty": "quantity",
    "addr": "address",
    "dob": "date_of_birth",
    "mfg": "manufacturing",
    "mfg_date": "manufacturing_date",
    "exp": "expiry",
    "exp_date": "expiry_date",
    "p_h": "ph",
    # medical
    "hb": "hemoglobin",
    "hgb": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "heamoglobin": "hemoglobin",
    "sgot": "ast",
    "aspartate_aminotransferase_ast_sgot": "ast",
    "sgot_aspartate_transaminase_ast": "ast",
    "sgpt": "alt",
    "alanine_aminotransferase_alt_sgpt": "alt",
    "sgpt_alanine_transaminase_alt": "alt",
    "wbc": "white_blood_cell_count",
    "rbc": "red_blood_cell_count",
    "bp": "blood_pressure",
    # milk
    "snf": "solids_not_fat",
    "milk_solids_not_fat": "solids_not_fat",
    # water
    "tds": "total_dissolved_solids",
    "bod": "biochemical_oxygen_demand",
    "cod": "chemical_oxygen_demand",
    # soil / chemical
    "npk": "nitrogen_phosphorus_potassium",
    "oc": "organic_carbon",
    "ec": "electrical_conductivity",
}

# Structural/boilerplate tokens that recur across many otherwise-unrelated
# fields in a lab report — a shared token from this set must never be what
# makes two keys look similar, or a bare key like "reference_range" (or
# "serum_biuret_conventional" vs "serum_bromocresolpurple_conventional")
# becomes a hub that transitively unions completely different analytes
# together (observed for real: alkaline_phosphatase, bilirubin_total, and
# total_protein all landed in one cluster purely because each has a
# "..._reference_range" variant). Removed from BOTH sides before comparing —
# see are_likely_synonyms.
_GENERIC_TOKENS = {
    "date", "no", "number", "name", "code", "id", "type", "of", "the", "and",
    "by", "on", "at", "for", "a", "total", "reference", "range", "interval",
    "conventional", "ip", "serum", "urine", "result", "results", "value",
    "values", "unit", "units", "method", "methods", "test", "tests",
    "parameter", "parameters",
}

# 0.6, not the more obvious 0.5: a 1-token set can never score above 0.5
# against a 2-token set no matter what that second token is (e.g.
# "bilirubin_total" filtered down to just {"bilirubin"} half-matching both
# {"bilirubin","direct"} and {"bilirubin","indirect"} — two clinically
# different values). 0.6 forces either an exact filtered-token match or
# genuine 2-vs-3-token overlap, not a single coincidental shared word.
_TOKEN_JACCARD_THRESHOLD = 0.6
_SEQUENCE_RATIO_THRESHOLD = 0.88


def normalize_key(key: str) -> str:
    return ABBREVIATION_EXPANSIONS.get(key, key)


def tokens(key: str) -> set[str]:
    return {t for t in key.split("_") if t}


def are_likely_synonyms(key_a: str, key_b: str, extra_generic_tokens: frozenset[str] = frozenset()) -> bool:
    """True only for pairs worth flagging as duplicate CANDIDATES — deliberately
    conservative. Two fields that are only semantically related (e.g.
    "sample_received_on" vs "date_of_receipt") deliberately stay separate:
    under-clustering just leaves two reviewable domain-specific keys, while
    over-clustering silently loses a real distinction, which is the failure
    mode this stage exists to avoid.

    Both the token-overlap AND the spelling-similarity checks run on keys
    with generic tokens stripped out first — comparing the raw, unstripped
    strings let pairs like "sgot_conventional"/"sgpt_conventional" look
    almost identical purely because of their shared "_conventional" suffix,
    even though SGOT and SGPT are different tests. `extra_generic_tokens`
    lets clustering.py add domain-specific boilerplate it detects
    dynamically (a token appearing in an unusually large share of that
    domain's keys), on top of this fixed list."""
    if key_a == key_b:
        return True

    generic = _GENERIC_TOKENS | extra_generic_tokens
    tokens_a = tokens(key_a) - generic
    tokens_b = tokens(key_b) - generic

    if not tokens_a and not tokens_b:
        # Both keys are made entirely of generic/boilerplate tokens (e.g.
        # "unit" vs "units") — nothing distinguishing to compare, so the raw
        # strings are all there is.
        return SequenceMatcher(None, key_a, key_b).ratio() >= _SEQUENCE_RATIO_THRESHOLD
    if not tokens_a or not tokens_b:
        # One side is all boilerplate, the other has real content — never
        # treat that as a match (this is exactly the failure mode that
        # let a bare "reference_range" bridge unrelated analytes together).
        return False

    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    if jaccard >= _TOKEN_JACCARD_THRESHOLD:
        return True

    filtered_a = "_".join(sorted(tokens_a))
    filtered_b = "_".join(sorted(tokens_b))
    return SequenceMatcher(None, filtered_a, filtered_b).ratio() >= _SEQUENCE_RATIO_THRESHOLD
