"""Rule-based, domain-agnostic key-value extraction from ALREADY-NORMALIZED
document text (dataset_normalization's output). No LLM, no classification —
pure text-pattern heuristics, since the actual next-stage need right now is
turning "Cane Sugar / - / Absent / Shall be absent" into a real key-value
pair, and that table shape (numbered rows, a parameter name, a result, an
optional method/unit, an optional limit) is common across domains (milk,
food, water, soil, ...), not specific to one.

Operates on dataset_normalization's NormalizedDocument JSON files as input —
this is deliberately a separate stage/package, not folded into
dataset_normalization itself, matching the pipeline direction: normalize ->
key-value extraction -> (later) domain classification -> (later)
domain-specific/LLM extraction.
"""
