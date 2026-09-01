"""Canonical output domains for this stage, plus a cheap keyword heuristic
used ONLY to spread the representative LLM sample across domains (see
sampling.py) so a run doesn't accidentally draw, say, 40 milk reports in a
row out of a mixed dataset. It is never written to a document's own output —
the LLM's own `candidate_domain` answer is what's authoritative there.
"""
from __future__ import annotations

CANONICAL_DOMAINS = ["medical", "milk", "food", "water", "soil", "chemical", "other"]

_HEURISTIC_KEYWORDS: dict[str, list[str]] = {
    "medical": [
        "patient", "hemoglobin", "haemoglobin", "diagnosis", "specimen",
        "pathology", "physician", "doctor", "blood group", "wbc", "rbc",
        "platelet", "urine", "biopsy", "cytology", "prescription",
        "clinical", "hospital", "histopath", "cln",
    ],
    "milk": [
        "milk", "snf", "lactometer", "lactose", "adulteration", "dahi",
        "ghee", "paneer", "chilling", "dairy", "fat %", "solids-not-fat",
        "solids not fat",
    ],
    "food": [
        "fssai", "moisture", "ash content", "net weight", "ingredient",
        "nutrition", "shelf life", "edible oil", "nutritional",
        "food safety", "rancidity", "adulterant", "nutritional label",
    ],
    "water": [
        "turbidity", "bod", "cod", "tds", "hardness", "chlorine",
        "potable", "effluent", "wastewater", "drinking water",
        "residual chlorine", "alkalinity",
    ],
    "soil": [
        "soil sample", "npk", "nitrogen content", "phosphorus", "potassium",
        "organic carbon", "electrical conductivity", "soil fertility",
        "agricultural",
    ],
    "chemical": [
        "assay", "purity", "msds", "reagent", "titration", "molarity",
        "cas no", "chemical composition", "solvent", "concentration (%)",
    ],
}


def heuristic_domain(text: str) -> str:
    """Best-effort, no-LLM-call guess used purely to bucket documents for
    sampling. Ties/no-hits fall to "other" rather than guessing wrong with
    false confidence."""
    text_lower = text.lower()
    scores = {domain: sum(text_lower.count(kw) for kw in kws) for domain, kws in _HEURISTIC_KEYWORDS.items()}
    best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_domain if best_score > 0 else "other"
