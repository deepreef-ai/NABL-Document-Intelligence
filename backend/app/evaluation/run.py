"""CLI: python -m app.evaluation.run --gold G --pred P --out report.json"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation import metrics as m


def load_documents(path: str) -> dict:
    """Load {doc_id: doc} from a JSON file ({"documents":[...]} or one doc) or a
    directory of such files. Each doc needs an "id"."""
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    docs = {}
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        items = data["documents"] if isinstance(data, dict) and "documents" in data else [data]
        for d in items:
            docs[d["id"]] = d
    return docs


def _rate(a, b):
    return a / b if b else None


def evaluate(gold: dict, pred: dict) -> dict:
    ids = sorted(set(gold) & set(pred))
    ce = ref_chars = we = ref_words = 0
    matched = total = ver = wrong = flagged = npred = 0
    lat: dict = {}
    for i in ids:
        g, p = gold[i], pred[i]
        if "text" in g and "text" in p:
            ce += m.levenshtein(g["text"], p["text"])
            ref_chars += len(g["text"])
            we += m.levenshtein(g["text"].split(), p["text"].split())
            ref_words += len(g["text"].split())
        gf, pf = g.get("fields", {}), p.get("fields", {})
        r = m.field_exact_match(gf, pf)
        matched += r["matched"]
        total += r["total"]
        r = m.false_verification_rate(gf, pf)
        ver += r["verified"]
        wrong += r["wrong_verified"]
        r = m.review_rate(pf)
        flagged += r["flagged"]
        npred += r["total"]
        for stage, ms in p.get("latency_ms", {}).items():
            lat.setdefault(stage, []).append(ms)

    return {
        "documents_evaluated": len(ids),
        "documents_missing_pred": sorted(set(gold) - set(pred)),
        "character_error_rate": _rate(ce, ref_chars),
        "word_error_rate": _rate(we, ref_words),
        "field_exact_match": {"matched": matched, "total": total, "rate": _rate(matched, total)},
        "false_verification_rate": {"wrong_verified": wrong, "verified": ver, "rate": _rate(wrong, ver)},
        "review_rate": {"flagged": flagged, "total": npred, "rate": _rate(flagged, npred)},
        "latency_ms": m.latency_summary(lat),
    }


def _f(x):
    return "n/a" if x is None else f"{x:.4f}"


def format_text(r: dict) -> str:
    fm, fv, rv = r["field_exact_match"], r["false_verification_rate"], r["review_rate"]
    lines = [
        f"documents evaluated: {r['documents_evaluated']}",
        f"CER: {_f(r['character_error_rate'])}   WER: {_f(r['word_error_rate'])}",
        f"field exact match: {fm['matched']}/{fm['total']} = {_f(fm['rate'])}",
        f"false verification: {fv['wrong_verified']}/{fv['verified']} = {_f(fv['rate'])}",
        f"review rate: {rv['flagged']}/{rv['total']} = {_f(rv['rate'])}",
    ]
    for s, v in r["latency_ms"].items():
        lines.append(f"latency {s}: n={v['count']} p50={v['p50']} p95={v['p95']} ms")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    report = evaluate(load_documents(a.gold), load_documents(a.pred))
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
