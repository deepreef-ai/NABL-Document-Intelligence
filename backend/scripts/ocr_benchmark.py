"""OCR-only benchmark: speed per stage and accuracy against an answer key.

Runs the same OCR code the graph uses for scanned pages (no LLM, no quota), so
an OCR change can be judged on numbers before it ships:

    .venv/bin/python scripts/ocr_benchmark.py <scanned.pdf> --key <answer-key.json> \
        [--pages 1-5,17] [--out results.json] [--compare previous.json]

Speed, per page: render, clean-up (preprocessing), OCR (detection +
recognition), text boxes, ms per box. Totals and p50/p95 per page.

Accuracy, per page, at the OCR level (what the extractor will be able to see):
- "exact" pages: each expected (test name, result) must appear on ONE OCR row
  line — every word of the name plus the result.
- "count" pages (long uniform tables): the number of row lines carrying each
  expected value must match the key.

The engine follows backend/.env (OCR_ENGINE, DEEPREEF_OCR_PATH, ...). The
report says whether deepreef-ocr ran with its pinned model or a stand-in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fold(text: str) -> str:
    return re.sub(r"[\s,]+", "", unicodedata.normalize("NFKC", str(text or "")).casefold())


def words(text: str) -> list[str]:
    return re.findall(r"[0-9a-z]+", unicodedata.normalize("NFKC", str(text or "")).casefold())


def parse_pages(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return [p for p in out if 1 <= p <= total]


def ocr_page(data: bytes, page_no: int, engine: str, dpi: int) -> dict:
    """Mirrors preprocess._ocr_scanned_page, with each stage timed."""
    from app.documents import pdf_utils

    if engine == "deepreef_local":
        from app.documents import deepreef_local

        eng = deepreef_local._engine()
        # The same render + clean-up functions the graph calls, so the
        # OCR_RENDER_TO_TARGET / OCR_CLEAN_SCAN_PROFILE switches are measured.
        t0 = time.perf_counter()
        png = deepreef_local.render_for_ocr(data, page_no, dpi)
        t_render = time.perf_counter() - t0
        t1 = time.perf_counter()
        image = deepreef_local.prepare_image(png, rendered_pdf_page=True)
        t_clean = time.perf_counter() - t1
        t2 = time.perf_counter()
        out = eng.extract(image, deepreef_local.SCRIPT, max_words=0)
        t_ocr = time.perf_counter() - t2
        lines = [ln for ln in (out.get("text_lines") or "").splitlines() if ln.strip()]
        boxes = int(out.get("region_count") or 0)
    else:
        from app.documents import local_ocr

        t0 = time.perf_counter()
        png = pdf_utils.rasterize_page(data, page_no - 1, dpi=dpi)
        t_render = time.perf_counter() - t0
        t1 = time.perf_counter()
        local_ocr._ensure_importable()
        from deepreef_ocr.preprocessor import decode_and_resize

        image, _ = decode_and_resize(png)
        t_clean = time.perf_counter() - t1
        t2 = time.perf_counter()
        result, _ = local_ocr._rapidocr_engine()(image)
        t_ocr = time.perf_counter() - t2
        lines = [r[1] for r in (result or [])]
        boxes = len(lines)
    return {"page": page_no, "render_s": t_render, "clean_s": t_clean, "ocr_s": t_ocr,
            "total_s": t_render + t_clean + t_ocr, "boxes": boxes, "lines": lines}


def score_page(page: dict, key: dict) -> tuple[int, int, list[str]]:
    """(found, expected, misses) for one page."""
    p = str(page["page"])
    lines = page["lines"]
    folded = [fold(ln) for ln in lines]
    if p in key.get("exact", {}):
        found, misses = 0, []
        for name, result in key["exact"][p]:
            ok = any(fold(result) in fl and all(w in fl for w in words(name)) for fl in folded)
            found += ok
            if not ok:
                misses.append(f"{name} = {result}")
        return found, len(key["exact"][p]), misses
    if p in key.get("count_pages", {}):
        spec = key["count_pages"][p]
        found, misses = 0, []
        for value, want in spec["values"].items():
            got = sum(1 for fl in folded if fold(value) in fl)
            found += min(got, want)
            if got < want:
                misses.append(f"{want - got} row(s) with {value} missing")
        return found, spec["rows"], misses
    return 0, 0, []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--key", required=True, help="answer key JSON")
    ap.add_argument("--pages", help="e.g. 1-5,17 (default: every scanned page)")
    ap.add_argument("--out", help="write per-page results JSON here")
    ap.add_argument("--compare", help="a previous --out JSON to diff against")
    ap.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args()

    import fitz
    from app.graph.config import get_graph_settings
    from app.graph.nodes.preprocess import _MIN_CHARS, _page_native_text, _quality_ok

    settings = get_graph_settings()
    engine = settings.ocr_engine
    dpi = args.dpi or settings.ocr_dpi
    data = open(args.pdf, "rb").read()
    key = json.load(open(args.key))
    with fitz.open(args.pdf) as doc:
        total = doc.page_count
        scanned = [i + 1 for i in range(total) if not _quality_ok(_page_native_text(doc[i]), _MIN_CHARS)]
    pages = [p for p in parse_pages(args.pages, total) if p in scanned]

    model = "rapidocr bundled model"
    if engine == "deepreef_local":
        from app.documents import deepreef_local

        report = deepreef_local.model_report()
        model = "deepreef-ocr " + (
            "STAND-IN model" if not report["genuine"]
            else "GENUINE pinned model" if report["variant"] == "fp32"
            else f"APPROVED {report['variant']} model")
    print(f"engine: {engine} ({model}), dpi {dpi}, {len(pages)} scanned page(s) of {total}, "
          f"render_to_target={settings.ocr_render_to_target}, clean_scan={settings.ocr_clean_scan_profile}")

    ocr_page(data, pages[0], engine, dpi)        # load models once, not timed
    rows = []
    t_all = time.perf_counter()
    for p in pages:
        r = ocr_page(data, p, engine, dpi)
        r["found"], r["expected"], r["misses"] = score_page(r, key)
        rows.append(r)
    wall = time.perf_counter() - t_all

    print(f"\n{'page':>4} {'render':>7} {'clean':>7} {'ocr':>7} {'total':>7} {'boxes':>6} {'ms/box':>7}  rows found")
    for r in rows:
        per_box = r["ocr_s"] / r["boxes"] * 1000 if r["boxes"] else 0
        acc = f"{r['found']}/{r['expected']}" if r["expected"] else "-"
        print(f"{r['page']:>4} {r['render_s']:6.2f}s {r['clean_s']:6.2f}s {r['ocr_s']:6.2f}s "
              f"{r['total_s']:6.2f}s {r['boxes']:>6} {per_box:6.0f}  {acc}"
              + (f"   MISSING: {'; '.join(r['misses'][:3])}" if r["misses"] else ""))
    tot = {k: sum(r[k] for r in rows) for k in ("render_s", "clean_s", "ocr_s", "total_s", "boxes", "found", "expected")}
    per_page = sorted(r["total_s"] for r in rows)
    p95 = per_page[min(len(per_page) - 1, int(round(0.95 * (len(per_page) - 1))))]
    print(f"\nTOTAL {len(rows)} pages: {wall:.1f}s wall | render {tot['render_s']:.1f}s, clean-up {tot['clean_s']:.1f}s, "
          f"ocr {tot['ocr_s']:.1f}s | per page p50 {statistics.median(per_page):.2f}s p95 {p95:.2f}s | "
          f"{tot['boxes']} boxes, {tot['ocr_s'] / max(1, tot['boxes']) * 1000:.0f} ms/box")
    if tot["expected"]:
        print(f"ACCURACY rows found by OCR: {tot['found']}/{tot['expected']} ({tot['found'] / tot['expected']:.1%})")

    result = {"engine": engine, "model": model, "dpi": dpi, "wall_s": wall, "totals": tot,
              "pages": [{k: v for k, v in r.items() if k != "lines"} | {"lines": r["lines"]} for r in rows]}
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"results written to {os.path.abspath(args.out)}")
    if args.compare:
        prev = json.load(open(args.compare))
        pt = prev["totals"]
        print(f"\nvs {args.compare}: wall {prev['wall_s']:.1f}s -> {wall:.1f}s "
              f"({(wall - prev['wall_s']) / prev['wall_s']:+.0%}); rows {pt['found']}/{pt['expected']} -> "
              f"{tot['found']}/{tot['expected']}")


if __name__ == "__main__":
    main()
