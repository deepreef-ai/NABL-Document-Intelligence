"""Pipeline test suite: fixtures, step tracing and the PR-ready report.

Run it with `python scripts/run_suite.py` (fast) or `--full` (adds real OCR
and the real dataset). Every run writes:

    backend/test-reports/suite_summary.md    <- paste into the PR
    backend/test-reports/suite_results.json  <- machine-readable, for diffing

Production-safety rules this file enforces:
  * The deepreef-ocr repo is imported read-only: bytecode writing is turned
    off before anything can import it, so no __pycache__ lands in that repo.
  * The dataset is read in place; no file from it is copied anywhere.
  * No test ever calls the production OCR Lambda unless SUITE_LIVE_LAMBDA=1.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.dont_write_bytecode = True  # must precede any import of deepreef_ocr

import pytest  # noqa: E402

BACKEND = Path(__file__).resolve().parents[2]
REPORT_DIR = BACKEND / "test-reports"
DATASET_DIR = Path(os.getenv(
    "SUITE_DATASET_DIR",
    r"G:\Shared drives\Engineering\Projects\NABL Document Intelligence\dataset"))
DEEPREEF_DIR = Path(os.getenv("SUITE_DEEPREEF_OCR_PATH", r"D:\ocr_new\deepreef-ocr"))

LAYERS = {
    "test_units.py": "Unit",
    "test_flow.py": "Integration (pipeline flow)",
    "test_ocr_quality.py": "Integration (real OCR + dataset)",
    "test_gateway.py": "Fix 1: OCR gateway",
}

_RESULTS: list[dict] = []
_SESSION_START = time.monotonic()


# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------


def pytest_configure(config):
    for name, doc in [
        ("realocr", "runs a real OCR engine (seconds per page)"),
        ("dataset", "reads real files from the shared dataset (read-only)"),
        ("live_lambda", "calls the deployed OCR Lambda (opt-in: SUITE_LIVE_LAMBDA=1)"),
    ]:
        config.addinivalue_line("markers", f"{name}: {doc}")


def pytest_collection_modifyitems(config, items):
    skip_live = pytest.mark.skip(reason="set SUITE_LIVE_LAMBDA=1 to call the deployed Lambda")
    for item in items:
        if "live_lambda" in item.keywords and os.getenv("SUITE_LIVE_LAMBDA") != "1":
            item.add_marker(skip_live)


# --------------------------------------------------------------------------
# Isolation (same contract as tests/graph/conftest.py)
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from app.graph import database
    from app.graph.config import get_graph_settings

    s = get_graph_settings()
    monkeypatch.setattr(s, "graph_database_url", "", raising=False)
    monkeypatch.setattr(s, "graph_sqlite_path", str(tmp_path / "graph.db"), raising=False)
    monkeypatch.setattr(s, "checkpoint_db_path", str(tmp_path / "ckpt.db"), raising=False)
    monkeypatch.setattr(s, "use_docling", False, raising=False)
    monkeypatch.setattr("app.graph.llm._sleep", lambda attempt: None, raising=False)
    database.reset_engine()
    yield
    database.reset_engine()


# --------------------------------------------------------------------------
# Step tracing + metrics (feed the report)
# --------------------------------------------------------------------------


class Trace:
    def __init__(self, item):
        self.item = item
        self.steps: list[tuple[str, float]] = []
        self.metrics: dict = {}

    @contextmanager
    def step(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.steps.append((name, dt))
            print(f"  [step] {name:<38} {dt * 1000:9.1f} ms")

    def metric(self, **values):
        self.metrics.update(values)
        print("  [metric] " + ", ".join(f"{k}={v}" for k, v in values.items()))


@pytest.fixture
def trace(request):
    t = Trace(request.node)
    # Attached at setup (the lists are filled in place during the test), because
    # the call-phase report is logged before fixture teardown runs.
    request.node.user_properties.append(("steps", t.steps))
    request.node.user_properties.append(("metrics", t.metrics))
    yield t


# --------------------------------------------------------------------------
# OCR fixtures
# --------------------------------------------------------------------------


class FakeOcr:
    """Deterministic stand-in for every OCR engine, so flow tests are fast and
    test the PIPELINE's handling of OCR output rather than OCR accuracy.

    `pages` is consumed in call order; an Exception entry is raised instead.
    Every call is recorded with the engine it impersonated."""

    def __init__(self, pages=None, default="Fake OCR line one\nFake OCR line two value 42"):
        self.pages = list(pages or [])
        self.default = default
        self.calls: list[dict] = []

    def _next(self, engine: str, script: str):
        self.calls.append({"engine": engine, "script": script})
        item = self.pages.pop(0) if self.pages else self.default
        if isinstance(item, Exception):
            raise item
        from app.documents.ocr_client import OcrResult

        lines = [ln for ln in item.split("\n") if ln.strip()]
        return OcrResult(text=" ".join(lines), lines=lines,
                         confidence=0.93 if lines else 0.0, boxes=[None] * 0,
                         model_used=f"fake-{engine}", region_count=len(lines))

    def install(self, monkeypatch):
        from app.documents import local_ocr, ocr_client

        monkeypatch.setattr(local_ocr, "extract_english",
                            lambda image_bytes: self._next("rapidocr", "english"))
        monkeypatch.setattr(ocr_client.OcrClient, "__init__", lambda self_, *a, **k: None)
        monkeypatch.setattr(ocr_client.OcrClient, "extract",
                            lambda self_, image_bytes, script: self._next("lambda", script))
        _install_gateway_fake(self, monkeypatch)
        return self


def _install_gateway_fake(fake: FakeOcr, monkeypatch) -> None:
    """Fix 1 adds app.ocr_gateway; route its engines to the same fake."""
    try:
        from app.ocr_gateway import engines
    except ImportError:
        return

    def _engine(name):
        def run(image_bytes, script, settings):
            r = fake._next(name, script)
            return engines.EngineResult.from_legacy(r, engine=name)
        return run

    for name in ("deepreef_local", "lambda", "rapidocr"):
        monkeypatch.setitem(engines.ENGINES, name, _engine(name))


@pytest.fixture
def fake_ocr(monkeypatch):
    def _make(pages=None, **kw) -> FakeOcr:
        return FakeOcr(pages, **kw).install(monkeypatch)
    return _make


@pytest.fixture
def deepreef_path(monkeypatch):
    """Point the app at the real (read-only) deepreef-ocr checkout."""
    if not (DEEPREEF_DIR / "deepreef_ocr" / "preprocessor.py").is_file():
        pytest.skip(f"deepreef-ocr not found at {DEEPREEF_DIR} (set SUITE_DEEPREEF_OCR_PATH)")
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "deepreef_ocr_path", str(DEEPREEF_DIR), raising=False)
    return DEEPREEF_DIR


@pytest.fixture
def dataset_file():
    def _get(name: str) -> Path:
        p = DATASET_DIR / name
        if not p.is_file():
            pytest.skip(f"dataset file not available: {p}")
        return p
    return _get


@pytest.fixture
def recording_llm(monkeypatch):
    from tests.suite._support import RecordingLlm

    stub = RecordingLlm()
    for module in ("app.graph.llm", "app.graph.nodes.analysis", "app.graph.nodes.extraction",
                   "app.graph.nodes.normalize", "app.graph.nodes.conflicts",
                   "app.graph.nodes.mapping"):
        monkeypatch.setattr(f"{module}.call_structured", stub, raising=False)
    return stub


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        props = dict(report.user_properties)
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfail" if report.skipped else "xpass"
        reason = ""
        if outcome == "failed":
            reason = str(report.longrepr).strip().splitlines()[-1][:180]
        elif outcome in ("xfail", "xpass"):
            reason = report.wasxfail
        elif outcome == "skipped":
            reason = str(report.longrepr[-1]) if isinstance(report.longrepr, tuple) else ""
        _RESULTS.append({
            "id": report.nodeid, "file": report.nodeid.split("::")[0].split("/")[-1],
            "outcome": outcome, "seconds": round(report.duration, 3), "reason": reason,
            "steps": props.get("steps", []), "metrics": props.get("metrics", {}),
        })


def _markdown(total_seconds: float) -> str:
    by = lambda o: [r for r in _RESULTS if r["outcome"] == o]  # noqa: E731
    failed = by("failed") + by("xpass")
    verdict = "FAIL" if failed else "PASS"
    lines = [
        f"### Pipeline test suite: **{verdict}**",
        "",
        f"`{len(by('passed'))} passed` | `{len(failed)} failed` | "
        f"`{len(by('xfail'))} known gaps (xfail)` | `{len(by('skipped'))} skipped` | "
        f"{total_seconds:.1f}s",
        "",
        "| Layer | Pass | Fail | Known gap | Skip | Time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fname, layer in LAYERS.items():
        rs = [r for r in _RESULTS if r["file"] == fname]
        if not rs:
            continue
        cnt = lambda o: sum(r["outcome"] == o for r in rs)  # noqa: E731
        lines.append(f"| {layer} | {cnt('passed')} | {cnt('failed') + cnt('xpass')} | "
                     f"{cnt('xfail')} | {cnt('skipped')} | {sum(r['seconds'] for r in rs):.1f}s |")

    quality = [r for r in _RESULTS if r["metrics"].get("exact_word_recall") is not None]
    if quality:
        lines += ["", "**OCR accuracy (exact, case-sensitive)**", "",
                  "| Case | Engine | Exact words | Dates | Numbers | Glued words | OCR time |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for r in quality:
            m = r["metrics"]
            case = r["id"].split("::")[-1]
            lines.append(
                f"| `{case}` | {m.get('engine', '-')} | {m['exact_word_recall']:.0%} | "
                f"{m.get('date_recall', 1):.0%} | {m.get('numbers_recall', 1):.0%} | "
                f"{m.get('glued_words', 0)} | {m.get('ocr_seconds', 0):.1f}s |")

    if failed:
        lines += ["", "**Failures**", ""]
        lines += [f"- `{r['id']}`: {r['reason']}" for r in failed]
    gaps = by("xfail")
    if gaps:
        lines += ["", "<details><summary>Known gaps tracked as xfail (expected, not regressions)"
                  "</summary>", ""]
        lines += [f"- `{r['id'].split('::')[-1]}`: {r['reason']}" for r in gaps]
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def pytest_sessionfinish(session, exitstatus):
    if not _RESULTS:
        return
    REPORT_DIR.mkdir(exist_ok=True)
    total = time.monotonic() - _SESSION_START
    (REPORT_DIR / "suite_summary.md").write_text(_markdown(total), encoding="utf-8")
    (REPORT_DIR / "suite_results.json").write_text(
        json.dumps({"seconds": round(total, 1), "results": _RESULTS}, indent=1), encoding="utf-8")


def pytest_terminal_summary(terminalreporter):
    if _RESULTS:
        terminalreporter.write_sep("=", "PR summary (also in backend/test-reports/suite_summary.md)")
        terminalreporter.write(_markdown(time.monotonic() - _SESSION_START))
