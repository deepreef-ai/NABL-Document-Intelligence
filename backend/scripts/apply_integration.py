#!/usr/bin/env python3
"""Re-apply the graph integration edits to pre-existing files. Idempotent.

Why this exists: every file that predates the graph work (main.py, App.tsx,
pipeline.py, chain.py, config.py, requirements.txt) has reverted at least once
during development, losing its integration edit. New files under app/graph/ and
the new frontend files have never reverted. Rather than hunt for which edit
vanished, run this — it checks each one and re-applies only what is missing.

    python scripts/apply_integration.py            # apply
    python scripts/apply_integration.py --check    # report only, exit 1 if any missing

If it reports nothing missing, the integration is intact.
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.abspath(os.path.join(ROOT, "..", "frontend"))
SNAPSHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "integration_snapshots")


class Edit:
    def __init__(self, path: str, name: str, marker: str, anchor: str, replacement: str,
                 count: int = 1):
        self.path = path
        self.name = name
        self.marker = marker      # present => already applied
        self.anchor = anchor      # text to replace
        self.replacement = replacement
        # How many occurrences to replace. The provider temperature edit lands
        # in three constructors that are textually identical, and replacing
        # only the first left Gemini and Groq still hardcoded.
        self.count = count

    def present(self, text: str) -> bool:
        return self.marker in text


class Snapshot:
    """A file that reverts WHOLESALE rather than losing one line.

    MEASURED: ReviewPage.tsx, client.ts and App.css came back as much older
    versions — App.css had lost every `rg-*` rule, not just the ones added
    last. Anchored edits cannot repair that: the anchors they need are
    themselves gone. So these files keep a pristine copy and are restored from
    it.

    Restoring only happens when the marker is ABSENT, i.e. the file has
    demonstrably lost the integration. A file that still has the marker is
    left alone however much else it has changed, so ordinary work on it is
    never clobbered. After deliberately changing one of these files, run
    `--snapshot` to record the new intended content.
    """

    def __init__(self, path: str, name: str, marker: str):
        self.path = path
        self.name = name
        self.marker = marker

    @property
    def store(self) -> str:
        return os.path.join(SNAPSHOTS, os.path.basename(self.path))

    def present(self, text: str) -> bool:
        return self.marker in text


EDITS: list[Edit] = [
    # ---- backend: register the router ------------------------------------
    Edit(
        os.path.join(ROOT, "app", "main.py"),
        "main.py: import graph router", "graph_documents",
        "from app.routers import chat, documents, review, wizard",
        "from app.routers import chat, documents, graph_documents, review, wizard",
    ),
    Edit(
        os.path.join(ROOT, "app", "main.py"),
        "main.py: include graph router", "include_router(graph_documents.router)",
        "app.include_router(chat.router)",
        "app.include_router(chat.router)\napp.include_router(graph_documents.router)",
    ),
    Edit(
        os.path.join(ROOT, "app", "main.py"),
        "main.py: init graph tables", "init_graph_db",
        "def on_startup() -> None:\n    init_db()",
        "def on_startup() -> None:\n    init_db()\n\n"
        "    # The graph keeps its own audit store (runs, fields, mappings,\n"
        "    # conflicts, errors, audit), separate from the app's operational\n"
        "    # tables: different lifecycle, different retention.\n"
        "    from app.graph.database import init_db as init_graph_db\n\n"
        "    init_graph_db()",
    ),
    # ---- backend: CORS ----------------------------------------------------
    Edit(
        os.path.join(ROOT, "app", "config.py"),
        "config.py: allow both dev origins", "127.0.0.1:5173",
        '    cors_origins: list[str] = ["http://localhost:5173"]',
        "    # Both spellings of the dev host. A browser treats http://localhost:5173\n"
        "    # and http://127.0.0.1:5173 as different origins, so allowing only one\n"
        "    # makes the UI silently fail to reach the API depending on how the dev\n"
        "    # server happened to be started.\n"
        '    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]',
    ),
    # ---- backend: chain fixes ---------------------------------------------
    Edit(
        os.path.join(ROOT, "app", "llm", "chain.py"),
        "chain.py: images kwarg only when present", 'extra = {"images"',
        "                raw = provider.generate(system, user_text, image, image_media_type, want_json=True, images=images)",
        "                # `images` is passed only when there are any: not every provider\n"
        "                # accepts the parameter (Nova does, Gemini and Groq do not), and\n"
        "                # sending images=None to one that lacks it raises TypeError and\n"
        "                # takes the chain down for a text-only call it could have served.\n"
        '                extra = {"images": images} if images else {}\n'
        "                raw = provider.generate(system, user_text, image, image_media_type, want_json=True, **extra)",
    ),
    Edit(
        os.path.join(ROOT, "app", "llm", "chain.py"),
        "chain.py: track provider + usage", "self.last_provider",
        "        self._status: dict[str, _ProviderStatus] = {p.name: _ProviderStatus() for p in providers}",
        "        self._status: dict[str, _ProviderStatus] = {p.name: _ProviderStatus() for p in providers}\n"
        "        # Which provider served the most recent successful call, and what it\n"
        "        # reported spending. Read by app/graph/llm.py to attribute cost per\n"
        "        # node; None until the first success.\n"
        "        self.last_usage: dict | None = None\n"
        "        self.last_provider: str | None = None",
    ),
    # NOTE: the /extraction debug route used to be re-applied here. It was
    # removed on purpose — the product is two screens, upload and review — so
    # re-adding it would fight the intended App.tsx rather than repair it.

    # ---- backend: deterministic provider temperature -----------------------
    # Without a settable temperature the providers hardcoded 0.2 and mapping
    # was non-deterministic: the SAME document filled six named form slots on
    # one run and none on the next. app/graph/llm.py pins it to 0.0.
    Edit(
        os.path.join(ROOT, "app", "llm", "providers.py"),
        "providers.py: temperature parameter", "temperature: float = 0.2)",
        "max_tokens: int = 8192):",
        "max_tokens: int = 8192, temperature: float = 0.2):",
        count=3,
    ),
    Edit(
        os.path.join(ROOT, "app", "llm", "providers.py"),
        "providers.py: store temperature", "self.temperature = temperature",
        "self.max_tokens = max_tokens",
        "self.max_tokens = max_tokens\n"
        "        # Settable so the graph can pin 0.0. Left hardcoded, the SAME\n"
        "        # document filled six named form slots on one run and none on the\n"
        "        # next, and no amount of prompt work makes that reproducible.\n"
        "        self.temperature = temperature",
        count=3,
    ),
    Edit(
        os.path.join(ROOT, "app", "llm", "providers.py"),
        "providers.py: nova uses it", '"temperature": self.temperature, "maxTokens"',
        'inferenceConfig={"temperature": 0.2, "maxTokens": self.max_tokens}',
        'inferenceConfig={"temperature": self.temperature, "maxTokens": self.max_tokens}',
    ),
    Edit(
        os.path.join(ROOT, "app", "llm", "providers.py"),
        "providers.py: gemini uses it", '"temperature": self.temperature, "maxOutputTokens"',
        'generation_config: dict = {"temperature": 0.2, "maxOutputTokens": self.max_tokens}',
        'generation_config: dict = {"temperature": self.temperature, "maxOutputTokens": self.max_tokens}',
    ),
    Edit(
        os.path.join(ROOT, "app", "llm", "providers.py"),
        "providers.py: groq uses it", '            "temperature": self.temperature,',
        '            "temperature": 0.2,\n',
        '            "temperature": self.temperature,\n',
    ),

    # ---- backend: the source sub-heading reaches the review screen ---------
    # Which heading a field sat under in the original document. Without it the
    # review screen can only show one flat list, which is exactly what the
    # two-level form replaced.
    Edit(
        os.path.join(ROOT, "app", "documents", "grounding.py"),
        "grounding.py: FieldResult.section", "section: str = \"\"",
        '    source: str = "llm"',
        '    source: str = "llm"\n'
        '    #: Sub-heading this field sits under in the source document — "URINE\n'
        '    #: CHEMISTRY", "Senior Management". Lets the review screen group fields\n'
        "    #: the way the document does instead of showing one flat list. Empty\n"
        "    #: when the extractor could not attribute one.\n"
        '    section: str = ""',
    ),
    Edit(
        os.path.join(ROOT, "app", "models.py"),
        "models.py: section column", "section: Mapped[str | None]",
        '    source: Mapped[str] = mapped_column(String, default="llm")',
        '    source: Mapped[str] = mapped_column(String, default="llm")\n'
        "    # The sub-heading this field sits under in the source document —\n"
        '    # "URINE CHEMISTRY", "Senior Management". Carried through so the review\n'
        "    # screen can group fields the way the document itself does rather than\n"
        "    # showing one flat list. Nullable: rows written before this existed.\n"
        "    section: Mapped[str | None] = mapped_column(String, nullable=True)",
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "documents.py"),
        "documents.py: persist section", 'section=getattr(f, "section"',
        "                source=f.source,",
        "                source=f.source,\n"
        '                section=getattr(f, "section", "") or None,',
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "documents.py"),
        "documents.py: serialise section", '"section": f.section',
        '                "source": f.source,',
        '                "source": f.source,\n'
        '                "section": f.section,',
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "review.py"),
        "review.py: serialise section", '"section": f.section',
        '                        "source_bbox": f.source_bbox,',
        '                        "source_bbox": f.source_bbox,\n'
        "                        # The sub-heading the field sat under in the source\n"
        "                        # document. The review screen files each field under\n"
        "                        # it, so leaving it out here collapsed every section\n"
        "                        # into one unlabelled block however well it was read.\n"
        '                        "section": f.section,',
    ),
    # ---- backend: the gold-dataset shape -----------------------------------
    # The labelled_dataset records are the output format, and structured.py's
    # route_field is the single place that decides which section a field lands
    # in. These edits are what carry that answer out to the API and the form;
    # without them the review screen classifies for itself and the two
    # disagree about the same document.
    Edit(
        os.path.join(ROOT, "app", "models.py"),
        "models.py: field_group column", "field_group: Mapped[str | None]",
        "    section: Mapped[str | None] = mapped_column(String, nullable=True)\n"
        "    accepted: Mapped[bool] = mapped_column(Boolean, default=False)",
        "    section: Mapped[str | None] = mapped_column(String, nullable=True)\n"
        "    # Which gold-dataset section this field belongs to — \"lab_info\",\n"
        "    # \"patient_info\", \"signatories\". Named field_group because `group`\n"
        "    # is a reserved word in SQL.\n"
        "    field_group: Mapped[str | None] = mapped_column(String, nullable=True)\n"
        "    accepted: Mapped[bool] = mapped_column(Boolean, default=False)",
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "documents.py"),
        "documents.py: persist field_group", "field_group=getattr(f",
        '                section=getattr(f, "section", "") or None,',
        '                section=getattr(f, "section", "") or None,\n'
        '                field_group=getattr(f, "group", "") or None,',
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "review.py"),
        "review.py: serialise group", '"group": f.field_group',
        '                        "section": f.section,',
        '                        "section": f.section,\n'
        '                        "group": f.field_group,',
    ),
    # ---- backend: the gold-shaped record ------------------------------------
    # The labelled_dataset shape is the output format. These carry it out of
    # the graph and into the API: without them the review form classifies for
    # itself and the exported JSON disagrees with what is on screen.
    Edit(
        os.path.join(ROOT, "app", "documents", "grounding.py"),
        "grounding.py: FieldResult.group", "group: str = \"\"",
        '    section: str = ""',
        '    section: str = ""\n'
        '    #: Which gold-dataset section this field belongs to — "lab_info",\n'
        '    #: "patient_info", "signatories". app/graph/structured.py\'s\n'
        "    #: route_field is the one place that decides; carrying the answer here\n"
        "    #: keeps the review form and the exported JSON filing the same field in\n"
        "    #: the same place.\n"
        '    group: str = ""',
    ),
    Edit(
        os.path.join(ROOT, "app", "documents", "grounding.py"),
        "grounding.py: PipelineResult.tests", "structured: dict = field(default_factory=dict)",
        "    page_count: int | None = None",
        "    page_count: int | None = None\n"
        "    # Result-table rows in the gold dataset's shape: one dict per analyte\n"
        "    # with test_name / result / unit / reference_range kept SEPARATE. A\n"
        "    # results table is not a list of fields — flattened into scalars it\n"
        '    # becomes ph = "6.5 5-9", with nothing left to sort or compare.\n'
        "    tests: list[dict] = field(default_factory=list)\n"
        "    # The whole document in the gold dataset's shape, ready to diff\n"
        "    # against a labelled record.\n"
        "    structured: dict = field(default_factory=dict)",
    ),
    Edit(
        os.path.join(ROOT, "app", "models.py"),
        "models.py: structured/tests columns", "structured_json: Mapped[dict | None]",
        '    # "uploaded" -> "processing" -> "extracted" -> "failed"\n'
        '    status: Mapped[str] = mapped_column(String, default="uploaded")',
        "    # The document in the gold dataset's shape. Stored whole rather than\n"
        "    # reassembled from extracted_fields on request: those rows have been\n"
        '    # through review edits, and what a person downloads as "the extraction"\n'
        "    # should be one coherent record, not a reconstruction that drifts from\n"
        "    # what the graph actually decided.\n"
        "    structured_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)\n"
        "    # Result-table rows, one dict per analyte, columns kept separate.\n"
        "    tests_json: Mapped[list | None] = mapped_column(JSON, nullable=True)\n"
        '    # "uploaded" -> "processing" -> "extracted" -> "failed"\n'
        '    status: Mapped[str] = mapped_column(String, default="uploaded")',
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "documents.py"),
        "documents.py: persist the gold record", "document.structured_json =",
        '    document.error = "; ".join(result.extraction_warnings) or None\n',
        '    document.error = "; ".join(result.extraction_warnings) or None\n'
        "    # The gold-shaped record and its results table, stored as the pipeline\n"
        "    # produced them. getattr keeps the legacy extraction paths working:\n"
        "    # they build a PipelineResult without either.\n"
        '    document.structured_json = getattr(result, "structured", None) or None\n'
        '    document.tests_json = getattr(result, "tests", None) or None\n',
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "documents.py"),
        "documents.py: serialise tests", '"tests": document.tests_json',
        '        "error": document.error,\n        "fields": [',
        '        "error": document.error,\n'
        "        # A results table is not a list of fields, so it travels as rows\n"
        "        # with its columns intact rather than flattened into scalars.\n"
        '        "tests": document.tests_json or [],\n'
        '        "fields": [',
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "documents.py"),
        "documents.py: serialise group", '"group": f.field_group',
        '                "section": f.section,',
        '                "section": f.section,\n'
        '                "group": f.field_group,',
    ),
    # ---- backend: unlock without the wizard ---------------------------------
    # The upload endpoint refuses an application still in "eligibility", and
    # the product no longer shows a start step, so without this the one screen
    # the user has cannot accept a file.
    Edit(
        os.path.join(ROOT, "app", "routers", "wizard.py"),
        "wizard.py: skip_eligibility flag", "skip_eligibility",
        "class CreateApplicationRequest(BaseModel):\n    form_type: NablFormType\n",
        "class CreateApplicationRequest(BaseModel):\n"
        "    form_type: NablFormType\n"
        "    #: Create the application already unlocked, skipping the eligibility\n"
        '    #: wizard. The upload endpoint refuses anything still in "eligibility"\n'
        "    #: status, so a deployment that does not present the wizard has no other\n"
        "    #: way to reach upload. Off by default: the gate stays the norm.\n"
        "    skip_eligibility: bool = False\n",
    ),
    Edit(
        os.path.join(ROOT, "app", "routers", "wizard.py"),
        "wizard.py: honour skip_eligibility", "Eligibility wizard skipped",
        "    state, message = WizardEngine(db).start(application)",
        "    if body.skip_eligibility:\n"
        "        # Unlock without asking the prerequisite questions. The caller has\n"
        "        # taken responsibility for eligibility; record that plainly rather\n"
        "        # than leaving it looking as though the wizard passed.\n"
        '        application.status = "unlocked"\n'
        "        db.commit()\n"
        "        db.refresh(application)\n"
        "        return {\n"
        '            "application": _serialize(application),\n'
        '            "state": None,\n'
        '            "message": "Eligibility wizard skipped; upload is unlocked.",\n'
        "        }\n\n"
        "    state, message = WizardEngine(db).start(application)",
    ),
    Edit(
        os.path.join(ROOT, "app", "db.py"),
        "db.py: run column migration", "_add_missing_columns()",
        "    Base.metadata.create_all(bind=engine)",
        "    Base.metadata.create_all(bind=engine)\n\n"
        "    # create_all only creates missing TABLES, never alters an existing\n"
        "    # one, so a database that predates a column keeps working until the\n"
        '    # first query touches it and dies with "no such column".\n'
        "    _add_missing_columns()",
    ),
]


#: Files that come back as WHOLESALE older versions rather than losing a line.
#: MEASURED: App.css returned with every `rg-*` rule gone, not just the newest;
#: ReviewPage.tsx returned rendering a flat field grid from before the review
#: form existed. There is no anchor left to repair those against, so they are
#: restored from a pristine copy instead.
#:
#: The marker is the test for "has this file lost the integration". A file that
#: still has its marker is never touched, so ordinary work on these files is
#: safe; run `--snapshot` after deliberately changing one.
SNAPSHOTS_TRACKED: list[Snapshot] = [
    Snapshot(os.path.join(FRONTEND, "src", "pages", "ReviewPage.tsx"),
             "ReviewPage.tsx: results table + JSON link", "TestResultsTable"),
    Snapshot(os.path.join(FRONTEND, "src", "api", "client.ts"),
             "client.ts: group, tests, structured URL", "structuredJsonUrl"),
    Snapshot(os.path.join(FRONTEND, "src", "App.css"),
             "App.css: review form + results table styles", "rg-subheading"),
    Snapshot(os.path.join(FRONTEND, "src", "App.tsx"),
             "App.tsx: two screens, upload first", 'path="/" element={<UploadPage'),
    Snapshot(os.path.join(FRONTEND, "src", "pages", "UploadPage.tsx"),
             "UploadPage.tsx: silently acquires an application", "DEFAULT_FORM_TYPE"),
    Snapshot(os.path.join(ROOT, "app", "db.py"),
             "db.py: the column migration itself", "def _add_missing_columns"),
    Snapshot(os.path.join(ROOT, "app", "benchmark", "compare.py"),
             "compare.py: value normalisation", "_SEPARATORS.sub"),
    Snapshot(os.path.join(ROOT, "tests", "test_pipeline.py"),
             "test_pipeline.py: legacy-path fixture", "_legacy_path"),
    Snapshot(os.path.join(ROOT, "tests", "test_lab_report_extraction.py"),
             "test_lab_report_extraction.py: import path", "from app.documents.lab_report import"),
]


# pipeline.py's graph branch is large enough that it lives here as a block.
PIPELINE_MARKER = "run_graph_pipeline"
# The anchor sits AFTER the call-budget block, not after _guess_kind: a budget
# is created per document between the two, and anchoring above it put the graph
# branch before the budget existed.
PIPELINE_ANCHOR = """    budget = cb.CallBudget.from_settings(document_id)

    if kind == "pdf":"""
PIPELINE_REPLACEMENT = """    budget = cb.CallBudget.from_settings(document_id)

    # The agentic LangGraph workflow (app/graph/) is the extraction path.
    # graph_pipeline_enabled is a rollback lever only: set it false to restore
    # the legacy paths below without a deploy.
    if get_graph_settings().graph_pipeline_enabled:
        import tempfile

        from app.graph.adapter import run_graph_pipeline

        suffix = (
            ".pdf" if kind == "pdf"
            else ".docx" if kind == "docx"
            else _image_suffix(content_type)
        )
        tmp_path = None
        try:
            # The graph validates and reads a real file — encryption, page
            # count, readability — so the upload bytes are staged to disk.
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            # `filename` is the upload's own name; tmp_path is where it was
            # staged. Without this the gold-shaped output recorded
            # original_filename as "tmpn0yclbyd.pdf".
            return run_graph_pipeline(
                tmp_path, document_id=document_id, form_type=form_type,
                display_name=filename,
            )
        except Exception as exc:  # noqa: BLE001 — a graph failure must not take
            # uploads down; fall through to the legacy path with the reason recorded.
            log.warning("graph pipeline failed, falling back to the legacy path: %s", exc)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    if kind == "pdf":"""

REQUIREMENTS_BLOCK = """

# --- Agentic extraction workflow (app/graph/) --------------------------------
# LangGraph orchestrates the multi-agent document workflow: a shared typed
# state, conditional routing, bounded retry loops and a checkpoint after every
# node (which is what makes a long run resumable rather than restartable).
langgraph==1.2.11
# SQLite checkpointer. Swap for langgraph-checkpoint-postgres in production —
# app/graph/graph.py's build_checkpointer() is the only place that changes.
langgraph-checkpoint-sqlite==3.1.1
langchain-core==1.6.2
"""


def has_conflict_markers(text: str) -> bool:
    return any(line.startswith("<<<<<<< ") for line in text.splitlines())


def resolve_to_head(text: str) -> str:
    """Drop the non-HEAD side of any conflict. Only used on pipeline.py, where
    HEAD is the only side that compiles (the shared body below the conflict
    uses suffix/script/ocr_client and never raw_fields/media_type)."""
    out, mode = [], "normal"
    for line in text.splitlines(keepends=True):
        if line.startswith("<<<<<<< "):
            mode = "head"; continue
        if line.startswith("=======") and mode == "head":
            mode = "theirs"; continue
        if line.startswith(">>>>>>> ") and mode == "theirs":
            mode = "normal"; continue
        if mode == "theirs":
            continue
        out.append(line)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    ap.add_argument("--snapshot", action="store_true",
                    help="record the CURRENT content of the snapshot-tracked files as the "
                         "intended version. Run this after deliberately changing one of them.")
    a = ap.parse_args()

    if a.snapshot:
        os.makedirs(SNAPSHOTS, exist_ok=True)
        for snap in SNAPSHOTS_TRACKED:
            if not os.path.exists(snap.path):
                print(f"  !! missing file: {snap.path}")
                continue
            text = io.open(snap.path, encoding="utf-8").read()
            if not snap.present(text):
                print(f"  !! {snap.name}: marker absent, refusing to snapshot a reverted file")
                continue
            io.open(snap.store, "w", encoding="utf-8", newline="\n").write(text)
            print("  recorded", snap.name)
        return 0

    missing: list[str] = []
    applied: list[str] = []

    # --- pipeline.py (conflicts + graph branch) ----------------------------
    pipeline = os.path.join(ROOT, "app", "documents", "pipeline.py")
    text = io.open(pipeline, encoding="utf-8").read()
    original = text

    if has_conflict_markers(text):
        missing.append("pipeline.py: unresolved merge conflicts")
        if not a.check:
            text = resolve_to_head(text)
            applied.append("pipeline.py: conflicts resolved to HEAD")

    if PIPELINE_MARKER not in text:
        missing.append("pipeline.py: graph branch")
        if not a.check:
            if PIPELINE_ANCHOR not in text:
                print("  !! pipeline.py: anchor not found, cannot re-apply the graph branch")
            else:
                text = text.replace(PIPELINE_ANCHOR, PIPELINE_REPLACEMENT, 1)
                if "\nimport os\n" not in text:
                    text = text.replace("import logging\nimport mimetypes\n",
                                        "import logging\nimport mimetypes\nimport os\n", 1)
                if "from app.graph.config import get_graph_settings" not in text:
                    text = text.replace(
                        "from app.documents.ocr_client import OcrClient, OcrResult, SUPPORTED_SCRIPTS\n",
                        "from app.documents.ocr_client import OcrClient, OcrResult, SUPPORTED_SCRIPTS\n"
                        "from app.graph.config import get_graph_settings\n", 1)
                applied.append("pipeline.py: graph branch")

    if text != original and not a.check:
        ast.parse(text)
        io.open(pipeline, "w", encoding="utf-8").write(text)

    # --- the simple anchored edits -----------------------------------------
    by_path: dict[str, str] = {}
    for edit in EDITS:
        if edit.path not in by_path:
            if not os.path.exists(edit.path):
                print(f"  !! missing file: {edit.path}")
                continue
            by_path[edit.path] = io.open(edit.path, encoding="utf-8").read()
        text = by_path[edit.path]
        if edit.present(text):
            continue
        missing.append(edit.name)
        if a.check:
            continue
        if edit.anchor not in text:
            print(f"  !! {edit.name}: anchor not found, skipped")
            continue
        by_path[edit.path] = text.replace(edit.anchor, edit.replacement, edit.count)
        applied.append(edit.name)

    if not a.check:
        for path, text in by_path.items():
            if path.endswith(".py"):
                ast.parse(text)
            io.open(path, "w", encoding="utf-8").write(text)

    # --- whole-file snapshots ------------------------------------------------
    for snap in SNAPSHOTS_TRACKED:
        if not os.path.exists(snap.path):
            print(f"  !! missing file: {snap.path}")
            continue
        if snap.present(io.open(snap.path, encoding="utf-8").read()):
            continue
        missing.append(snap.name)
        if a.check:
            continue
        if not os.path.exists(snap.store):
            print(f"  !! {snap.name}: no snapshot recorded, cannot restore")
            continue
        io.open(snap.path, "w", encoding="utf-8", newline="\n").write(
            io.open(snap.store, encoding="utf-8").read())
        applied.append(f"{snap.name} (restored from snapshot)")

    # --- requirements -------------------------------------------------------
    req = os.path.join(ROOT, "requirements.txt")
    rtext = io.open(req, encoding="utf-8").read()
    if "langgraph==" not in rtext:
        missing.append("requirements.txt: langgraph pins")
        if not a.check:
            io.open(req, "w", encoding="utf-8").write(rtext.rstrip() + REQUIREMENTS_BLOCK)
            applied.append("requirements.txt: langgraph pins")

    # --- report --------------------------------------------------------------
    if a.check:
        if missing:
            print(f"{len(missing)} integration edit(s) MISSING:")
            for m in missing:
                print("   -", m)
            return 1
        print("integration intact — nothing missing")
        return 0

    if applied:
        print(f"re-applied {len(applied)} edit(s):")
        for m in applied:
            print("   -", m)
    else:
        print("nothing to do — integration already intact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
