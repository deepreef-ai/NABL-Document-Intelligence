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


class Edit:
    def __init__(self, path: str, name: str, marker: str, anchor: str, replacement: str):
        self.path = path
        self.name = name
        self.marker = marker      # present => already applied
        self.anchor = anchor      # text to replace
        self.replacement = replacement

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
        "providers.py: temperature parameter", "temperature: float = 0.2",
        "                 max_tokens: int = 8192):",
        "                 max_tokens: int = 8192, temperature: float = 0.2):",
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
    Edit(
        os.path.join(ROOT, "app", "documents", "pipeline.py"),
        "pipeline.py: real filename to the graph", "display_name=filename",
        "            return run_graph_pipeline(tmp_path, document_id=document_id, form_type=form_type)",
        "            # `filename` is the upload's own name; tmp_path is where it was\n"
        "            # staged. Without this the gold-shaped output recorded\n"
        '            # original_filename as "tmpn0yclbyd.pdf".\n'
        "            return run_graph_pipeline(\n"
        "                tmp_path, document_id=document_id, form_type=form_type,\n"
        "                display_name=filename,\n"
        "            )",
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

# pipeline.py's graph branch is large enough that it lives here as a block.
PIPELINE_MARKER = "run_graph_pipeline"
PIPELINE_ANCHOR = """    ocr_client = ocr_client or OcrClient()
    kind = _guess_kind(filename, content_type)

    if kind == "pdf":"""
PIPELINE_REPLACEMENT = """    ocr_client = ocr_client or OcrClient()
    kind = _guess_kind(filename, content_type)

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
            return run_graph_pipeline(tmp_path, document_id=document_id, form_type=form_type)
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
    a = ap.parse_args()

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
        by_path[edit.path] = text.replace(edit.anchor, edit.replacement, 1)
        applied.append(edit.name)

    if not a.check:
        for path, text in by_path.items():
            if path.endswith(".py"):
                ast.parse(text)
            io.open(path, "w", encoding="utf-8").write(text)

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
