"""Groups extractor.py's flattened field templates by top-level schema
section, and retrieves only the chunks most relevant to each section from
Qdrant — one embedding-similarity lookup per section, no LLM call, instead of
sending every page to the LLM for every section."""
from collections import defaultdict

from app.config import get_settings
from app.documents import vector_store
from app.documents.chunking import Chunk
from app.documents.embeddings import embed_texts


def group_templates_by_section(field_templates: list[str]) -> dict[str, list[str]]:
    """"organisation.gst_number" -> section "organisation";
    "equipment[i].name" -> section "equipment"."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for template in field_templates:
        section = template.split(".", 1)[0].split("[", 1)[0]
        grouped[section].append(template)
    return dict(grouped)


def _section_query(section: str, templates: list[str]) -> str:
    leaf_names = [t.rsplit(".", 1)[-1].replace("_", " ") for t in templates]
    return f"{section.replace('_', ' ')}: {' '.join(leaf_names)}"


def index_document_chunks(document_id: str, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    vector_store.index_chunks(document_id, [c.page_number for c in chunks], texts, vectors)


def retrieve_chunks_for_section(
    document_id: str, section: str, templates: list[str], chunks_by_page: dict[int, Chunk]
) -> list[Chunk]:
    query = _section_query(section, templates)
    [query_vector] = embed_texts([query])
    top_k = get_settings().retrieval_top_k
    page_numbers = vector_store.query_top_k(document_id, query_vector, top_k)
    return [chunks_by_page[p] for p in page_numbers if p in chunks_by_page]


def select_relevant_pages(
    document_id: str,
    field_templates: list[str],
    pages: list,
    min_score: float | None = None,
) -> list[int]:
    """Page numbers (1-based, matching NormalizedPage) that are plausible
    evidence for `field_templates`, best-first.

    Combines three deterministic-to-semi-deterministic signals rather than
    relying on embeddings alone:

    1. dense similarity per schema section (the existing index), now
       score-thresholded instead of "always take the top 4 whatever they
       scored";
    2. literal leaf-name/alias matching in the page text — a page that
       actually prints "GST Number" is evidence for organisation.gst_number
       regardless of what a 384-dim MiniLM embedding thinks;
    3. table-bearing pages, kept whenever any requested field is a repeating
       table row (`attr[i].sub`), because that is exactly where test rows are.

    Falls back to EVERY page when no signal fires, so a retrieval miss
    degrades to today's behaviour (send everything) rather than silently
    extracting from nothing.
    """
    if not pages:
        return []
    min_score = get_settings().retrieval_min_score if min_score is None else min_score
    page_numbers = [p.page_number for p in pages]
    scored: dict[int, float] = {}

    # 1. dense retrieval, per section, thresholded
    sections = group_templates_by_section(field_templates)
    for section, templates in sections.items():
        try:
            [query_vector] = embed_texts([_section_query(section, templates)])
            hits = vector_store.query_top_k_scored(
                document_id, query_vector, get_settings().retrieval_top_k
            )
        except Exception:  # noqa: BLE001 — no index/model available: skip this signal
            hits = []
        for chunk_id, score in hits:
            if score >= min_score:
                # chunk_id is 0-based (chunking uses PageText.page_number);
                # NormalizedPage is 1-based.
                scored[chunk_id + 1] = max(scored.get(chunk_id + 1, 0.0), score)

    # 2. literal leaf-name matching
    leaves = {t.rsplit(".", 1)[-1].replace("_", " ").lower() for t in field_templates}
    leaves = {leaf for leaf in leaves if len(leaf) >= 4}
    for page in pages:
        low = (page.text or "").lower()
        if not low:
            continue
        hits = sum(1 for leaf in leaves if leaf in low)
        if hits:
            scored[page.page_number] = max(scored.get(page.page_number, 0.0), min(0.3 + 0.1 * hits, 1.0))

    # 3. table pages, when a repeating table was asked for
    if any("[i]" in t for t in field_templates):
        for page in pages:
            if getattr(page, "has_table", False):
                scored[page.page_number] = max(scored.get(page.page_number, 0.0), min_score)

    if not scored:
        return page_numbers
    return sorted(scored, key=lambda pn: (-scored[pn], pn))
