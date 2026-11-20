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
