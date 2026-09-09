from app.documents import retrieval, vector_store
from app.documents.chunking import Chunk
from app.documents.retrieval import group_templates_by_section, index_document_chunks, retrieve_chunks_for_section


def test_group_templates_by_section_groups_flat_and_indexed_paths():
    templates = [
        "organisation.gst_number",
        "organisation.pan_number",
        "equipment[0].name",
        "equipment[0].serial_number",
        "staff[0].name",
    ]

    grouped = group_templates_by_section(templates)

    assert grouped == {
        "organisation": ["organisation.gst_number", "organisation.pan_number"],
        "equipment": ["equipment[0].name", "equipment[0].serial_number"],
        "staff": ["staff[0].name"],
    }


def test_index_document_chunks_embeds_all_chunk_texts_and_stores_by_page_number(monkeypatch):
    calls = {}
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(
        vector_store,
        "index_chunks",
        lambda document_id, chunk_ids, texts, vectors: calls.update(
            document_id=document_id, chunk_ids=chunk_ids, texts=texts, vectors=vectors
        ),
    )

    chunks = [Chunk(page_number=0, text="page zero"), Chunk(page_number=1, text="page one")]
    index_document_chunks("doc1", chunks)

    assert calls["document_id"] == "doc1"
    assert calls["chunk_ids"] == [0, 1]
    assert calls["texts"] == ["page zero", "page one"]
    assert calls["vectors"] == [[0.1, 0.2], [0.1, 0.2]]


def test_index_document_chunks_with_no_chunks_skips_embedding_entirely(monkeypatch):
    monkeypatch.setattr(
        retrieval, "embed_texts", lambda texts: (_ for _ in ()).throw(AssertionError("should not embed an empty chunk list"))
    )

    index_document_chunks("doc1", [])  # must not raise


def test_retrieve_chunks_for_section_maps_page_numbers_back_to_chunks_in_ranked_order(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [[0.5, 0.5]])
    monkeypatch.setattr(vector_store, "query_top_k", lambda document_id, query_vector, top_k: [1, 0])

    chunks_by_page = {
        0: Chunk(page_number=0, text="page zero"),
        1: Chunk(page_number=1, text="page one"),
    }

    result = retrieve_chunks_for_section("doc1", "equipment", ["equipment[0].name"], chunks_by_page)

    assert [c.page_number for c in result] == [1, 0]


def test_retrieve_chunks_for_section_skips_page_numbers_missing_from_the_chunk_map(monkeypatch):
    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [[0.5, 0.5]])
    monkeypatch.setattr(vector_store, "query_top_k", lambda document_id, query_vector, top_k: [5])

    result = retrieve_chunks_for_section(
        "doc1", "equipment", ["equipment[0].name"], {0: Chunk(page_number=0, text="x")}
    )

    assert result == []
