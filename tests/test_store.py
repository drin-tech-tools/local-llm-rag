from local_llm_rag.store import ChunkInput, Store, collection_name


def test_collection_name_is_valid_for_chroma():
    assert collection_name("nomic-embed-text") == "chunks-nomic-embed-text"
    assert collection_name("hf.co/org/model:Q4_K_M") == "chunks-hf.co-org-model-Q4_K_M"


def test_add_search_scope_and_delete(tmp_path):
    store = Store(tmp_path)
    a = store.add_document(
        "a.pdf",
        2,
        "m1",
        [
            ChunkInput(1, 0, "cats", [1, 0, 0]),
            ChunkInput(2, 1, "dogs", [0, 1, 0]),
        ],
    )
    b = store.add_document("b.pdf", 1, "m2", [ChunkInput(1, 0, "other model", [1, 0])])
    assert (a.chunks, a.dims) == (2, 3)

    hits = store.search([0.9, 0.1, 0], "m1", k=5)
    assert [h.content for h in hits] == ["cats", "dogs"]
    assert hits[0].document_name == "a.pdf"
    assert hits[0].page == 1
    assert hits[0].score > hits[1].score

    # Each embedding model is searched separately; a document can be targeted.
    assert [h.content for h in store.search([1, 0], "m2", k=5, document_id=b.id)] == ["other model"]
    assert store.search([1, 0, 0], "m1", k=5, document_id=b.id) == []

    # Survives a restart.
    reopened = Store(tmp_path)
    assert sorted(d.name for d in reopened.list_documents()) == ["a.pdf", "b.pdf"]

    assert reopened.delete_document(a.id)
    assert not reopened.delete_document(a.id)
    assert reopened.search([1, 0, 0], "m1", k=5) == []


def test_search_with_no_documents(tmp_path):
    assert Store(tmp_path).search([1.0, 0.0], "m1", k=5) == []
