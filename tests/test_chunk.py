from local_llm_rag.chunk import chunk_pages


def test_keeps_page_numbers_and_overlaps_windows():
    words = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_pages(["", words], size=200, overlap=40)
    assert len(chunks) == 2
    assert all(c.page == 2 for c in chunks)
    assert chunks[1].content.startswith("w160 ")
    assert [c.index for c in chunks] == [0, 1]


def test_skips_near_empty_pages():
    assert chunk_pages(["   ", "tiny"]) == []
