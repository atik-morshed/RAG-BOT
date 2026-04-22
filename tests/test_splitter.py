from rag_chatbot.ingestion.splitter import chunk_text


def test_chunk_text_returns_chunks() -> None:
    text = " ".join(["token"] * 1500)
    chunks = chunk_text(text, {"document_name": "sample.txt", "page": 1}, chunk_size=256, overlap_ratio=0.1)

    assert len(chunks) > 1
    assert all("text" in chunk for chunk in chunks)
    assert all("metadata" in chunk for chunk in chunks)
