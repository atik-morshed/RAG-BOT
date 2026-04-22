from __future__ import annotations

import hashlib

from rag_chatbot.ingestion.loaders import load_documents
from rag_chatbot.ingestion.splitter import chunk_text
from rag_chatbot.retrieval.embeddings import embed_texts
from rag_chatbot.vectorstore import upsert_chunks


UPSERT_BATCH_SIZE = 128


def _chunk_id(document_name: str, page: int, chunk_index: int, text: str) -> str:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    return f"{document_name}-p{page}-c{chunk_index}-{digest}"


def _embed_and_upsert(batch: list[dict]) -> int:
    if not batch:
        return 0

    embeddings = embed_texts([chunk["text"] for chunk in batch])
    records = []
    for chunk, embedding in zip(batch, embeddings):
        records.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": embedding,
            }
        )

    upsert_chunks(records)
    return len(records)


def ingest_documents(data_dir: str, chunk_size: int, overlap_ratio: float) -> dict[str, int]:
    docs = load_documents(data_dir)
    pending_chunks: list[dict] = []
    unique_docs: set[str] = set()
    total_chunks = 0

    for doc in docs:
        unique_docs.add(str(doc["metadata"].get("document_name", "unknown")))
        for chunk in chunk_text(
            text=doc["text"],
            base_metadata=doc["metadata"],
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
        ):
            chunk_id = _chunk_id(
                document_name=chunk["metadata"]["document_name"],
                page=int(chunk["metadata"].get("page", 1)),
                chunk_index=int(chunk["metadata"].get("chunk_index", 0)),
                text=chunk["text"],
            )
            pending_chunks.append({"id": chunk_id, "text": chunk["text"], "metadata": chunk["metadata"]})

            if len(pending_chunks) >= UPSERT_BATCH_SIZE:
                total_chunks += _embed_and_upsert(pending_chunks)
                pending_chunks = []

    if pending_chunks:
        total_chunks += _embed_and_upsert(pending_chunks)

    if total_chunks == 0:
        return {"documents": 0, "chunks": 0}

    return {"documents": len(unique_docs), "chunks": total_chunks}
