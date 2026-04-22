from __future__ import annotations

from rag_chatbot.generation.llm import generate_text
from rag_chatbot.generation.prompting import build_prompt
from rag_chatbot.retrieval.retriever import retrieve


def _build_citations(chunks):
    seen = set()
    citations = []
    for chunk in chunks:
        doc = chunk.metadata.get("document_name", "unknown")
        page = chunk.metadata.get("page", "?")
        key = (doc, page)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"document": doc, "page": page})
    return citations


async def answer_query(query: str, where: dict | None = None, top_k: int | None = None) -> dict:
    chunks = retrieve(query=query, where=where, top_k=top_k)

    if not chunks:
        return {
            "answer": "I could not find relevant context in the ingested documents.",
            "citations": [],
            "retrieved_chunks": [],
        }

    prompt = build_prompt(user_query=query, chunks=chunks)
    answer = await generate_text(prompt)

    return {
        "answer": answer.strip(),
        "citations": _build_citations(chunks),
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "metadata": chunk.metadata,
                "text": chunk.text,
            }
            for chunk in chunks
        ],
    }
