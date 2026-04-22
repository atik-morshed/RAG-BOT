from __future__ import annotations

from rag_chatbot.types import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        doc = chunk.metadata.get("document_name", "unknown")
        page = chunk.metadata.get("page", "?")
        lines.append(f"[Chunk {i}] Source: {doc} (page {page})")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(user_query: str, chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks)
    system_block = (
        "You are a grounded document assistant. "
        "Only answer based on provided context. "
        "If context is insufficient, say you are unsure and ask for more documents. "
        "Always include citations in the format [document_name p.X]."
    )

    return (
        f"SYSTEM INSTRUCTIONS:\n{system_block}\n\n"
        f"CONTEXT CHUNKS:\n{context}\n\n"
        f"USER QUESTION:\n{user_query}\n\n"
        "Answer concisely and include source citations."
    )
