from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag_chatbot.config import get_settings
from rag_chatbot.types import RetrievedChunk


@lru_cache
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.rerank_model)


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []

    model = get_reranker()
    pairs = [[query, chunk.text] for chunk in chunks]
    scores = model.predict(pairs)

    rescored = []
    for chunk, score in zip(chunks, scores):
        rescored.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=float(score),
            )
        )

    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored[:top_k]
