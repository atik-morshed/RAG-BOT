from __future__ import annotations

from collections import defaultdict
from typing import Any

from rank_bm25 import BM25Okapi

from rag_chatbot.config import get_settings
from rag_chatbot.retrieval.embeddings import embed_query
from rag_chatbot.retrieval.reranker import rerank
from rag_chatbot.types import RetrievedChunk
from rag_chatbot.vectorstore import get_collection


def _apply_filters(where: dict[str, Any] | None) -> dict[str, Any] | None:
    if not where:
        return None
    # Chroma supports a mongo-like where object.
    return {k: v for k, v in where.items() if v is not None}


def dense_retrieve(query: str, top_k: int, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
    collection = get_collection()
    vector = embed_query(query)
    result = collection.query(
        query_embeddings=[vector],
        n_results=max(top_k, 8),
        where=_apply_filters(where),
        include=["documents", "metadatas", "distances"],
    )

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    chunks: list[RetrievedChunk] = []
    for cid, text, metadata, distance in zip(ids, docs, metas, distances):
        score = 1.0 - float(distance)
        chunks.append(RetrievedChunk(chunk_id=cid, text=text, metadata=metadata or {}, score=score))

    return chunks[:top_k]


def hybrid_retrieve(query: str, top_k: int, where: dict[str, Any] | None = None) -> list[RetrievedChunk]:
    # Candidate pool from dense search first, then lexical fusion via BM25.
    dense_candidates = dense_retrieve(query=query, top_k=max(top_k * 3, 15), where=where)
    if not dense_candidates:
        return []

    tokenized = [chunk.text.lower().split() for chunk in dense_candidates]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    max_dense = max(chunk.score for chunk in dense_candidates) or 1.0
    max_bm25 = max(bm25_scores) if len(bm25_scores) else 1.0
    if max_bm25 == 0:
        max_bm25 = 1.0

    fused: list[RetrievedChunk] = []
    for chunk, bm25_score in zip(dense_candidates, bm25_scores):
        dense_norm = chunk.score / max_dense
        sparse_norm = float(bm25_score) / max_bm25
        fused_score = 0.6 * dense_norm + 0.4 * sparse_norm
        fused.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=fused_score,
            )
        )

    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:top_k]


def retrieve(query: str, where: dict[str, Any] | None = None, top_k: int | None = None) -> list[RetrievedChunk]:
    settings = get_settings()
    k = top_k or settings.top_k

    if settings.use_hybrid:
        chunks = hybrid_retrieve(query=query, top_k=max(k * 2, k), where=where)
    else:
        chunks = dense_retrieve(query=query, top_k=max(k * 2, k), where=where)

    if settings.use_rerank:
        chunks = rerank(query=query, chunks=chunks, top_k=k)
    else:
        chunks = chunks[:k]

    # Deduplicate by chunk id while preserving order.
    deduped: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append(chunk)

    return deduped[:k]


def build_retrieval_metrics(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
) -> dict[str, float]:
    precision_vals = []
    recall_vals = []

    for qid, predicted_ids in predictions.items():
        gold_ids = set(ground_truth.get(qid, []))
        pred_set = set(predicted_ids)
        if not pred_set:
            precision_vals.append(0.0)
            recall_vals.append(0.0)
            continue

        tp = len(pred_set & gold_ids)
        precision_vals.append(tp / len(pred_set))
        recall_vals.append(tp / len(gold_ids) if gold_ids else 0.0)

    avg_precision = sum(precision_vals) / len(precision_vals) if precision_vals else 0.0
    avg_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0
    return {
        "precision": round(avg_precision, 4),
        "recall": round(avg_recall, 4),
    }
