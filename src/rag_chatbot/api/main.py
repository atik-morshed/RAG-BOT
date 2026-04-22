from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

from rag_chatbot.api.auth import require_api_key
from rag_chatbot.api.schemas import (
    DocumentsResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RemoveDocumentRequest,
    RemoveDocumentResponse,
)
from rag_chatbot.config import get_settings
from rag_chatbot.generation.service import answer_query
from rag_chatbot.ingestion.pipeline import ingest_documents
from rag_chatbot.logging_utils import log_query
from rag_chatbot.vectorstore import clear_documents, delete_document, list_document_names

app = FastAPI(title="RAG Chatbot API", version="0.1.0")


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "RAG Chatbot API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(payload: IngestRequest) -> IngestResponse:
    try:
        stats = ingest_documents(
            data_dir=payload.docs_path,
            chunk_size=payload.chunk_size,
            overlap_ratio=payload.overlap_ratio,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(**stats)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def query(payload: QueryRequest) -> QueryResponse:
    result = await answer_query(
        query=payload.query,
        where=payload.metadata_filter,
        top_k=payload.top_k,
    )

    settings = get_settings()
    log_query(
        settings.query_log_path,
        {
            "query": payload.query,
            "top_k": payload.top_k,
            "metadata_filter": payload.metadata_filter,
            "answer": result["answer"],
            "citations": result["citations"],
            "retrieved_chunk_ids": [c["chunk_id"] for c in result["retrieved_chunks"]],
        },
    )

    return QueryResponse(**result)


@app.get("/documents", response_model=DocumentsResponse, dependencies=[Depends(require_api_key)])
async def documents() -> DocumentsResponse:
    return DocumentsResponse(documents=list_document_names())


@app.post("/documents/remove", response_model=RemoveDocumentResponse, dependencies=[Depends(require_api_key)])
async def remove_document(payload: RemoveDocumentRequest) -> RemoveDocumentResponse:
    removed = delete_document(payload.document_name)
    return RemoveDocumentResponse(removed_chunks=removed)


@app.post("/documents/clear", response_model=RemoveDocumentResponse, dependencies=[Depends(require_api_key)])
async def clear_all_documents() -> RemoveDocumentResponse:
    removed = clear_documents()
    return RemoveDocumentResponse(removed_chunks=removed)
