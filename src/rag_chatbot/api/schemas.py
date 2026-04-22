from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    docs_path: str = Field(default="./data")
    chunk_size: int = Field(default=512, ge=128, le=2048)
    overlap_ratio: float = Field(default=0.12, ge=0.0, le=0.5)


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    metadata_filter: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]


class DocumentsResponse(BaseModel):
    documents: list[str]


class RemoveDocumentRequest(BaseModel):
    document_name: str


class RemoveDocumentResponse(BaseModel):
    removed_chunks: int
