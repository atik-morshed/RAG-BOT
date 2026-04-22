from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from rag_chatbot.config import get_settings


def _build_client() -> chromadb.ClientAPI:
    settings = get_settings()
    if settings.chroma_mode.lower() == "http":
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection() -> Collection:
    settings = get_settings()
    client = _build_client()
    return client.get_or_create_collection(name=settings.chroma_collection, metadata={"hnsw:space": "cosine"})


def upsert_chunks(chunks: list[dict[str, Any]]) -> None:
    collection = get_collection()
    collection.upsert(
        ids=[item["id"] for item in chunks],
        documents=[item["text"] for item in chunks],
        metadatas=[item["metadata"] for item in chunks],
        embeddings=[item["embedding"] for item in chunks],
    )


def list_document_names(limit: int = 1000) -> list[str]:
    collection = get_collection()
    result = collection.get(limit=limit, include=["metadatas"])
    names = set()
    for metadata in result.get("metadatas", []):
        if metadata and metadata.get("document_name"):
            names.add(str(metadata["document_name"]))
    return sorted(names)


def _get_ids_for_document(document_name: str) -> list[str]:
    collection = get_collection()
    result = collection.get(where={"document_name": document_name}, include=[])
    return result.get("ids", []) or []


def _get_all_ids(limit: int = 100000) -> list[str]:
    collection = get_collection()
    result = collection.get(limit=limit, include=[])
    return result.get("ids", []) or []


def delete_document(document_name: str) -> int:
    collection = get_collection()
    ids = _get_ids_for_document(document_name)
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)


def clear_documents() -> int:
    collection = get_collection()
    ids = _get_all_ids()
    if not ids:
        return 0
    collection.delete(ids=ids)
    return len(ids)
