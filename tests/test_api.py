from fastapi.testclient import TestClient

from rag_chatbot.api.main import app


client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"x-api-key": "dev-secret"}


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "RAG Chatbot API"
    assert payload["status"] == "ok"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_documents_requires_api_key() -> None:
    response = client.get("/documents")
    assert response.status_code == 401


def test_documents_returns_list(monkeypatch) -> None:
    monkeypatch.setattr("rag_chatbot.api.main.list_document_names", lambda: ["a.pdf", "b.pdf"])
    response = client.get("/documents", headers=_auth_headers())
    assert response.status_code == 200
    assert response.json() == {"documents": ["a.pdf", "b.pdf"]}


def test_ingest_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_chatbot.api.main.ingest_documents",
        lambda data_dir, chunk_size, overlap_ratio: {"documents": 2, "chunks": 17},
    )
    payload = {"docs_path": "./data/uploads", "chunk_size": 512, "overlap_ratio": 0.12}
    response = client.post("/ingest", json=payload, headers=_auth_headers())
    assert response.status_code == 200
    assert response.json() == {"documents": 2, "chunks": 17}


def test_query_endpoint(monkeypatch) -> None:
    async def _fake_answer_query(query: str, where: dict | None = None, top_k: int | None = None) -> dict:
        return {
            "answer": f"Echo: {query}",
            "citations": [{"document": "demo.pdf", "page": 1}],
            "retrieved_chunks": [
                {
                    "chunk_id": "demo-1",
                    "score": 0.9,
                    "metadata": {"document_name": "demo.pdf", "page": 1},
                    "text": "demo text",
                }
            ],
        }

    monkeypatch.setattr("rag_chatbot.api.main.answer_query", _fake_answer_query)
    monkeypatch.setattr("rag_chatbot.api.main.log_query", lambda *args, **kwargs: None)

    payload = {"query": "test", "top_k": 3}
    response = client.post("/query", json=payload, headers=_auth_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Echo: test"
    assert body["citations"] == [{"document": "demo.pdf", "page": 1}]
    assert len(body["retrieved_chunks"]) == 1


def test_remove_and_clear_endpoints(monkeypatch) -> None:
    monkeypatch.setattr("rag_chatbot.api.main.delete_document", lambda document_name: 11)
    monkeypatch.setattr("rag_chatbot.api.main.clear_documents", lambda: 29)

    remove_response = client.post(
        "/documents/remove",
        json={"document_name": "demo.pdf"},
        headers=_auth_headers(),
    )
    assert remove_response.status_code == 200
    assert remove_response.json() == {"removed_chunks": 11}

    clear_response = client.post("/documents/clear", headers=_auth_headers())
    assert clear_response.status_code == 200
    assert clear_response.json() == {"removed_chunks": 29}