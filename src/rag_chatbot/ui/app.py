from __future__ import annotations

from pathlib import Path

import httpx
import streamlit as st

API_BASE_URL = st.secrets.get("api_base_url", "http://api:8001")
API_KEY = st.secrets.get("api_key", "dev-secret")
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _headers() -> dict[str, str]:
    return {"x-api-key": API_KEY}


def _list_documents() -> list[str]:
    try:
        r = httpx.get(f"{API_BASE_URL}/documents", headers=_headers(), timeout=30)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception:
        return []


def _ingest(path: str, chunk_size: int, overlap: float) -> dict:
    payload = {
        "docs_path": path,
        "chunk_size": chunk_size,
        "overlap_ratio": overlap,
    }
    r = httpx.post(
        f"{API_BASE_URL}/ingest",
        headers=_headers(),
        json=payload,
        timeout=httpx.Timeout(600.0),
    )
    r.raise_for_status()
    return r.json()


def _query(text: str, top_k: int) -> dict:
    payload = {"query": text, "top_k": top_k}
    r = httpx.post(f"{API_BASE_URL}/query", headers=_headers(), json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def _remove_document(document_name: str) -> dict:
    payload = {"document_name": document_name}
    r = httpx.post(f"{API_BASE_URL}/documents/remove", headers=_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def _clear_documents() -> dict:
    r = httpx.post(f"{API_BASE_URL}/documents/clear", headers=_headers(), timeout=120)
    r.raise_for_status()
    return r.json()


def _remove_uploaded_file(document_name: str) -> None:
    target = UPLOAD_DIR / document_name
    if target.exists():
        target.unlink()


def _clear_uploaded_files() -> int:
    removed = 0
    for path in UPLOAD_DIR.glob("*"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


st.set_page_config(page_title="RAG Portfolio Bot", layout="wide")
st.title("RAG Portfolio Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Ingestion")
    files = st.file_uploader(
        "Upload documents (.pdf, .docx, .txt, .md)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    chunk_size = st.select_slider("Chunk size (tokens)", options=[256, 512, 1024], value=512)
    overlap = st.slider("Overlap ratio", min_value=0.05, max_value=0.25, value=0.12, step=0.01)

    if st.button("Save + Ingest", use_container_width=True):
        selected_files = files or []
        if not selected_files:
            st.warning("Please upload at least one document before ingesting")
            st.stop()

        saved_count = 0
        for file in selected_files:
            payload = file.getvalue()
            if not payload:
                continue
            target = UPLOAD_DIR / file.name
            target.write_bytes(payload)
            saved_count += 1

        if saved_count == 0:
            st.error("Uploaded files appear empty. Please re-upload the documents and try again.")
            st.stop()

        try:
            stats = _ingest(str(UPLOAD_DIR), chunk_size, overlap)
            st.success(f"Ingested {stats['documents']} docs and {stats['chunks']} chunks")
        except Exception as exc:
            st.error(str(exc))

    st.header("Documents")
    docs = _list_documents()
    if not docs:
        st.caption("No ingested documents")
    else:
        for doc in docs:
            st.caption(doc)

    st.subheader("Manage Documents")
    selected_doc = st.selectbox("Select document to remove", options=["-- select --", *docs], index=0)

    if st.button("Remove Selected Document", use_container_width=True):
        if selected_doc == "-- select --":
            st.warning("Select a document first")
        else:
            try:
                result = _remove_document(selected_doc)
                _remove_uploaded_file(selected_doc)
                st.success(f"Removed {selected_doc} ({result.get('removed_chunks', 0)} chunks)")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    if st.button("Clear All Documents", use_container_width=True):
        try:
            result = _clear_documents()
            removed_files = _clear_uploaded_files()
            st.success(
                f"Cleared vector store ({result.get('removed_chunks', 0)} chunks) and {removed_files} uploaded files"
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

st.subheader("Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            st.caption("Sources: " + ", ".join(f"{c['document']} p.{c['page']}" for c in message["citations"]))

prompt = st.chat_input("Ask a question about your documents")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = _query(prompt, top_k=5)
                answer = result.get("answer", "")
                citations = result.get("citations", [])
                st.markdown(answer)
                if citations:
                    st.caption("Sources: " + ", ".join(f"{c['document']} p.{c['page']}" for c in citations))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                })
            except Exception as exc:
                err = f"Request failed: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
