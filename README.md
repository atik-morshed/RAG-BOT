# RAG Chatbot

Minimal RAG chatbot project with FastAPI, Streamlit, Chroma, and local embeddings.

## Run

1. Install dependencies:
   - pip install -e .
2. Copy env file:
   - copy .env.example .env
3. Start API:
   - uvicorn rag_chatbot.api.main:app --host 0.0.0.0 --port 8001
4. Start UI:
   - streamlit run src/rag_chatbot/ui/app.py

## API

- GET /health
- POST /ingest
- POST /query
- GET /documents
- POST /documents/remove
- POST /documents/clear

Use x-api-key header for protected endpoints.
