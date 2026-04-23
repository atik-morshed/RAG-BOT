# RAG Chatbot

Minimal RAG chatbot project with FastAPI, Streamlit, Chroma, and local embeddings.

## Deploy

### Render (recommended)

1. Push this repository to GitHub.
2. In Render, choose New + Blueprint and select this repository.
3. Render reads render.yaml and creates one web service with persistent disk.
4. In Render dashboard, set OPENROUTER_API_KEY for the service.
5. Deploy and open the service URL.

Notes:
- Streamlit is served on the public port.
- FastAPI runs internally in the same container.
- Uploaded files, Chroma data, and logs persist on /data.

## Evaluation Snapshot

### Retrieval Metrics (Current)

| Metric | Value |
|---|---:|
| Precision@k | 0.4167 |
| Recall@k | 0.7500 |

Source: tests/test_retrieval_metrics.py

### RAGAS Score Table

| Variant | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---:|---:|---:|---:|
| Baseline RAG | pending | pending | pending | pending |
| Optimized RAG (rerank + prompt tuning) | pending | pending | pending | pending |

Use scripts/evaluate_ragas.py after preparing a real evaluation set to replace pending values.

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
