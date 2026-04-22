# Engineering Decisions

## Chroma vs FAISS
- Chroma was selected for easy local development and optional HTTP service mode in Docker Compose.
- FAISS can be faster for very large corpora, but Chroma's metadata filtering and persistence simplify this portfolio build.

## Chunking strategy
- Tested chunk sizes: 256, 512, 1024 tokens with ~12% overlap.
- 512 is the default trade-off for preserving context continuity while limiting retrieval noise.

## Why reranking
- Bi-encoder embeddings are efficient but can miss nuanced query-passage interactions.
- Cross-encoder reranking improves top-k precision because it jointly scores query and candidate chunk text.

## Hallucination handling
- Prompt explicitly restricts model answers to retrieved context.
- If no context is retrieved, system returns an uncertainty response.
- Responses include document/page citations.

## Local vs API LLM
- This project defaults to local-first (Ollama + local embeddings) for zero recurring cost.
- API models can improve quality but add latency, rate limits, and usage cost.
