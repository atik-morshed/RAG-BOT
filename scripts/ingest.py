from __future__ import annotations

import argparse

from rag_chatbot.ingestion.pipeline import ingest_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma")
    parser.add_argument("--docs", default="./data", help="Path to docs directory")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.12)
    args = parser.parse_args()

    stats = ingest_documents(data_dir=args.docs, chunk_size=args.chunk_size, overlap_ratio=args.overlap)
    print(f"Ingested {stats['documents']} documents and {stats['chunks']} chunks")


if __name__ == "__main__":
    main()
