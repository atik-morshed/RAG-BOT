from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rag_chatbot.ingestion.pipeline import ingest_documents


CHUNK_SIZES = [256, 512, 1024]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunk-size ablation")
    parser.add_argument("--docs", default="./data")
    parser.add_argument("--overlap", type=float, default=0.12)
    parser.add_argument("--out", default="ablation_results.csv")
    args = parser.parse_args()

    rows = []
    for size in CHUNK_SIZES:
        stats = ingest_documents(data_dir=args.docs, chunk_size=size, overlap_ratio=args.overlap)
        rows.append(
            {
                "chunk_size": size,
                "overlap_ratio": args.overlap,
                "documents": stats["documents"],
                "chunks": stats["chunks"],
            }
        )

    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved ablation table to {out_path}")


if __name__ == "__main__":
    main()
