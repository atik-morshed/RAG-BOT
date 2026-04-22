from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chatbot.retrieval.retriever import build_retrieval_metrics, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on QA ground-truth file")
    parser.add_argument("--qa-file", default="data/qa_ground_truth.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    qa_path = Path(args.qa_file)
    payload = json.loads(qa_path.read_text(encoding="utf-8"))

    predictions: dict[str, list[str]] = {}
    ground_truth: dict[str, list[str]] = {}

    for item in payload:
        qid = item["id"]
        query = item["question"]
        expected_ids = item["expected_chunk_ids"]
        chunks = retrieve(query=query, top_k=args.top_k)
        predictions[qid] = [c.chunk_id for c in chunks]
        ground_truth[qid] = expected_ids

    metrics = build_retrieval_metrics(predictions, ground_truth)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
