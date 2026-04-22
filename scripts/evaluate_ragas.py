from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_chatbot.evaluation.ragas_eval import run_ragas


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation from json input")
    parser.add_argument("--input", default="data/ragas_eval.json")
    parser.add_argument("--out", default="ragas_scores.csv")
    args = parser.parse_args()

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    df = run_ragas(records)
    df.to_csv(args.out, index=False)
    print(f"Saved RAGAS scores to {args.out}")


if __name__ == "__main__":
    main()
