from __future__ import annotations

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness


def run_ragas(records: list[dict]) -> pd.DataFrame:
    dataset = Dataset.from_list(records)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result.to_pandas()
