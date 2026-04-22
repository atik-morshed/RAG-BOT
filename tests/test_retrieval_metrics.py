from rag_chatbot.retrieval.retriever import build_retrieval_metrics


def test_retrieval_metrics_non_empty() -> None:
    predictions = {
        "q1": ["a", "b", "c"],
        "q2": ["x", "y"],
    }
    ground_truth = {
        "q1": ["a", "d"],
        "q2": ["x"],
    }

    metrics = build_retrieval_metrics(predictions, ground_truth)
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert metrics["precision"] == 0.4167
    assert metrics["recall"] == 0.75
