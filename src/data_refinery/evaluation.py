from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def train_and_evaluate(train_data: Dataset, test_data: Dataset, seed: int, condition: str) -> dict[str, Any]:
    """Train the fixed downstream classifier and evaluate it."""
    classifier = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1_000, random_state=seed)),
        ]
    )

    train_texts = list(train_data["text"])
    train_labels = list(train_data["label"])
    test_texts = list(test_data["text"])
    test_labels = list(test_data["label"])

    classifier.fit(train_texts, train_labels)
    predictions = classifier.predict(test_texts)

    return {
        "seed": seed,
        "condition": condition,
        "training_examples": len(train_data),
        "accuracy": accuracy_score(test_labels, predictions),
        "macro_precision": precision_score(test_labels, predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(test_labels, predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(test_labels, predictions, average="macro", zero_division=0),
    }


def save_metrics(metrics: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save experiment metrics to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(path, index=False)
