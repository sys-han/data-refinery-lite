from pathlib import Path

import pandas as pd

from .config import (
    METRICS_PATH,
    PARAPHRASE_EXAMPLES_PATH,
    PARAPHRASE_PROMPT_PATH,
    PARAPHRASE_RESPONSE_PATH,
    RECOVERY_EXAMPLES_PATH,
    RECOVERY_PROMPT_PATH,
    RECOVERY_RESPONSE_PATH,
    SOURCE_PATH,
)


def expected_artifacts() -> list[Path]:
    return [
        SOURCE_PATH,
        RECOVERY_PROMPT_PATH,
        PARAPHRASE_PROMPT_PATH,
        RECOVERY_RESPONSE_PATH,
        PARAPHRASE_RESPONSE_PATH,
        RECOVERY_EXAMPLES_PATH,
        PARAPHRASE_EXAMPLES_PATH,
        METRICS_PATH,
    ]


def print_status() -> None:
    for path in expected_artifacts():
        marker = "OK" if path.exists() else "--"
        print(f"[{marker}] {path}")


def load_metrics(path: str | Path = METRICS_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return pd.read_csv(path)


def format_results_table(metrics: pd.DataFrame) -> str:
    frame = metrics.copy()
    frame["accuracy"] = frame["accuracy"].map(lambda value: f"{value:.4f}")
    frame["macro_f1"] = frame["macro_f1"].map(lambda value: f"{value:.4f}")
    frame["macro_f1_delta_vs_32_real"] = frame["macro_f1_delta_vs_32_real"].map(
        lambda value: "—" if abs(value) < 1e-12 else f"{value:+.4f}"
    )
    return frame[
        [
            "condition",
            "training_examples",
            "accuracy",
            "macro_f1",
            "macro_f1_delta_vs_32_real",
        ]
    ].to_markdown(index=False)


def print_report(path: str | Path = METRICS_PATH) -> None:
    metrics = load_metrics(path)
    print("Data Refinery Lite — fixed seed-11 pilot")
    print(format_results_table(metrics))

    best = metrics.sort_values("macro_f1", ascending=False).iloc[0]
    baseline = metrics.loc[metrics["condition"] == "32_real"].iloc[0]
    print("\nBest condition:", best["condition"])
    print(f"Macro-F1: {best['macro_f1']:.4f}")
    print(f"Delta vs 32 real: {best['macro_f1'] - baseline['macro_f1']:+.4f}")
