import argparse
from typing import Any

import pandas as pd

from .config import (
    EXECUTION_MODE,
    EXAMPLES_PER_CLASS,
    EXPECTED_EXAMPLES,
    MASK_TOKEN,
    MASKING_RATE,
    MAX_SPAN_TOKENS,
    METRICS_PATH,
    MIN_CONTEXT_TOKENS,
    PARAPHRASE_EXAMPLES_PATH,
    PARAPHRASE_PROMPT_PATH,
    PARAPHRASE_PROMPT_VERSION,
    PARAPHRASE_RESPONSE_PATH,
    RECOVERY_EXAMPLES_PATH,
    RECOVERY_PROMPT_PATH,
    RECOVERY_PROMPT_VERSION,
    RECOVERY_RESPONSE_PATH,
    RESULTS_DIR,
    SEED,
    SOURCE_PATH,
)
from .report import print_report, print_status


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _build_source_examples() -> pd.DataFrame:
    from .augment import build_masked_dataset
    from .data import load_rotten_tomatoes, sample_balanced_train

    dataset = load_rotten_tomatoes()
    real_data = sample_balanced_train(dataset["train"], EXAMPLES_PER_CLASS, SEED)
    masked_data, records = build_masked_dataset(
        real_data=real_data,
        masking_rate=MASKING_RATE,
        seed=SEED,
        mask_token=MASK_TOKEN,
        min_context_tokens=MIN_CONTEXT_TOKENS,
        max_span_tokens=MAX_SPAN_TOKENS,
    )

    if len(real_data) != EXPECTED_EXAMPLES or len(masked_data) != EXPECTED_EXAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_EXAMPLES} real and masked examples.")

    source = pd.DataFrame(
        [
            {
                "id": index,
                "label": int(record["label"]),
                "sentiment": record["sentiment"],
                "original": record["original"],
                "masked": record["masked"],
                "masked_span": record["masked_span"],
                "masked_token_count": record["masked_token_count"],
                "actual_masking_rate": record["actual_masking_rate"],
            }
            for index, record in enumerate(records, start=1)
        ]
    )

    if source["label"].value_counts().to_dict() != {0: 16, 1: 16}:
        raise RuntimeError("The source sample is not balanced 16 per class.")
    return source


def _load_or_create_source() -> pd.DataFrame:
    _ensure_results_dir()
    if SOURCE_PATH.exists():
        source = pd.read_csv(SOURCE_PATH)
    else:
        source = _build_source_examples()
        source.to_csv(SOURCE_PATH, index=False)

    if len(source) != EXPECTED_EXAMPLES:
        raise ValueError(f"{SOURCE_PATH} must contain exactly {EXPECTED_EXAMPLES} rows.")
    if source["id"].astype(int).tolist() != list(range(1, EXPECTED_EXAMPLES + 1)):
        raise ValueError(f"{SOURCE_PATH} IDs must be 1 through {EXPECTED_EXAMPLES}.")
    return source


def prepare() -> None:
    from .augment import build_corruption_guided_prompt, build_direct_paraphrase_prompt

    source = _load_or_create_source()
    RECOVERY_PROMPT_PATH.write_text(build_corruption_guided_prompt(source), encoding="utf-8")
    PARAPHRASE_PROMPT_PATH.write_text(build_direct_paraphrase_prompt(source), encoding="utf-8")
    print(f"Saved fixed source examples: {SOURCE_PATH}")
    print(f"Saved corruption-guided prompt: {RECOVERY_PROMPT_PATH}")
    print(f"Saved direct-paraphrase prompt: {PARAPHRASE_PROMPT_PATH}")
    print("\nPaste each prompt into a fresh hosted-LLM chat and save JSON replies to:")
    print(f"  {RECOVERY_RESPONSE_PATH}")
    print(f"  {PARAPHRASE_RESPONSE_PATH}")


def ingest() -> None:
    from .augment import read_json_response

    source = _load_or_create_source()
    recovery_items = read_json_response(RECOVERY_RESPONSE_PATH, text_field="rewrite")
    paraphrase_items = read_json_response(PARAPHRASE_RESPONSE_PATH, text_field="paraphrase")

    recovery_by_id = {item["id"]: item["rewrite"] for item in recovery_items}
    paraphrase_by_id = {item["id"]: item["paraphrase"] for item in paraphrase_items}

    recovery_records: list[dict[str, Any]] = []
    paraphrase_records: list[dict[str, Any]] = []

    for row in source.itertuples(index=False):
        item_id = int(row.id)
        original = str(row.original).strip()
        rewrite = recovery_by_id[item_id]
        paraphrase = paraphrase_by_id[item_id]

        if not rewrite:
            raise ValueError(f"Recovery output {item_id} is empty.")
        if MASK_TOKEN in rewrite:
            raise ValueError(f"Recovery output {item_id} still contains {MASK_TOKEN}.")
        if not paraphrase:
            raise ValueError(f"Paraphrase output {item_id} is empty.")

        recovery_records.append(
            {
                "id": item_id,
                "label": int(row.label),
                "sentiment": row.sentiment,
                "original": original,
                "masked": str(row.masked).strip(),
                "text": rewrite,
                "exact_copy_of_original": rewrite.lower() == original.lower(),
                "contains_mask_token": MASK_TOKEN in rewrite,
                "execution_mode": EXECUTION_MODE,
                "prompt_version": RECOVERY_PROMPT_VERSION,
            }
        )
        paraphrase_records.append(
            {
                "id": item_id,
                "label": int(row.label),
                "sentiment": row.sentiment,
                "original": original,
                "text": paraphrase,
                "exact_copy_of_original": paraphrase.lower() == original.lower(),
                "execution_mode": EXECUTION_MODE,
                "prompt_version": PARAPHRASE_PROMPT_VERSION,
            }
        )

    pd.DataFrame(recovery_records).to_csv(RECOVERY_EXAMPLES_PATH, index=False)
    pd.DataFrame(paraphrase_records).to_csv(PARAPHRASE_EXAMPLES_PATH, index=False)
    print(f"Saved: {RECOVERY_EXAMPLES_PATH}")
    print(f"Saved: {PARAPHRASE_EXAMPLES_PATH}")


def _dataframe_to_dataset(frame: pd.DataFrame, real_data):
    from datasets import Dataset

    return Dataset.from_dict(
        {"text": frame["text"].astype(str).tolist(), "label": frame["label"].astype(int).tolist()},
        features=real_data.features,
    )


def _validate_alignment(source: pd.DataFrame, real_data) -> None:
    if source["original"].astype(str).tolist() != list(real_data["text"]):
        raise RuntimeError("Saved source examples do not match the current seed-11 real sample.")
    if source["label"].astype(int).tolist() != list(real_data["label"]):
        raise RuntimeError("Saved source labels do not match the current seed-11 real sample.")


def run_downstream() -> None:
    from datasets import Dataset

    from .data import combine_real_and_synthetic, load_rotten_tomatoes, sample_balanced_train
    from .evaluation import save_metrics, train_and_evaluate

    source = _load_or_create_source()
    if not RECOVERY_EXAMPLES_PATH.exists() or not PARAPHRASE_EXAMPLES_PATH.exists():
        raise FileNotFoundError("Missing generated examples. Run `data-refinery ingest` first.")

    dataset = load_rotten_tomatoes()
    real_data = sample_balanced_train(dataset["train"], EXAMPLES_PER_CLASS, SEED)
    _validate_alignment(source, real_data)

    masked_data = Dataset.from_dict(
        {"text": source["masked"].astype(str).tolist(), "label": source["label"].astype(int).tolist()},
        features=real_data.features,
    )
    recovery_data = _dataframe_to_dataset(pd.read_csv(RECOVERY_EXAMPLES_PATH), real_data)
    paraphrase_data = _dataframe_to_dataset(pd.read_csv(PARAPHRASE_EXAMPLES_PATH), real_data)

    conditions = {
        "32_real": real_data,
        "32_real_plus_32_masked_unrecovered": combine_real_and_synthetic(real_data, masked_data, SEED),
        "32_real_plus_32_corruption_guided_rewrites": combine_real_and_synthetic(real_data, recovery_data, SEED),
        "32_real_plus_32_direct_paraphrases": combine_real_and_synthetic(real_data, paraphrase_data, SEED),
    }

    metrics = [
        train_and_evaluate(train, dataset["test"], SEED, condition)
        for condition, train in conditions.items()
    ]
    metrics_frame = pd.DataFrame(metrics)
    baseline = metrics_frame.loc[metrics_frame["condition"] == "32_real"].iloc[0]
    metrics_frame["macro_f1_delta_vs_32_real"] = metrics_frame["macro_f1"] - float(baseline["macro_f1"])
    metrics_frame["accuracy_delta_vs_32_real"] = metrics_frame["accuracy"] - float(baseline["accuracy"])

    save_metrics(metrics_frame.to_dict(orient="records"), METRICS_PATH)
    print(
        metrics_frame[
            ["condition", "training_examples", "accuracy", "macro_f1", "macro_f1_delta_vs_32_real"]
        ].to_string(index=False)
    )
    print(f"\nSaved metrics to {METRICS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="data-refinery",
        description="Small synthetic-augmentation workbench for a low-data text-classification pilot.",
    )
    parser.add_argument(
        "command",
        choices=("prepare", "ingest", "run", "status", "report"),
        help="show saved metrics, check artifacts, or rerun the fixed pilot pipeline",
    )
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "ingest":
        ingest()
    elif args.command == "run":
        run_downstream()
    elif args.command == "status":
        print_status()
    else:
        print_report()


if __name__ == "__main__":
    main()
