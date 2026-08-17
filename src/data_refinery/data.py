from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from .config import DATASET_ID


def load_rotten_tomatoes() -> DatasetDict:
    """Load the Rotten Tomatoes dataset and verify required splits."""
    dataset = load_dataset(DATASET_ID)
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits.difference(dataset.keys())
    if missing_splits:
        raise ValueError(f"Dataset is missing required splits: {sorted(missing_splits)}")
    return dataset


def sample_balanced_train(train_split: Dataset, examples_per_class: int, seed: int) -> Dataset:
    """Sample an equal number of negative and positive examples."""
    if examples_per_class <= 0:
        raise ValueError("examples_per_class must be greater than zero.")

    sampled_classes: list[Dataset] = []
    for label in (0, 1):
        class_examples = train_split.filter(lambda example: example["label"] == label)
        if len(class_examples) < examples_per_class:
            raise ValueError(
                f"Class {label} contains only {len(class_examples)} examples, "
                f"but {examples_per_class} were requested."
            )
        sampled_classes.append(
            class_examples.shuffle(seed=seed).select(range(examples_per_class))
        )

    return concatenate_datasets(sampled_classes).shuffle(seed=seed)


def combine_real_and_synthetic(real_data: Dataset, synthetic_data: Dataset, seed: int) -> Dataset:
    """Align features, combine real and synthetic examples, and shuffle."""
    synthetic_data = synthetic_data.cast(real_data.features)
    return concatenate_datasets([real_data, synthetic_data]).shuffle(seed=seed)
