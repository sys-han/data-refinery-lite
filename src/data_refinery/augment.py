import json
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset

from .config import EXPECTED_EXAMPLES, MASK_TOKEN


def _looks_like_initial_sequence(span_tokens: list[str]) -> bool:
    cleaned = [token.strip(".,!?;:'\"()[]{}-") for token in span_tokens]
    single_letters = sum(1 for token in cleaned if len(token) == 1 and token.isalpha())
    empty_after_strip = sum(1 for token in cleaned if not token)
    return single_letters >= 2 or (single_letters >= 1 and empty_after_strip >= 1)


def mask_contiguous_span(
    text: str,
    masking_rate: float,
    seed: int,
    mask_token: str = MASK_TOKEN,
    min_context_tokens: int = 1,
    max_span_tokens: int = 4,
) -> dict[str, Any]:
    if not 0.0 < masking_rate < 1.0:
        raise ValueError("masking_rate must be between 0 and 1.")
    if min_context_tokens < 0:
        raise ValueError("min_context_tokens cannot be negative.")
    if max_span_tokens <= 0:
        raise ValueError("max_span_tokens must be greater than zero.")

    tokens = text.split()
    if not tokens:
        raise ValueError("Cannot mask an empty string.")

    context = min_context_tokens
    span_capacity = len(tokens) - 2 * context
    if span_capacity < 1:
        context = 0
        span_capacity = len(tokens) - 1 if len(tokens) > 1 else 1

    span_length = max(1, round(len(tokens) * masking_rate))
    span_length = min(span_length, max_span_tokens, span_capacity)

    first_start = context
    last_start = len(tokens) - context - span_length
    candidate_starts = list(range(first_start, last_start + 1))

    preferred_starts = []
    for start in candidate_starts:
        span = tokens[start : start + span_length]
        has_word = any(any(ch.isalnum() for ch in token) for token in span)
        crosses_sentence_boundary = any(token in {".", "!", "?"} for token in span)
        if has_word and not _looks_like_initial_sequence(span) and not crosses_sentence_boundary:
            preferred_starts.append(start)

    valid_starts = preferred_starts or candidate_starts
    if not valid_starts:
        raise ValueError("Could not find a valid span-masking position.")

    start = random.Random(seed).choice(valid_starts)
    end = start + span_length
    masked_tokens = tokens[:start] + [mask_token] + tokens[end:]

    return {
        "masked_text": " ".join(masked_tokens),
        "masked_span": " ".join(tokens[start:end]),
        "total_token_count": len(tokens),
        "masked_token_count": span_length,
        "mask_start_index": start,
        "mask_end_index": end,
        "left_context_token_count": start,
        "right_context_token_count": len(tokens) - end,
        "actual_masking_rate": span_length / len(tokens),
    }


def build_masked_dataset(
    real_data: Dataset,
    masking_rate: float,
    seed: int,
    mask_token: str = MASK_TOKEN,
    min_context_tokens: int = 1,
    max_span_tokens: int = 4,
) -> tuple[Dataset, list[dict[str, Any]]]:
    masked_texts: list[str] = []
    records: list[dict[str, Any]] = []

    # Keep one deterministic masked variant per fixed real example so that all
    # augmentation conditions share the same 32-example source set.
    for index, (original, label) in enumerate(zip(real_data["text"], real_data["label"])):
        mask = mask_contiguous_span(
            text=original,
            masking_rate=masking_rate,
            seed=seed * 100_000 + index,
            mask_token=mask_token,
            min_context_tokens=min_context_tokens,
            max_span_tokens=max_span_tokens,
        )
        masked_texts.append(mask["masked_text"])
        records.append(
            {
                "label": label,
                "sentiment": "positive" if label == 1 else "negative",
                "original": original,
                "masked": mask["masked_text"],
                "masked_span": mask["masked_span"],
                "total_token_count": mask["total_token_count"],
                "masked_token_count": mask["masked_token_count"],
                "mask_start_index": mask["mask_start_index"],
                "mask_end_index": mask["mask_end_index"],
                "left_context_token_count": mask["left_context_token_count"],
                "right_context_token_count": mask["right_context_token_count"],
                "actual_masking_rate": mask["actual_masking_rate"],
            }
        )

    masked_dataset = Dataset.from_dict(
        {"text": masked_texts, "label": list(real_data["label"])},
        features=real_data.features,
    )
    return masked_dataset, records


def build_corruption_guided_prompt(source: pd.DataFrame) -> str:
    lines = [
        "You are generating nonredundant training examples for a controlled text-augmentation experiment.",
        "",
        "For each masked movie review, reconstruct the missing meaning and rewrite the complete review using wording that is meaningfully different from the visible source text.",
        "",
        "Rules:",
        "- Preserve the original evaluative meaning and sentiment.",
        "- Do not reproduce the visible review verbatim.",
        "- Use different wording or sentence structure.",
        "- Do not add or remove people, events, claims, opinions, or factual details.",
        "- Do not strengthen, weaken, or reverse the evaluation.",
        "- Do not mention the task or explain your answers.",
        f"- Do not include {MASK_TOKEN} in any output.",
        "- Return valid JSON only, with no Markdown code fence.",
        "- Return exactly one object per item in the same order.",
        "",
        "Required JSON format:",
        '[{"id": 1, "rewrite": "complete rewritten review"}, {"id": 2, "rewrite": "complete rewritten review"}]',
        "",
        "Items:",
    ]
    lines.extend(f"{int(row.id)}. {row.masked}" for row in source.itertuples(index=False))
    return "\n".join(lines) + "\n"


def build_direct_paraphrase_prompt(source: pd.DataFrame) -> str:
    lines = [
        "You are generating direct paraphrases for a controlled text-augmentation experiment.",
        "",
        "Rewrite each movie review using meaningfully different wording while preserving its original evaluative meaning and sentiment.",
        "",
        "Rules:",
        "- Preserve the original sentiment and central evaluation.",
        "- Do not reproduce the review verbatim.",
        "- Use different wording or sentence structure.",
        "- Do not add or remove people, events, claims, opinions, or factual details.",
        "- Do not strengthen, weaken, or reverse the evaluation.",
        "- Do not explain your answers.",
        "- Return valid JSON only, with no Markdown code fence.",
        "- Return exactly one object per item in the same order.",
        "",
        "Required JSON format:",
        '[{"id": 1, "paraphrase": "rewritten review"}, {"id": 2, "paraphrase": "rewritten review"}]',
        "",
        "Items:",
    ]
    lines.extend(f"{int(row.id)}. {row.original}" for row in source.itertuples(index=False))
    return "\n".join(lines) + "\n"


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def read_json_response(path: Path, text_field: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing response file: {path}")

    parsed = json.loads(_strip_markdown_fence(path.read_text(encoding="utf-8")))
    if isinstance(parsed, dict):
        if isinstance(parsed.get("items"), list):
            parsed = parsed["items"]
        elif isinstance(parsed.get("results"), list):
            parsed = parsed["results"]

    if not isinstance(parsed, list):
        raise ValueError(f"{path} must contain a JSON list of objects.")

    normalized: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError(f"Every item in {path} must be an object.")
        if "id" not in item or text_field not in item:
            raise ValueError(f"Every item in {path} must contain 'id' and '{text_field}'.")
        normalized.append({"id": int(item["id"]), text_field: str(item[text_field]).strip()})

    expected_ids = list(range(1, EXPECTED_EXAMPLES + 1))
    received_ids = sorted(item["id"] for item in normalized)
    if received_ids != expected_ids:
        raise ValueError(f"{path} must contain IDs 1 through {EXPECTED_EXAMPLES}. Received: {received_ids}")

    return normalized
