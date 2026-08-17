# Data Refinery Lite

**A lightweight synthetic-data workbench for testing whether generated training examples actually improve low-data classifiers.**

This project turns a small augmentation experiment into a reproducible evaluation flow:

1. sample a fixed low-data training set;
2. create controlled corruptions;
3. generate two hosted-LLM augmentation prompts;
4. ingest manually collected model outputs;
5. compare downstream performance across real-only, corrupted, corruption-guided, and paraphrase-based conditions.

The current release is intentionally small: it is a workbench based on **one-seed downstream pilot**, not a general claim across datasets or seeds.

---

## Why this exists

Synthetic data is easy to generate, but generated examples are not automatically useful. This project treats augmentation as an evaluation problem:

> Generate variants, apply simple quality gates, and measure whether the downstream classifier improves.

The pilot asks whether **corruption-guided rewriting** produces more useful low-data augmentation than direct paraphrasing or simply adding damaged text.

---

## Experiment setup

- Dataset: `cornell-movie-review-data/rotten_tomatoes`
- Task: binary sentiment classification
- Real-data budget: 16 negative + 16 positive examples
- Augmentation budget: one synthetic example per real example
- Downstream model: TF-IDF + Logistic Regression
- Final pilot seed: `11`
- Evaluation split: official test split

Every augmented condition uses the same original 32 real examples and exactly 32 additional examples. Test examples are never used for training, generation, or prompt selection.

---

## Conditions

| Condition | Training set |
|---|---|
| `32_real` | 32 original labeled examples |
| `32_real_plus_32_masked_unrecovered` | 32 real + 32 masked/damaged examples |
| `32_real_plus_32_corruption_guided_rewrites` | 32 real + 32 hosted-LLM rewrites from masked inputs |
| `32_real_plus_32_direct_paraphrases` | 32 real + 32 hosted-LLM direct paraphrases |

## Design notes

I framed this as an evaluation problem rather than a generation problem. Synthetic examples are cheap to create, but plausible-looking examples are not automatically useful for downstream learning.

The pipeline therefore compares corruption-guided rewriting against three controls: real-only training, unrecovered masked examples, and direct paraphrases. All augmented conditions use the same original 32 real examples, the same augmentation budget, and the same official test split.

The goal is not to claim that this method generalizes across domains. The goal is to make the augmentation effect measurable instead of relying on surface-level output quality.

---

## Results

| Condition | Training examples | Accuracy | Macro-F1 | Macro-F1 delta vs. 32 real |
|---|---:|---:|---:|---:|
| 32 real | 32 | 0.5319 | 0.5296 | — |
| + 32 masked, unrecovered | 64 | 0.5253 | 0.5240 | -0.0056 |
| + 32 corruption-guided rewrites | 64 | **0.5516** | **0.5452** | **+0.0157** |
| + 32 direct paraphrases | 64 | 0.5356 | 0.5309 | +0.0014 |

Reference baselines:

| Reference condition | Macro-F1 |
|---|---:|
| 64 real, seed 11 | 0.5462 |
| Full real training data | 0.7824 |

### Interpretation

In this fixed seed-11 pilot, corruption-guided rewrites produced the strongest downstream result. Direct paraphrases barely improved over the 32-example baseline, and adding masked but unrecovered examples reduced macro-F1.

This supports a directional result:

> In this one-seed low-data pilot, corruption-guided rewriting produced more useful augmentation than direct paraphrasing or unrecovered corruption.

Yet, there is no sufficient evidence that the method reliably improves performance across seeds, datasets, classifiers, or model snapshots.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
data-refinery report
```

This prints the saved result table from the fixed seed-11 pilot.

## Reproducing the pipeline

Check whether all expected artifacts exist:

```bash
data-refinery status
```

Recreate the fixed source set and prompt files:

```bash
data-refinery prepare
```

After saving hosted-LLM JSON responses into `results/`, ingest them:

```bash
data-refinery ingest
```

Run the downstream comparison:

```bash
data-refinery run
```

Final metrics are saved to:

```text
results/st04_one_seed_downstream_metrics.csv
```

---

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── app.py
├── src/
│   └── data_refinery/
│       ├── __init__.py
│       ├── __main__.py
│       ├── augment.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       └── report.py
├── scripts/
│   ├── run_pipeline.py
│   └── show_results.py
├── results/
│   ├── st04_source_examples.csv
│   ├── st04_corruption_guided_prompt.txt
│   ├── st04_direct_paraphrase_prompt.txt
│   ├── st04_corruption_guided_response.json
│   ├── st04_direct_paraphrase_response.json
│   ├── st04_corruption_guided_examples.csv
│   ├── st04_direct_paraphrase_examples.csv
│   └── st04_one_seed_downstream_metrics.csv
└── docs/
    ├── experiment_log.md
    └── website_copy.md
```

---

## What failed and why it matters

The project includes useful negative results:

- Unmarked deletion gave the model too little information about what had been removed.
- Small local recovery models often generated awkward, incomplete, or sentiment-altering text.
- Strong exact recovery could reproduce the original review without creating nonredundant training data.
- Corruption alone did not improve downstream performance.

These failures motivated the final corruption-guided rewriting setup and the use of explicit quality gates before downstream evaluation.

---

## Limitations

- The final downstream comparison uses only seed 11.
- Hosted-LLM generations were collected manually through the ChatGPT UI.
- The hosted model snapshot and decoding parameters were not controlled through an API.
- The quality review was manual and small-scale.
- Results come from one dataset, one downstream classifier, and one augmentation budget.

---

## Future work

- Repeat generation and downstream evaluation across additional seeds.
- Add automated semantic-similarity and sentiment-consistency filters.
- Compare hosted models under fixed API snapshots.
- Test additional low-data text domains after leakage and label-boundary audits.
- Package the workflow into a reusable augmentation-audit harness.
