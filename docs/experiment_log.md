# Experiment Log


## Reference Baselines

In addition to the 32-example low-data baseline, the experiment includes:

- **64-real baseline:** 32 real examples per class, used to compare
  32 real + 32 synthetic examples against the same total number of
  real training examples.
- **Full-data baseline:** the complete official training split, used as
  a sanity check that the downstream classification pipeline can learn
  the task when sufficient labeled data is available.


## Experiment 001 — Real-Only Baseline

- Dataset: Rotten Tomatoes
- Date: 2026-08-03
- Training budget: 16 examples per class
- Total training examples: 32
- Classifier: TF-IDF + Logistic Regression
- Seeds: [11, 12, 132, 144]
- Test split: Official Rotten Tomatoes test split


### Results

| Condition | Training Examples | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 32 real | 32 | 0.5279 ± 0.0060 | 0.5255 ± 0.0085 |
| 64 real | 64 | 0.5485 ± 0.0101 | 0.5447 ± 0.0112 |
| Full real | 8,530 | 0.7824 | 0.7824 |

The full-data result confirms that the downstream pipeline can learn the
task when sufficient labeled data is available. Doubling the low-data
training budget from 32 to 64 real examples improved mean macro-F1 by
0.0192, establishing a real-data reference point for evaluating the value
of 32 synthetic examples.


## Smoke-Test Checklist

### ST-00 — Unmarked deletion recovery
- [x] End-to-end pipeline executes
- [x] Generated samples are saved
- [ ] Outputs are coherent
- [ ] Original sentiment is preserved
- [ ] Repetition and generic sentiment shortcuts are rare

Result: Failed quality gate.

FLAN-T5-small frequently generated repetitive generic sentiment phrases,
copied corrupted inputs, or changed the original sentiment. Unmarked token
deletion will not be scaled in its current form.


### ST-01 — Span-masking correctness
- [x] Use 8 examples from seed 11
- [x] Replace one contiguous span with `<extra_id_0>`
- [x] Mask approximately 20% of tokens
- [x] Never produce an empty sentence
- [x] Preserve the original label metadata
- [x] The same seed produces the same corruption
- [x] Different examples do not always mask the same position

Result: Passed.


### ST-02 v1 — FLAN-T5 instruction-based span recovery

- [x] No empty outputs
- [x] No `sloppy/sluggish/stale` repetition loops
- [ ] At least 6 of 8 outputs are grammatical and relevant
- [ ] At least 7 of 8 preserve the original sentiment
- [x] At least 4 of 8 are not exact copies of the original
- [x] Outputs do not introduce unrelated people, events, or opinions

Result: Failed quality gate.

All eight generations were non-empty and avoided the repetition loops seen
in ST-00. However, FLAN-T5-base frequently copied nearby words rather than
reconstructing the missing spans, producing duplicated or ungrammatical
reviews. Because the recovered sentences were not semantically reliable,
sentiment preservation could not be confidently established.


### ST-02 v2 — Native T5 sentinel infilling

Result: Failed quality gate.

The native T5 interface successfully produced parseable sentinel-formatted
outputs, confirming that the input and extraction pipeline worked. However,
the generated spans were generally incomplete, ungrammatical, or
semantically inconsistent with the original reviews. Only a small minority
of recovered sentences were plausible, falling well below the required
quality and sentiment-preservation thresholds.


### ST-02 v3 — BART native denoising

- [x] No empty outputs
- [x] No repetition loops
- [ ] At least 6 of 8 outputs are grammatical and semantically reliable
- [ ] At least 7 of 8 confidently preserve the original sentiment
- [x] At least 4 of 8 are not exact copies of the original
- [x] Outputs do not introduce unrelated people, events, or opinions

Result: Failed quality gate, but substantially improved over both T5
configurations.

BART-base produced eight format-valid full-sentence reconstructions and
removed every `<mask>` token. Several outputs were fluent and preserved the
review's general meaning. However, unrestricted 20% masking sometimes
removed long sentence openings, sentence-final sentiment cues, or
punctuation-heavy name spans.

Manual review found fewer than 6 of 8 outputs clearly grammatical and
semantically reliable. One negative review was explicitly changed from
`naughty` to `good`, demonstrating label-flip risk.

This result suggests that BART is a viable recovery engine, but the original
corruption policy is too destructive for short reviews. The next test keeps
BART-base fixed while reducing and constraining the masked span.


### ST-02 v4 — Constrained BART denoising

- [x] No empty outputs
- [x] No repetition loops
- [ ] At least 6 of 8 outputs are grammatical and semantically reliable
- [ ] At least 7 of 8 confidently preserve the original sentiment
- [x] No explicit label flips
- [x] Outputs do not introduce unrelated people, events, or opinions

Result: Failed quality gate, but produced the strongest recovery quality
among the tested configurations.

Constrained masking reduced the major failure modes observed in the
unrestricted BART experiment. All eight outputs were non-empty, removed
the mask token, avoided repetition, and introduced no explicit label
flips or unrelated content.

However, conservative manual review found approximately 5 of 8 outputs
grammatical and semantically reliable and approximately 6 of 8 outputs
with clearly preserved sentiment. Several reconstructions remained
awkward, incomplete, or changed the relationship between important
sentence elements.

The constrained corruption policy improved recovery quality but did not
reach the predefined threshold required for downstream scaling.
Zero-shot recovery will therefore not be expanded to the full
augmentation budget in the current MVP.


### ST-02 v5 — Hosted LLM corruption-guided rewriting

Configuration:

- Execution mode: Manual batch inference through ChatGPT UI
- Response collection: Copied manually from the chat output
- API used: No
- Examples: The same 8 seed-11 examples used in ST-02 v4
- Gold sentiment labels supplied: No
- Solved examples supplied: No
- Prompt version: `st02-v5-chatgpt-ui-zero-shot-v2`

Quality gate:

- [x] No empty outputs
- [x] No repetition loops
- [x] At least 6 of 8 outputs are grammatical and semantically reliable
- [x] At least 7 of 8 outputs preserve the original sentiment
- [x] At least 4 of 8 outputs are not exact copies of the original
- [x] No unrelated people, events, or opinions are introduced

Result: Passed.

The hosted LLM produced eight fluent and non-identical rewrites. All
eight preserved the original sentiment, while seven of eight
conservatively preserved the original semantic content. One output
introduced minor additional specificity by rewriting "bits" as
"comedy bits," but this did not alter the central evaluation.

The experiment therefore passed the predefined quality gate and
advanced to the one-seed downstream pilot.


### ST-03 — Direct paraphrase quality
- [x] Generate 8 paraphrases from the same real examples
- [x] At least 7 of 8 preserve sentiment
- [x] No empty or repetitive outputs
- [x] At least 4 of 8 differ meaningfully from the original wording

Result: Passed.

All eight direct paraphrases were fluent, non-empty, and distinct from
the original wording. All eight preserved the original sentiment and
central evaluation. Two outputs made minor stylistic adjustments to the
strength or metaphor of the source review, but neither changed its label
or substantive meaning.

The direct-paraphrase method passed the quality gate and was advanced to
the one-seed downstream pilot.


### ST-04 — One-seed downstream utility
- [x] Seed 11 only
- [x] 32 real examples
- [x] 32 masked/unrecovered examples
- [x] 32 corruption-guided rewrite examples
- [x] 32 direct paraphrase examples
- [x] All augmented conditions use exactly 32 added examples
- [x] All conditions use the same original 32 real examples
- [x] All conditions are evaluated on the same official test split
- [x] Save macro-F1 for all four conditions

Result: Passed.

In the one-seed downstream pilot, adding unrecovered masked examples
reduced macro-F1 from 0.5296 to 0.5240, showing that corruption alone did
not improve the classifier.

Direct paraphrases produced only a marginal increase to 0.5309.
In contrast, corruption-guided rewrites increased macro-F1 to 0.5452,
a gain of 1.57 percentage points over the 32-example baseline.

This result was also close to the seed-11 performance obtained with 64
real examples (0.5462 macro-F1). While a single-seed pilot cannot establish
statistical reliability, it provides preliminary evidence that
corruption-guided rewriting can produce more useful low-data augmentation
than direct paraphrasing.