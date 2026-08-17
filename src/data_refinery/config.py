from pathlib import Path

SEED = 11
EXAMPLES_PER_CLASS = 16
EXPECTED_EXAMPLES = 32

MASKING_RATE = 0.15
MAX_SPAN_TOKENS = 4
MIN_CONTEXT_TOKENS = 2
MASK_TOKEN = "<mask>"

RESULTS_DIR = Path("results")

SOURCE_PATH = RESULTS_DIR / "st04_source_examples.csv"
RECOVERY_PROMPT_PATH = RESULTS_DIR / "st04_corruption_guided_prompt.txt"
PARAPHRASE_PROMPT_PATH = RESULTS_DIR / "st04_direct_paraphrase_prompt.txt"
RECOVERY_RESPONSE_PATH = RESULTS_DIR / "st04_corruption_guided_response.json"
PARAPHRASE_RESPONSE_PATH = RESULTS_DIR / "st04_direct_paraphrase_response.json"
RECOVERY_EXAMPLES_PATH = RESULTS_DIR / "st04_corruption_guided_examples.csv"
PARAPHRASE_EXAMPLES_PATH = RESULTS_DIR / "st04_direct_paraphrase_examples.csv"
METRICS_PATH = RESULTS_DIR / "st04_one_seed_downstream_metrics.csv"

RECOVERY_PROMPT_VERSION = "st04-chatgpt-ui-zero-shot-corruption-guided-v1"
PARAPHRASE_PROMPT_VERSION = "st04-chatgpt-ui-zero-shot-direct-paraphrase-v1"
EXECUTION_MODE = "ChatGPT UI manual batch"

DATASET_ID = "cornell-movie-review-data/rotten_tomatoes"
