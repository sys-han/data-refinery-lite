"""Optional Streamlit summary app.

Run with:
    pip install streamlit
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd


METRICS_PATH = Path("results/st04_one_seed_downstream_metrics.csv")


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install streamlit first: pip install streamlit") from exc

    st.set_page_config(page_title="Data Refinery", layout="wide")
    st.title("Data Refinery Lite")
    st.caption("Synthetic augmentation evaluation for low-data text classification")

    st.markdown(
        """
        This demo summarizes a fixed one-seed pilot comparing real-only training,
        masked unrecovered examples, corruption-guided rewrites, and direct paraphrases.
        """
    )

    metrics = pd.read_csv(METRICS_PATH)
    display = metrics[
        [
            "condition",
            "training_examples",
            "accuracy",
            "macro_f1",
            "macro_f1_delta_vs_32_real",
        ]
    ].copy()
    st.dataframe(display, use_container_width=True)

    best = metrics.sort_values("macro_f1", ascending=False).iloc[0]
    st.metric("Best condition", best["condition"])
    st.metric("Best macro-F1", f"{best['macro_f1']:.4f}")
    st.metric("Delta vs 32 real", f"{best['macro_f1_delta_vs_32_real']:+.4f}")

    st.warning(
        "This is a one-seed pilot, not statistically reliable multi-seed evidence."
    )


if __name__ == "__main__":
    main()
