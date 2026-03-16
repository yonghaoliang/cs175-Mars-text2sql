import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    BASELINE_SCORED, FEWSHOT_SCORED, COT_SCORED,
    REFINEMENT_COMPARED, REFINEMENT_OPTIMIZED,
)


# Early-stopping optimizer

def multi_step_early_stop(input_csv: str = REFINEMENT_COMPARED,
                          output_csv: str = REFINEMENT_OPTIMIZED) -> None:
    """
    Simulate optimal early stopping:
      - If Attempt 1 scored 10/10  → keep Attempt 1, discard the rest.
      - If Attempt 2 scored 10/10  → keep Attempt 2, discard Attempt 3.
      - Otherwise                  → keep Attempt 3.
    """
    df = pd.read_csv(input_csv)

    for col in ["Score_1", "Score_2", "Score_3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Final_Predicted_SQL" not in df.columns:
        df["Final_Predicted_SQL"] = df["Attempt_3_SQL"]

    mask_1 = df["Score_1"] == 10
    mask_2 = (df["Score_1"] != 10) & (df["Score_2"] == 10)

    # Early-stop at Attempt 1
    df.loc[mask_1, "Final_Predicted_SQL"] = df.loc[mask_1, "Attempt_1_SQL"]
    for col in ["Attempt_2_SQL", "Score_2", "Reason_2", "Attempt_3_SQL", "Score_3", "Reason_3"]:
        if col in df.columns:
            df.loc[mask_1, col] = ""

    # Early-stop at Attempt 2
    df.loc[mask_2, "Final_Predicted_SQL"] = df.loc[mask_2, "Attempt_2_SQL"]
    for col in ["Attempt_3_SQL", "Score_3", "Reason_3"]:
        if col in df.columns:
            df.loc[mask_2, col] = ""

    df.to_csv(output_csv, index=False)

    total     = len(df)
    cleared_1 = int(mask_1.sum())
    cleared_2 = int(mask_2.sum())
    kept      = total - cleared_1 - cleared_2

    print("Optimization complete! Saved to:", output_csv)
    print(f"  Total Rows:                    {total}")
    print(f"  Stopped at Attempt 1 (10/10):  {cleared_1}")
    print(f"  Stopped at Attempt 2 (10/10):  {cleared_2}")
    print(f"  Went all the way to Attempt 3: {kept}")

# Per-step improvement charts

def plot_refinement_improvement(input_csv: str = REFINEMENT_COMPARED) -> None:
    """
    Two side-by-side bar charts:
      Left  – improvement from Attempt 1 → Attempt 2
      Right – improvement from Attempt 2 → Attempt 3
    """
    df = pd.read_csv(input_csv)
    for col in ["Score_1", "Score_2", "Score_3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Imp_2_vs_1"] = df["Score_2"] - df["Score_1"]
    df["Imp_3_vs_2"] = df["Score_3"] - df["Score_2"]

    improved_2_vs_1 = df[df["Imp_2_vs_1"] > 0]["Imp_2_vs_1"].value_counts().sort_index()
    improved_3_vs_2 = df[df["Imp_3_vs_2"] > 0]["Imp_3_vs_2"].value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    def _annotated_bar(ax, series, color, title):
        bars = ax.bar(series.index, series.values, color=color, edgecolor="white", alpha=0.9)
        ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
        ax.set_xlabel("Points Improved By", fontsize=13)
        ax.set_ylabel("Number of Queries", fontsize=13)
        if not series.empty:
            ax.set_xticks(np.arange(1, max(series.index) + 1))
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{int(h)}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold", fontsize=11,
                )

    _annotated_bar(ax1, improved_2_vs_1, "#ff7f0e", "First Loop: Attempt 1 → Attempt 2")
    _annotated_bar(ax2, improved_3_vs_2, "#2ca02c", "Second Loop: Attempt 2 → Attempt 3")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "refinement_improvement.png"), dpi=150)
    plt.show()

    print("=== TEXT SUMMARY ===")
    print(f"Queries that improved on Attempt 2 (vs 1): {improved_2_vs_1.sum()}")
    for pts, cnt in improved_2_vs_1.items():
        print(f"  -> {int(cnt)} queries improved by {int(pts)} points")
    print(f"\nQueries that improved on Attempt 3 (vs 2): {improved_3_vs_2.sum()}")
    for pts, cnt in improved_3_vs_2.items():
        print(f"  -> {int(cnt)} queries improved by {int(pts)} points")


# Score distribution across methods


def plot_score_distribution(
    baseline_csv: str = BASELINE_SCORED,
    cot_csv:      str = COT_SCORED,
    fewshot_csv:  str = FEWSHOT_SCORED,
) -> None:
    """Grouped bar chart comparing AI_Score distributions across all three methods."""
    baseline_df = pd.read_csv(baseline_csv)
    cot_df      = pd.read_csv(cot_csv)
    fewshot_df  = pd.read_csv(fewshot_csv)

    for df in (baseline_df, cot_df, fewshot_df):
        df["Score"] = pd.to_numeric(df["AI_Score"], errors="coerce")

    all_scores = (
        set(baseline_df["Score"].dropna().unique())
        | set(cot_df["Score"].dropna().unique())
        | set(fewshot_df["Score"].dropna().unique())
    )
    scores = sorted(int(x) for x in all_scores if pd.notna(x))

    vc_base = baseline_df["Score"].value_counts().reindex(scores, fill_value=0)
    vc_cot  = cot_df["Score"].value_counts().reindex(scores, fill_value=0)
    vc_few  = fewshot_df["Score"].value_counts().reindex(scores, fill_value=0)

    x     = np.arange(len(scores))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    r1 = ax.bar(x - width, vc_base, width, label="Baseline",              color="#1f77b4", edgecolor="white", alpha=0.9)
    r2 = ax.bar(x,          vc_cot,  width, label="Chain-of-Thought (CoT)",color="#ff7f0e", edgecolor="white", alpha=0.9)
    r3 = ax.bar(x + width,  vc_few,  width, label="Few-Shot",              color="#2ca02c", edgecolor="white", alpha=0.9)

    ax.set_ylabel("Number of Queries", fontsize=14)
    ax.set_xlabel("AI Execution Score (0 = Failed Completely, 10 = Perfect)", fontsize=14)
    ax.set_title("Score Distribution Across Prompting Strategies", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(scores)
    ax.legend(fontsize=12, title="Prompting Strategy")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    def _autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{int(h)}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10,
                )

    _autolabel(r1)
    _autolabel(r2)
    _autolabel(r3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score_distribution.png"), dpi=150)
    plt.show()


# Entry point

if __name__ == "__main__":
    multi_step_early_stop()
    plot_refinement_improvement()
    plot_score_distribution()