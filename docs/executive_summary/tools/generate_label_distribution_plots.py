#!/usr/bin/env python3
"""
Generate dark-theme label distribution plots for the executive summary.

Outputs:
  - docs/executive_summary/assets/label_distribution_generalized.png
  - docs/executive_summary/assets/label_distribution_specific.png

Counts are computed across all homes after label cleanup and filtering.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


GENERAL_ACTIVITY_MAP: dict[str, str] = {
    "Bathe": "Hygiene",
    "Personal_Hygiene": "Hygiene",
    "Groom": "Hygiene",
    "Toilet": "Hygiene",
    "Bed_Toilet_Transition": "Hygiene",
    "Dress": "Hygiene",
    "Cook": "Meal_Prep",
    "Cook_Breakfast": "Meal_Prep",
    "Cook_Lunch": "Meal_Prep",
    "Cook_Dinner": "Meal_Prep",
    "Wash_Dishes": "Meal_Cleanup",
    "Wash_Breakfast_Dishes": "Meal_Cleanup",
    "Wash_Lunch_Dishes": "Meal_Cleanup",
    "Wash_Dinner_Dishes": "Meal_Cleanup",
    "Eat": "Eating",
    "Drink": "Eating",
    "Eat_Breakfast": "Eating",
    "Eat_Lunch": "Eating",
    "Eat_Dinner": "Eating",
    "Sleep": "Sleep_Rest",
    "Go_To_Sleep": "Sleep_Rest",
    "Wake_Up": "Sleep_Rest",
    "Nap": "Sleep_Rest",
    "Sleep_Out_Of_Bed": "Sleep_Rest",
    "Work": "Work_Study",
    "Work_On_Computer": "Work_Study",
    "Work_At_Desk": "Work_Study",
    "Work_At_Table": "Work_Study",
    "Exercise": "Exercise",
    "Read": "Leisure",
    "Phone": "Leisure",
    "Relax": "Leisure",
    "Watch_TV": "Leisure",
    "Entertain_Guests": "Leisure",
    "Morning_Meds": "Medication",
    "Evening_Meds": "Medication",
    "Take_Medicine": "Medication",
    "Enter_Home": "Home_Transition",
    "Leave_Home": "Home_Transition",
    "Step_Out": "Home_Transition",
    "Laundry": "Household",
}


def dark_mpl_theme() -> None:
    # Minimal seaborn styling, with explicit dark colors that match the site.
    sns.set_theme(style="ticks", font_scale=0.95)
    plt.rcParams.update(
        {
            "figure.facecolor": "#0f1722",
            "axes.facecolor": "#0f1722",
            "axes.edgecolor": "0.55",
            "axes.labelcolor": "#e6edf3",
            "text.color": "#e6edf3",
            "xtick.color": "#a8b3bf",
            "ytick.color": "#a8b3bf",
            "grid.color": (1.0, 1.0, 1.0, 0.10),
            "grid.linestyle": "-",
            "axes.grid": True,
            "axes.titleweight": "semibold",
            "savefig.facecolor": "#0f1722",
            "savefig.edgecolor": "#0f1722",
        }
    )


def compute_label_counts(data_files: list[Path]) -> tuple[Counter[str], Counter[str], int]:
    specific = Counter()
    general = Counter()
    total_rows = 0

    for file_path in data_files:
        lf = (
            pl.scan_csv(str(file_path))
            .select(pl.col("activity"))
            .with_columns(pl.col("activity").str.replace(r"^r[12]\\.", "").alias("activity"))
            .filter(pl.col("activity") != "Other_Activity")
            .with_columns(pl.col("activity").replace(GENERAL_ACTIVITY_MAP).alias("activity_general"))
            .group_by(["activity", "activity_general"])
            .len()
        )
        combo = lf.collect()
        for act, act_gen, n in combo.iter_rows():
            specific[str(act)] += int(n)
            general[str(act_gen)] += int(n)
            total_rows += int(n)

    return specific, general, total_rows


def plot_distribution(
    labels: list[str],
    counts: list[int],
    *,
    title: str,
    xlabel: str,
    color: str,
    output_path: Path,
) -> None:
    # Scale height so category labels remain readable.
    height = max(6.0, 0.28 * len(labels))
    fig, ax = plt.subplots(figsize=(12.5, height))

    sns.barplot(x=counts, y=labels, ax=ax, color=color)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")

    # Clean spines and keep grid subtle.
    for spine in ax.spines.values():
        spine.set_alpha(0.35)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = repo_root / "data" / "raw"
    data_files = sorted(data_dir.glob("csh*/csh*.ann.features.csv"))
    if not data_files:
        raise SystemExit("No data files found under data/raw/csh*/csh*.ann.features.csv")

    dark_mpl_theme()
    specific_counts, general_counts, total_rows = compute_label_counts(data_files)
    print(f"total_rows_post_cleaning={total_rows:,}")

    # Sort high-to-low for readability.
    spec_sorted = sorted(specific_counts.items(), key=lambda kv: kv[1], reverse=True)
    gen_sorted = sorted(general_counts.items(), key=lambda kv: kv[1], reverse=True)

    spec_labels = [k for k, _ in spec_sorted]
    spec_vals = [v for _, v in spec_sorted]
    gen_labels = [k for k, _ in gen_sorted]
    gen_vals = [v for _, v in gen_sorted]

    assets_dir = repo_root / "docs" / "executive_summary" / "assets"
    plot_distribution(
        gen_labels,
        gen_vals,
        title="Generalized label distribution (log scale)",
        xlabel="Window count (log scale)",
        color="#F58518",
        output_path=assets_dir / "label_distribution_generalized.png",
    )
    plot_distribution(
        spec_labels,
        spec_vals,
        title="Specific label distribution (log scale)",
        xlabel="Window count (log scale)",
        color="#4C78A8",
        output_path=assets_dir / "label_distribution_specific.png",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

