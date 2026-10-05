"""Plot pooled and per-poem inter-stress interval CV distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from adapt_eeg.poem_rhythm import IRREGULAR_CV_MIN, REGULAR_CV_MAX


def plot(input_csv: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_csv)
    valid = frame.dropna(subset=["coefficient_of_variation"]).copy()
    poems = sorted(valid["poem"].astype(int).unique())

    figure, axes = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    axes[0].hist(
        valid["coefficient_of_variation"],
        bins=30,
        color="#4c78a8",
        edgecolor="white",
        alpha=0.9,
    )
    axes[0].axvline(
        REGULAR_CV_MAX,
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label=f"Regular < {REGULAR_CV_MAX:.2f}",
    )
    axes[0].axvline(
        IRREGULAR_CV_MIN,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Irregular > {IRREGULAR_CV_MIN:.2f}",
    )
    axes[0].set(
        xlabel="Coefficient of variation of inter-stress intervals",
        ylabel="Selected lines",
        title=f"Pooled CV distribution (n={len(valid)})",
    )
    axes[0].legend()

    values = [
        valid.loc[valid["poem"] == poem, "coefficient_of_variation"].to_numpy()
        for poem in poems
    ]
    axes[1].boxplot(values, tick_labels=[str(poem) for poem in poems], showfliers=True)
    axes[1].axhspan(
        0,
        REGULAR_CV_MAX,
        color="#2ca02c",
        alpha=0.08,
        label="Regular region",
    )
    axes[1].axhspan(
        REGULAR_CV_MAX,
        IRREGULAR_CV_MIN,
        color="#7f7f7f",
        alpha=0.08,
        label="Ambiguous region",
    )
    axes[1].axhspan(
        IRREGULAR_CV_MIN,
        max(float(valid["coefficient_of_variation"].max()), IRREGULAR_CV_MIN),
        color="#d62728",
        alpha=0.08,
        label="Irregular region",
    )
    axes[1].axhline(REGULAR_CV_MAX, color="#ff7f0e", linestyle="--")
    axes[1].axhline(IRREGULAR_CV_MIN, color="#d62728", linestyle="--")
    axes[1].set(
        xlabel="Poem",
        ylabel="Coefficient of variation",
        title="CV distribution by poem",
    )
    axes[1].legend(loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/poem_itpc/poem_line_rhythm.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/exploration/cv_distribution.png"),
    )
    args = parser.parse_args()
    plot(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
