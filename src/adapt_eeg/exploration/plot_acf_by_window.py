"""Plot the line-level ACF peak windows used by the poem selector."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adapt_eeg.exploration.stress_rhythm_evidence import EEG_POEM_IDS, analyze_poem


def plot(poems: list[int], output_path: Path) -> None:
    lines = pd.concat([analyze_poem(poem) for poem in poems], ignore_index=True)
    lines["acf_peak_window_s"] = 1.0 / lines["intensity_autocorrelation_peak_hz"]
    lines[
        [
            "poem",
            "line_number",
            "intensity_autocorrelation_peak_hz",
            "acf_peak_window_s",
        ]
    ].to_csv(output_path.with_name("acf_line_peak_windows.csv"), index=False)

    figure, axis = plt.subplots(figsize=(9, 5.8))
    rng = np.random.default_rng(0)
    plotted = []
    for position, poem in enumerate(poems, start=1):
        frame = lines.loc[lines["poem"] == poem]
        windows = frame["acf_peak_window_s"].dropna().to_numpy()
        plotted.append(windows)
        jitter = rng.uniform(-0.09, 0.09, len(windows))
        axis.scatter(
            position + jitter,
            windows,
            alpha=0.55,
            s=24,
            color="tab:blue",
        )
        selected_frequency = frame["intensity_autocorrelation_peak_hz"].median()
        selected_window = 1.0 / selected_frequency
        axis.scatter(
            position,
            selected_window,
            marker="D",
            color="black",
            s=65,
            zorder=3,
            label="Selected poem window" if position == 1 else None,
        )
        axis.annotate(
            f"{selected_window:.3f}s",
            (position, selected_window),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    axis.boxplot(plotted, positions=range(1, len(poems) + 1), widths=0.45)
    axis.set(
        xticks=range(1, len(poems) + 1),
        xticklabels=[f"Poem {poem}" for poem in poems],
        ylim=(0, 1.02),
        ylabel="Line-level prominent ACF peak window (s)",
        title="Line-level ACF peak windows and poem-level selection",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poems", type=int, nargs="+", default=list(EEG_POEM_IDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/exploration/acf_by_window_size.png"),
    )
    args = parser.parse_args()
    plot(args.poems, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
