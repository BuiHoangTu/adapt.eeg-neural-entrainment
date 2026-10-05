"""Plot channel-wise one-sided p-values for the chosen ITPC configuration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


SUMMARY_CSV = Path("results/frequency_comparison/chosen_cycle_2-1_by_channel_summary.csv")
OUTPUT_DIR = Path("results/frequency_comparison")
CONTRASTS = {
    "Regular minus irregular at f": (
        "Regular minus irregular at poetic frequency",
        "chosen_2-1_regular_minus_irregular_one_sided_p.png",
    ),
    "Regular lines: f minus 2f": (
        "Regular lines: poetic frequency greater than doubled frequency",
        "chosen_2-1_regular_f_vs_2f_one_sided_p.png",
    ),
}


def plot_contrast(frame: pd.DataFrame, title: str, output_name: str) -> None:
    frame = frame.loc[frame["channel"] != "All EEG channels"].copy()
    channel_names = frame["channel"].tolist()
    p_values = frame["wilcoxon_p_one_sided"].to_numpy(dtype=float)

    info = mne.create_info(channel_names, sfreq=1.0, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False)
    significant = p_values < 0.05
    log_p_values = -np.log10(p_values)

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image, _ = mne.viz.plot_topomap(
        log_p_values,
        info,
        axes=axis,
        show=False,
        sensors=True,
        names=channel_names,
        mask=significant,
        mask_params={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "linewidth": 1.5,
            "markersize": 9,
            "linestyle": "none",
        },
        contours=(-np.log10(np.array([0.10, 0.05, 0.01]))),
        cmap="viridis",
        vlim=(0.0, 3.0),
    )
    axis.set_title(f"{title}\nOne-sided Wilcoxon p-value", pad=18)
    colorbar = figure.colorbar(image, ax=axis, shrink=0.78, pad=0.08)
    colorbar.set_label("One-sided p-value")
    p_ticks = np.array([1.0, 0.10, 0.05, 0.01, 0.001])
    colorbar.set_ticks(-np.log10(p_ticks))
    colorbar.set_ticklabels(["1", "0.10", "0.05", "0.01", "0.001"])
    figure.text(
        0.5,
        0.015,
        "Outlined sensors: p < 0.05; channel-wise p-values are uncorrected",
        ha="center",
        fontsize=9,
    )
    figure.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_CSV)
    for contrast, (title, output_name) in CONTRASTS.items():
        frame = summary.loc[summary["contrast"] == contrast]
        if frame.empty:
            raise ValueError(f"Missing contrast in {SUMMARY_CSV}: {contrast}")
        plot_contrast(frame, title, output_name)


if __name__ == "__main__":
    main()
