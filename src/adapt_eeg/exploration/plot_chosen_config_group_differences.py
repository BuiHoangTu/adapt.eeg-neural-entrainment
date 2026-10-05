"""Plot chosen-config ITPC differences separately by language group."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats


POETIC_FREQUENCY_CSV = Path("results/poem_itpc/itpc_by_line_channel.csv")
DOUBLE_FREQUENCY_CSV = Path(
    "results/frequency_comparison/cycle_2-1_double_poetic_frequency_by_channel.csv"
)
OUTPUT_DIR = Path("results/frequency_comparison")
GROUP_LABELS = {
    "english_competence": "Bilingual",
    "non_english_competence": "Monolingual",
}


def _wilcoxon_row(
    contrast: str,
    group: str,
    channel: str,
    left: pd.Series,
    right: pd.Series,
) -> dict:
    one_sided = stats.wilcoxon(left, right, alternative="greater")
    two_sided = stats.wilcoxon(left, right, alternative="two-sided")
    return {
        "contrast": contrast,
        "language_group": group,
        "channel": channel,
        "n_participants": len(left),
        "itpc_difference": float((left - right).mean()),
        "wilcoxon_W_one_sided": float(one_sided.statistic),
        "wilcoxon_p_one_sided": float(one_sided.pvalue),
        "wilcoxon_W_two_sided": float(two_sided.statistic),
        "wilcoxon_p_two_sided": float(two_sided.pvalue),
    }


def group_wilcoxon_tests(
    poetic: pd.DataFrame, doubled: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    participant_condition = (
        poetic.groupby(
            ["participant", "language_group", "channel", "rhythm_condition"],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
        .pivot(
            index=["participant", "language_group", "channel"],
            columns="rhythm_condition",
            values="itpc_squared_debiased",
        )
        .dropna(subset=["regular", "irregular"])
        .reset_index()
    )
    for (group, channel), frame in participant_condition.groupby(
        ["language_group", "channel"], sort=True
    ):
        rows.append(
            _wilcoxon_row(
                "Regular minus irregular at f",
                group,
                channel,
                frame["regular"],
                frame["irregular"],
            )
        )
    overall_condition = (
        poetic.groupby(
            ["participant", "language_group", "rhythm_condition"], as_index=False
        )["itpc_squared_debiased"]
        .mean()
        .pivot(
            index=["participant", "language_group"],
            columns="rhythm_condition",
            values="itpc_squared_debiased",
        )
        .dropna(subset=["regular", "irregular"])
        .reset_index()
    )
    for group, frame in overall_condition.groupby("language_group", sort=True):
        rows.append(
            _wilcoxon_row(
                "Regular minus irregular at f",
                group,
                "All EEG channels",
                frame["regular"],
                frame["irregular"],
            )
        )

    keys = [
        "poem",
        "line",
        "participant",
        "language_group",
        "rhythm_condition",
        "channel",
    ]
    at_f = poetic.loc[
        poetic["rhythm_condition"] == "regular",
        keys + ["itpc_squared_debiased"],
    ].rename(columns={"itpc_squared_debiased": "itpc_f"})
    at_2f = doubled.loc[
        doubled["rhythm_condition"] == "regular",
        keys + ["itpc_squared_debiased"],
    ].rename(columns={"itpc_squared_debiased": "itpc_2f"})
    matched = at_f.merge(at_2f, on=keys, how="inner", validate="one_to_one")
    participant_frequency = matched.groupby(
        ["participant", "language_group", "channel"], as_index=False
    )[["itpc_f", "itpc_2f"]].mean()
    for (group, channel), frame in participant_frequency.groupby(
        ["language_group", "channel"], sort=True
    ):
        rows.append(
            _wilcoxon_row(
                "Regular lines: f minus 2f",
                group,
                channel,
                frame["itpc_f"],
                frame["itpc_2f"],
            )
        )
    overall_frequency = matched.groupby(
        ["participant", "language_group"], as_index=False
    )[["itpc_f", "itpc_2f"]].mean()
    for group, frame in overall_frequency.groupby("language_group", sort=True):
        rows.append(
            _wilcoxon_row(
                "Regular lines: f minus 2f",
                group,
                "All EEG channels",
                frame["itpc_f"],
                frame["itpc_2f"],
            )
        )
    return pd.DataFrame(rows)


def regular_minus_irregular(poetic: pd.DataFrame) -> pd.DataFrame:
    participant = (
        poetic.groupby(
            ["participant", "language_group", "channel", "rhythm_condition"],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
        .pivot(
            index=["participant", "language_group", "channel"],
            columns="rhythm_condition",
            values="itpc_squared_debiased",
        )
        .dropna(subset=["regular", "irregular"])
        .reset_index()
    )
    participant["itpc_difference"] = participant["regular"] - participant["irregular"]
    result = (
        participant.groupby(["language_group", "channel"], as_index=False)
        .agg(
            itpc_difference=("itpc_difference", "mean"),
            n_participants=("participant", "nunique"),
        )
    )
    result.insert(0, "contrast", "Regular minus irregular at f")
    return result


def regular_f_minus_2f(poetic: pd.DataFrame, doubled: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "poem",
        "line",
        "participant",
        "language_group",
        "rhythm_condition",
        "channel",
    ]
    at_f = poetic.loc[
        poetic["rhythm_condition"] == "regular",
        keys + ["itpc_squared_debiased"],
    ].rename(columns={"itpc_squared_debiased": "itpc_f"})
    at_2f = doubled.loc[
        doubled["rhythm_condition"] == "regular",
        keys + ["itpc_squared_debiased"],
    ].rename(columns={"itpc_squared_debiased": "itpc_2f"})
    matched = at_f.merge(at_2f, on=keys, how="inner", validate="one_to_one")
    participant = matched.groupby(
        ["participant", "language_group", "channel"], as_index=False
    )[["itpc_f", "itpc_2f"]].mean()
    participant["itpc_difference"] = participant["itpc_f"] - participant["itpc_2f"]
    result = (
        participant.groupby(["language_group", "channel"], as_index=False)
        .agg(
            itpc_difference=("itpc_difference", "mean"),
            n_participants=("participant", "nunique"),
        )
    )
    result.insert(0, "contrast", "Regular lines: f minus 2f")
    return result


def plot_groups(frame: pd.DataFrame, title: str, output_name: str) -> None:
    channels = sorted(frame["channel"].unique())
    info = mne.create_info(channels, sfreq=1.0, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False)
    limit = float(np.max(np.abs(frame["itpc_difference"])))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    image = None
    for axis, (group, label) in zip(axes, GROUP_LABELS.items(), strict=True):
        values = (
            frame.loc[frame["language_group"] == group]
            .set_index("channel")
            .reindex(channels)["itpc_difference"]
            .to_numpy()
        )
        image, _ = mne.viz.plot_topomap(
            values,
            info,
            axes=axis,
            show=False,
            sensors=True,
            names=channels,
            contours=6,
            cmap="RdBu_r",
            vlim=(-limit, limit),
        )
        n_participants = int(
            frame.loc[frame["language_group"] == group, "n_participants"].max()
        )
        axis.set_title(f"{label} (n={n_participants})", pad=16)

    figure.suptitle(title, fontsize=15, y=1.08)
    colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.04)
    colorbar.set_label("Mean debiased squared ITPC difference")
    figure.savefig(OUTPUT_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    poetic = pd.read_csv(POETIC_FREQUENCY_CSV)
    doubled = pd.read_csv(DOUBLE_FREQUENCY_CSV)
    regularity = regular_minus_irregular(poetic)
    frequency = regular_f_minus_2f(poetic, doubled)
    pd.concat([regularity, frequency], ignore_index=True).to_csv(
        OUTPUT_DIR / "chosen_cycle_2-1_group_differences_by_channel.csv",
        index=False,
        float_format="%.9g",
    )
    group_wilcoxon_tests(poetic, doubled).to_csv(
        OUTPUT_DIR / "chosen_cycle_2-1_group_wilcoxon_by_channel.csv",
        index=False,
        float_format="%.9g",
    )
    plot_groups(
        regularity,
        "Regular minus irregular ITPC at poetic frequency",
        "chosen_2-1_regular_minus_irregular_by_language_group.png",
    )
    plot_groups(
        frequency,
        "Regular-line ITPC: poetic frequency minus doubled frequency",
        "chosen_2-1_regular_f_vs_2f_by_language_group.png",
    )


if __name__ == "__main__":
    main()
