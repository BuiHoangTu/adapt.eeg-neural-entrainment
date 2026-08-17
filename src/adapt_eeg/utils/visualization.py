from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_NAME = "itpc_by_channel_line.csv"


def _exclude_lines(channel_itpc: pd.DataFrame, excluded_lines: set[int]) -> pd.DataFrame:
    if not excluded_lines:
        return channel_itpc
    return channel_itpc.loc[~channel_itpc["line"].isin(excluded_lines)].copy()


def _frequency_order(frame: pd.DataFrame) -> list[str]:
    frequencies = frame[["frequency_name", "frequency_hz"]].drop_duplicates()
    frequencies = frequencies.sort_values("frequency_hz")
    return frequencies["frequency_name"].astype(str).tolist()


def _participant_frequency_condition(channel_itpc: pd.DataFrame) -> pd.DataFrame:
    return (
        channel_itpc.groupby(
            ["participant", "language_group", "rhythm_condition", "frequency_name", "frequency_hz"],
            as_index=False,
        )["itpc"]
        .mean()
        .sort_values(["frequency_hz", "participant", "rhythm_condition"])
    )


def _rhythm_effect_by_participant(channel_itpc: pd.DataFrame) -> pd.DataFrame:
    participant_condition = _participant_frequency_condition(channel_itpc)
    pivot = participant_condition.pivot_table(
        index=["participant", "language_group", "frequency_name", "frequency_hz"],
        columns="rhythm_condition",
        values="itpc",
    ).dropna(subset=["regular", "irregular"])
    if pivot.empty:
        return pd.DataFrame()
    effect = pivot.reset_index()
    effect["rhythm_effect"] = effect["regular"] - effect["irregular"]
    return effect.sort_values(["frequency_hz", "participant"])


def _mean_sem(frame: pd.DataFrame, group_columns: list[str], value_column: str) -> pd.DataFrame:
    summary = (
        frame.groupby(group_columns, as_index=False)[value_column]
        .agg(mean="mean", sem="sem", n="count")
        .sort_values(group_columns)
    )
    summary["sem"] = summary["sem"].fillna(0.0)
    return summary


def _save_frequency_condition_plot(channel_itpc: pd.DataFrame, output_dir: Path) -> None:
    participant_condition = _participant_frequency_condition(channel_itpc)
    summary = _mean_sem(
        participant_condition,
        ["frequency_name", "frequency_hz", "rhythm_condition"],
        "itpc",
    )
    frequency_order = _frequency_order(channel_itpc)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = range(len(frequency_order))
    offsets = {"regular": -0.08, "irregular": 0.08}
    for condition in ["regular", "irregular"]:
        data = summary.loc[summary["rhythm_condition"] == condition]
        data = data.set_index("frequency_name").reindex(frequency_order).reset_index()
        ax.errorbar(
            [x + offsets[condition] for x in x_positions],
            data["mean"],
            yerr=data["sem"],
            marker="o",
            capsize=3,
            label=condition,
        )
    ax.set_xticks(list(x_positions), frequency_order)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Mean ITPC")
    ax.set_title("Participant-mean ITPC by rhythm condition")
    ax.legend(title="Rhythm condition")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_01_itpc_by_frequency_condition.png", dpi=300)
    plt.close(fig)


def _save_rhythm_effect_plot(channel_itpc: pd.DataFrame, output_dir: Path) -> None:
    effect = _rhythm_effect_by_participant(channel_itpc)
    if effect.empty:
        return
    summary = _mean_sem(effect, ["frequency_name", "frequency_hz"], "rhythm_effect")
    frequency_order = _frequency_order(channel_itpc)
    summary = summary.set_index("frequency_name").reindex(frequency_order).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.errorbar(
        range(len(summary)),
        summary["mean"],
        yerr=summary["sem"],
        marker="o",
        capsize=3,
    )
    ax.set_xticks(range(len(summary)), summary["frequency_name"])
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Regular - irregular ITPC")
    ax.set_title("Rhythm effect by frequency")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_02_rhythm_effect_by_frequency.png", dpi=300)
    plt.close(fig)


def _save_rhythm_effect_by_language_plot(channel_itpc: pd.DataFrame, output_dir: Path) -> None:
    effect = _rhythm_effect_by_participant(channel_itpc)
    if effect.empty:
        return
    summary = _mean_sem(
        effect,
        ["frequency_name", "frequency_hz", "language_group"],
        "rhythm_effect",
    )
    frequency_order = _frequency_order(channel_itpc)
    groups = summary["language_group"].drop_duplicates().sort_values().tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, color="black", linewidth=1)
    spread = 0.16 if len(groups) > 1 else 0.0
    for index, group in enumerate(groups):
        offset = (index - (len(groups) - 1) / 2) * spread
        data = summary.loc[summary["language_group"] == group]
        data = data.set_index("frequency_name").reindex(frequency_order).reset_index()
        ax.errorbar(
            [x + offset for x in range(len(frequency_order))],
            data["mean"],
            yerr=data["sem"],
            marker="o",
            capsize=3,
            label=group,
        )
    ax.set_xticks(range(len(frequency_order)), frequency_order)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Regular - irregular ITPC")
    ax.set_title("Rhythm effect by language group")
    ax.legend(title="Language group")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_03_rhythm_effect_by_language_group.png", dpi=300)
    plt.close(fig)


def _save_participant_preference_ratios(channel_itpc: pd.DataFrame, output_dir: Path) -> None:
    participant_condition = _participant_frequency_condition(channel_itpc)
    pivot = participant_condition.pivot_table(
        index=["participant", "language_group", "frequency_name", "frequency_hz"],
        columns="rhythm_condition",
        values="itpc",
    ).dropna(subset=["regular", "irregular"])
    if pivot.empty:
        return

    comparison = pivot.reset_index()
    comparison["higher_itpc_condition"] = "tie"
    comparison.loc[comparison["regular"] > comparison["irregular"], "higher_itpc_condition"] = "regular"
    comparison.loc[comparison["irregular"] > comparison["regular"], "higher_itpc_condition"] = "irregular"
    comparison["regular_minus_irregular"] = comparison["regular"] - comparison["irregular"]
    comparison.to_csv(output_dir / "participant_regular_irregular_preference.csv", index=False)

    counts = (
        comparison.groupby(["frequency_name", "frequency_hz", "higher_itpc_condition"], as_index=False)
        .size()
        .rename(columns={"size": "n_participants"})
    )
    totals = counts.groupby(["frequency_name", "frequency_hz"])["n_participants"].transform("sum")
    counts["participant_ratio"] = counts["n_participants"] / totals
    counts.sort_values(["frequency_hz", "higher_itpc_condition"]).to_csv(
        output_dir / "participant_regular_irregular_preference_ratio.csv",
        index=False,
    )

    frequency_order = _frequency_order(channel_itpc)
    condition_order = ["regular", "irregular", "tie"]
    ratio_matrix = counts.pivot_table(
        index="frequency_name",
        columns="higher_itpc_condition",
        values="participant_ratio",
        fill_value=0.0,
    ).reindex(index=frequency_order, columns=condition_order, fill_value=0.0)
    count_matrix = counts.pivot_table(
        index="frequency_name",
        columns="higher_itpc_condition",
        values="n_participants",
        fill_value=0,
    ).reindex(index=frequency_order, columns=condition_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(7, max(4.0, len(ratio_matrix) * 0.5)))
    image = ax.imshow(ratio_matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(condition_order)), condition_order)
    ax.set_yticks(range(len(ratio_matrix.index)), ratio_matrix.index)
    ax.set_xlabel("Condition with higher participant-mean ITPC")
    ax.set_ylabel("Frequency")
    ax.set_title("Participant preference ratio: regular vs irregular")
    for y_index, frequency_name in enumerate(ratio_matrix.index):
        for x_index, condition in enumerate(condition_order):
            ratio = ratio_matrix.loc[frequency_name, condition]
            count = int(count_matrix.loc[frequency_name, condition])
            ax.text(
                x_index,
                y_index,
                f"{ratio:.0%}\nn={count}",
                ha="center",
                va="center",
                color="white" if ratio >= 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, label="Participant ratio")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_04_participant_preference_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = [0.0] * len(frequency_order)
    colors = {"regular": "#4c78a8", "irregular": "#f58518", "tie": "#bab0ac"}
    for condition in condition_order:
        values = ratio_matrix[condition].tolist()
        ax.bar(frequency_order, values, bottom=bottom, label=condition, color=colors[condition])
        bottom = [current + value for current, value in zip(bottom, values, strict=False)]
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Participant ratio")
    ax.set_title("Which condition has higher participant-mean ITPC?")
    ax.legend(title="Higher ITPC")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_05_participant_preference_stacked_ratio.png", dpi=300)
    plt.close(fig)


def _save_channel_frequency_heatmap(channel_itpc: pd.DataFrame, output_dir: Path) -> None:
    pivot = channel_itpc.pivot_table(
        index="channel",
        columns="frequency_name",
        values="itpc",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=_frequency_order(channel_itpc))
    fig_height = max(5.0, len(pivot) * 0.25)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    image = ax.imshow(pivot, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Channel")
    ax.set_title("Mean ITPC by channel and frequency")
    fig.colorbar(image, ax=ax, label="Mean ITPC")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_06_channel_frequency_itpc_heatmap.png", dpi=300)
    plt.close(fig)


def _save_channel_rhythm_effect_heatmap(channel_itpc: pd.DataFrame, output_dir: Path) -> None:
    participant_channel_condition = channel_itpc.groupby(
        ["participant", "channel", "frequency_name", "frequency_hz", "rhythm_condition"],
        as_index=False,
    )["itpc"].mean()
    pivot = participant_channel_condition.pivot_table(
        index=["participant", "channel", "frequency_name", "frequency_hz"],
        columns="rhythm_condition",
        values="itpc",
    ).dropna(subset=["regular", "irregular"])
    if pivot.empty:
        return
    effect = pivot.reset_index()
    effect["rhythm_effect"] = effect["regular"] - effect["irregular"]
    heatmap = effect.pivot_table(
        index="channel",
        columns="frequency_name",
        values="rhythm_effect",
        aggfunc="mean",
    ).reindex(columns=_frequency_order(channel_itpc))
    max_abs = float(heatmap.abs().max().max())
    fig_height = max(5.0, len(heatmap) * 0.25)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    image = ax.imshow(
        heatmap,
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    ax.set_xticks(range(len(heatmap.columns)), heatmap.columns)
    ax.set_yticks(range(len(heatmap.index)), heatmap.index)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Channel")
    ax.set_title("Regular - irregular ITPC by channel and frequency")
    fig.colorbar(image, ax=ax, label="Regular - irregular ITPC")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_07_channel_frequency_rhythm_effect_heatmap.png", dpi=300)
    plt.close(fig)


def write_itpc_visualizations(results_dir: Path, excluded_lines: set[int] | None = None) -> None:
    channel_lines_path = results_dir / CSV_NAME
    if not channel_lines_path.exists():
        raise FileNotFoundError(f"Missing {channel_lines_path}")

    channel_itpc = _exclude_lines(pd.read_csv(channel_lines_path), excluded_lines or set())
    if channel_itpc.empty:
        return

    _save_frequency_condition_plot(channel_itpc, results_dir)
    _save_rhythm_effect_plot(channel_itpc, results_dir)
    _save_rhythm_effect_by_language_plot(channel_itpc, results_dir)
    _save_participant_preference_ratios(channel_itpc, results_dir)
    _save_channel_frequency_heatmap(channel_itpc, results_dir)
    _save_channel_rhythm_effect_heatmap(channel_itpc, results_dir)
