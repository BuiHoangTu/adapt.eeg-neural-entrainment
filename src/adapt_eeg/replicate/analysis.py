from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from adapt_eeg.baseline import IRREGULAR_LINES, REGULAR_LINES
from adapt_eeg.replicate.stats import reject_studentized_residuals
from adapt_eeg.utils.visualization import write_itpc_visualizations

CSV_FLOAT_FORMAT = "%.6f"

PAPER_BEAT_DISTANCES = pd.DataFrame(
    [
        (1, 0.503735, 0.226681, 0.804039, 0.581233, 0.24),
        (2, 0.296426, 0.234429, 0.635476, 0.619976, 0.21),
        (3, 0.569443, 0.551074, 0.658992, 0.631463, 0.05),
        (4, 0.499655, 0.685341, 0.623506, 0.533329, 0.08),
        (5, 0.595007, 0.554708, 0.474109, 1.052522, 0.26),
        (6, 0.366510, 0.168261, 0.476463, 0.336523, 0.13),
        (7, 0.692475, 0.282643, 0.383334, 0.201383, 0.21),
        (8, 0.177542, 0.448696, 0.078000, 0.887585, 0.36),
        (9, 0.255697, 0.135978, 0.743443, 0.285257, 0.27),
        (10, 0.387441, 0.356132, 0.495063, 0.532242, 0.08),
        (15, 0.532279, 0.638617, 0.502741, 0.654469, 0.08),
        (23, 0.354290, 0.365276, 0.538512, 0.527316, 0.10),
        (27, 0.316022, 0.505636, 0.423119, 0.503880, 0.09),
        (30, 0.462049, 0.430738, 0.454202, 0.544886, 0.05),
    ],
    columns=["line", "dist1", "dist2", "dist3", "dist4", "sd"],
)


def _partial_eta_squared_from_f(f_value: float, df_effect: int, df_error: int) -> float:
    if not np.isfinite(f_value):
        return np.nan
    return (f_value * df_effect) / (f_value * df_effect + df_error)


def _ttest_to_f_row(test, df_error: int) -> tuple[float, float, float]:
    f_value = float(test.statistic**2)
    return f_value, float(test.pvalue), _partial_eta_squared_from_f(f_value, 1, df_error)


def _participant_condition_pivot(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=["participant", "language_group"],
        columns="rhythm_condition",
        values="itpc",
        aggfunc="mean",
    ).dropna()
    required = {"regular", "irregular"}
    if required - set(pivot.columns):
        return pd.DataFrame()
    return pivot.reset_index()


def paper_table_1() -> pd.DataFrame:
    table = PAPER_BEAT_DISTANCES.copy()
    table["rhythm_condition"] = np.where(
        table["line"].isin(REGULAR_LINES), "regular", "irregular"
    )
    return table


def paper_table_2(included_itpc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (frequency_hz, channel), group in included_itpc.groupby(["frequency_hz", "channel"]):
        pivot = _participant_condition_pivot(group)
        if pivot.empty or len(pivot) < 3:
            continue

        rhythm_test = stats.ttest_1samp(pivot["regular"] - pivot["irregular"], 0.0)
        rhythm_f, rhythm_p, rhythm_eta = _ttest_to_f_row(rhythm_test, len(pivot) - 1)

        participant_means = pivot.assign(
            mean_itpc=(pivot["regular"] + pivot["irregular"]) / 2
        )
        english = participant_means.loc[
            participant_means["language_group"] == "english_competence", "mean_itpc"
        ]
        non_english = participant_means.loc[
            participant_means["language_group"] == "non_english_competence", "mean_itpc"
        ]
        if len(english) >= 2 and len(non_english) >= 2:
            group_test = stats.ttest_ind(non_english, english, equal_var=True)
            group_f, group_p, group_eta = _ttest_to_f_row(group_test, len(pivot) - 2)
        else:
            group_f, group_p, group_eta = np.nan, np.nan, np.nan

        rows.append(
            {
                "eeg_channel": channel,
                "frequency_hz": frequency_hz,
                "rhythm_F": rhythm_f,
                "rhythm_p": rhythm_p,
                "rhythm_partial_eta_squared": rhythm_eta,
                "group_F": group_f,
                "group_p": group_p,
                "group_partial_eta_squared": group_eta,
                "n_participants": len(pivot),
            }
        )
    return pd.DataFrame(rows).sort_values(["frequency_hz", "eeg_channel"])


def paper_table_3(included_itpc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (frequency_hz, channel), group in included_itpc.groupby(["frequency_hz", "channel"]):
        participant_means = group.groupby(
            ["participant", "language_group"], as_index=False
        )["itpc"].mean()
        english = participant_means.loc[
            participant_means["language_group"] == "english_competence", "itpc"
        ]
        non_english = participant_means.loc[
            participant_means["language_group"] == "non_english_competence", "itpc"
        ]
        if len(english) < 2 or len(non_english) < 2:
            continue

        difference = float(non_english.mean() - english.mean())
        se = float(np.sqrt(non_english.var(ddof=1) / len(non_english) + english.var(ddof=1) / len(english)))
        welch = stats.ttest_ind(non_english, english, equal_var=False)
        ci_low, ci_high = stats.t.interval(
            0.95,
            df=float(welch.df),
            loc=difference,
            scale=se,
        )
        rows.append(
            {
                "eeg_channel": channel,
                "frequency_hz": frequency_hz,
                "mean_itpc_difference_non_english_minus_english": difference,
                "standard_error": se,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "p": float(welch.pvalue),
                "n_english": len(english),
                "n_non_english": len(non_english),
            }
        )
    return pd.DataFrame(rows).sort_values(["frequency_hz", "eeg_channel"])


def included_itpc_from_channel_lines(channel_itpc: pd.DataFrame) -> pd.DataFrame:
    participant_condition_channel = (
        channel_itpc.groupby(
            [
                "participant",
                "language_group",
                "rhythm_condition",
                "frequency_hz",
                "channel",
            ],
            as_index=False,
        )
        .agg(itpc=("itpc", "mean"), n_lines=("line", "nunique"))
        .sort_values(["frequency_hz", "channel", "participant", "rhythm_condition"])
    )
    cleaned = reject_studentized_residuals(participant_condition_channel)
    return cleaned.loc[~cleaned["excluded_as_outlier"]].copy()


def write_paper_tables(results_dir: Path) -> None:
    channel_lines_path = results_dir / "itpc_by_channel_line.csv"
    if not channel_lines_path.exists():
        raise FileNotFoundError(
            f"Missing {channel_lines_path}. Run adapt_eeg.replicate.run before table analysis."
        )

    included_itpc = included_itpc_from_channel_lines(pd.read_csv(channel_lines_path))
    outputs = {
        "paper_table_1_beat_distances.csv": paper_table_1(),
        "paper_table_2_mixed_anova.csv": paper_table_2(included_itpc),
        "paper_table_3_group_differences.csv": paper_table_3(included_itpc),
    }
    for filename, frame in outputs.items():
        frame.to_csv(results_dir / filename, index=False, float_format=CSV_FLOAT_FORMAT)
    write_itpc_visualizations(results_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-style tables from ADAPT EEG replication outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/replicate"),
        help="Directory containing outputs from adapt_eeg.replicate.run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_paper_tables(args.results_dir)


if __name__ == "__main__":
    main()
