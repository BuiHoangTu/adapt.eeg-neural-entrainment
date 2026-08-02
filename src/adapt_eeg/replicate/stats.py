from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def studentized_residuals(values: pd.Series) -> pd.Series:
    clean = values.astype(float)
    if len(clean) < 3:
        return pd.Series(np.zeros(len(clean)), index=clean.index)

    mean = clean.mean()
    centered = clean - mean
    leverage = 1 / len(clean)
    mse = (centered**2).sum() / (len(clean) - 1)
    if mse <= np.finfo(float).eps:
        return pd.Series(np.zeros(len(clean)), index=clean.index)
    return centered / np.sqrt(mse * (1 - leverage))


def reject_studentized_residuals(
    frame: pd.DataFrame,
    value_column: str = "itpc",
    group_columns: tuple[str, ...] = ("frequency_hz", "channel", "rhythm_condition", "language_group"),
    threshold: float = 3.0,
) -> pd.DataFrame:
    result = frame.copy()
    result["studentized_residual"] = result.groupby(list(group_columns), group_keys=False)[
        value_column
    ].apply(studentized_residuals)
    result["excluded_as_outlier"] = result["studentized_residual"].abs() > threshold
    return result


def paired_rhythm_tests(
    frame: pd.DataFrame,
    value_column: str = "itpc",
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = frame.groupby(["frequency_hz", "channel"])
    for (frequency_hz, channel), group in grouped:
        frequency_name = None
        if "frequency_name" in group.columns:
            frequency_name = (
                group["frequency_name"].dropna().iloc[0]
                if not group["frequency_name"].dropna().empty
                else None
            )
        pivot = group.pivot_table(
            index="participant",
            columns="rhythm_condition",
            values=value_column,
            aggfunc="mean",
        ).dropna()
        if {"regular", "irregular"} - set(pivot.columns) or len(pivot) < 2:
            continue

        test = stats.ttest_rel(pivot["regular"], pivot["irregular"])
        diff = pivot["regular"] - pivot["irregular"]
        rows.append(
            {
                "frequency_name": frequency_name,
                "frequency_hz": frequency_hz,
                "channel": channel,
                "n_participants": len(pivot),
                "regular_mean": pivot["regular"].mean(),
                "irregular_mean": pivot["irregular"].mean(),
                "regular_minus_irregular": diff.mean(),
                "t": test.statistic,
                "p": test.pvalue,
            }
        )
    return pd.DataFrame(rows)


def group_difference_tests(
    frame: pd.DataFrame,
    value_column: str = "itpc",
) -> pd.DataFrame:
    rows: list[dict] = []
    for (frequency_hz, channel), group in frame.groupby(["frequency_hz", "channel"]):
        frequency_name = None
        if "frequency_name" in group.columns:
            frequency_name = (
                group["frequency_name"].dropna().iloc[0]
                if not group["frequency_name"].dropna().empty
                else None
            )
        participant_means = group.groupby(
            ["participant", "language_group"], as_index=False
        )[value_column].mean()
        english = participant_means.loc[
            participant_means["language_group"] == "english_competence", value_column
        ]
        non_english = participant_means.loc[
            participant_means["language_group"] == "non_english_competence", value_column
        ]
        if len(english) < 2 or len(non_english) < 2:
            continue

        test = stats.ttest_ind(non_english, english, equal_var=False)
        rows.append(
            {
                "frequency_name": frequency_name,
                "frequency_hz": frequency_hz,
                "channel": channel,
                "n_english": len(english),
                "n_non_english": len(non_english),
                "english_mean": english.mean(),
                "non_english_mean": non_english.mean(),
                "non_english_minus_english": non_english.mean() - english.mean(),
                "t": test.statistic,
                "p": test.pvalue,
            }
        )
    return pd.DataFrame(rows)


def mixed_anova_approximation(frame: pd.DataFrame, value_column: str = "itpc") -> pd.DataFrame:
    rhythm = paired_rhythm_tests(frame, value_column)
    group = group_difference_tests(frame, value_column)
    if rhythm.empty and group.empty:
        return pd.DataFrame(columns=["frequency_name", "frequency_hz", "channel"])
    if rhythm.empty:
        return group.copy()
    if group.empty:
        return rhythm.copy()
    return rhythm.merge(
        group,
        on=["frequency_hz", "channel"],
        how="outer",
        suffixes=("_rhythm", "_group"),
    )
