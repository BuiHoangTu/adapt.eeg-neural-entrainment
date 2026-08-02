from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires statsmodels. Install with: pip install statsmodels"
    ) from exc


REQUIRED_COLUMNS = {
    "participant",
    "line",
    "rhythm_condition",
    "language_group",
    "frequency_hz",
    "peak_index",
    "peak_seconds",
    "channel",
    "stimulus_plv",
    "mean_phase_lag_radians",
}

NUMERIC_COLUMNS = [
    "line",
    "frequency_hz",
    "peak_index",
    "peak_seconds",
    "window_start",
    "window_end",
    "window_duration",
    "window_radius",
    "n_window_samples",
    "stimulus_plv",
    "mean_phase_lag_radians",
]


def read_results(path: Path) -> pd.DataFrame:
    """Read comma-, semicolon-, or tab-delimited output."""
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = frame.columns.str.strip()

    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in NUMERIC_COLUMNS:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ["participant", "rhythm_condition", "language_group", "channel"]:
        frame[column] = frame[column].astype(str).str.strip()

    frame = frame.dropna(
        subset=[
            "participant",
            "rhythm_condition",
            "frequency_hz",
            "peak_index",
            "channel",
            "stimulus_plv",
        ]
    ).copy()

    invalid = ~frame["stimulus_plv"].between(0, 1, inclusive="both")
    if invalid.any():
        warnings.warn(f"Dropping {int(invalid.sum())} rows with PLV outside [0, 1].")
        frame = frame.loc[~invalid].copy()

    frame["peak_index"] = frame["peak_index"].astype(int)
    frame["line"] = frame["line"].astype("Int64")
    frame["phase_lag_degrees"] = np.degrees(frame["mean_phase_lag_radians"])

    eps = 1e-4
    bounded = frame["stimulus_plv"].clip(eps, 1 - eps)
    frame["plv_logit"] = np.log(bounded / (1 - bounded))
    frame["peak_index_centered"] = frame["peak_index"] - frame["peak_index"].mean()
    return frame


def mean_ci(values: pd.Series, confidence: float = 0.95) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    n = x.size
    if n == 0:
        return pd.Series(
            {
                "mean": np.nan,
                "sd": np.nan,
                "sem": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": 0,
            }
        )
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    delta = stats.t.ppf((1 + confidence) / 2, n - 1) * sem if n > 1 else np.nan
    return pd.Series(
        {
            "mean": mean,
            "sd": sd,
            "sem": sem,
            "ci_low": mean - delta if n > 1 else np.nan,
            "ci_high": mean + delta if n > 1 else np.nan,
            "n": n,
        }
    )


def circular_summary(group: pd.DataFrame) -> pd.Series:
    angles = group["mean_phase_lag_radians"].dropna().to_numpy()
    if angles.size == 0:
        return pd.Series(
            {
                "mean_phase_lag_radians": np.nan,
                "mean_phase_lag_degrees": np.nan,
                "resultant_length": np.nan,
            }
        )
    vector = np.mean(np.exp(1j * angles))
    angle = np.angle(vector)
    return pd.Series(
        {
            "mean_phase_lag_radians": angle,
            "mean_phase_lag_degrees": np.degrees(angle),
            "resultant_length": np.abs(vector),
        }
    )


def participant_peak_table(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "participant",
        "language_group",
        "rhythm_condition",
        "frequency_hz",
        "channel",
        "peak_index",
    ]
    result = frame.groupby(keys, as_index=False).agg(
        stimulus_plv=("stimulus_plv", "mean"),
        plv_logit=("plv_logit", "mean"),
        mean_peak_seconds=("peak_seconds", "mean"),
        n_lines=("line", "nunique"),
        n_observations=("stimulus_plv", "size"),
    )
    result["peak_index_centered"] = result["peak_index"] - result["peak_index"].mean()
    return result


def descriptive_tables(
    frame: pd.DataFrame, participant_peak: pd.DataFrame, output_dir: Path
) -> None:
    coverage = pd.DataFrame(
        {
            "metric": [
                "rows",
                "participants",
                "lines",
                "channels",
                "conditions",
                "language_groups",
                "frequencies",
                "peaks",
            ],
            "value": [
                len(frame),
                frame["participant"].nunique(),
                frame["line"].nunique(),
                frame["channel"].nunique(),
                frame["rhythm_condition"].nunique(),
                frame["language_group"].nunique(),
                frame["frequency_hz"].nunique(),
                frame["peak_index"].nunique(),
            ],
        }
    )
    coverage.to_csv(output_dir / "table_00_dataset_coverage.csv", index=False)

    participant_counts = frame.groupby(
        ["rhythm_condition", "language_group"], as_index=False
    ).agg(
        participants=("participant", "nunique"),
        lines=("line", "nunique"),
        observations=("stimulus_plv", "size"),
    )
    participant_counts.to_csv(output_dir / "table_01_sample_counts.csv", index=False)

    summary = (
        participant_peak.groupby(
            ["frequency_hz", "channel", "rhythm_condition", "peak_index"],
            sort=True,
        )["stimulus_plv"]
        .apply(mean_ci)
        .unstack()
        .reset_index()
        .rename(
            columns={
                "mean": "mean_plv",
                "sd": "sd_plv",
                "sem": "sem_plv",
                "ci_low": "ci95_low",
                "ci_high": "ci95_high",
                "n": "n_participant_cells",
            }
        )
    )
    summary.to_csv(
        output_dir / "table_02_plv_by_condition_peak_channel.csv", index=False
    )

    overall = (
        participant_peak.groupby(["frequency_hz", "rhythm_condition", "peak_index"])[
            "stimulus_plv"
        ]
        .apply(mean_ci)
        .unstack()
        .reset_index()
        .rename(
            columns={
                "mean": "mean_plv",
                "sd": "sd_plv",
                "sem": "sem_plv",
                "ci_low": "ci95_low",
                "ci_high": "ci95_high",
                "n": "n_participant_channel_cells",
            }
        )
    )
    overall.to_csv(
        output_dir / "table_03_plv_by_condition_peak_overall.csv", index=False
    )

    circular = (
        frame.groupby(["frequency_hz", "channel", "rhythm_condition", "peak_index"])
        .apply(circular_summary, include_groups=False)
        .reset_index()
    )
    circular.to_csv(output_dir / "table_04_phase_lag_circular_summary.csv", index=False)


def fit_channel_models(
    participant_peak: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients: list[pd.DataFrame] = []
    model_status: list[dict] = []

    grouping = participant_peak.groupby(["frequency_hz", "channel"], sort=True)
    for (frequency_hz, channel), data in grouping:
        data = data.copy()
        n_participants = data["participant"].nunique()
        n_conditions = data["rhythm_condition"].nunique()
        n_peaks = data["peak_index"].nunique()

        status = {
            "frequency_hz": frequency_hz,
            "channel": channel,
            "n_rows": len(data),
            "n_participants": n_participants,
            "n_conditions": n_conditions,
            "n_peaks": n_peaks,
            "model_type": "NOT_FIT",
            "converged": False,
            "error": "",
        }

        if n_participants < 3 or n_conditions < 2 or n_peaks < 2:
            status["error"] = "Insufficient participants, conditions, or peak levels"
            model_status.append(status)
            continue

        formula = (
            "plv_logit ~ C(rhythm_condition) * peak_index_centered + C(language_group)"
        )
        fitted = None
        errors: list[str] = []

        # Preferred: random participant intercept. This is usually more stable than
        # a channel-wise random slope model with a small number of observations.
        try:
            model = smf.mixedlm(
                formula, data=data, groups=data["participant"], re_formula="1"
            )
            for optimizer in ("lbfgs", "powell", "cg"):
                try:
                    candidate = model.fit(
                        reml=False, method=optimizer, maxiter=2000, disp=False
                    )
                    if bool(getattr(candidate, "converged", False)):
                        fitted = candidate
                        status["model_type"] = f"MixedLM_random_intercept[{optimizer}]"
                        status["converged"] = True
                        break
                except Exception as exc:
                    errors.append(f"MixedLM {optimizer}: {exc!r}")
        except Exception as exc:
            errors.append(f"MixedLM build: {exc!r}")

        # Transparent fallback: participant-clustered OLS.
        if fitted is None:
            try:
                fitted = smf.ols(formula, data=data).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": data["participant"]},
                )
                status["model_type"] = "OLS_participant_clustered"
                status["converged"] = True
            except Exception as exc:
                errors.append(f"Clustered OLS: {exc!r}")

        status["error"] = " | ".join(errors)
        model_status.append(status)
        if fitted is None:
            continue

        conf = fitted.conf_int()
        terms = pd.DataFrame(
            {
                "term": fitted.params.index,
                "estimate": fitted.params.values,
                "std_error": fitted.bse.values,
                "statistic": fitted.tvalues.values,
                "p_value": fitted.pvalues.values,
                "ci95_low": conf.iloc[:, 0].values,
                "ci95_high": conf.iloc[:, 1].values,
            }
        )
        terms.insert(0, "model_type", status["model_type"])
        terms.insert(0, "channel", channel)
        terms.insert(0, "frequency_hz", frequency_hz)
        coefficients.append(terms)

    coefficient_frame = (
        pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    )
    if not coefficient_frame.empty:
        coefficient_frame["p_fdr_bh"] = np.nan
        for term, idx in coefficient_frame.groupby("term").groups.items():
            p = coefficient_frame.loc[idx, "p_value"].to_numpy()
            finite = np.isfinite(p)
            if finite.any():
                adjusted = multipletests(p[finite], method="fdr_bh")[1]
                coefficient_frame.loc[np.asarray(idx)[finite], "p_fdr_bh"] = adjusted

    return coefficient_frame, pd.DataFrame(model_status)


def paired_regular_irregular_tests(participant_peak: pd.DataFrame) -> pd.DataFrame:
    conditions = list(participant_peak["rhythm_condition"].dropna().unique())
    if len(conditions) != 2:
        return pd.DataFrame(
            [
                {
                    "note": f"Paired tests require exactly two rhythm conditions; found {conditions}"
                }
            ]
        )

    condition_a, condition_b = sorted(conditions)
    rows: list[dict] = []
    keys = ["frequency_hz", "channel", "peak_index"]
    for group_key, data in participant_peak.groupby(keys, sort=True):
        wide = data.pivot_table(
            index="participant",
            columns="rhythm_condition",
            values="stimulus_plv",
            aggfunc="mean",
        )
        if condition_a not in wide or condition_b not in wide:
            continue
        pair = wide[[condition_a, condition_b]].dropna()
        if len(pair) < 3:
            continue
        difference = pair[condition_b] - pair[condition_a]
        test = stats.ttest_rel(pair[condition_b], pair[condition_a], nan_policy="omit")
        dz = (
            difference.mean() / difference.std(ddof=1)
            if difference.std(ddof=1) > 0
            else np.nan
        )
        rows.append(
            {
                "frequency_hz": group_key[0],
                "channel": group_key[1],
                "peak_index": group_key[2],
                "condition_a": condition_a,
                "condition_b": condition_b,
                "contrast": f"{condition_b} - {condition_a}",
                "n_pairs": len(pair),
                "mean_difference": difference.mean(),
                "t_statistic": test.statistic,
                "p_value": test.pvalue,
                "cohens_dz": dz,
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_fdr_bh"] = multipletests(result["p_value"], method="fdr_bh")[1]
    return result


def plot_overall_progression(participant_peak: pd.DataFrame, output_dir: Path) -> None:
    plot_data = participant_peak.groupby(
        ["participant", "rhythm_condition", "peak_index"], as_index=False
    ).agg(stimulus_plv=("stimulus_plv", "mean"))
    summary = (
        plot_data.groupby(["rhythm_condition", "peak_index"])["stimulus_plv"]
        .apply(mean_ci)
        .unstack()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for condition, data in summary.groupby("rhythm_condition"):
        data = data.sort_values("peak_index")
        ax.plot(data["peak_index"], data["mean"], marker="o", label=condition)
        ax.fill_between(data["peak_index"], data["ci_low"], data["ci_high"], alpha=0.2)
    ax.set_xlabel("Peak position in line")
    ax.set_ylabel("Mean EEG–audio PLV")
    ax.set_title("Stimulus locking across the phrase")
    ax.set_ylim(0, 1)
    ax.legend(title="Rhythm condition")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_01_overall_plv_progression.png", dpi=300)
    plt.close(fig)


def plot_channel_heatmaps(participant_peak: pd.DataFrame, output_dir: Path) -> None:
    conditions = sorted(participant_peak["rhythm_condition"].unique())
    for condition in conditions:
        data = participant_peak.loc[participant_peak["rhythm_condition"] == condition]
        matrix = data.pivot_table(
            index="channel", columns="peak_index", values="stimulus_plv", aggfunc="mean"
        )
        if matrix.empty:
            continue
        fig_height = max(5, 0.25 * len(matrix.index))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        image = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
        ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
        ax.set_xlabel("Peak position")
        ax.set_ylabel("Channel")
        ax.set_title(f"Mean EEG–audio PLV: {condition}")
        fig.colorbar(image, ax=ax, label="PLV")
        fig.tight_layout()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in condition)
        fig.savefig(output_dir / f"figure_02_heatmap_{safe}.png", dpi=300)
        plt.close(fig)


def plot_condition_difference_heatmap(
    participant_peak: pd.DataFrame, output_dir: Path
) -> None:
    conditions = sorted(participant_peak["rhythm_condition"].unique())
    if len(conditions) != 2:
        return
    a, b = conditions
    means = participant_peak.pivot_table(
        index=["channel", "peak_index"],
        columns="rhythm_condition",
        values="stimulus_plv",
        aggfunc="mean",
    )
    if a not in means or b not in means:
        return
    difference = (means[b] - means[a]).unstack("peak_index")
    limit = np.nanmax(np.abs(difference.to_numpy()))
    if not np.isfinite(limit) or limit == 0:
        limit = 1e-6

    fig_height = max(5, 0.25 * len(difference.index))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    image = ax.imshow(
        difference.to_numpy(), aspect="auto", vmin=-limit, vmax=limit, cmap="coolwarm"
    )
    ax.set_xticks(np.arange(len(difference.columns)), labels=difference.columns)
    ax.set_yticks(np.arange(len(difference.index)), labels=difference.index)
    ax.set_xlabel("Peak position")
    ax.set_ylabel("Channel")
    ax.set_title(f"Condition difference in PLV: {b} − {a}")
    fig.colorbar(image, ax=ax, label="PLV difference")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_03_condition_difference_heatmap.png", dpi=300)
    plt.close(fig)


def plot_participant_slopes(participant_peak: pd.DataFrame, output_dir: Path) -> None:
    collapsed = participant_peak.groupby(
        ["participant", "rhythm_condition", "peak_index"], as_index=False
    ).agg(stimulus_plv=("stimulus_plv", "mean"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for (_, condition), data in collapsed.groupby(["participant", "rhythm_condition"]):
        data = data.sort_values("peak_index")
        ax.plot(data["peak_index"], data["stimulus_plv"], alpha=0.15)
    means = collapsed.groupby(["rhythm_condition", "peak_index"], as_index=False)[
        "stimulus_plv"
    ].mean()
    for condition, data in means.groupby("rhythm_condition"):
        ax.plot(
            data["peak_index"],
            data["stimulus_plv"],
            marker="o",
            linewidth=3,
            label=condition,
        )
    ax.set_xlabel("Peak position in line")
    ax.set_ylabel("Participant mean PLV")
    ax.set_title("Participant trajectories and condition means")
    ax.set_ylim(0, 1)
    ax.legend(title="Rhythm condition")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_04_participant_trajectories.png", dpi=300)
    plt.close(fig)


def plot_plv_distribution(frame: pd.DataFrame, output_dir: Path) -> None:
    conditions = sorted(frame["rhythm_condition"].unique())
    values = [
        frame.loc[frame["rhythm_condition"] == condition, "stimulus_plv"].dropna()
        for condition in conditions
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(values, tick_labels=conditions, showfliers=False)
    ax.set_xlabel("Rhythm condition")
    ax.set_ylabel("EEG–audio PLV")
    ax.set_title("Distribution of stimulus-locking values")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_05_plv_distribution.png", dpi=300)
    plt.close(fig)


def write_report(
    output_dir: Path,
    frame: pd.DataFrame,
    coefficients: pd.DataFrame,
    status: pd.DataFrame,
) -> None:
    interaction = pd.DataFrame()
    if not coefficients.empty:
        interaction = coefficients.loc[
            coefficients["term"].str.contains("rhythm_condition", regex=False)
            & coefficients["term"].str.contains("peak_index_centered", regex=False)
        ].copy()

    report = {
        "rows_analyzed": int(len(frame)),
        "participants": int(frame["participant"].nunique()),
        "lines": int(frame["line"].nunique()),
        "channels": int(frame["channel"].nunique()),
        "conditions": sorted(frame["rhythm_condition"].unique().tolist()),
        "peaks": sorted(frame["peak_index"].unique().astype(int).tolist()),
        "models_fit": int(status["converged"].sum()) if not status.empty else 0,
        "models_failed": int((~status["converged"]).sum()) if not status.empty else 0,
        "significant_interactions_fdr_0_05": int(
            (interaction.get("p_fdr_bh", pd.Series(dtype=float)) < 0.05).sum()
        ),
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def main(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_results(args.input)
    participant_peak = participant_peak_table(frame)

    frame.to_csv(args.output_dir / "cleaned_input.csv", index=False)
    participant_peak.to_csv(
        args.output_dir / "participant_peak_channel.csv", index=False
    )

    descriptive_tables(frame, participant_peak, args.output_dir)

    coefficients, status = fit_channel_models(participant_peak)
    coefficients.to_csv(
        args.output_dir / "table_05_peak_progression_models.csv", index=False
    )
    status.to_csv(args.output_dir / "table_06_model_status.csv", index=False)

    paired = paired_regular_irregular_tests(participant_peak)
    paired.to_csv(
        args.output_dir / "table_07_paired_condition_tests_by_peak.csv", index=False
    )

    plot_overall_progression(participant_peak, args.output_dir)
    plot_channel_heatmaps(participant_peak, args.output_dir)
    plot_condition_difference_heatmap(participant_peak, args.output_dir)
    plot_participant_slopes(participant_peak, args.output_dir)
    plot_plv_distribution(frame, args.output_dir)
    write_report(args.output_dir, frame, coefficients, status)

    print(f"Wrote tables, model results, and figures to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create analysis tables and graphs from stimulus-locked EEG PLV CSV output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/plv_peak/stimulus_plv_by_peak_line_channel.csv"),
        # help="Path to plv_by_peak_line_channel CSV",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/plv_analysis"))
    args = parser.parse_args()

    main(args)
