"""Test the continuous association between verse-line CV and ITPC.

All selected lines with a finite stress-interval CV are analyzed.  The
regular/ambiguous/irregular label is intentionally not used in this analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.formula.api as smf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This analysis requires statsmodels.") from exc

from adapt_eeg.exploration.stress_rhythm_evidence import EEG_POEM_IDS
from adapt_eeg.poem_eeg import (
    discover_ica_cleaned_files,
    ids_from_path,
    load_poem_recording,
    unique_recording_paths,
)
from adapt_eeg.poem_itpc import analyze_recording
from adapt_eeg.poem_itpc_acf import DEFAULT_RHYTHM_CSV, load_or_derive_timings
from adapt_eeg.poem_rhythm import poem_stress_frequency

CSV_FLOAT_FORMAT = "%.9g"


def valid_cv_lines(timings: dict[int, list]) -> dict[int, list]:
    """Keep all and only selected lines whose CV can be calculated."""
    return {
        poem: [
            line
            for line in lines
            if np.isfinite(line.coefficient_of_variation)
        ]
        for poem, lines in timings.items()
    }


def fit_primary_model(participant_line: pd.DataFrame) -> pd.DataFrame:
    """Fit participant/poem fixed effects with line-clustered robust errors."""
    data = participant_line.copy()
    data["cv_within_poem"] = data["rhythm_cv"] - data.groupby("poem")[
        "rhythm_cv"
    ].transform("mean")
    data["poem_line"] = "P" + data["poem"].astype(str) + "_L" + data["line"].astype(str)
    model = smf.ols(
        "itpc_squared_debiased ~ cv_within_poem + C(poem) + C(participant)",
        data=data,
    ).fit(cov_type="cluster", cov_kwds={"groups": data["poem_line"]})
    coefficient = float(model.params["cv_within_poem"])
    p_two_sided = float(model.pvalues["cv_within_poem"])
    p_lower_cv_higher_itpc = p_two_sided / 2 if coefficient < 0 else 1 - p_two_sided / 2
    return pd.DataFrame(
        [
            {
                "analysis": "participant_and_poem_fixed_effects_line_clustered",
                "outcome": "channel_averaged_debiased_squared_itpc",
                "predictor": "CV_centered_within_poem",
                "coefficient": coefficient,
                "standard_error": float(model.bse["cv_within_poem"]),
                "ci95_low": float(model.conf_int().loc["cv_within_poem", 0]),
                "ci95_high": float(model.conf_int().loc["cv_within_poem", 1]),
                "p_two_sided": p_two_sided,
                "p_one_sided_lower_cv_higher_itpc": p_lower_cv_higher_itpc,
                "n_participant_line_rows": len(data),
                "n_lines": data["poem_line"].nunique(),
                "n_participants": data["participant"].nunique(),
                "n_poems": data["poem"].nunique(),
                "r_squared": float(model.rsquared),
            }
        ]
    )


def line_spearman(line_mean: pd.DataFrame) -> pd.DataFrame:
    """Descriptive Spearman association across unique scoreable poem-lines."""
    value = stats.spearmanr(
        line_mean["rhythm_cv"], line_mean["itpc_squared_debiased"],
    )
    return pd.DataFrame(
        [
            {
                "analysis": "descriptive_spearman_across_line_means",
                "spearman_rho": float(value.statistic),
                "p_two_sided": float(value.pvalue),
                "n_lines": len(line_mean),
            }
        ]
    )


def plot_line_means(line_mean: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    axis.scatter(
        line_mean["rhythm_cv"],
        line_mean["itpc_squared_debiased"],
        s=28,
        alpha=0.65,
        color="#4c78a8",
        edgecolor="none",
    )
    slope, intercept, _, _, _ = stats.linregress(
        line_mean["rhythm_cv"],
        line_mean["itpc_squared_debiased"],
    )
    x = np.linspace(line_mean["rhythm_cv"].min(), line_mean["rhythm_cv"].max(), 100)
    axis.plot(x, intercept + slope * x, color="black", linewidth=2, label="Linear guide")
    correlation = stats.spearmanr(
        line_mean["rhythm_cv"], line_mean["itpc_squared_debiased"]
    )
    y_limits = line_mean["itpc_squared_debiased"].quantile([0.01, 0.99]).to_numpy()
    axis.set(
        xlabel="Line stress-interval coefficient of variation (CV)",
        ylabel="Mean debiased squared ITPC (central 98%)",
        title="Line regularity (CV) and ITPC at poetic frequency",
        ylim=tuple(y_limits),
    )
    axis.text(
        0.98,
        0.96,
        f"Spearman $\\rho$ = {correlation.statistic:.3f}\n"
        f"two-sided $p$ = {correlation.pvalue:.4f}\n"
        f"n = {len(line_mean)} lines",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    axis.legend(loc="lower right")
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def run(data_root: Path, output_dir: Path, rhythm_csv: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = valid_cv_lines(load_or_derive_timings(list(EEG_POEM_IDS), rhythm_csv))
    frequencies = {poem: poem_stress_frequency(lines) for poem, lines in timings.items()}
    discovered = [
        path
        for path in discover_ica_cleaned_files(data_root)
        if ids_from_path(path)[0] in set(timings)
    ]
    paths, duplicates = unique_recording_paths(discovered)
    rows: list[dict] = []
    errors: list[dict] = [
        {
            "stage": "discovery",
            "poem": poem,
            "participant": participant,
            "error": f"Duplicate recordings: {len(candidates)}",
        }
        for (poem, participant), candidates in sorted(duplicates.items())
    ]
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}", flush=True)
        try:
            recording = load_poem_recording(path, preload=True)
            found, failures = analyze_recording(
                recording,
                timings[recording.poem],
                frequencies[recording.poem],
                include_ambiguous=True,
            )
            rows.extend(found)
            errors.extend(failures)
        except Exception as exc:
            poem, participant = ids_from_path(path)
            errors.append(
                {
                    "stage": "recording",
                    "poem": poem,
                    "participant": participant,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    full = pd.DataFrame(rows)
    if full.empty:
        raise RuntimeError("No ITPC rows were produced")
    full.to_csv(output_dir / "itpc_by_line_channel.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    pd.DataFrame(errors).to_csv(output_dir / "load_errors.csv", index=False)

    participant_line = (
        full.groupby(
            ["poem", "line", "participant", "language_group", "rhythm_cv"],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
    )
    participant_line.to_csv(
        output_dir / "participant_line_itpc.csv", index=False, float_format=CSV_FLOAT_FORMAT
    )
    line_mean = (
        participant_line.groupby(["poem", "line", "rhythm_cv"], as_index=False)[
            "itpc_squared_debiased"
        ]
        .mean()
    )
    line_mean["cv_within_poem"] = line_mean["rhythm_cv"] - line_mean.groupby("poem")[
        "rhythm_cv"
    ].transform("mean")
    line_mean["itpc_within_poem"] = line_mean["itpc_squared_debiased"] - line_mean.groupby("poem")[
        "itpc_squared_debiased"
    ].transform("mean")
    line_mean.to_csv(output_dir / "line_mean_itpc.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    fit_primary_model(participant_line).to_csv(
        output_dir / "primary_model_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT
    )
    line_spearman(line_mean).to_csv(
        output_dir / "line_mean_spearman.csv", index=False, float_format=CSV_FLOAT_FORMAT
    )
    plot_line_means(line_mean, output_dir / "cv_vs_itpc.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw-eeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cv_itpc"))
    parser.add_argument("--rhythm-csv", type=Path, default=DEFAULT_RHYTHM_CSV)
    args = parser.parse_args()
    run(args.data_root, args.output_dir, args.rhythm_csv)


if __name__ == "__main__":
    main()
