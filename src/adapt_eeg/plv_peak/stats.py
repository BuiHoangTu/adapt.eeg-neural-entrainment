from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from adapt_eeg.replicate.stats import (
    group_difference_tests,
    mixed_anova_approximation,
    paired_rhythm_tests,
    reject_studentized_residuals,
    studentized_residuals,
)


def add_logit_plv(
    frame: pd.DataFrame,
    value_column: str = "stimulus_plv",
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    result = frame.copy()
    clipped = np.clip(result[value_column].astype(float), epsilon, 1.0 - epsilon)
    result["plv_logit"] = np.log(clipped / (1.0 - clipped))
    return result


def fit_peak_progression_models(
    frame: pd.DataFrame,
    value_column: str = "plv_logit",
) -> pd.DataFrame:
    """Fit regularity × peak-position mixed models separately by frequency/channel."""
    rows: list[dict] = []
    required = {
        "participant",
        "rhythm_condition",
        "language_group",
        "frequency_hz",
        "channel",
        "peak_index",
        value_column,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for progression model: {sorted(missing)}")

    for (frequency_hz, channel), subset in frame.groupby(
        ["frequency_hz", "channel"], sort=True
    ):
        data = subset.dropna(subset=[value_column, "peak_index"]).copy()
        data["participant"] = data["participant"].astype(str)
        data["peak_index_centered"] = (
            data["peak_index"].astype(float) - data["peak_index"].astype(float).mean()
        )
        base = {
            "frequency_hz": frequency_hz,
            "channel": channel,
            "n_rows": len(data),
            "n_participants": data["participant"].nunique(),
        }
        try:
            model = smf.mixedlm(
                f"{value_column} ~ C(rhythm_condition) * peak_index_centered "
                "+ C(language_group)",
                data=data,
                groups=data["participant"],
                re_formula="~peak_index_centered",
            )
            fit = model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
            conf = fit.conf_int()
            for term, estimate in fit.params.items():
                rows.append(
                    {
                        **base,
                        "term": term,
                        "estimate": float(estimate),
                        "std_error": float(fit.bse.get(term, np.nan)),
                        "z_value": float(fit.tvalues.get(term, np.nan)),
                        "p_value": float(fit.pvalues.get(term, np.nan)),
                        "ci_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                        "ci_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                        "converged": bool(fit.converged),
                        "error": "",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "term": "MODEL_FAILED",
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "z_value": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "converged": False,
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "add_logit_plv",
    "fit_peak_progression_models",
    "group_difference_tests",
    "mixed_anova_approximation",
    "paired_rhythm_tests",
    "reject_studentized_residuals",
    "studentized_residuals",
]
