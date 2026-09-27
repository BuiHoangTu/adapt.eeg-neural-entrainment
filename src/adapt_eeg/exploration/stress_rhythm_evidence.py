from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from adapt_eeg.auto_annotation.classes import Contour, LinedAudio
from adapt_eeg.auto_annotation.stress_identifier import (
    calculate_inter_stress_distances,
)
from adapt_eeg.auto_annotation.textgrid_annotation_fix import derive_lined_audio
from adapt_eeg.constants import POEMS_CONFIG
from adapt_eeg.poem_rhythm import (
    IRREGULAR_CV_MIN,
    REGULAR_CV_MAX,
    classify_rhythm,
    inter_stress_interval_cv,
)

EEG_POEM_IDS = (1, 7, 11, 13)
MIN_RHYTHM_HZ = 1.0
MAX_RHYTHM_HZ = 5.0


def _line_ranges(lined_audio: LinedAudio) -> list[tuple[float, float]]:
    starts = [0.0, *lined_audio.line_boundaries]
    ends = [*lined_audio.line_boundaries, len(lined_audio.audio) / 1000.0]
    return list(zip(starts, ends, strict=True))


def _uniform_values(
    contour: Contour,
    start_s: float,
    end_s: float,
) -> tuple[np.ndarray, float] | None:
    selected = (
        (contour.times >= start_s)
        & (contour.times < end_s)
        & np.isfinite(contour.values)
    )
    times = contour.times[selected]
    values = contour.values[selected]
    if len(times) < 3:
        return None

    time_step_s = float(np.median(np.diff(times)))
    if not np.isfinite(time_step_s) or time_step_s <= 0:
        return None
    uniform_times = np.arange(times[0], times[-1] + time_step_s / 2, time_step_s)
    uniform_values = np.interp(uniform_times, times, values)
    uniform_values -= np.mean(uniform_values)
    return uniform_values, time_step_s


def _spectral_peak_hz(values: np.ndarray, time_step_s: float) -> float:
    if len(values) < 3 or np.allclose(values, 0):
        return np.nan
    padded_length = max(len(values), 8 * len(values))
    spectrum = np.abs(
        np.fft.rfft(values * np.hanning(len(values)), n=padded_length)
    ) ** 2
    frequencies = np.fft.rfftfreq(padded_length, time_step_s)
    eligible = (frequencies >= MIN_RHYTHM_HZ) & (frequencies <= MAX_RHYTHM_HZ)
    if not np.any(eligible):
        return np.nan
    eligible_indices = np.flatnonzero(eligible)
    return float(frequencies[eligible_indices[np.argmax(spectrum[eligible])]])


def _autocorrelation_peak_hz(values: np.ndarray, time_step_s: float) -> float:
    if len(values) < 3 or np.allclose(values, 0):
        return np.nan
    autocorrelation = np.correlate(values, values, mode="full")[len(values) - 1 :]
    overlap_counts = np.arange(len(values), 0, -1)
    autocorrelation = autocorrelation / overlap_counts
    if autocorrelation[0] <= np.finfo(float).eps:
        return np.nan
    autocorrelation /= autocorrelation[0]

    minimum_lag = max(1, int(np.ceil(1 / (MAX_RHYTHM_HZ * time_step_s))))
    maximum_lag = min(
        len(autocorrelation) - 1,
        int(np.floor(1 / (MIN_RHYTHM_HZ * time_step_s))),
    )
    if maximum_lag <= minimum_lag:
        return np.nan

    region = autocorrelation[minimum_lag : maximum_lag + 1]
    peaks, properties = find_peaks(region, prominence=0)
    if len(peaks) == 0:
        return np.nan
    best = int(peaks[np.argmax(properties["prominences"])]) + minimum_lag
    return float(1 / (best * time_step_s))


def analyze_poem(poem_id: int) -> pd.DataFrame:
    config = POEMS_CONFIG[poem_id - 1]
    boundary_module = import_module(
        f"adapt_eeg.auto_annotation.textgrid_annotation_fix.poem{poem_id}"
    )
    lined_audio = derive_lined_audio(
        config["audio_url"],
        config["textgrid_url"],
        boundary_module.textgrid_eol_indices,
        intro_end_ms=config["intro_end"],
    )
    distances_by_line = calculate_inter_stress_distances(lined_audio)
    line_ranges = _line_ranges(lined_audio)
    if len(distances_by_line) != len(line_ranges):
        raise RuntimeError(
            f"Poem {poem_id}: distance rows and line ranges have different lengths"
        )

    rows = []
    for line_index, (distances, (start_s, end_s)) in enumerate(
        zip(distances_by_line, line_ranges, strict=True)
    ):
        interval_values = np.asarray(distances, dtype=float)
        mean_s = float(np.mean(interval_values)) if len(interval_values) else np.nan
        median_s = float(np.median(interval_values)) if len(interval_values) else np.nan
        std_s = float(np.std(interval_values, ddof=0)) if len(interval_values) else np.nan
        cv = inter_stress_interval_cv(interval_values)
        uniform = _uniform_values(lined_audio.intensity, start_s, end_s)
        spectral_hz = np.nan
        autocorrelation_hz = np.nan
        if uniform is not None:
            values, time_step_s = uniform
            spectral_hz = _spectral_peak_hz(values, time_step_s)
            autocorrelation_hz = _autocorrelation_peak_hz(values, time_step_s)

        rows.append(
            {
                "poem": poem_id,
                "line_index": line_index,
                "line_number": line_index + 1,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "n_stresses": len(distances) + 1 if distances else 0,
                "n_intervals": len(distances),
                "mean_interval_s": mean_s,
                "median_interval_s": median_s,
                "std_interval_s": std_s,
                "coefficient_of_variation": cv,
                "stress_frequency_hz": 1 / median_s if median_s > 0 else np.nan,
                "intensity_spectral_peak_hz": spectral_hz,
                "intensity_autocorrelation_peak_hz": autocorrelation_hz,
                "rhythm_condition": classify_rhythm(cv),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_poem(lines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for poem, frame in lines.groupby("poem", sort=True):
        counts = frame["rhythm_condition"].value_counts()
        rows.append(
            {
                "poem": poem,
                "n_lines": len(frame),
                "n_regular": int(counts.get("regular", 0)),
                "n_irregular": int(counts.get("irregular", 0)),
                "n_ambiguous": int(counts.get("ambiguous", 0)),
                "median_line_mean_interval_s": frame["mean_interval_s"].median(),
                "median_line_median_interval_s": frame["median_interval_s"].median(),
                "median_line_std_interval_s": frame["std_interval_s"].median(),
                "median_line_coefficient_of_variation": frame[
                    "coefficient_of_variation"
                ].median(),
                "median_stress_frequency_hz": frame["stress_frequency_hz"].median(),
                "median_intensity_spectral_peak_hz": frame[
                    "intensity_spectral_peak_hz"
                ].median(),
                "median_intensity_autocorrelation_peak_hz": frame[
                    "intensity_autocorrelation_peak_hz"
                ].median(),
            }
        )
    return pd.DataFrame(rows)


def window_feasibility(lines: pd.DataFrame) -> pd.DataFrame:
    """Count complete overlapping windows for candidate frequency policies."""
    rows = []
    frequency_policies = {
        "fixed_2hz": pd.Series(2.0, index=lines.index),
        "line_stress_frequency": lines["stress_frequency_hz"],
    }
    for frequency_source, frequencies in frequency_policies.items():
        for cycles_per_window in (2.0, 3.0, 4.0):
            for index, line in lines.iterrows():
                frequency_hz = float(frequencies.loc[index])
                window_s = cycles_per_window / frequency_hz
                shift_s = 1.0 / frequency_hz
                available = float(line["duration_s"])
                n_windows = (
                    1 + int(np.floor((available - window_s) / shift_s))
                    if np.isfinite(window_s) and available >= window_s
                    else 0
                )
                rows.append(
                    {
                        "poem": int(line["poem"]),
                        "line_number": int(line["line_number"]),
                        "rhythm_condition": line["rhythm_condition"],
                        "frequency_source": frequency_source,
                        "frequency_hz": frequency_hz,
                        "cycles_per_window": cycles_per_window,
                        "shift_cycles": 1.0,
                        "window_duration_s": window_s,
                        "window_shift_s": shift_s,
                        "n_complete_windows": n_windows,
                        "itpc_eligible": n_windows >= 2,
                    }
                )
    return pd.DataFrame(rows)


def save_distribution(lines: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(lines["coefficient_of_variation"].dropna(), bins=20)
    axes[0].axvline(REGULAR_CV_MAX, color="green", linestyle="--")
    axes[0].axvline(IRREGULAR_CV_MIN, color="red", linestyle="--")
    axes[0].set(xlabel="Inter-stress interval CV", ylabel="Lines")

    frequency_columns = {
        "stress_frequency_hz": "Stress intervals",
        "intensity_spectral_peak_hz": "Intensity spectrum",
        "intensity_autocorrelation_peak_hz": "Intensity autocorrelation",
    }
    axes[1].boxplot(
        [lines[column].dropna() for column in frequency_columns],
        tick_labels=list(frequency_columns.values()),
    )
    axes[1].axhline(2.0, color="black", linestyle="--", label="Literature: 2 Hz")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/exploration"))
    parser.add_argument("--poems", type=int, nargs="+", default=list(EEG_POEM_IDS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines = pd.concat([analyze_poem(poem) for poem in args.poems], ignore_index=True)
    summary = summarize_by_poem(lines)
    feasibility = window_feasibility(lines)
    lines.to_csv(args.output_dir / "stress_rhythm_by_line.csv", index=False)
    summary.to_csv(args.output_dir / "stress_rhythm_by_poem.csv", index=False)
    feasibility.to_csv(args.output_dir / "itpc_window_feasibility.csv", index=False)
    save_distribution(lines, args.output_dir / "stress_rhythm_distribution.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
