"""Approximate paper Figure 5 from trigger-aligned continuous EEG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft

from adapt_eeg.poem_eeg import (
    discover_ica_cleaned_files,
    ids_from_path,
    load_poem_recording,
    unique_recording_paths,
)
from adapt_eeg.poem_itpc import usable_eeg_channels


def normalized_acf(data: np.ndarray, max_lag_samples: int) -> np.ndarray:
    """Compute an unbiased, zero-mean normalized ACF for every row."""
    centered = data - data.mean(axis=1, keepdims=True)
    n_times = centered.shape[1]
    transform_length = next_fast_len(2 * n_times - 1)
    spectrum = rfft(centered, n=transform_length, axis=1)
    autocovariance = irfft(
        spectrum * spectrum.conj(), n=transform_length, axis=1
    )[:, : max_lag_samples + 1]
    autocovariance /= np.arange(n_times, n_times - max_lag_samples - 1, -1)
    variance = autocovariance[:, :1]
    valid = variance[:, 0] > np.finfo(float).eps
    return autocovariance[valid] / variance[valid]


def participant_acf(path: Path, max_lag_s: float) -> tuple[np.ndarray, float]:
    recording = load_poem_recording(path, preload=True)
    channels = usable_eeg_channels(recording.raw)
    cropped = recording.raw.copy().pick(channels).crop(
        tmin=recording.usable_start_s,
        tmax=recording.usable_end_s,
        include_tmax=False,
    )
    sfreq = float(cropped.info["sfreq"])
    max_lag_samples = int(round(max_lag_s * sfreq))
    channel_acf = normalized_acf(cropped.get_data(), max_lag_samples)
    if len(channel_acf) == 0:
        raise ValueError(f"{path}: no non-flat channels available for ACF")
    return channel_acf.mean(axis=0), sfreq


def run(
    data_root: Path,
    output_dir: Path,
    poem: int,
    max_lag_s: float,
) -> None:
    candidates = [
        path
        for path in discover_ica_cleaned_files(data_root)
        if ids_from_path(path)[0] == poem
    ]
    paths, duplicates = unique_recording_paths(candidates)
    errors = [
        {
            "path": " | ".join(map(str, values)),
            "error": f"duplicate recording for poem {poem}, participant {participant}",
        }
        for (_, participant), values in sorted(duplicates.items())
    ]

    participant_curves = []
    sample_rate = None
    included = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}")
        try:
            curve, current_rate = participant_acf(path, max_lag_s)
            if sample_rate is None:
                sample_rate = current_rate
            elif not np.isclose(current_rate, sample_rate):
                raise ValueError(
                    f"sample rate {current_rate:g} differs from {sample_rate:g} Hz"
                )
            participant_curves.append(curve)
            included.append(ids_from_path(path)[1])
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if not participant_curves or sample_rate is None:
        raise RuntimeError("No valid EEG recordings were available for ACF")
    curves = np.asarray(participant_curves)
    lags_s = np.arange(curves.shape[1]) / sample_rate
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0, ddof=1) / np.sqrt(len(curves))

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "lag_s": lags_s,
            "mean_acf": mean,
            "sem_acf": sem,
            "ci95_lower": mean - 1.96 * sem,
            "ci95_upper": mean + 1.96 * sem,
            "n_participants": len(curves),
        }
    ).to_csv(output_dir / f"poem{poem}_eeg_mean_acf.csv", index=False)
    pd.DataFrame(errors, columns=["path", "error"]).to_csv(
        output_dir / f"poem{poem}_eeg_acf_errors.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.plot(lags_s, mean, color="#4C9A9A", linewidth=1.6, label="Mean EEG ACF")
    axis.fill_between(
        lags_s,
        mean - 1.96 * sem,
        mean + 1.96 * sem,
        color="#4C9A9A",
        alpha=0.2,
        label="95% CI across participants",
    )
    axis.axvline(0.3, color="grey", linestyle="--", label="Paper cutoff: 0.3 s")
    axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axis.set(
        xlim=(0, max_lag_s),
        xlabel="Lag (s)",
        ylabel="Normalized ACF",
        title=f"Poem {poem}: mean continuous-EEG autocorrelation",
    )
    axis.grid(alpha=0.15)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / f"poem{poem}_eeg_mean_acf.png", dpi=200)
    plt.close(figure)
    print(f"Included {len(curves)} participants: {included}")
    print(f"Recorded {len(errors)} exclusions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw-eeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exploration"))
    parser.add_argument("--poem", type=int, default=1)
    parser.add_argument("--max-lag-s", type=float, default=1.0)
    args = parser.parse_args()
    run(args.data_root, args.output_dir, args.poem, args.max_lag_s)


if __name__ == "__main__":
    main()
