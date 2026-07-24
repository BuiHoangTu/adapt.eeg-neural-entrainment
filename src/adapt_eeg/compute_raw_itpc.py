from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from adapt_eeg.constants import FREQUENCIES, MIN_DURATION, MIN_SHIFT
from adapt_eeg.data_reader import Sample, read_raw_data

CSV_FLOAT_FORMAT = "%.4f"


def fourier_itpc(
    data: np.ndarray, times: np.ndarray, freqs: dict[str, float]
) -> dict[str, np.ndarray]:
    data = data - data.mean(axis=-1, keepdims=True)
    window = np.hanning(data.shape[-1])
    out = {}
    for frequency_name, freq in freqs.items():
        kernel = window * np.exp(-2j * np.pi * freq * times)
        coef = np.sum(data * kernel[None, None, :], axis=-1)
        valid = np.abs(coef) > np.finfo(float).eps
        unit_phase = np.zeros_like(coef, dtype=np.complex128)
        unit_phase[valid] = coef[valid] / np.abs(coef[valid])
        out[frequency_name] = np.abs(unit_phase.mean(axis=0))
    return out



def _sample_duration(raw: mne.io.BaseRaw) -> float:
    # return raw.n_times / float(raw.info["sfreq"])
    return raw.duration


def raw_to_epochs(
    raw: mne.io.BaseRaw,
    window_duration: float,
    window_overlap: float,
) -> mne.Epochs:
    duration = _sample_duration(raw)
    if duration < window_duration:
        raise ValueError(
            f"Raw segment is {duration:.3f}s, shorter than the requested "
            f"{window_duration:.3f}s ITPC window"
        )

    return mne.make_fixed_length_epochs(
        raw,
        duration=window_duration,
        overlap=window_overlap,
        preload=True,
        verbose="ERROR",
    )


def analyze_raw_sample(
    sample: Sample,
    window_duration: float,
    window_overlap: float,
) -> tuple[list[dict], list[dict]]:
    raw = sample.data
    # if not isinstance(raw, mne.io.BaseRaw):
    #     raise TypeError(f"Expected an MNE Raw object, got {type(raw).__name__}")

    epochs = raw_to_epochs(raw, window_duration, window_overlap)
    data = epochs.get_data(copy=True)
    itpc_by_freq = fourier_itpc(data, epochs.times, FREQUENCIES)

    file_rows = []
    channel_rows = []

    for frequency_name, channel_values in itpc_by_freq.items():
        freq = FREQUENCIES[frequency_name]
        file_rows.append(
            {
                "participant": sample.participant,
                "line": sample.line,
                "rhythm_condition": sample.rhythm_type.value,
                "language_group": sample.language_understanding.value,
                "frequency_name": frequency_name,
                "frequency_hz": freq,
                "itpc_mean": float(channel_values.mean()),
            }
        )
        for channel, value in zip(epochs.ch_names, channel_values, strict=True):
            channel_rows.append(
                {
                    "participant": sample.participant,
                    "line": sample.line,
                    "rhythm_condition": sample.rhythm_type.value,
                    "language_group": sample.language_understanding.value,
                    "frequency_name": frequency_name,
                    "frequency_hz": freq,
                    "channel": channel,
                    "itpc": float(value),
                }
            )

    return file_rows, channel_rows


def summarize_outputs(by_file: pd.DataFrame, by_channel: pd.DataFrame, output_dir: Path) -> None:
    participant_condition_channel = by_channel.groupby(
        [
            "participant",
            "language_group",
            "rhythm_condition",
            "frequency_name",
            "frequency_hz",
            "channel",
        ],
        as_index=False,
    ).agg(itpc=("itpc", "mean"))
    participant_condition_channel.to_csv(
        output_dir / "itpc_by_participant_condition_channel.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )

    participant_condition = by_file.groupby(
        [
            "participant",
            "language_group",
            "rhythm_condition",
            "frequency_name",
            "frequency_hz",
        ],
        as_index=False,
    ).agg(itpc=("itpc_mean", "mean"))
    participant_condition.to_csv(
        output_dir / "itpc_by_participant_condition.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )


def main(args) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.window_shift <= 0:
        raise ValueError("--window-shift must be positive")
    if args.window_shift > args.window_duration:
        raise ValueError("--window-shift must be less than or equal to --window-duration")

    samples = read_raw_data(args.data_root)
    if not samples:
        raise RuntimeError(f"No Raw samples found under {args.data_root}")

    file_rows = []
    channel_rows = []
    error_rows = []

    for sample in samples:
        try:
            one_file_rows, one_channel_rows = analyze_raw_sample(
                sample,
                window_duration=args.window_duration,
                window_overlap=args.window_duration - args.window_shift,
            )
        except Exception as exc:
            error_rows.append(
                {
                    "participant": sample.participant,
                    "line": sample.line,
                    "rhythm_condition": sample.rhythm_type.value,
                    "language_group": sample.language_understanding.value,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        file_rows.extend(one_file_rows)
        channel_rows.extend(one_channel_rows)

    pd.DataFrame(error_rows).to_csv(
        output_dir / "load_errors.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    by_file = pd.DataFrame(file_rows)
    by_channel = pd.DataFrame(channel_rows)
    if by_file.empty or by_channel.empty:
        raise RuntimeError("No samples could be analyzed; see load_errors.csv")

    by_file.to_csv(
        output_dir / "itpc_by_file.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    by_channel.to_csv(
        output_dir / "itpc_by_line_channel.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )

    summarize_outputs(by_file, by_channel, output_dir)

    print(f"Read {len(samples)} Raw samples from {args.data_root}")
    print(f"Skipped {len(error_rows)} samples; see {output_dir / 'load_errors.csv'}")
    print(f"Wrote {len(by_file)} file-frequency rows to {output_dir / 'itpc_by_file.csv'}")
    print(
        f"Wrote {len(by_channel)} channel rows to "
        f"{output_dir / 'itpc_by_line_channel.csv'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute paper-aligned ITPC outputs from Raw line-level EEGLAB files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/datasets_4Hz"),
        help="Root containing v*/4hz_datasets/p*.set Raw line segments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/datasets_4Hz"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=MIN_DURATION,
        help="Fixed Raw window duration in seconds for across-window ITPC.",
    )
    parser.add_argument(
        "--window-shift",
        type=float,
        default=MIN_SHIFT,
        help="Fixed Raw window shifting in seconds.",
    )
    args = parser.parse_args()
    main(args)
