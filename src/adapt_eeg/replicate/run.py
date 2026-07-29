from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd

from adapt_eeg.replicate.design import (
    CHANNELS_OF_INTEREST,
    EXPECTED_CHANNELS,
    FREQUENCIES_HZ,
    PaperSample,
    discover_samples,
    read_eeglab_sample,
    validate_design_coverage,
)
from adapt_eeg.replicate.itpc import fixed_length_itpc
from adapt_eeg.replicate.stats import (
    group_difference_tests,
    mixed_anova_approximation,
    paired_rhythm_tests,
    reject_studentized_residuals,
)

CSV_FLOAT_FORMAT = "%.6f"


def _pick_eeg_channels(inst: mne.Epochs | mne.io.BaseRaw) -> list[str]:
    picks = mne.pick_types(inst.info, eeg=True, exclude=[])
    channel_names = [inst.ch_names[pick] for pick in picks]
    if len(channel_names) != EXPECTED_CHANNELS:
        print(
            f"Warning: expected {EXPECTED_CHANNELS} EEG channels, found "
            f"{len(channel_names)} in {getattr(inst, 'filename', '<memory>')}"
        )
    return channel_names



def _itpc_from_raw(
    raw: mne.io.BaseRaw,
    frequency_hz: float,
    window_seconds: float,
    step_seconds: float,
) -> np.ndarray:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    return fixed_length_itpc(
        data,
        float(raw.info["sfreq"]),
        frequency_hz,
        window_seconds,
        step_seconds,
    )


def analyze_sample(
    sample: PaperSample,
    window_seconds: float,
    step_seconds: float,
) -> tuple[list[dict], dict | None]:
    try:
        raw = read_eeglab_sample(sample)
        channel_names = _pick_eeg_channels(raw)
        rows: list[dict] = []
        for frequency_hz in FREQUENCIES_HZ:
            values = _itpc_from_raw(raw, frequency_hz, window_seconds, step_seconds)

            for channel, itpc in zip(channel_names, values, strict=False):
                rows.append(
                    {
                        "participant": sample.participant,
                        "line": sample.line,
                        "rhythm_condition": sample.rhythm_condition.value,
                        "language_group": sample.language_group.value,
                        "frequency_hz": frequency_hz,
                        "channel": channel,
                        "itpc": float(itpc),
                        "set_path": str(sample.set_path),
                    }
                )
        return rows, None
    except Exception as exc:
        return [], {
            "participant": sample.participant,
            "line": sample.line,
            "set_path": str(sample.set_path),
            "error": repr(exc),
        }


def summarize_outputs(channel_itpc: pd.DataFrame) -> dict[str, pd.DataFrame]:
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
    included = cleaned.loc[~cleaned["excluded_as_outlier"]].copy()

    condition_stats = (
        included.groupby(["frequency_hz", "channel", "rhythm_condition"], as_index=False)
        .agg(
            mean_itpc=("itpc", "mean"),
            sd_itpc=("itpc", "std"),
            n_participants=("participant", "nunique"),
        )
        .sort_values(["frequency_hz", "channel", "rhythm_condition"])
    )

    group_stats = (
        included.groupby(["frequency_hz", "channel", "language_group"], as_index=False)
        .agg(
            mean_itpc=("itpc", "mean"),
            sd_itpc=("itpc", "std"),
            n_participants=("participant", "nunique"),
        )
        .sort_values(["frequency_hz", "channel", "language_group"])
    )

    channels_of_interest = included.loc[included["channel"].isin(CHANNELS_OF_INTEREST)]

    return {
        "itpc_by_channel_line": channel_itpc,
        "itpc_by_participant_condition_channel": cleaned,
        "itpc_included_after_outlier_rejection": included,
        "condition_stats_by_channel": condition_stats,
        "group_stats_by_channel": group_stats,
        "paired_rhythm_tests": paired_rhythm_tests(included),
        "group_difference_tests": group_difference_tests(included),
        "mixed_anova_approximation": mixed_anova_approximation(included),
        "channels_of_interest": channels_of_interest,
    }


def run_replication(
    data_root: Path,
    output_dir: Path,
    window_seconds: float,
    step_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(data_root)
    validate_design_coverage(samples)

    all_rows: list[dict] = []
    errors: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] participant {sample.participant}, line {sample.line}")
        rows, error = analyze_sample(
            sample,
            window_seconds,
            step_seconds,
        )
        all_rows.extend(rows)
        if error is not None:
            errors.append(error)

    metadata = pd.DataFrame(
        [
            {
                "participant": sample.participant,
                "line": sample.line,
                "rhythm_condition": sample.rhythm_condition.value,
                "language_group": sample.language_group.value,
                "set_path": str(sample.set_path),
            }
            for sample in samples
        ]
    )
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "load_errors.csv", index=False)
    if errors:
        raise RuntimeError(
            f"Failed to load or analyze {len(errors)} required samples; see load_errors.csv"
        )

    channel_itpc = pd.DataFrame(all_rows)
    if channel_itpc.empty:
        raise RuntimeError("No ITPC rows were produced; see load_errors.csv")

    for name, frame in summarize_outputs(channel_itpc).items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    print(f"Wrote replication outputs to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replicate the ADAPT EEG poetry-rhythm ITPC experiment."
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
        default=Path("results/replicate"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.3,
        help="Fixed window length from the paper autocorrelation analysis.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=0.05,
        help="Step size when slicing continuous line segments into ITPC trials.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_replication(
        args.data_root,
        args.output_dir,
        args.window_seconds,
        args.step_seconds,
    )


if __name__ == "__main__":
    main()
