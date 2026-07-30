from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd

from adapt_eeg.itpc_3cycles.design import (
    CHANNELS_OF_INTEREST,
    EXPECTED_CHANNELS,
    FREQUENCIES_HZ,
    PaperSample,
    discover_samples,
    read_eeglab_sample,
    validate_design_coverage,
)
from adapt_eeg.itpc_3cycles.itpc import epoch_timing_for_frequency, three_cycle_itpc
from adapt_eeg.itpc_3cycles.stats import (
    group_difference_tests,
    mixed_anova_approximation,
    paired_rhythm_tests,
    reject_studentized_residuals,
)

CSV_FLOAT_FORMAT = "%.6f"


def _pick_eeg_channels(inst: mne.io.BaseRaw) -> list[str]:
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
    cycles_per_epoch: float,
    shift_cycles: float,
) -> list[float]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw_eeg = raw.copy().pick(picks)
    return three_cycle_itpc(
        raw_eeg,
        frequency_hz,
        cycles_per_epoch=cycles_per_epoch,
        shift_cycles=shift_cycles,
    ).tolist()


def analyze_sample(
    sample: PaperSample,
    cycles_per_epoch: float,
    shift_cycles: float,
) -> tuple[list[dict], list[dict]]:
    try:
        raw = read_eeglab_sample(sample)
    except Exception as exc:
        return [], [
            {
                "participant": sample.participant,
                "line": sample.line,
                "frequency_hz": None,
                "set_path": str(sample.set_path),
                "reason": "read_failed",
                "error": repr(exc),
            }
        ]

    channel_names = _pick_eeg_channels(raw)
    rows: list[dict] = []
    skipped: list[dict] = []
    for frequency_hz in FREQUENCIES_HZ:
        epoch_seconds, shift_seconds = epoch_timing_for_frequency(
            frequency_hz,
            cycles_per_epoch=cycles_per_epoch,
            shift_cycles=shift_cycles,
        )
        try:
            values = _itpc_from_raw(raw, frequency_hz, cycles_per_epoch, shift_cycles)
        except Exception as exc:
            skipped.append(
                {
                    "participant": sample.participant,
                    "line": sample.line,
                    "frequency_hz": frequency_hz,
                    "epoch_seconds": epoch_seconds,
                    "shift_seconds": shift_seconds,
                    "cycles_per_epoch": cycles_per_epoch,
                    "shift_cycles": shift_cycles,
                    "set_path": str(sample.set_path),
                    "reason": "not_enough_usable_signal_for_3_cycle_itpc",
                    "error": repr(exc),
                }
            )
            continue

        for channel, itpc in zip(channel_names, values, strict=False):
            rows.append(
                {
                    "participant": sample.participant,
                    "line": sample.line,
                    "rhythm_condition": sample.rhythm_condition.value,
                    "language_group": sample.language_group.value,
                    "frequency_hz": frequency_hz,
                    "epoch_seconds": epoch_seconds,
                    "shift_seconds": shift_seconds,
                    "cycles_per_epoch": cycles_per_epoch,
                    "shift_cycles": shift_cycles,
                    "channel": channel,
                    "itpc": float(itpc),
                    "set_path": str(sample.set_path),
                }
            )
    return rows, skipped


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
        .agg(
            itpc=("itpc", "mean"),
            n_lines=("line", "nunique"),
            epoch_seconds=("epoch_seconds", "first"),
            shift_seconds=("shift_seconds", "first"),
            cycles_per_epoch=("cycles_per_epoch", "first"),
            shift_cycles=("shift_cycles", "first"),
        )
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
            epoch_seconds=("epoch_seconds", "first"),
            shift_seconds=("shift_seconds", "first"),
        )
        .sort_values(["frequency_hz", "channel", "rhythm_condition"])
    )

    group_stats = (
        included.groupby(["frequency_hz", "channel", "language_group"], as_index=False)
        .agg(
            mean_itpc=("itpc", "mean"),
            sd_itpc=("itpc", "std"),
            n_participants=("participant", "nunique"),
            epoch_seconds=("epoch_seconds", "first"),
            shift_seconds=("shift_seconds", "first"),
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
    cycles_per_epoch: float,
    shift_cycles: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(data_root)
    validate_design_coverage(samples)

    all_rows: list[dict] = []
    skipped: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] participant {sample.participant}, line {sample.line}")
        rows, sample_skips = analyze_sample(sample, cycles_per_epoch, shift_cycles)
        all_rows.extend(rows)
        skipped.extend(sample_skips)

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
    pd.DataFrame(skipped).to_csv(output_dir / "skipped_samples.csv", index=False)

    channel_itpc = pd.DataFrame(all_rows)
    if channel_itpc.empty:
        print(f"No ITPC rows were produced; see skipped_samples.csv in {output_dir}")
        return

    for name, frame in summarize_outputs(channel_itpc).items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    print(f"Wrote 3-cycle ITPC outputs to {output_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} participant-line-frequency samples; see skipped_samples.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 3-cycle ITPC replication pipeline."
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
        default=Path("results/itpc_3cycles"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--cycles-per-epoch",
        type=float,
        default=3.0,
        help="Cycles per epoch at each analyzed frequency.",
    )
    parser.add_argument(
        "--shift-cycles",
        type=float,
        default=1.0,
        help="Epoch shift in cycles; must be at least 1.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_replication(
        args.data_root,
        args.output_dir,
        args.cycles_per_epoch,
        args.shift_cycles,
    )


if __name__ == "__main__":
    main()
