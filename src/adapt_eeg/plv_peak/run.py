from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd

from adapt_eeg.plv_peak.design import (
    EXPECTED_CHANNELS,
    FREQUENCIES_HZ,
    PaperSample,
    discover_samples,
    read_eeglab_sample,
    validate_design_coverage,
)
from adapt_eeg.plv_peak.peaks import (
    PEAK_WINDOW_SECONDS,
    PEAKS_PER_EPOCH,
    infer_first_peak_offsets,
    peak_windows_for_line,
    table1_peak_times,
)
from adapt_eeg.plv_peak.plv import centered_plv, channel_phase_signal, peak_window_plv
from adapt_eeg.plv_peak.stats import (
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
        print(f"Warning: expected {EXPECTED_CHANNELS} EEG channels, found {len(channel_names)}")
    return channel_names


def analyze_sample(
    sample: PaperSample,
    window_radius: float,
    peaks_per_epoch: int,
    first_peak_offsets: pd.DataFrame | None,
) -> tuple[list[dict], list[dict]]:
    try:
        raw = read_eeglab_sample(sample)
    except Exception as exc:
        return [], [{"participant": sample.participant, "line": sample.line, "set_path": str(sample.set_path), "reason": "read_failed", "error": repr(exc)}]

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw_eeg = raw.copy().pick(picks)
    channel_names = _pick_eeg_channels(raw_eeg)
    duration = raw_eeg.n_times / float(raw_eeg.info["sfreq"])

    rows: list[dict] = []
    skipped: list[dict] = []
    for frequency_hz in FREQUENCIES_HZ:
        try:
            phases = channel_phase_signal(raw_eeg, frequency_hz)
            windows = peak_windows_for_line(
                sample.line,
                duration,
                window_radius=window_radius,
                peaks_per_epoch=peaks_per_epoch,
                first_peak_offsets=first_peak_offsets,
            )
        except Exception as exc:
            skipped.append(
                {
                    "participant": sample.participant,
                    "line": sample.line,
                    "frequency_hz": frequency_hz,
                    "set_path": str(sample.set_path),
                    "reason": "phase_or_peak_window_failed",
                    "error": repr(exc),
                }
            )
            continue

        for window in windows:
            try:
                values = peak_window_plv(phases, float(raw_eeg.info["sfreq"]), window)
            except Exception as exc:
                skipped.append(
                    {
                        "participant": sample.participant,
                        "line": sample.line,
                        "frequency_hz": frequency_hz,
                        "peak_index": window.peak_index,
                        "peak_seconds": window.peak_seconds,
                        "set_path": str(sample.set_path),
                        "reason": "peak_plv_failed",
                        "error": repr(exc),
                    }
                )
                continue

            for channel, plv in zip(channel_names, values, strict=False):
                rows.append(
                    {
                        "participant": sample.participant,
                        "line": sample.line,
                        "rhythm_condition": sample.rhythm_condition.value,
                        "language_group": sample.language_group.value,
                        "frequency_hz": frequency_hz,
                        "peak_index": window.peak_index,
                        "peak_seconds": window.peak_seconds,
                        "window_start": window.window_start,
                        "window_end": window.window_end,
                        "window_radius": window_radius,
                        "clipped_start": window.clipped_start,
                        "clipped_end": window.clipped_end,
                        "channel": channel,
                        "plv": float(plv),
                        "set_path": str(sample.set_path),
                    }
                )
    return rows, skipped


def summarize_outputs(
    peak_plv: pd.DataFrame,
    first_peak_offsets: pd.DataFrame,
    peaks_per_epoch: int,
) -> dict[str, pd.DataFrame]:
    centered = centered_plv(peak_plv)
    participant_condition_channel = (
        centered.groupby(
            ["participant", "language_group", "rhythm_condition", "frequency_hz", "channel"],
            as_index=False,
        )
        .agg(
            plv=("plv", "mean"),
            centered_plv=("centered_plv", "mean"),
            n_lines=("line", "nunique"),
            n_peaks=("peak_index", "count"),
        )
        .rename(columns={"centered_plv": "itpc"})
        .sort_values(["frequency_hz", "channel", "participant", "rhythm_condition"])
    )

    cleaned = reject_studentized_residuals(participant_condition_channel, value_column="itpc")
    included = cleaned.loc[~cleaned["excluded_as_outlier"]].copy()

    condition_stats = (
        included.groupby(["frequency_hz", "channel", "rhythm_condition"], as_index=False)
        .agg(
            mean_centered_plv=("itpc", "mean"),
            sd_centered_plv=("itpc", "std"),
            n_participants=("participant", "nunique"),
        )
        .sort_values(["frequency_hz", "channel", "rhythm_condition"])
    )

    group_stats = (
        included.groupby(["frequency_hz", "channel", "language_group"], as_index=False)
        .agg(
            mean_centered_plv=("itpc", "mean"),
            sd_centered_plv=("itpc", "std"),
            n_participants=("participant", "nunique"),
        )
        .sort_values(["frequency_hz", "channel", "language_group"])
    )

    return {
        "first_peak_offsets_by_line": first_peak_offsets,
        "peak_times_by_line": table1_peak_times(
            peaks_per_epoch,
            first_peak_offsets=first_peak_offsets,
        ),
        "plv_by_peak_line_channel": peak_plv,
        "plv_centered_by_peak_line_channel": centered,
        "plv_by_participant_condition_channel": cleaned,
        "plv_included_after_outlier_rejection": included,
        "condition_stats_by_channel": condition_stats,
        "group_stats_by_channel": group_stats,
        "paired_rhythm_tests": paired_rhythm_tests(included, value_column="itpc"),
        "group_difference_tests": group_difference_tests(included, value_column="itpc"),
        "mixed_anova_approximation": mixed_anova_approximation(included, value_column="itpc"),
    }


def run_peak_plv(
    data_root: Path,
    output_dir: Path,
    audio_root: Path,
    window_radius: float,
    peaks_per_epoch: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(data_root)
    validate_design_coverage(samples)
    first_peak_offsets = infer_first_peak_offsets(audio_root)

    all_rows: list[dict] = []
    skipped: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] participant {sample.participant}, line {sample.line}")
        rows, sample_skips = analyze_sample(
            sample,
            window_radius,
            peaks_per_epoch,
            first_peak_offsets,
        )
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

    peak_plv = pd.DataFrame(all_rows)
    if peak_plv.empty:
        print(f"No peak PLV rows were produced; see skipped_samples.csv in {output_dir}")
        return

    for name, frame in summarize_outputs(
        peak_plv,
        first_peak_offsets,
        peaks_per_epoch,
    ).items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    if skipped:
        print(f"Skipped {len(skipped)} sample/peak entries; see skipped_samples.csv")
    print(f"Wrote peak PLV outputs to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run peak-locked poem PLV analysis.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/datasets_4Hz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plv_peak"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/poem1_versuri"))
    parser.add_argument("--window-radius", type=float, default=PEAK_WINDOW_SECONDS)
    parser.add_argument("--peaks-per-epoch", type=int, default=PEAKS_PER_EPOCH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_peak_plv(
        args.data_root,
        args.output_dir,
        args.audio_root,
        args.window_radius,
        args.peaks_per_epoch,
    )


if __name__ == "__main__":
    main()
