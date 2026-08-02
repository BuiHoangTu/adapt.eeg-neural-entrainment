from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
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
    beat_intervals_for_line,
    infer_first_peak_offsets,
    load_line_audio,
    peak_windows_for_line,
    table1_peak_times,
)
from adapt_eeg.plv_peak.plv import (
    audio_envelope_phase,
    channel_phase_signal,
    peak_window_stimulus_plv,
)
from adapt_eeg.plv_peak.stats import add_logit_plv, fit_peak_progression_models

CSV_FLOAT_FORMAT = "%.6f"


def _pick_eeg_channels(inst: mne.io.BaseRaw) -> list[str]:
    picks = mne.pick_types(inst.info, eeg=True, exclude=[])
    channel_names = [inst.ch_names[pick] for pick in picks]
    if len(channel_names) != EXPECTED_CHANNELS:
        print(f"Warning: expected {EXPECTED_CHANNELS} EEG channels, found {len(channel_names)}")
    return channel_names


def _alignment_is_valid(first_peak_offsets: pd.DataFrame, line: int) -> tuple[bool, str]:
    match = first_peak_offsets.loc[first_peak_offsets["line"].astype(int) == int(line)]
    if match.empty:
        return False, "No alignment record for line"
    row = match.iloc[0]
    valid = bool(row.get("alignment_valid", pd.notna(row["first_peak_offset_seconds"])))
    return valid, str(row.get("alignment_error", ""))


def analyze_sample(
    sample: PaperSample,
    audio_root: Path,
    window_radius: float,
    peaks_per_epoch: int,
    first_peak_offsets: pd.DataFrame,
    window_mode: str = "beat_interval",
    bandwidth_hz: float = 1.0,
    padding_seconds: float = 3.0,
) -> tuple[list[dict], list[dict]]:
    alignment_valid, alignment_error = _alignment_is_valid(first_peak_offsets, sample.line)
    if not alignment_valid:
        return [], [
            {
                "participant": sample.participant,
                "line": sample.line,
                "set_path": str(sample.set_path),
                "reason": "invalid_audio_alignment",
                "error": alignment_error,
            }
        ]

    try:
        raw = read_eeglab_sample(sample)
    except Exception as exc:
        return [], [
            {
                "participant": sample.participant,
                "line": sample.line,
                "set_path": str(sample.set_path),
                "reason": "eeg_read_failed",
                "error": repr(exc),
            }
        ]

    try:
        audio_sfreq, audio = load_line_audio(audio_root, sample.line)
    except Exception as exc:
        return [], [
            {
                "participant": sample.participant,
                "line": sample.line,
                "set_path": str(sample.set_path),
                "reason": "audio_read_failed",
                "error": repr(exc),
            }
        ]

    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw_eeg = raw.copy().pick(picks)
    channel_names = _pick_eeg_channels(raw_eeg)
    sfreq = float(raw_eeg.info["sfreq"])
    duration = raw_eeg.n_times / sfreq
    eeg_times = np.arange(raw_eeg.n_times, dtype=float) / sfreq

    rows: list[dict] = []
    skipped: list[dict] = []
    for frequency_hz in FREQUENCIES_HZ:
        try:
            eeg_phases = channel_phase_signal(
                raw_eeg,
                frequency_hz=frequency_hz,
                bandwidth_hz=bandwidth_hz,
                padding_seconds=padding_seconds,
            )
            stimulus_phase = audio_envelope_phase(
                audio,
                sample_rate=float(audio_sfreq),
                eeg_times=eeg_times,
                frequency_hz=frequency_hz,
                bandwidth_hz=bandwidth_hz,
                padding_seconds=padding_seconds,
            )
            if window_mode == "beat_interval":
                windows = beat_intervals_for_line(
                    sample.line,
                    duration,
                    peaks_per_epoch=peaks_per_epoch,
                    first_peak_offsets=first_peak_offsets,
                )
            elif window_mode == "fixed_radius":
                windows = peak_windows_for_line(
                    sample.line,
                    duration,
                    window_radius=window_radius,
                    peaks_per_epoch=peaks_per_epoch,
                    first_peak_offsets=first_peak_offsets,
                )
            else:
                raise ValueError(f"Unknown window mode: {window_mode}")
        except Exception as exc:
            skipped.append(
                {
                    "participant": sample.participant,
                    "line": sample.line,
                    "frequency_hz": frequency_hz,
                    "set_path": str(sample.set_path),
                    "reason": "phase_or_alignment_failed",
                    "error": repr(exc),
                }
            )
            continue

        for window in windows:
            try:
                values, phase_lags, n_window_samples = peak_window_stimulus_plv(
                    eeg_phases,
                    stimulus_phase,
                    sfreq,
                    window,
                )
            except Exception as exc:
                skipped.append(
                    {
                        "participant": sample.participant,
                        "line": sample.line,
                        "frequency_hz": frequency_hz,
                        "peak_index": window.peak_index,
                        "peak_seconds": window.peak_seconds,
                        "set_path": str(sample.set_path),
                        "reason": "stimulus_plv_failed",
                        "error": repr(exc),
                    }
                )
                continue

            for channel, plv, phase_lag in zip(
                channel_names, values, phase_lags, strict=False
            ):
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
                        "window_duration": window.window_end - window.window_start,
                        "window_mode": window_mode,
                        "window_radius": window_radius if window_mode == "fixed_radius" else np.nan,
                        "n_window_samples": n_window_samples,
                        "clipped_start": window.clipped_start,
                        "clipped_end": window.clipped_end,
                        "channel": channel,
                        "stimulus_plv": float(plv),
                        "mean_phase_lag_radians": float(phase_lag),
                        "set_path": str(sample.set_path),
                    }
                )
    return rows, skipped


def summarize_outputs(
    peak_plv: pd.DataFrame,
    first_peak_offsets: pd.DataFrame,
    peaks_per_epoch: int,
) -> dict[str, pd.DataFrame]:
    participant_peak = (
        peak_plv.groupby(
            [
                "participant",
                "language_group",
                "rhythm_condition",
                "frequency_hz",
                "channel",
                "peak_index",
            ],
            as_index=False,
        )
        .agg(
            stimulus_plv=("stimulus_plv", "mean"),
            mean_peak_seconds=("peak_seconds", "mean"),
            mean_phase_lag_radians=("mean_phase_lag_radians", "mean"),
            n_lines=("line", "nunique"),
            n_observations=("line", "size"),
        )
        .sort_values(
            ["frequency_hz", "channel", "participant", "rhythm_condition", "peak_index"]
        )
    )
    participant_peak = add_logit_plv(participant_peak)

    condition_peak_stats = (
        participant_peak.groupby(
            ["frequency_hz", "channel", "rhythm_condition", "peak_index"],
            as_index=False,
        )
        .agg(
            mean_stimulus_plv=("stimulus_plv", "mean"),
            sd_stimulus_plv=("stimulus_plv", "std"),
            mean_plv_logit=("plv_logit", "mean"),
            n_participants=("participant", "nunique"),
        )
        .sort_values(["frequency_hz", "channel", "rhythm_condition", "peak_index"])
    )

    group_peak_stats = (
        participant_peak.groupby(
            ["frequency_hz", "channel", "language_group", "peak_index"],
            as_index=False,
        )
        .agg(
            mean_stimulus_plv=("stimulus_plv", "mean"),
            sd_stimulus_plv=("stimulus_plv", "std"),
            n_participants=("participant", "nunique"),
        )
        .sort_values(["frequency_hz", "channel", "language_group", "peak_index"])
    )

    return {
        "alignment_quality": first_peak_offsets,
        "peak_times_by_line": table1_peak_times(
            peaks_per_epoch,
            first_peak_offsets=first_peak_offsets,
        ),
        "stimulus_plv_by_peak_line_channel": peak_plv,
        "participant_peak_condition_channel": participant_peak,
        "condition_by_peak_summary": condition_peak_stats,
        "group_by_peak_summary": group_peak_stats,
        "peak_progression_model": fit_peak_progression_models(participant_peak),
    }


def run_peak_plv(
    data_root: Path,
    output_dir: Path,
    audio_root: Path,
    window_radius: float,
    peaks_per_epoch: int,
    window_mode: str = "beat_interval",
    bandwidth_hz: float = 1.0,
    padding_seconds: float = 3.0,
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
            audio_root,
            window_radius,
            peaks_per_epoch,
            first_peak_offsets,
            window_mode=window_mode,
            bandwidth_hz=bandwidth_hz,
            padding_seconds=padding_seconds,
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
        first_peak_offsets.to_csv(output_dir / "alignment_quality.csv", index=False)
        print(f"No stimulus-PLV rows were produced; see skipped_samples.csv in {output_dir}")
        return

    for name, frame in summarize_outputs(
        peak_plv,
        first_peak_offsets,
        peaks_per_epoch,
    ).items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    if skipped:
        print(f"Skipped {len(skipped)} sample/peak entries; see skipped_samples.csv")
    print(f"Wrote stimulus-locked PLV outputs to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EEG–audio phase-locking analysis.")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/datasets_4Hz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plv_peak"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/poem1_versuri"))
    parser.add_argument("--window-radius", type=float, default=PEAK_WINDOW_SECONDS)
    parser.add_argument("--peaks-per-epoch", type=int, default=PEAKS_PER_EPOCH)
    parser.add_argument(
        "--window-mode",
        choices=("beat_interval", "fixed_radius"),
        default="beat_interval",
    )
    parser.add_argument("--bandwidth-hz", type=float, default=1.0)
    parser.add_argument("--padding-seconds", type=float, default=3.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_peak_plv(
        args.data_root,
        args.output_dir,
        args.audio_root,
        args.window_radius,
        args.peaks_per_epoch,
        window_mode=args.window_mode,
        bandwidth_hz=args.bandwidth_hz,
        padding_seconds=args.padding_seconds,
    )


if __name__ == "__main__":
    main()
