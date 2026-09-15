"""Poem ITPC using poem-specific ACF windows and 30 equal line shifts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats

from adapt_eeg.exploration.stress_rhythm_evidence import analyze_poem, summarize_by_poem
from adapt_eeg.poem_eeg import (
    InvalidPoemRecordingError,
    PoemRecording,
    discover_ica_cleaned_files,
    ids_from_path,
    load_poem_recording,
    unique_recording_paths,
)
from adapt_eeg.poem_itpc import (
    CSV_FLOAT_FORMAT,
    MIN_WINDOWS,
    condition_stats_table,
    condition_stats_by_poem,
    debiased_squared_itpc,
    export_xlsx,
    fourier_itpc,
    language_group,
    language_group_stats_table,
    participant_condition_table,
    usable_eeg_channels,
)
from adapt_eeg.poem_rhythm import PoemLineTiming, derive_poem_line_timings, poem_stress_frequency

N_LINE_INTERVALS = 30
PRE_ANCHOR_FRACTION = 1.0 / 3.0


def poem_acf_configuration(poems: list[int]) -> pd.DataFrame:
    """Use the median line-level intensity ACF peak as each poem's ACF estimate."""
    lines = pd.concat([analyze_poem(poem) for poem in poems], ignore_index=True)
    summary = summarize_by_poem(lines)
    summary = summary[["poem", "median_intensity_autocorrelation_peak_hz"]].copy()
    summary = summary.rename(
        columns={"median_intensity_autocorrelation_peak_hz": "acf_frequency_hz"}
    )
    summary["acf_window_duration_s"] = 1.0 / summary["acf_frequency_hz"]
    summary["itpc_frequency_source"] = "poem_mean_stress_interval"
    summary["window_source"] = "median_line_intensity_acf_peak_lag"
    summary["line_anchor_count"] = N_LINE_INTERVALS + 1
    return summary


def acf_condition_stats_by_poem(full: pd.DataFrame) -> pd.DataFrame:
    """Add the ACF-derived configuration to the poem-level statistics table."""
    table = condition_stats_by_poem(full).drop(columns="frequency_hz")
    windows = full.groupby("poem")["acf_window_duration_s"].first()
    table.insert(
        1,
        "acf_frequency_hz",
        table["poem"].map(1.0 / windows),
    )
    table.insert(
        2,
        "acf_window_duration_s",
        table["poem"].map(windows),
    )

    participant = (
        full.groupby(
            ["participant", "language_group", "rhythm_condition"],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
    )
    paired = participant.pivot(
        index=["participant", "language_group"],
        columns="rhythm_condition",
        values="itpc_squared_debiased",
    ).dropna(subset=["regular", "irregular"])
    differences = paired["regular"] - paired["irregular"]
    grouped = {
        group: values.to_numpy()
        for group, values in differences.groupby(level="language_group")
    }
    wilcoxon = stats.wilcoxon(paired["regular"], paired["irregular"])
    mannwhitney = stats.mannwhitneyu(
        grouped["english_competence"],
        grouped["non_english_competence"],
        alternative="two-sided",
    )
    total = pd.DataFrame(
        [
            {
                "poem": "Total",
                "acf_frequency_hz": np.nan,
                "acf_window_duration_s": np.nan,
                "n_paired": len(paired),
                "regular_mean": paired["regular"].mean(),
                "irregular_mean": paired["irregular"].mean(),
                "regular_minus_irregular": differences.mean(),
                "wilcoxon_statistic": wilcoxon.statistic,
                "wilcoxon_pvalue": wilcoxon.pvalue,
                "english_n": len(grouped["english_competence"]),
                "non_english_n": len(grouped["non_english_competence"]),
                "english_mean_difference": grouped["english_competence"].mean(),
                "non_english_mean_difference": grouped[
                    "non_english_competence"
                ].mean(),
                "mannwhitney_statistic": mannwhitney.statistic,
                "mannwhitney_pvalue": mannwhitney.pvalue,
            }
        ]
    )
    return pd.concat([table, total], ignore_index=True)


def acf_line_epochs(
    recording: PoemRecording,
    line: PoemLineTiming,
    channel_names: list[str],
    window_s: float,
    *,
    is_final_line: bool,
) -> mne.Epochs:
    start_s = recording.usable_start_s + line.start_s
    end_s = recording.usable_end_s if is_final_line else recording.usable_start_s + line.end_s
    if start_s >= end_s:
        raise ValueError(f"poem {line.poem} line {line.line_number}: non-positive interval")

    # Match the legacy construction: 31 anchors define 30 equal shifts. Epochs
    # extend one third of the ACF window before and two thirds after each anchor;
    # MNE drops anchors whose complete window lies outside the recording/line.
    anchor_times = np.linspace(start_s, end_s, N_LINE_INTERVALS + 1)
    sfreq = float(recording.raw.info["sfreq"])
    event_samples = recording.raw.time_as_index(anchor_times, use_rounding=True)
    events = np.column_stack(
        [event_samples, np.zeros(len(event_samples), dtype=int), np.ones(len(event_samples), dtype=int)]
    )
    tmin = -PRE_ANCHOR_FRACTION * window_s
    n_samples = max(2, int(round(window_s * sfreq)))
    tmax = tmin + (n_samples - 1) / sfreq
    epochs = mne.Epochs(
        recording.raw.copy().pick(channel_names),
        events,
        event_id={"line_anchor": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    # Enforce line containment; otherwise interior poem lines could borrow EEG
    # from their neighbours even though the epoch exists in the full recording.
    kept = epochs.selection
    epoch_starts = anchor_times[kept] + tmin
    epoch_ends = anchor_times[kept] + tmax
    epochs = epochs[(epoch_starts >= start_s) & (epoch_ends <= end_s)]
    if len(epochs) < MIN_WINDOWS:
        raise ValueError(f"ITPC requires at least {MIN_WINDOWS} windows; found {len(epochs)}")
    return epochs


def analyze_recording(
    recording: PoemRecording,
    lines: list[PoemLineTiming],
    frequency_hz: float,
    window_s: float,
) -> tuple[list[dict], list[dict]]:
    rows, errors = [], []
    channel_names = usable_eeg_channels(recording.raw)
    for line in (item for item in lines if item.rhythm_condition != "ambiguous"):
        try:
            final = line.line_index == len(lines) - 1
            epochs = acf_line_epochs(recording, line, channel_names, window_s, is_final_line=final)
            values, counts = fourier_itpc(epochs, frequency_hz)
            debiased = debiased_squared_itpc(values, counts)
            end_s = recording.usable_end_s if final else recording.usable_start_s + line.end_s
            start_s = recording.usable_start_s + line.start_s
            shift_s = (end_s - start_s) / N_LINE_INTERVALS
            for channel, value, corrected, count in zip(
                epochs.ch_names, values, debiased, counts, strict=True
            ):
                if np.isfinite(value):
                    rows.append(
                        {
                            "participant": recording.participant,
                            "language_group": language_group(recording.participant),
                            "poem": recording.poem,
                            "line": line.line_number,
                            "rhythm_condition": line.rhythm_condition,
                            "rhythm_std_s": line.std_interval_s,
                            "median_inter_stress_interval_s": line.median_interval_s,
                            "stress_frequency_hz": line.stress_frequency_hz,
                            "frequency_source": "poem_mean_stress_interval",
                            "frequency_hz": frequency_hz,
                            "window_source": "poem_intensity_acf_peak_lag",
                            "acf_window_duration_s": window_s,
                            "line_anchor_count": N_LINE_INTERVALS + 1,
                            "window_shift_s": shift_s,
                            "n_windows": len(epochs),
                            "n_valid_phase_windows": int(count),
                            "line_start_eeg_s": start_s,
                            "line_end_eeg_s": end_s,
                            "line_duration_s": end_s - start_s,
                            "channel": channel,
                            "itpc": float(value),
                            "itpc_squared_debiased": float(corrected),
                            "set_path": str(recording.path),
                        }
                    )
        except Exception as exc:
            errors.append(
                {"participant": recording.participant, "poem": recording.poem,
                 "line": line.line_number, "set_path": str(recording.path),
                 "stage": "line_itpc", "error_type": type(exc).__name__, "error": str(exc)}
            )
    return rows, errors


def run(data_root: Path, output_dir: Path, *, fail_on_errors: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_ica_cleaned_files(data_root)
    paths, duplicates = unique_recording_paths(discovered)
    poems = sorted({ids_from_path(path)[0] for path in discovered})
    timings = {poem: derive_poem_line_timings(poem) for poem in poems}
    frequencies = {poem: poem_stress_frequency(lines) for poem, lines in timings.items()}
    config = poem_acf_configuration(poems)
    config["itpc_frequency_hz"] = config["poem"].map(frequencies)
    config.to_csv(output_dir / "acf_itpc_configuration.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    windows = config.set_index("poem")["acf_window_duration_s"].to_dict()
    pd.DataFrame([asdict(line) for lines in timings.values() for line in lines]).to_csv(
        output_dir / "poem_line_rhythm.csv", index=False, float_format=CSV_FLOAT_FORMAT
    )

    errors = [
        {"participant": participant, "poem": poem, "line": None,
         "set_path": " | ".join(map(str, candidates)), "stage": "discovery",
         "error_type": "DuplicatePoemRecordingError",
         "error": f"Found {len(candidates)} recordings; none selected"}
        for (poem, participant), candidates in sorted(duplicates.items())
    ]
    rows = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path}")
        try:
            recording = load_poem_recording(path, preload=True)
            found, failures = analyze_recording(
                recording, timings[recording.poem], frequencies[recording.poem], windows[recording.poem]
            )
            rows.extend(found)
            errors.extend(failures)
        except Exception as exc:
            try:
                poem, participant = ids_from_path(path)
            except Exception:
                poem, participant = None, None
            errors.append({"participant": participant, "poem": poem, "line": None,
                           "set_path": str(path), "stage": "recording",
                           "error_type": type(exc).__name__, "error": str(exc)})

    full = pd.DataFrame(rows)
    pd.DataFrame(errors).to_csv(output_dir / "load_errors.csv", index=False)
    if full.empty:
        raise RuntimeError("No ACF-influenced ITPC results were produced")
    full.to_csv(output_dir / "itpc_by_line_channel.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    participant = participant_condition_table(full)
    participant.to_csv(output_dir / "itpc_by_participant_condition_channel.csv", index=False,
                       float_format=CSV_FLOAT_FORMAT)
    condition_stats_table(participant).to_csv(output_dir / "condition_stats_by_channel.csv", index=False,
                                              float_format=CSV_FLOAT_FORMAT)
    language_group_stats_table(participant).to_csv(
        output_dir / "language_group_stats_by_channel.csv", index=False, float_format=CSV_FLOAT_FORMAT
    )
    acf_condition_stats_by_poem(full).to_csv(
        output_dir / "condition_stats_by_poem.csv", index=False, float_format=CSV_FLOAT_FORMAT
    )
    export_xlsx(full, output_dir / "poem_itpc_by_channel.xlsx")
    print(config.to_string(index=False))
    print(f"Wrote {len(full)} line-channel rows from {len(paths)} recordings")
    print(f"Recorded {len(errors)} errors in {output_dir / 'load_errors.csv'}")
    if errors and fail_on_errors:
        raise InvalidPoemRecordingError(f"{len(errors)} errors; see {output_dir / 'load_errors.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw-eeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/poem_itpc_acf"))
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()
    run(args.data_root, args.output_dir, fail_on_errors=not args.allow_errors)


if __name__ == "__main__":
    main()
