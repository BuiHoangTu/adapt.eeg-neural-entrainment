"""Poem ITPC using poem-specific ACF windows and 30 equal line shifts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats

from adapt_eeg.exploration.stress_rhythm_evidence import EEG_POEM_IDS
from adapt_eeg.poem_eeg import (
    InvalidPoemRecordingError,
    PoemRecording,
    aligned_line_end_s,
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
from adapt_eeg.poem_rhythm import (
    PoemLineTiming,
    classify_rhythm,
    derive_poem_line_timings,
    poem_stress_frequency,
    select_iambic_lines,
)

N_LINE_INTERVALS = 30
PRE_ANCHOR_FRACTION = 1.0 / 3.0
DEFAULT_RHYTHM_CSV = Path("results/poem_itpc/poem_line_rhythm.csv")
PAPER_WINDOW_S = 0.3


def paper_window_configuration(poems: list[int]) -> pd.DataFrame:
    """Use the paper's EEG-ACF-motivated fixed 0.3-second window."""
    return pd.DataFrame(
        {
            "poem": poems,
            "window_duration_s": PAPER_WINDOW_S,
            "itpc_frequency_source": "poem_mean_stress_interval",
            "window_source": "paper_eeg_acf_fixed_0.3s",
            "line_anchor_count": N_LINE_INTERVALS + 1,
        }
    )


def load_or_derive_timings(
    poems: list[int],
    rhythm_csv: Path,
    *,
    insufficient_as_ambiguous: bool = False,
) -> dict[int, list[PoemLineTiming]]:
    """Reuse the cycle-aligned line classification so both analyses match."""
    if rhythm_csv.exists():
        frame = pd.read_csv(rhythm_csv)
        available = set(frame["poem"].astype(int))
        missing = set(poems) - available
        if missing:
            raise ValueError(f"{rhythm_csv} is missing poems {sorted(missing)}")
        if "coefficient_of_variation" not in frame.columns:
            frame["coefficient_of_variation"] = np.where(
                frame["mean_interval_s"] > 0,
                frame["std_interval_s"] / frame["mean_interval_s"],
                np.nan,
            )
        timings = {}
        for poem in poems:
            rows = frame.loc[frame["poem"] == poem].sort_values("line_index")
            timings[poem] = [
                PoemLineTiming(
                    poem=int(row.poem),
                    line_index=int(row.line_index),
                    line_number=int(row.line_number),
                    start_s=float(row.start_s),
                    end_s=float(row.end_s),
                    mean_interval_s=float(row.mean_interval_s),
                    median_interval_s=float(row.median_interval_s),
                    std_interval_s=float(row.std_interval_s),
                    coefficient_of_variation=float(row.coefficient_of_variation),
                    stress_frequency_hz=float(row.stress_frequency_hz),
                    rhythm_condition=classify_rhythm(
                        float(row.coefficient_of_variation)
                    ),
                )
                for row in rows.itertuples(index=False)
            ]
        return timings
    return {
        poem: select_iambic_lines(
            poem,
            derive_poem_line_timings(
                poem,
                insufficient_as_ambiguous=insufficient_as_ambiguous,
            ),
        )
        for poem in poems
    }


def paper_condition_stats_by_poem(full: pd.DataFrame) -> pd.DataFrame:
    """Add the fixed paper-window configuration to poem-level statistics."""
    table = condition_stats_by_poem(full).drop(columns="frequency_hz")
    windows = full.groupby("poem")["window_duration_s"].first()
    table.insert(
        1,
        "window_duration_s",
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
    wilcoxon = stats.wilcoxon(
        paired["regular"],
        paired["irregular"],
        alternative="greater",
    )
    mannwhitney = stats.mannwhitneyu(
        grouped["english_competence"],
        grouped["non_english_competence"],
        alternative="two-sided",
    )
    total = pd.DataFrame(
        [
            {
                "poem": "Total",
                "window_duration_s": PAPER_WINDOW_S,
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
) -> mne.Epochs:
    start_s = recording.usable_start_s + line.start_s
    end_s = aligned_line_end_s(recording, line.end_s)
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
    window_source: str,
) -> tuple[list[dict], list[dict]]:
    rows, errors = [], []
    channel_names = usable_eeg_channels(recording.raw)
    for line in (item for item in lines if item.rhythm_condition != "ambiguous"):
        try:
            epochs = acf_line_epochs(recording, line, channel_names, window_s)
            values, counts = fourier_itpc(epochs, frequency_hz)
            debiased = debiased_squared_itpc(values, counts)
            end_s = aligned_line_end_s(recording, line.end_s)
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
                            "rhythm_cv": line.coefficient_of_variation,
                            "median_inter_stress_interval_s": line.median_interval_s,
                            "stress_frequency_hz": line.stress_frequency_hz,
                            "frequency_source": "poem_mean_stress_interval",
                            "frequency_hz": frequency_hz,
                            "window_source": window_source,
                            "window_duration_s": window_s,
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


def run(
    data_root: Path,
    output_dir: Path,
    *,
    rhythm_csv: Path = DEFAULT_RHYTHM_CSV,
    fail_on_errors: bool = True,
    insufficient_as_ambiguous: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered = [
        path
        for path in discover_ica_cleaned_files(data_root)
        if ids_from_path(path)[0] in EEG_POEM_IDS
    ]
    paths, duplicates = unique_recording_paths(discovered)
    poems = sorted({ids_from_path(path)[0] for path in discovered})
    timings = load_or_derive_timings(
        poems,
        rhythm_csv,
        insufficient_as_ambiguous=insufficient_as_ambiguous,
    )
    frequencies = {poem: poem_stress_frequency(lines) for poem, lines in timings.items()}
    config = paper_window_configuration(poems)
    config["itpc_frequency_hz"] = config["poem"].map(frequencies)
    config.to_csv(output_dir / "acf_itpc_configuration.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    configurations = config.set_index("poem").to_dict(orient="index")
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
            poem_config = configurations[recording.poem]
            found, failures = analyze_recording(
                recording,
                timings[recording.poem],
                frequencies[recording.poem],
                float(poem_config["window_duration_s"]),
                str(poem_config["window_source"]),
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
    paper_condition_stats_by_poem(full).to_csv(
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
    parser.add_argument("--rhythm-csv", type=Path, default=DEFAULT_RHYTHM_CSV)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--insufficient-as-ambiguous", action="store_true")
    args = parser.parse_args()
    run(
        args.data_root,
        args.output_dir,
        rhythm_csv=args.rhythm_csv,
        fail_on_errors=not args.allow_errors,
        insufficient_as_ambiguous=args.insufficient_as_ambiguous,
    )


if __name__ == "__main__":
    main()
