from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats

from adapt_eeg.constants import ENGLISH_COMPETENCE_PARTICIPANTS
from adapt_eeg.poem_eeg import (
    InvalidPoemRecordingError,
    PoemRecording,
    aligned_line_end_s,
    discover_ica_cleaned_files,
    ids_from_path,
    load_poem_recording,
    unique_recording_paths,
)
from adapt_eeg.poem_rhythm import (
    PoemLineTiming,
    derive_poem_line_timings,
    poem_has_iambic_lines,
    poem_stress_frequency,
    select_iambic_lines,
)

CYCLES_PER_WINDOW = 2.0
SHIFT_CYCLES = 1.0
MIN_WINDOWS = 2
CSV_FLOAT_FORMAT = "%.6f"


def language_group(participant: int) -> str:
    return (
        "english_competence"
        if participant in ENGLISH_COMPETENCE_PARTICIPANTS
        else "non_english_competence"
    )


def epoch_timing(
    frequency_hz: float,
    cycles_per_window: float = CYCLES_PER_WINDOW,
    shift_cycles: float = SHIFT_CYCLES,
) -> tuple[float, float]:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if cycles_per_window <= 0 or shift_cycles <= 0:
        raise ValueError("cycle window and slide must be positive")
    if shift_cycles > cycles_per_window:
        raise ValueError("shift_cycles cannot exceed cycles_per_window")
    return cycles_per_window / frequency_hz, shift_cycles / frequency_hz


def usable_eeg_channels(raw: mne.io.BaseRaw) -> list[str]:
    """Select EEG channels while removing data-driven flat reference channels."""
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks)
    scales = np.std(data, axis=1)
    positive_scales = scales[scales > np.finfo(float).eps]
    if len(positive_scales) == 0:
        raise ValueError("Recording has no non-flat EEG channels")
    minimum_scale = float(np.median(positive_scales)) * 1e-6
    return [
        raw.ch_names[pick]
        for pick, scale in zip(picks, scales, strict=True)
        if scale >= minimum_scale
    ]


def line_epochs(
    recording: PoemRecording,
    line: PoemLineTiming,
    channel_names: list[str],
    frequency_hz: float,
    cycles_per_window: float = CYCLES_PER_WINDOW,
    shift_cycles: float = SHIFT_CYCLES,
) -> mne.Epochs:
    absolute_start_s = recording.usable_start_s + line.start_s
    absolute_end_s = aligned_line_end_s(recording, line.end_s)
    if absolute_start_s >= absolute_end_s:
        raise ValueError(
            f"poem {line.poem} line {line.line_number}: non-positive EEG interval"
        )

    cropped = recording.raw.copy().pick(channel_names).crop(
        tmin=absolute_start_s,
        tmax=absolute_end_s,
        include_tmax=False,
    )
    window_s, shift_s = epoch_timing(
        frequency_hz,
        cycles_per_window,
        shift_cycles,
    )
    if cropped.duration < window_s:
        raise ValueError(
            f"line duration {cropped.duration:.6f}s is shorter than the "
            f"{window_s:.6f}s ITPC window"
        )
    epochs = mne.make_fixed_length_epochs(
        cropped,
        duration=window_s,
        overlap=window_s - shift_s,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )
    if len(epochs) < MIN_WINDOWS:
        raise ValueError(
            f"ITPC requires at least {MIN_WINDOWS} windows; found {len(epochs)}"
        )
    return epochs


def fourier_itpc(
    epochs: mne.Epochs,
    frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = epochs.get_data(copy=True)
    data -= data.mean(axis=-1, keepdims=True)
    window = np.hanning(data.shape[-1])
    kernel = window * np.exp(-2j * np.pi * frequency_hz * epochs.times)
    coefficients = np.sum(data * kernel[None, None, :], axis=-1)
    magnitudes = np.abs(coefficients)
    valid = magnitudes > np.finfo(float).eps
    unit_phase = np.zeros_like(coefficients, dtype=np.complex128)
    unit_phase[valid] = coefficients[valid] / magnitudes[valid]
    valid_counts = valid.sum(axis=0)
    values = np.full(coefficients.shape[1], np.nan)
    enough_phase_samples = valid_counts >= MIN_WINDOWS
    values[enough_phase_samples] = np.abs(
        unit_phase[:, enough_phase_samples].sum(axis=0)
        / valid_counts[enough_phase_samples]
    )
    return values, valid_counts


def debiased_squared_itpc(
    itpc: np.ndarray,
    n_windows: np.ndarray,
) -> np.ndarray:
    """Remove the finite-sample bias from squared ITPC."""
    result = np.full_like(itpc, np.nan, dtype=float)
    valid = n_windows >= MIN_WINDOWS
    result[valid] = (
        n_windows[valid] * np.square(itpc[valid]) - 1.0
    ) / (n_windows[valid] - 1.0)
    return result


def analyze_recording(
    recording: PoemRecording,
    lines: list[PoemLineTiming],
    frequency_hz: float,
    cycles_per_window: float = CYCLES_PER_WINDOW,
    shift_cycles: float = SHIFT_CYCLES,
) -> tuple[list[dict], list[dict]]:
    rows = []
    errors = []
    channel_names = usable_eeg_channels(recording.raw)
    eligible_lines = [line for line in lines if line.rhythm_condition != "ambiguous"]
    for line in eligible_lines:
        try:
            epochs = line_epochs(
                recording,
                line,
                channel_names,
                frequency_hz,
                cycles_per_window,
                shift_cycles,
            )
            values, valid_counts = fourier_itpc(epochs, frequency_hz)
            debiased_values = debiased_squared_itpc(values, valid_counts)
            channel_names = epochs.ch_names
            absolute_start_s = recording.usable_start_s + line.start_s
            absolute_end_s = aligned_line_end_s(recording, line.end_s)
            window_s, shift_s = epoch_timing(
                frequency_hz,
                cycles_per_window,
                shift_cycles,
            )
            for channel, value, debiased_value, valid_count in zip(
                channel_names,
                values,
                debiased_values,
                valid_counts,
                strict=True,
            ):
                if not np.isfinite(value):
                    continue
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
                        "cycles_per_window": cycles_per_window,
                        "shift_cycles": shift_cycles,
                        "window_duration_s": window_s,
                        "window_shift_s": shift_s,
                        "n_windows": len(epochs),
                        "n_valid_phase_windows": int(valid_count),
                        "line_start_eeg_s": absolute_start_s,
                        "line_end_eeg_s": absolute_end_s,
                        "line_duration_s": absolute_end_s - absolute_start_s,
                        "channel": channel,
                        "itpc": float(value),
                        "itpc_squared_debiased": float(debiased_value),
                        "set_path": str(recording.path),
                    }
                )
        except Exception as exc:
            errors.append(
                {
                    "participant": recording.participant,
                    "poem": recording.poem,
                    "line": line.line_number,
                    "set_path": str(recording.path),
                    "stage": "line_itpc",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return rows, errors


def participant_condition_table(full: pd.DataFrame) -> pd.DataFrame:
    return (
        full.groupby(
            ["participant", "language_group", "rhythm_condition", "channel"],
            as_index=False,
        )
        .agg(
            itpc_mean=("itpc", "mean"),
            itpc_sd_across_lines=("itpc", "std"),
            itpc_squared_debiased_mean=("itpc_squared_debiased", "mean"),
            itpc_squared_debiased_sd_across_lines=(
                "itpc_squared_debiased",
                "std",
            ),
            n_lines=("line", "count"),
        )
    )


def condition_stats_table(participant_condition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = {
        "raw_itpc": "itpc_mean",
        "debiased_squared_itpc": "itpc_squared_debiased_mean",
    }
    for metric, value_column in metrics.items():
        for channel, frame in participant_condition.groupby("channel", sort=True):
            paired = frame.pivot(
                index="participant",
                columns="rhythm_condition",
                values=value_column,
            )
            if not {"regular", "irregular"}.issubset(paired.columns):
                continue
            paired = paired.dropna(subset=["regular", "irregular"])
            if len(paired) < 2:
                continue
            differences = paired["regular"] - paired["irregular"]
            test = stats.ttest_rel(paired["regular"], paired["irregular"])
            try:
                wilcoxon = stats.wilcoxon(
                    paired["regular"],
                    paired["irregular"],
                    zero_method="wilcox",
                    alternative="greater",
                )
                wilcoxon_statistic = float(wilcoxon.statistic)
                wilcoxon_pvalue = float(wilcoxon.pvalue)
            except ValueError:
                wilcoxon_statistic = np.nan
                wilcoxon_pvalue = np.nan
            difference_sd = float(differences.std(ddof=1))
            rows.append(
                {
                    "metric": metric,
                    "channel": channel,
                    "n_participants": len(paired),
                    "regular_mean": float(paired["regular"].mean()),
                    "irregular_mean": float(paired["irregular"].mean()),
                    "mean_difference_regular_minus_irregular": float(
                        differences.mean()
                    ),
                    "sd_difference": difference_sd,
                    "cohens_dz": (
                        float(differences.mean() / difference_sd)
                        if difference_sd > 0
                        else np.nan
                    ),
                    "paired_t_statistic": float(test.statistic),
                    "paired_t_pvalue": float(test.pvalue),
                    "wilcoxon_statistic": wilcoxon_statistic,
                    "wilcoxon_pvalue": wilcoxon_pvalue,
                }
            )
    return pd.DataFrame(rows)


def language_group_stats_table(participant_condition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = {
        "raw_itpc": "itpc_mean",
        "debiased_squared_itpc": "itpc_squared_debiased_mean",
    }
    for metric, value_column in metrics.items():
        for channel, frame in participant_condition.groupby("channel", sort=True):
            paired = frame.pivot(
                index=["participant", "language_group"],
                columns="rhythm_condition",
                values=value_column,
            )
            if not {"regular", "irregular"}.issubset(paired.columns):
                continue
            paired = paired.dropna(subset=["regular", "irregular"]).reset_index()
            paired["regular_minus_irregular"] = (
                paired["regular"] - paired["irregular"]
            )
            groups = {
                name: group["regular_minus_irregular"].to_numpy()
                for name, group in paired.groupby("language_group")
            }
            required = {"english_competence", "non_english_competence"}
            if not required.issubset(groups):
                continue
            test = stats.mannwhitneyu(
                groups["english_competence"],
                groups["non_english_competence"],
                alternative="two-sided",
            )
            rows.append(
                {
                    "metric": metric,
                    "channel": channel,
                    "english_competence_n": len(groups["english_competence"]),
                    "non_english_competence_n": len(
                        groups["non_english_competence"]
                    ),
                    "english_competence_mean_difference": float(
                        groups["english_competence"].mean()
                    ),
                    "non_english_competence_mean_difference": float(
                        groups["non_english_competence"].mean()
                    ),
                    "mannwhitney_statistic": float(test.statistic),
                    "mannwhitney_pvalue": float(test.pvalue),
                }
            )
    return pd.DataFrame(rows)


def condition_stats_by_poem(full: pd.DataFrame) -> pd.DataFrame:
    """Summarize debiased squared ITPC contrasts separately for each poem."""
    participant = (
        full.groupby(
            ["poem", "participant", "language_group", "rhythm_condition"],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
    )
    rows = []
    for poem, frame in participant.groupby("poem", sort=True):
        paired = frame.pivot(
            index=["participant", "language_group"],
            columns="rhythm_condition",
            values="itpc_squared_debiased",
        ).dropna(subset=["regular", "irregular"])
        differences = paired["regular"] - paired["irregular"]
        wilcoxon = stats.wilcoxon(
            paired["regular"],
            paired["irregular"],
            alternative="greater",
        )
        grouped = {
            group: values.to_numpy()
            for group, values in differences.groupby(level="language_group")
        }
        mannwhitney = stats.mannwhitneyu(
            grouped["english_competence"],
            grouped["non_english_competence"],
            alternative="two-sided",
        )
        poem_rows = full.loc[full["poem"] == poem]
        rows.append(
            {
                "poem": int(poem),
                "frequency_hz": float(poem_rows["frequency_hz"].iloc[0]),
                "n_paired": len(paired),
                "regular_mean": float(paired["regular"].mean()),
                "irregular_mean": float(paired["irregular"].mean()),
                "regular_minus_irregular": float(differences.mean()),
                "wilcoxon_statistic": float(wilcoxon.statistic),
                "wilcoxon_pvalue": float(wilcoxon.pvalue),
                "english_n": len(grouped["english_competence"]),
                "non_english_n": len(grouped["non_english_competence"]),
                "english_mean_difference": float(
                    grouped["english_competence"].mean()
                ),
                "non_english_mean_difference": float(
                    grouped["non_english_competence"].mean()
                ),
                "mannwhitney_statistic": float(mannwhitney.statistic),
                "mannwhitney_pvalue": float(mannwhitney.pvalue),
            }
        )
    return pd.DataFrame(rows)


def export_xlsx(full: pd.DataFrame, output_path: Path) -> None:
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        regular = workbook.add_format({"bold": True, "bg_color": "#92D050"})
        irregular = workbook.add_format({"bold": True, "bg_color": "#F4B183"})
        corner = workbook.add_format({"bold": True})
        english_participant = workbook.add_format(
            {"bold": True, "bg_color": "#FF6666"}
        )
        value_format = workbook.add_format({"num_format": "0.0000"})

        for channel, frame in full.groupby("channel", sort=True):
            frame = frame.copy()
            frame["poem_line"] = frame.apply(
                lambda row: f"P{int(row['poem'])} L{int(row['line'])}", axis=1
            )
            ordered = (
                frame[["poem", "line", "poem_line", "rhythm_condition"]]
                .drop_duplicates()
                .sort_values(["poem", "line"])
            )
            columns = ordered["poem_line"].tolist()
            conditions = ordered.set_index("poem_line")["rhythm_condition"].to_dict()
            matrix = frame.pivot_table(
                index="participant", columns="poem_line", values="itpc", aggfunc="mean"
            ).reindex(columns=columns)

            sheet_name = str(channel)[:31]
            worksheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = worksheet
            worksheet.write(0, 0, "Participant / line", corner)
            for column_index, label in enumerate(columns, start=1):
                header_format = regular if conditions[label] == "regular" else irregular
                worksheet.write(0, column_index, label, header_format)
            for row_index, participant in enumerate(matrix.index, start=1):
                participant_format = (
                    english_participant
                    if int(participant) in ENGLISH_COMPETENCE_PARTICIPANTS
                    else corner
                )
                worksheet.write(row_index, 0, int(participant), participant_format)
                for column_index, label in enumerate(columns, start=1):
                    value = matrix.loc[participant, label]
                    if pd.isna(value):
                        worksheet.write_blank(row_index, column_index, None)
                    else:
                        worksheet.write_number(
                            row_index, column_index, float(value), value_format
                        )
            worksheet.freeze_panes(1, 1)
            worksheet.set_column(0, 0, 18)
            worksheet.set_column(1, len(columns), 10)


def run(
    data_root: Path,
    output_dir: Path,
    *,
    fail_on_errors: bool = True,
    insufficient_as_ambiguous: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    discovered_paths = [
        path
        for path in discover_ica_cleaned_files(data_root)
        if poem_has_iambic_lines(ids_from_path(path)[0])
    ]
    paths, duplicate_paths = unique_recording_paths(discovered_paths)
    poems = sorted({ids_from_path(path)[0] for path in discovered_paths})
    all_timings = {
        poem: derive_poem_line_timings(
            poem,
            insufficient_as_ambiguous=insufficient_as_ambiguous,
        )
        for poem in poems
    }
    timings = {
        poem: select_iambic_lines(poem, lines)
        for poem, lines in all_timings.items()
    }
    frequencies = {
        poem: poem_stress_frequency(lines) for poem, lines in timings.items()
    }

    rhythm_rows = [asdict(line) for lines in timings.values() for line in lines]
    pd.DataFrame(rhythm_rows).to_csv(
        output_dir / "poem_line_rhythm.csv", index=False, float_format=CSV_FLOAT_FORMAT
    )

    all_rows = []
    errors = [
        {
            "participant": participant,
            "poem": poem,
            "line": None,
            "set_path": " | ".join(str(path) for path in candidates),
            "stage": "discovery",
            "error_type": "DuplicatePoemRecordingError",
            "error": (
                f"Found {len(candidates)} ICA-cleaned recordings for poem {poem}, "
                f"participant {participant}; none were selected"
            ),
        }
        for (poem, participant), candidates in sorted(duplicate_paths.items())
    ]
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}")
        try:
            recording = load_poem_recording(path, preload=True)
            rows, line_errors = analyze_recording(
                recording,
                timings[recording.poem],
                frequencies[recording.poem],
            )
            all_rows.extend(rows)
            errors.extend(line_errors)
        except Exception as exc:
            try:
                poem, participant = ids_from_path(path)
            except Exception:
                poem, participant = None, None
            errors.append(
                {
                    "participant": participant,
                    "poem": poem,
                    "line": None,
                    "set_path": str(path),
                    "stage": "recording",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    full = pd.DataFrame(all_rows)
    error_table = pd.DataFrame(errors)
    error_table.to_csv(output_dir / "load_errors.csv", index=False)
    if full.empty:
        raise RuntimeError("No ITPC results were produced")

    full.to_csv(
        output_dir / "itpc_by_line_channel.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    participant_condition = participant_condition_table(full)
    participant_condition.to_csv(
        output_dir / "itpc_by_participant_condition_channel.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    condition_stats_table(participant_condition).to_csv(
        output_dir / "condition_stats_by_channel.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    language_group_stats_table(participant_condition).to_csv(
        output_dir / "language_group_stats_by_channel.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    condition_stats_by_poem(full).to_csv(
        output_dir / "condition_stats_by_poem.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    export_xlsx(full, output_dir / "poem_itpc_by_channel.xlsx")

    print(f"Wrote {len(full)} line-channel ITPC rows from {len(paths)} recordings")
    print(f"Recorded {len(errors)} errors in {output_dir / 'load_errors.csv'}")
    if errors and fail_on_errors:
        raise InvalidPoemRecordingError(
            f"ITPC outputs were written, but {len(errors)} errors occurred; "
            f"see {output_dir / 'load_errors.csv'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute poem-rate ITPC from trigger-aligned ICA-cleaned EEG."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw-eeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/poem_itpc"))
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Return success after writing valid outputs even when files are rejected.",
    )
    parser.add_argument(
        "--insufficient-as-ambiguous",
        action="store_true",
        help="Warn and mark selected lines with fewer than five scoreable syllables ambiguous.",
    )
    args = parser.parse_args()
    run(
        args.data_root,
        args.output_dir,
        fail_on_errors=not args.allow_errors,
        insufficient_as_ambiguous=args.insufficient_as_ambiguous,
    )


if __name__ == "__main__":
    main()
