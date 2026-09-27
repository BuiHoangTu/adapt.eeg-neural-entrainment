"""Run and summarize the four staged poem-ITPC analysis variants."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
from scipy import stats

from adapt_eeg.baseline import (
    FREQUENCY_HZ as BASELINE_FREQUENCY_HZ,
    IRREGULAR_LINES,
    REGULAR_LINES,
    SELECTED_LINES,
    WINDOW_DURATION_S,
)
from adapt_eeg.exploration.stress_rhythm_evidence import EEG_POEM_IDS
from adapt_eeg.poem_eeg import (
    discover_ica_cleaned_files,
    ids_from_path,
    load_poem_recording,
    unique_recording_paths,
)
from adapt_eeg.poem_itpc_acf import (
    analyze_recording,
    load_or_derive_timings,
)
from adapt_eeg.poem_rhythm import PoemLineTiming

CSV_FLOAT_FORMAT = "%.9g"


def baseline_timings(lines: list[PoemLineTiming]) -> list[PoemLineTiming]:
    """Apply the paper's seven regular/seven irregular Poem 1 labels."""
    selected = []
    for line in lines:
        if line.line_number not in SELECTED_LINES:
            continue
        condition = "regular" if line.line_number in REGULAR_LINES else "irregular"
        selected.append(replace(line, rhythm_condition=condition))
    found = {line.line_number for line in selected}
    if found != set(SELECTED_LINES):
        raise ValueError(f"Missing legacy Poem 1 lines: {sorted(set(SELECTED_LINES) - found)}")
    return selected


def compute_fixed_variant(
    data_root: Path,
    output_dir: Path,
    timings: dict[int, list[PoemLineTiming]],
    frequencies: dict[int, float],
    frequency_source: str,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    poem_ids = set(timings)
    discovered = [
        path
        for path in discover_ica_cleaned_files(data_root)
        if ids_from_path(path)[0] in poem_ids
    ]
    paths, duplicates = unique_recording_paths(discovered)
    errors = [
        {
            "participant": participant,
            "poem": poem,
            "path": " | ".join(map(str, candidates)),
            "error": f"DuplicatePoemRecordingError: found {len(candidates)} recordings",
        }
        for (poem, participant), candidates in sorted(duplicates.items())
    ]
    rows = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}")
        try:
            recording = load_poem_recording(path, preload=True)
            found, failures = analyze_recording(
                recording,
                timings[recording.poem],
                frequencies[recording.poem],
                WINDOW_DURATION_S,
                "paper_eeg_acf_fixed_0.3s",
            )
            for row in found:
                row["frequency_source"] = frequency_source
            rows.extend(found)
            errors.extend(failures)
        except Exception as exc:
            poem, participant = ids_from_path(path)
            errors.append(
                {
                    "participant": participant,
                    "poem": poem,
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    full = pd.DataFrame(rows)
    if full.empty:
        raise RuntimeError(f"No ITPC rows produced for {output_dir.name}")
    full.to_csv(
        output_dir / "itpc_by_line_channel.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    pd.DataFrame(errors).to_csv(output_dir / "load_errors.csv", index=False)
    return full


def _test_row(label: int | str, participant_poem: pd.DataFrame) -> dict:
    paired = participant_poem.pivot_table(
        index=["participant", "language_group"],
        columns="rhythm_condition",
        values="itpc_squared_debiased",
        aggfunc="mean",
    ).dropna(subset=["regular", "irregular"])
    differences = paired["regular"] - paired["irregular"]
    wilcoxon = stats.wilcoxon(
        paired["regular"],
        paired["irregular"],
        alternative="greater",
    )
    groups = {
        name: values.to_numpy()
        for name, values in differences.groupby(level="language_group")
    }
    mannwhitney = stats.mannwhitneyu(
        groups["english_competence"],
        groups["non_english_competence"],
        alternative="two-sided",
    )
    english = groups["english_competence"]
    non_english = groups["non_english_competence"]
    return {
        "poem": label,
        "n_paired": len(paired),
        "regular_mean": paired["regular"].mean(),
        "irregular_mean": paired["irregular"].mean(),
        "regular_minus_irregular": differences.mean(),
        "wilcoxon_W": wilcoxon.statistic,
        "wilcoxon_p": wilcoxon.pvalue,
        "know_language_n": len(english),
        "know_language_effect": english.mean(),
        "do_not_know_n": len(non_english),
        "do_not_know_effect": non_english.mean(),
        "language_effect_difference": english.mean() - non_english.mean(),
        "mann_whitney_U": mannwhitney.statistic,
        "mann_whitney_p": mannwhitney.pvalue,
    }


def summarize(full: pd.DataFrame) -> pd.DataFrame:
    """Summarize poems separately and pool all lines for the Total row."""
    participant_poem = (
        full.groupby(
            ["poem", "participant", "language_group", "rhythm_condition"],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
    )
    rows = [
        _test_row(int(poem), frame)
        for poem, frame in participant_poem.groupby("poem", sort=True)
    ]
    participant_lines = (
        full.groupby(
            [
                "poem",
                "line",
                "participant",
                "language_group",
                "rhythm_condition",
            ],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
    )
    pooled_line_total = (
        participant_lines.groupby(
            ["participant", "language_group", "rhythm_condition"],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
    )
    rows.append(_test_row("Total", pooled_line_total))
    return pd.DataFrame(rows)


def run(data_root: Path, results_root: Path, rhythm_csv: Path) -> None:
    timings = load_or_derive_timings(list(EEG_POEM_IDS), rhythm_csv)
    baseline = compute_fixed_variant(
        data_root,
        results_root / "1_baseline_poem1_fixed_2hz_0.3s",
        {1: baseline_timings(timings[1])},
        {1: BASELINE_FREQUENCY_HZ},
        "fixed_2hz",
    )
    fixed_2hz = compute_fixed_variant(
        data_root,
        results_root / "2_four_poems_fixed_2hz_0.3s",
        timings,
        {poem: 2.0 for poem in timings},
        "fixed_2hz",
    )
    poem_frequency_fixed = pd.read_csv(
        "results/poem_itpc_acf/itpc_by_line_channel.csv"
    )
    cycle_aligned = pd.read_csv("results/poem_itpc/itpc_by_line_channel.csv")

    variants = {
        "1_baseline": baseline,
        "2_four_poems_2hz": fixed_2hz,
        "3_poem_frequency_fixed_0.3s": poem_frequency_fixed,
        "4_poem_frequency_cycle_aligned": cycle_aligned,
    }
    summaries = {name: summarize(frame) for name, frame in variants.items()}
    results_root.mkdir(parents=True, exist_ok=True)
    for name, table in summaries.items():
        table.to_csv(
            results_root / f"{name}_summary.csv",
            index=False,
            float_format=CSV_FLOAT_FORMAT,
        )
    with pd.ExcelWriter(results_root / "four_method_summary.xlsx") as writer:
        for name, table in summaries.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"Wrote four comparison tables to {results_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw-eeg"))
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/method_comparison")
    )
    parser.add_argument(
        "--rhythm-csv",
        type=Path,
        default=Path("results/poem_itpc/poem_line_rhythm.csv"),
    )
    args = parser.parse_args()
    run(args.data_root, args.results_root, args.rhythm_csv)


if __name__ == "__main__":
    main()
