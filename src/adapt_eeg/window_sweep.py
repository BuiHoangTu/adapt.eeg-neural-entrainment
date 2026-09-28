"""Compare fixed-second and cycle-aligned ITPC window configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adapt_eeg.exploration.stress_rhythm_evidence import EEG_POEM_IDS
from adapt_eeg.method_comparison import summarize
from adapt_eeg.poem_eeg import (
    discover_ica_cleaned_files,
    ids_from_path,
    load_poem_recording,
    unique_recording_paths,
)
from adapt_eeg.poem_itpc import analyze_recording as analyze_cycle_recording
from adapt_eeg.poem_itpc_acf import (
    DEFAULT_RHYTHM_CSV,
    analyze_recording as analyze_fixed_recording,
    load_or_derive_timings,
)
from adapt_eeg.poem_rhythm import poem_stress_frequency

FIXED_WINDOWS_S = (0.1, 0.2, 0.3, 0.4)
CYCLE_CONFIGS = ((1.0, 1.0), (2.0, 1.0), (1.0, 0.5), (2.0, 0.5))
GROUP_COLUMNS = [
    "poem",
    "line",
    "participant",
    "language_group",
    "rhythm_condition",
    "frequency_hz",
]


def _compact_rows(rows: list[dict]) -> pd.DataFrame:
    """Average channels so every participant-line has one value."""
    if not rows:
        return pd.DataFrame(columns=[*GROUP_COLUMNS, "itpc_squared_debiased"])
    return (
        pd.DataFrame(rows)
        .groupby(GROUP_COLUMNS, as_index=False)["itpc_squared_debiased"]
        .mean()
    )


def _compact_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return (
        frame.groupby(GROUP_COLUMNS, as_index=False)["itpc_squared_debiased"]
        .mean()
    )


def _participant_totals(frame: pd.DataFrame) -> pd.DataFrame:
    participant = (
        frame.groupby(
            ["participant", "language_group", "rhythm_condition"],
            as_index=False,
        )["itpc_squared_debiased"]
        .mean()
    )
    paired = participant.pivot_table(
        index=["participant", "language_group"],
        columns="rhythm_condition",
        values="itpc_squared_debiased",
    ).dropna(subset=["regular", "irregular"])
    paired["difference"] = paired["regular"] - paired["irregular"]
    return paired.reset_index()


def _plot_method(
    method: str,
    labels: list[str],
    frames: list[pd.DataFrame],
    summaries: pd.DataFrame,
    output_path: Path,
) -> None:
    participant = [_participant_totals(frame) for frame in frames]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

    for condition, offset, color in (
        ("regular", -0.07, "#4c78a8"),
        ("irregular", 0.07, "#f58518"),
    ):
        means = [item[condition].mean() for item in participant]
        sems = [item[condition].sem() for item in participant]
        axes[0].errorbar(
            x + offset,
            means,
            yerr=sems,
            marker="o",
            capsize=4,
            color=color,
            label=condition,
        )
    axes[0].set_ylabel("Debiased squared ITPC")
    axes[0].set_title(f"{method}: participant condition means ± SEM")
    axes[0].legend()

    differences = [item["difference"].mean() for item in participant]
    difference_sems = [item["difference"].sem() for item in participant]
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].errorbar(x, differences, yerr=difference_sems, marker="o", capsize=4)
    totals = summaries.loc[summaries["poem"].astype(str) == "Total"]
    for position, (_, row) in enumerate(totals.iterrows()):
        axes[1].annotate(
            f"p={row.wilcoxon_p:.3g}",
            (position, differences[position]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axes[1].set_ylabel("Regular − irregular")
    axes[1].set_title("Pooled-line participant effect ± SEM; one-sided Wilcoxon p")

    poem_rows = summaries.loc[summaries["poem"].astype(str) != "Total"].copy()
    poem_rows["poem"] = poem_rows["poem"].astype(int)
    heatmap = poem_rows.pivot(
        index="poem",
        columns="setting",
        values="regular_minus_irregular",
    ).reindex(columns=labels)
    maximum = float(np.nanmax(np.abs(heatmap.to_numpy())))
    image = axes[2].imshow(
        heatmap,
        aspect="auto",
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
    )
    axes[2].set_yticks(
        range(len(heatmap.index)),
        [f"Poem {poem}" for poem in heatmap.index],
    )
    axes[2].set_ylabel("Poem")
    axes[2].set_title("Regular − irregular by poem")
    fig.colorbar(image, ax=axes[2], label="Debiased squared ITPC difference")

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Window configuration")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run(
    data_root: Path,
    output_dir: Path,
    rhythm_csv: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    poems = list(EEG_POEM_IDS)
    timings = load_or_derive_timings(poems, rhythm_csv)
    frequencies = {
        poem: poem_stress_frequency(lines) for poem, lines in timings.items()
    }
    discovered = [
        path
        for path in discover_ica_cleaned_files(data_root)
        if ids_from_path(path)[0] in set(poems)
    ]
    paths, duplicates = unique_recording_paths(discovered)
    errors = [
        {
            "method": "discovery",
            "setting": "all",
            "path": " | ".join(map(str, candidates)),
            "error": f"Duplicate recording for poem {poem}, participant {participant}",
        }
        for (poem, participant), candidates in sorted(duplicates.items())
    ]

    configurations = [
        ("fixed", f"{window:.1f}s", window, None)
        for window in FIXED_WINDOWS_S
    ] + [
        ("cycle", f"{window:g}-{slide:g}", window, slide)
        for window, slide in CYCLE_CONFIGS
    ]
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    compact_parts: dict[tuple[str, str], list[pd.DataFrame]] = {}
    missing = []
    for method, label, window, slide in configurations:
        sweep_cache = output_dir / f"{method}_{label.replace('.', 'p')}_line_results.csv"
        cached: Path | None = sweep_cache if sweep_cache.exists() else None
        if method == "fixed" and window == 0.3:
            cached = Path("results/poem_itpc_acf/itpc_by_line_channel.csv")
        elif method == "cycle" and window == 2.0 and slide == 1.0:
            cached = Path("results/poem_itpc/itpc_by_line_channel.csv")
        if cached is not None and cached.exists():
            compact = _compact_file(cached)
            print(f"Reused {cached} for {method} {label}")
            frames[(method, label)] = compact
        else:
            missing.append((method, label, window, slide))
            compact_parts[(method, label)] = []

    for index, path in enumerate(paths, 1):
        print(f"[recording {index}/{len(paths)}] {path}")
        try:
            recording = load_poem_recording(path, preload=True)
        except Exception as exc:
            for method, label, _, _ in missing:
                errors.append(
                    {
                        "method": method,
                        "setting": label,
                        "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        for method, label, window, slide in missing:
            try:
                if method == "fixed":
                    rows, failures = analyze_fixed_recording(
                        recording,
                        timings[recording.poem],
                        frequencies[recording.poem],
                        window,
                        f"fixed_{window:.1f}s",
                    )
                else:
                    rows, failures = analyze_cycle_recording(
                        recording,
                        timings[recording.poem],
                        frequencies[recording.poem],
                        window,
                        float(slide),
                    )
                compact_parts[(method, label)].append(_compact_rows(rows))
                errors.extend(
                    {"method": method, "setting": label, **failure}
                    for failure in failures
                )
            except Exception as exc:
                errors.append(
                    {
                        "method": method,
                        "setting": label,
                        "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    for method, label, _, _ in missing:
        parts = compact_parts[(method, label)]
        if not parts:
            raise RuntimeError(f"No results produced for {method} {label}")
        frames[(method, label)] = pd.concat(parts, ignore_index=True)

    summaries = []
    for method, label, _, _ in configurations:
        compact = frames[(method, label)]
        compact.to_csv(
            output_dir / f"{method}_{label.replace('.', 'p')}_line_results.csv",
            index=False,
        )
        setting_summary = summarize(compact)
        setting_summary.insert(0, "setting", label)
        setting_summary.insert(0, "method", method)
        summaries.append(setting_summary)

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output_dir / "window_sweep_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "window_sweep_errors.csv", index=False)
    for method, labels in (
        ("fixed", [f"{value:.1f}s" for value in FIXED_WINDOWS_S]),
        ("cycle", [f"{window:g}-{slide:g}" for window, slide in CYCLE_CONFIGS]),
    ):
        method_summary = summary.loc[summary["method"] == method]
        _plot_method(
            "Fixed-second windows" if method == "fixed" else "Cycle-aligned windows",
            labels,
            [frames[(method, label)] for label in labels],
            method_summary,
            output_dir / f"{method}_window_sweep.png",
        )
    print(f"Wrote sweep results and figures to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw-eeg"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/window_sweep"),
    )
    parser.add_argument("--rhythm-csv", type=Path, default=DEFAULT_RHYTHM_CSV)
    args = parser.parse_args()
    run(args.data_root, args.output_dir, args.rhythm_csv)


if __name__ == "__main__":
    main()
