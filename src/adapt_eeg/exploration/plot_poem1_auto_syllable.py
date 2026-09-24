"""Plot poem-1 auto syllables with the largest midpoint/nucleus gaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adapt_eeg.auto_annotation.syllable_identifier import syllablize_audio
from adapt_eeg.constants import POEMS_CONFIG


def _midpoint_gap(syllable) -> float:
    midpoint_s = (syllable.start + syllable.end) / 2.0
    return abs(syllable.nucleus - midpoint_s)


def plot_auto_syllables(
    output_path: Path,
    *,
    top: int = 10,
    syllable_number: int | None = None,
    context_s: float = 0.1,
) -> list[tuple[int, float, float, float, float]]:
    """Plot selected syllables with start, nucleus, end, and midpoint markers."""
    if top < 1:
        raise ValueError("top must be positive")
    if syllable_number is not None and syllable_number < 1:
        raise ValueError("syllable_number must be one-based and positive")

    detected_audio = syllablize_audio(POEMS_CONFIG[0]["audio_url"])
    syllables = detected_audio.syllables
    if syllable_number is not None:
        if syllable_number > len(syllables):
            raise ValueError(
                f"Poem 1 has {len(syllables)} automatic syllables; "
                f"cannot select syllable {syllable_number}."
            )
        selected_indices = [syllable_number - 1]
    else:
        selected_indices = sorted(
            range(len(syllables)),
            key=lambda index: _midpoint_gap(syllables[index]),
            reverse=True,
        )[:top]

    figure, axes = plt.subplots(
        len(selected_indices),
        1,
        figsize=(11, 3.2 * len(selected_indices)),
        squeeze=False,
    )
    results = []
    for plot_index, (axis, syllable_index) in enumerate(
        zip(axes[:, 0], selected_indices, strict=True)
    ):
        syllable = syllables[syllable_index]
        number = syllable_index + 1
        midpoint_s = (syllable.start + syllable.end) / 2.0
        gap_ms = abs(syllable.nucleus - midpoint_s) * 1000.0
        window_start_s = max(0.0, syllable.start - context_s)
        window_end_s = min(
            len(detected_audio.samples) / detected_audio.sample_rate,
            syllable.end + context_s,
        )
        sample_start = round(window_start_s * detected_audio.sample_rate)
        sample_end = round(window_end_s * detected_audio.sample_rate)
        samples = np.asarray(
            detected_audio.samples[sample_start:sample_end], dtype=float
        )
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        maximum = float(np.max(np.abs(samples)))
        if maximum:
            samples /= maximum
        times = np.arange(sample_start, sample_end) / detected_audio.sample_rate

        axis.plot(times, samples, color="#737373", linewidth=0.7, label="waveform")
        for label, time_s, linestyle in (
            ("start", syllable.start, "-"),
            ("nucleus", syllable.nucleus, "--"),
            ("end", syllable.end, ":"),
        ):
            axis.axvline(
                time_s,
                color="#ef4444",
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )
        axis.axvline(
            midpoint_s,
            color="#2563eb",
            linewidth=2.2,
            label="interval midpoint",
        )
        axis.set(
            xlim=(window_start_s, window_end_s),
            ylim=(-1.05, 1.05),
            ylabel="Normalized amplitude",
            title=f"#{plot_index + 1}: auto syllable {number} — gap {gap_ms:.1f} ms",
        )
        axis.grid(axis="x", alpha=0.15)
        if plot_index == 0:
            axis.legend(loc="lower left", ncols=3)
        results.append(
            (number, syllable.start, syllable.nucleus, syllable.end, midpoint_s)
        )

    axes[-1, 0].set_xlabel("Time in full poem recording (s)")
    figure.suptitle("Poem 1: largest automatic-syllable midpoint/nucleus gaps")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--syllable",
        type=int,
        help="one-based syllable number; overrides --top",
    )
    parser.add_argument("--context-s", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/exploration/"
            "poem1_auto_syllable_top10_midpoint_nucleus_gaps.png"
        ),
    )
    args = parser.parse_args()
    results = plot_auto_syllables(
        args.output,
        top=args.top,
        syllable_number=args.syllable,
        context_s=args.context_s,
    )
    for rank, (number, start_s, nucleus_s, end_s, midpoint_s) in enumerate(
        results, start=1
    ):
        print(
            f"{rank}. syllable {number}: start={start_s:.6f}s, "
            f"nucleus={nucleus_s:.6f}s, end={end_s:.6f}s, "
            f"midpoint={midpoint_s:.6f}s, "
            f"gap={abs(nucleus_s - midpoint_s) * 1000:.3f}ms"
        )
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
