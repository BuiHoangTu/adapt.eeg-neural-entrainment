"""Find a poem-1 syllable whose TextGrid/automatic edges differ by ~100 ms.

Run from the repository root with::

    PYTHONPATH=src conda run -n nbm python \
        exploration/find_poem1_100ms_difference.py
"""

from __future__ import annotations

import argparse
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from adapt_eeg.auto_annotation.syllable_from_textgrid import load_textgrid_syllable
from adapt_eeg.auto_annotation.syllable_identifier import syllablize_audio
from adapt_eeg.constants import POEMS_CONFIG


@dataclass(frozen=True)
class SyllableDifference:
    textgrid_index: int
    automatic_index: int
    transcription: str
    edge: str
    textgrid_s: float
    automatic_s: float
    nucleus_distance_ms: float
    textgrid_interval: tuple[float, float, float]
    automatic_interval: tuple[float, float, float]

    @property
    def difference_ms(self) -> float:
        return (self.automatic_s - self.textgrid_s) * 1000.0


def poem1_syllable_differences() -> list[SyllableDifference]:
    """Return edge differences for mutually nearest syllable-nucleus matches."""
    config = POEMS_CONFIG[0]
    manual = load_textgrid_syllable(
        config["audio_url"],
        config["textgrid_url"],
        intensity_floor_hz=50,
        pitch_timestep_s=0.01,
        pitch_range_hz=(75, 500),
        silence_threshold_relative_db=-25,
        syllable_pitch_percentile=90,
        min_nucleus_prominence_db=1,
    ).syllables
    automatic = syllablize_audio(config["audio_url"]).syllables

    differences = []
    for manual_index, manual_syllable in enumerate(manual):
        automatic_index = min(
            range(len(automatic)),
            key=lambda index: abs(
                automatic[index].nucleus - manual_syllable.nucleus
            ),
        )
        automatic_syllable = automatic[automatic_index]
        nearest_manual_index = min(
            range(len(manual)),
            key=lambda index: abs(
                manual[index].nucleus - automatic_syllable.nucleus
            ),
        )
        # Mutual-nearest matching prevents multiple TextGrid syllables from
        # being paired with the same automatic syllable.
        if nearest_manual_index != manual_index:
            continue
        nucleus_distance_ms = abs(
            automatic_syllable.nucleus - manual_syllable.nucleus
        ) * 1000.0
        for edge in ("start", "end"):
            differences.append(
                SyllableDifference(
                    textgrid_index=manual_index,
                    automatic_index=automatic_index,
                    transcription=manual_syllable.transcription,
                    edge=edge,
                    textgrid_s=getattr(manual_syllable, edge),
                    automatic_s=getattr(automatic_syllable, edge),
                    nucleus_distance_ms=nucleus_distance_ms,
                    textgrid_interval=(
                        manual_syllable.start,
                        manual_syllable.nucleus,
                        manual_syllable.end,
                    ),
                    automatic_interval=(
                        automatic_syllable.start,
                        automatic_syllable.nucleus,
                        automatic_syllable.end,
                    ),
                )
            )
    return differences


def export_context(
    input_path: Path, output_path: Path, center_s: float, context_s: float
) -> tuple[float, float]:
    """Export a WAV window around the discrepant syllable edge."""
    with wave.open(str(input_path), "rb") as source:
        params = source.getparams()
        start_frame = max(0, round((center_s - context_s) * params.framerate))
        end_frame = min(params.nframes, round((center_s + context_s) * params.framerate))
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)
    return start_frame / params.framerate, end_frame / params.framerate


def draw_comparison(
    input_path: Path,
    output_path: Path,
    difference: SyllableDifference,
    section_start_s: float,
    section_end_s: float,
) -> None:
    """Draw the waveform with manual (blue) and detected (red) intervals."""
    sample_rate, raw_samples = wavfile.read(input_path)
    samples = np.asarray(raw_samples, dtype=float)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    start_sample = round(section_start_s * sample_rate)
    end_sample = round(section_end_s * sample_rate)
    samples = samples[start_sample:end_sample]
    scale = float(np.max(np.abs(samples)))
    if scale:
        samples /= scale
    times = np.arange(start_sample, end_sample) / sample_rate

    manual_color = "#2563eb"
    detected_color = "#ef4444"
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(times, samples, color="#737373", linewidth=0.65, label="waveform")

    for label, interval, color, linestyle in (
        ("Manual TextGrid", difference.textgrid_interval, manual_color, "-"),
        ("Automatic detection", difference.automatic_interval, detected_color, "--"),
    ):
        start_s, nucleus_s, end_s = interval
        ax.axvspan(start_s, end_s, color=color, alpha=0.10)
        ax.axvline(start_s, color=color, linestyle=linestyle, linewidth=2, label=label)
        ax.axvline(end_s, color=color, linestyle=linestyle, linewidth=2)
        ax.scatter(
            [nucleus_s],
            [0.88 if label == "Manual TextGrid" else -0.88],
            color=color,
            marker="v" if label == "Manual TextGrid" else "^",
            s=55,
            zorder=5,
        )

    ax.annotate(
        f"{difference.edge} difference = {difference.difference_ms:+.1f} ms",
        xy=((difference.textgrid_s + difference.automatic_s) / 2.0, 0.0),
        xytext=(0, 35),
        textcoords="offset points",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "black"},
    )
    ax.set_xlim(section_start_s, section_end_s)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time in full poem recording (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(
        f"Poem 1 syllable {difference.transcription!r}: "
        "manual vs automatic annotation"
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-ms", type=float, default=100.0)
    parser.add_argument("--context-s", type=float, default=0.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exploration/poem1_100ms_difference.wav"),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("exploration/poem1_100ms_difference.png"),
    )
    args = parser.parse_args()
    best = min(
        poem1_syllable_differences(),
        key=lambda item: abs(abs(item.difference_ms) - args.target_ms),
    )
    center_s = (best.textgrid_s + best.automatic_s) / 2.0
    section_start_s, section_end_s = export_context(
        POEMS_CONFIG[0]["audio_url"], args.output, center_s, args.context_s
    )
    draw_comparison(
        POEMS_CONFIG[0]["audio_url"],
        args.plot_output,
        best,
        section_start_s,
        section_end_s,
    )
    print(f"TextGrid syllable:  {best.textgrid_index} ({best.transcription!r})")
    print(f"automatic syllable: {best.automatic_index}")
    print(f"edge:               {best.edge}")
    print(f"TextGrid:           {best.textgrid_s:.6f} s")
    print(f"automatic:          {best.automatic_s:.6f} s")
    print(f"signed delta:       {best.difference_ms:+.3f} ms")
    print(f"absolute delta:     {abs(best.difference_ms):.3f} ms")
    print(f"nucleus distance:   {best.nucleus_distance_ms:.3f} ms")
    print(f"audio section:      {section_start_s:.6f}-{section_end_s:.6f} s")
    print(f"wrote:              {args.output}")
    print(f"plot:               {args.plot_output}")


if __name__ == "__main__":
    main()
