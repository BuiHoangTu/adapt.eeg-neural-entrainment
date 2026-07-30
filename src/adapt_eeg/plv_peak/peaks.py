from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks, hilbert

from adapt_eeg.replicate.analysis import PAPER_BEAT_DISTANCES

POEM_FREQUENCY_HZ = 2.0
PEAK_WINDOW_SECONDS = 0.05
PEAKS_PER_EPOCH = 3


@dataclass(frozen=True)
class PeakWindow:
    line: int
    peak_index: int
    peak_seconds: float
    window_start: float
    window_end: float
    clipped_start: bool
    clipped_end: bool


def _mono_float_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    audio = np.asarray(data, dtype=float)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs
    return sample_rate, audio


def infer_first_peak_offset(wav_path: Path) -> float:
    sample_rate, audio = _mono_float_audio(wav_path)
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {wav_path}")

    envelope = np.abs(hilbert(audio))
    smooth_samples = max(1, int(round(sample_rate * 0.02)))
    smooth = np.convolve(envelope, np.ones(smooth_samples) / smooth_samples, mode="same")
    if np.max(smooth) <= np.finfo(float).eps:
        return 0.0

    active = np.flatnonzero(smooth >= np.max(smooth) * 0.10)
    onset_sample = int(active[0]) if active.size else 0
    peaks, _ = find_peaks(
        smooth,
        height=np.max(smooth) * 0.15,
        prominence=np.max(smooth) * 0.05,
        distance=max(1, int(round(sample_rate * 0.15))),
    )
    peaks_after_onset = peaks[peaks >= onset_sample]
    first_peak_sample = int(peaks_after_onset[0]) if peaks_after_onset.size else onset_sample
    return first_peak_sample / sample_rate


def infer_first_peak_offsets(audio_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for line in sorted(PAPER_BEAT_DISTANCES["line"].astype(int)):
        wav_path = audio_root / f"vers{line}.wav"
        if not wav_path.exists():
            rows.append(
                {
                    "line": line,
                    "first_peak_offset_seconds": 0.0,
                    "source": "line_start_anchor_missing_audio",
                    "audio_path": str(wav_path),
                }
            )
            continue
        try:
            offset = infer_first_peak_offset(wav_path)
            source = "audio_envelope_first_peak"
        except Exception:
            offset = 0.0
            source = "line_start_anchor_audio_peak_failed"
        rows.append(
            {
                "line": line,
                "first_peak_offset_seconds": offset,
                "source": source,
                "audio_path": str(wav_path),
            }
        )
    return pd.DataFrame(rows)


def table1_peak_times(
    peaks_per_epoch: int = PEAKS_PER_EPOCH,
    first_peak_offsets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    offset_by_line = {}
    source_by_line = {}
    if first_peak_offsets is not None and not first_peak_offsets.empty:
        offset_by_line = dict(
            zip(
                first_peak_offsets["line"].astype(int),
                first_peak_offsets["first_peak_offset_seconds"].astype(float),
                strict=False,
            )
        )
        source_by_line = dict(
            zip(
                first_peak_offsets["line"].astype(int),
                first_peak_offsets["source"],
                strict=False,
            )
        )

    rows: list[dict] = []
    distance_columns = ["dist1", "dist2", "dist3", "dist4"]
    for _, row in PAPER_BEAT_DISTANCES.iterrows():
        line = int(row["line"])
        elapsed = float(offset_by_line.get(line, 0.0))
        peak_times = [elapsed]
        for column in distance_columns:
            elapsed += float(row[column])
            peak_times.append(elapsed)

        source = source_by_line.get(line, "table_1_distances_line_start_anchor")
        for peak_index, peak_seconds in enumerate(peak_times[:peaks_per_epoch], start=1):
            rows.append(
                {
                    "line": line,
                    "peak_index": peak_index,
                    "peak_seconds": peak_seconds,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def peak_windows_for_line(
    line: int,
    segment_duration: float,
    window_radius: float = PEAK_WINDOW_SECONDS,
    peaks_per_epoch: int = PEAKS_PER_EPOCH,
    first_peak_offsets: pd.DataFrame | None = None,
) -> list[PeakWindow]:
    peaks = table1_peak_times(peaks_per_epoch, first_peak_offsets=first_peak_offsets)
    line_peaks = peaks.loc[peaks["line"] == line]
    if line_peaks.empty:
        raise ValueError(f"No Table 1 peak timings for line {line}")

    windows: list[PeakWindow] = []
    for _, peak in line_peaks.iterrows():
        peak_seconds = float(peak["peak_seconds"])
        raw_start = peak_seconds - window_radius
        raw_end = peak_seconds + window_radius
        window_start = max(0.0, raw_start)
        window_end = min(segment_duration, raw_end)
        if window_end <= window_start:
            raise ValueError(
                f"Peak {int(peak['peak_index'])} at {peak_seconds:.3f}s is outside "
                f"line {line} segment duration {segment_duration:.3f}s"
            )
        windows.append(
            PeakWindow(
                line=line,
                peak_index=int(peak["peak_index"]),
                peak_seconds=peak_seconds,
                window_start=window_start,
                window_end=window_end,
                clipped_start=window_start != raw_start,
                clipped_end=window_end != raw_end,
            )
        )
    return windows

if __name__ == "__main__":
    peaks = infer_first_peak_offsets(Path("data/raw/poem1_versuri"))
    print(peaks)