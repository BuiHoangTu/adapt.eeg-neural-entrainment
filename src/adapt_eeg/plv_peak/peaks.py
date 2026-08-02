from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks, hilbert

from adapt_eeg.replicate.analysis import PAPER_BEAT_DISTANCES

POEM_FREQUENCY_HZ = 2.0
PEAK_WINDOW_SECONDS = 0.25
PEAKS_PER_EPOCH = 5


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
    if audio.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0:
        audio = audio / max_abs
    return int(sample_rate), audio


def load_line_audio(audio_root: Path, line: int) -> tuple[int, np.ndarray]:
    wav_path = audio_root / f"vers{line}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing audio for line {line}: {wav_path}")
    return _mono_float_audio(wav_path)


def infer_first_peak_offset(wav_path: Path) -> float:
    sample_rate, audio = _mono_float_audio(wav_path)
    envelope = np.abs(hilbert(audio))
    smooth_samples = max(1, int(round(sample_rate * 0.02)))
    smooth = np.convolve(
        envelope,
        np.ones(smooth_samples, dtype=float) / smooth_samples,
        mode="same",
    )
    maximum = float(np.max(smooth))
    if maximum <= np.finfo(float).eps:
        raise ValueError(f"No detectable audio envelope in {wav_path}")

    active = np.flatnonzero(smooth >= maximum * 0.10)
    if active.size == 0:
        raise ValueError(f"No detectable audio onset in {wav_path}")
    onset_sample = int(active[0])

    peaks, _ = find_peaks(
        smooth,
        height=maximum * 0.15,
        prominence=maximum * 0.05,
        distance=max(1, int(round(sample_rate * 0.15))),
    )
    peaks_after_onset = peaks[peaks >= onset_sample]
    if peaks_after_onset.size == 0:
        raise ValueError(f"No envelope peak detected after onset in {wav_path}")
    return float(peaks_after_onset[0]) / sample_rate


def infer_first_peak_offsets(audio_root: Path) -> pd.DataFrame:
    """Infer audio anchors without silently substituting invalid zero offsets."""
    rows: list[dict] = []
    for line in sorted(PAPER_BEAT_DISTANCES["line"].astype(int)):
        wav_path = audio_root / f"vers{line}.wav"
        try:
            offset = infer_first_peak_offset(wav_path)
            rows.append(
                {
                    "line": line,
                    "first_peak_offset_seconds": offset,
                    "source": "audio_envelope_first_peak",
                    "alignment_valid": True,
                    "alignment_error": "",
                    "audio_path": str(wav_path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "line": line,
                    "first_peak_offset_seconds": np.nan,
                    "source": "invalid_audio_alignment",
                    "alignment_valid": False,
                    "alignment_error": repr(exc),
                    "audio_path": str(wav_path),
                }
            )
    return pd.DataFrame(rows)


def _valid_offset_maps(
    first_peak_offsets: pd.DataFrame | None,
) -> tuple[dict[int, float], dict[int, str]]:
    if first_peak_offsets is None or first_peak_offsets.empty:
        return {}, {}

    valid = first_peak_offsets.copy()
    if "alignment_valid" in valid:
        valid = valid.loc[valid["alignment_valid"].astype(bool)]
    valid = valid.dropna(subset=["first_peak_offset_seconds"])
    return (
        dict(
            zip(
                valid["line"].astype(int),
                valid["first_peak_offset_seconds"].astype(float),
                strict=False,
            )
        ),
        dict(zip(valid["line"].astype(int), valid["source"], strict=False)),
    )


def table1_peak_times(
    peaks_per_epoch: int = PEAKS_PER_EPOCH,
    first_peak_offsets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    offset_by_line, source_by_line = _valid_offset_maps(first_peak_offsets)

    rows: list[dict] = []
    distance_columns = ["dist1", "dist2", "dist3", "dist4"]
    for _, row in PAPER_BEAT_DISTANCES.iterrows():
        line = int(row["line"])
        if first_peak_offsets is not None and line not in offset_by_line:
            continue
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
    """Create fixed-radius windows, retained for backward-compatible runs."""
    peaks = table1_peak_times(peaks_per_epoch, first_peak_offsets=first_peak_offsets)
    line_peaks = peaks.loc[peaks["line"] == line]
    if line_peaks.empty:
        raise ValueError(f"No valid peak timings for line {line}")

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


def beat_intervals_for_line(
    line: int,
    segment_duration: float,
    peaks_per_epoch: int = PEAKS_PER_EPOCH,
    first_peak_offsets: pd.DataFrame | None = None,
) -> list[PeakWindow]:
    """Create non-overlapping intervals bounded by midpoints between beats."""
    peaks = table1_peak_times(peaks_per_epoch, first_peak_offsets=first_peak_offsets)
    line_peaks = peaks.loc[peaks["line"] == line].sort_values("peak_index")
    if len(line_peaks) < 2:
        raise ValueError(f"At least two valid peak timings are required for line {line}")

    times = line_peaks["peak_seconds"].to_numpy(dtype=float)
    indices = line_peaks["peak_index"].to_numpy(dtype=int)
    boundaries = np.empty(times.size + 1, dtype=float)
    boundaries[1:-1] = (times[:-1] + times[1:]) / 2.0
    boundaries[0] = times[0] - (times[1] - times[0]) / 2.0
    boundaries[-1] = times[-1] + (times[-1] - times[-2]) / 2.0

    windows: list[PeakWindow] = []
    for position, (peak_index, peak_seconds) in enumerate(zip(indices, times, strict=True)):
        raw_start = float(boundaries[position])
        raw_end = float(boundaries[position + 1])
        window_start = max(0.0, raw_start)
        window_end = min(segment_duration, raw_end)
        if window_end <= window_start:
            raise ValueError(
                f"Beat interval {peak_index} is outside line {line} duration "
                f"{segment_duration:.3f}s"
            )
        windows.append(
            PeakWindow(
                line=line,
                peak_index=int(peak_index),
                peak_seconds=float(peak_seconds),
                window_start=window_start,
                window_end=window_end,
                clipped_start=window_start != raw_start,
                clipped_end=window_end != raw_end,
            )
        )
    return windows


if __name__ == "__main__":
    print(infer_first_peak_offsets(Path("data/raw/poem1_versuri")))
