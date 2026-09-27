from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import numpy as np

from adapt_eeg.auto_annotation.stress_identifier import (
    calculate_inter_stress_distances,
)
from adapt_eeg.auto_annotation.textgrid_annotation_fix import derive_lined_audio
from adapt_eeg.constants import POEMS_CONFIG

REGULAR_CV_MAX = 0.40
IRREGULAR_CV_MIN = 0.55


@dataclass(frozen=True)
class PoemLineTiming:
    poem: int
    line_index: int
    line_number: int
    start_s: float
    end_s: float
    mean_interval_s: float
    median_interval_s: float
    std_interval_s: float
    coefficient_of_variation: float
    stress_frequency_hz: float
    rhythm_condition: str


def inter_stress_interval_cv(intervals: np.ndarray | list[float]) -> float:
    """Return population-SD CV for stress-to-stress intervals."""
    values = np.asarray(intervals, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return np.nan
    mean_s = float(np.mean(values))
    if mean_s <= 0:
        return np.nan
    return float(np.std(values, ddof=0) / mean_s)


def classify_rhythm(coefficient_of_variation: float) -> str:
    """Classify a line from the CV of its stress-to-stress intervals."""
    if not np.isfinite(coefficient_of_variation):
        return "ambiguous"
    if coefficient_of_variation < REGULAR_CV_MAX:
        return "regular"
    if coefficient_of_variation > IRREGULAR_CV_MIN:
        return "irregular"
    return "ambiguous"


def derive_poem_line_timings(poem: int) -> list[PoemLineTiming]:
    config = POEMS_CONFIG[poem - 1]
    boundary_module = import_module(
        f"adapt_eeg.auto_annotation.textgrid_annotation_fix.poem{poem}"
    )
    lined_audio = derive_lined_audio(
        config["audio_url"],
        config["textgrid_url"],
        boundary_module.textgrid_eol_indices,
        intro_end_ms=config["intro_end"],
    )
    distances_by_line = calculate_inter_stress_distances(lined_audio)
    starts = [0.0, *lined_audio.line_boundaries]
    ends = [*lined_audio.line_boundaries, len(lined_audio.audio) / 1000.0]

    lines = []
    for line_index, (start_s, end_s, distances) in enumerate(
        zip(starts, ends, distances_by_line, strict=True)
    ):
        values = np.asarray(distances, dtype=float)
        mean_s = float(np.mean(values)) if len(values) else np.nan
        median_s = float(np.median(values)) if len(values) else np.nan
        std_s = float(np.std(values, ddof=0)) if len(values) else np.nan
        cv = inter_stress_interval_cv(values)
        lines.append(
            PoemLineTiming(
                poem=poem,
                line_index=line_index,
                line_number=line_index + 1,
                start_s=start_s,
                end_s=end_s,
                mean_interval_s=mean_s,
                median_interval_s=median_s,
                std_interval_s=std_s,
                coefficient_of_variation=cv,
                stress_frequency_hz=1 / median_s if median_s > 0 else np.nan,
                rhythm_condition=classify_rhythm(cv),
            )
        )
    return lines


def poem_stress_frequency(lines: list[PoemLineTiming]) -> float:
    """Return the reciprocal of the poem's mean inter-stress interval."""
    intervals = [line.mean_interval_s for line in lines if np.isfinite(line.mean_interval_s)]
    if not intervals:
        raise ValueError("Cannot estimate poem frequency without stress intervals")
    mean_interval_s = float(np.mean(intervals))
    if mean_interval_s <= 0:
        raise ValueError("Mean inter-stress interval must be positive")
    return 1.0 / mean_interval_s
