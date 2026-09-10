from dataclasses import dataclass, field, fields, replace
from typing import TypeAlias

import numpy as np
import parselmouth
from pydub import AudioSegment


@dataclass(frozen=True)
class StressIdentifyParams:
    # pitch range
    pitch_range_hz: tuple[float, float] = (75, 500)

    # Neighboring-region baseline for pitch/intensity prominence
    prominence_neighbor_half_window_s: float = 0.05
    prominence_min_nucleus_distance_s: float = 0.05

    # Duration prominence compares against nearby syllables.
    duration_neighbor_radius: int = 3

    # Stress score aggregation
    stress_weights: tuple[float, float, float] = (5, 3, 2)
    stressed_syllables_per_line: int = 5


@dataclass(frozen=True)
class Nucleus:
    time: float
    intensity_db: float
    prominence_db: float


@dataclass(frozen=True)
class Syllable:
    start: float
    _nucleus: Nucleus = field(repr=False)
    nucleus: float = field(init=False)
    end: float

    _audio: AudioSegment = field(repr=False)

    pitch_hz: float
    intensity_db: float

    duration_s: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "nucleus", self._nucleus.time)
        object.__setattr__(self, "duration_s", self.end - self.start)


@dataclass(frozen=True)
class TranscriptedSyllable(Syllable):
    transcription: str

    @classmethod
    def from_syllable(
        cls,
        syllable: Syllable,
        transcription: str,
    ) -> "TranscriptedSyllable":
        values = {f.name: getattr(syllable, f.name) for f in fields(Syllable) if f.init}

        return cls(
            **values,
            transcription=transcription,
        )


@dataclass(frozen=True)
class SyllableIdentifyParams:
    # Intensity / nucleus detection
    intensity_floor_hz: float = 50.0
    silence_threshold_relative_db: float = -25.0
    min_nucleus_prominence_db: float = 1.0

    # Pitch
    pitch_range_hz: tuple[float, float] = (75.0, 500.0)
    pitch_time_step_s: float = 0.01

    # Voicing
    voicing_half_window_s: float = 0.04
    min_voiced_fraction: float = 0.50

    # Pause handling
    min_pause_duration_s: float = 0.3

    # Pitch/intensity co-occurrence
    pitch_percentile: float = 90.0


@dataclass(frozen=True)
class Contour:
    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class SyllablizedAudio:
    sound: parselmouth.Sound
    audio: AudioSegment
    samples: np.ndarray
    sample_rate: int
    intensity: Contour
    f0: Contour
    silence_threshold_db: float
    syllables: list[Syllable]

    def __getitem__(self, key: slice) -> "SyllablizedAudio":
        if not isinstance(key, slice):
            raise TypeError("SyllablizedAudio indices must be slices")
        if key.step is not None:
            raise ValueError("SyllablizedAudio slicing does not support steps")

        start_ms, end_ms, _ = key.indices(len(self.audio))
        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0

        sample_start = round(start_s * self.sample_rate)
        sample_end = round(end_s * self.sample_rate)

        def slice_contour(contour: Contour) -> Contour:
            mask = (contour.times >= start_s) & (contour.times <= end_s)
            return Contour(
                times=contour.times[mask] - start_s,
                values=contour.values[mask],
            )

        syllables = []
        for syllable in self.syllables:
            if syllable.start < start_s or syllable.end > end_s:
                continue

            shifted_start = syllable.start - start_s
            shifted_end = syllable.end - start_s
            syllables.append(
                replace(
                    syllable,
                    start=shifted_start,
                    _nucleus=replace(
                        syllable._nucleus,
                        time=syllable._nucleus.time - start_s,
                    ),
                    end=shifted_end,
                    _audio=syllable._audio,
                )
            )

        return SyllablizedAudio(
            sound=self.sound.extract_part(
                from_time=start_s,
                to_time=end_s,
                preserve_times=False,
            ),
            audio=self.audio[start_ms:end_ms],  # type: ignore # not generator without steps
            samples=self.samples[sample_start:sample_end],
            sample_rate=self.sample_rate,
            intensity=slice_contour(self.intensity),
            f0=slice_contour(self.f0),
            silence_threshold_db=self.silence_threshold_db,
            syllables=syllables,
        )


@dataclass(frozen=True)
class LinedAudio(SyllablizedAudio):
    line_boundaries: tuple[float, ...]

    @classmethod
    def from_syllablized_audio(
        cls,
        syllablized_audio: SyllablizedAudio,
        line_boundaries: tuple[float, ...],
    ) -> "LinedAudio":
        values = {
            field.name: getattr(syllablized_audio, field.name)
            for field in fields(SyllablizedAudio)
            if field.init
        }
        return cls(**values, line_boundaries=line_boundaries)

    def __getitem__(self, key: slice) -> "LinedAudio":
        sliced_audio = super().__getitem__(key)
        start_ms, end_ms, _ = key.indices(len(self.audio))
        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0

        return LinedAudio.from_syllablized_audio(
            sliced_audio,
            tuple(
                boundary - start_s
                for boundary in self.line_boundaries
                if start_s < boundary < end_s
            ),
        )


SyllableInterval: TypeAlias = tuple[float, Nucleus, float]
"""Tuple of (start_time, nucleus, end_time)"""
StressEvidence: TypeAlias = tuple[float, float, float]
"""Tuple of (pitch_prominence_hz, intensity_prominence_db, duration_prominence_s)"""
