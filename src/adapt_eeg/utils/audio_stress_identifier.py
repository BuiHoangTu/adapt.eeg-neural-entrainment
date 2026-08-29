import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth
from pydub import AudioSegment
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Contour:
    times: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class Nucleus:
    time: float
    intensity_db: float
    prominence_db: float


@dataclass(frozen=True)
class Syllable:
    start: float
    nucleus: float
    end: float

    pitch_hz: float
    intensity_db: float
    duration_s: float


@dataclass(frozen=True)
class AnalysisParams:
    # Intensity / nucleus detection
    intensity_floor_hz: float = 50.0
    silence_threshold_relative_db: float = -25.0
    min_nucleus_prominence_db: float = 2.0

    # F0 / voicing
    pitch_floor_hz: float = 75.0
    pitch_ceiling_hz: float = 500.0
    pitch_time_step_s: float = 0.01

    voicing_half_window_s: float = 0.04
    min_voiced_fraction: float = 0.50

    min_pause_duration_s: float = 0.3

    # Syllable-level pitch measurement
    pitch_percentile: float = 90.0

    # Local prominence
    prominence_neighbor_radius: int = 3
    prominence_baseline_quantile: float = 0.25

    # Stress weights
    pitch_weight: float = 0.5
    intensity_weight: float = 0.3
    duration_weight: float = 0.2

    # Final classification
    stress_score_threshold: float = 0.50
    require_pitch_intensity_cooccurrence: bool = True


DEFAULT_PARAMS = AnalysisParams()


SyllableInterval = tuple[float, float, float]
Prominence = tuple[float, float, float]


def load_audio(
    audio_input: str | Path | AudioSegment,
) -> tuple[parselmouth.Sound, np.ndarray, int]:
    """
    Load audio, convert to mono, and return:
        Praat Sound,
        normalized waveform samples,
        sample rate.
    """

    if isinstance(audio_input, AudioSegment):
        audio = audio_input
    else:
        audio = AudioSegment.from_file(audio_input)

    # convert to mono
    audio = audio.set_channels(1)

    sample_rate = audio.frame_rate
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float64)

    # Normalize. 8 * n_byte_per_sample - 1 (sign bit <+/->). 2^n_bit
    full_scale = float(1 << (8 * audio.sample_width - 1))
    samples = samples / full_scale

    sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)

    return sound, samples, sample_rate


def get_intensity_contour(
    sound: parselmouth.Sound,
    intensity_floor_hz: float,
) -> Contour:

    intensity = sound.to_intensity(
        minimum_pitch=intensity_floor_hz,
        time_step=None,
        subtract_mean=True,
    )

    return Contour(
        times=np.asarray(intensity.xs(), dtype=np.float64),
        values=np.asarray(intensity.values[0], dtype=np.float64),
    )


def get_f0_contour(
    sound: parselmouth.Sound,
    pitch_time_step_s,
    pitch_floor_hz,
    pitch_ceiling_hz,
) -> Contour:

    pitch = sound.to_pitch_ac(
        time_step=pitch_time_step_s,
        pitch_floor=pitch_floor_hz,
        pitch_ceiling=pitch_ceiling_hz,
    )

    times = np.asarray(pitch.xs(), dtype=np.float64)

    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)

    # Praat represents unvoiced frames as 0.
    f0[f0 <= 0] = np.nan

    return Contour(
        times=times,
        values=f0,
    )


def get_silence_threshold_db(intensity, silence_threshold_relative_db):
    reference_db = np.nanquantile(intensity.values, 0.99)

    return reference_db + silence_threshold_relative_db


def detect_nucleus_candidates(
    intensity: Contour,
    silence_threshold_db,
    min_nucleus_prominence_db,
) -> list[Nucleus]:

    if len(intensity.values) == 0:
        return []

    peak_indices, properties = find_peaks(
        intensity.values,
        height=silence_threshold_db,
        prominence=min_nucleus_prominence_db,
    )

    prominences = properties["prominences"]

    return [
        Nucleus(
            time=float(intensity.times[index]),
            intensity_db=float(intensity.values[index]),
            prominence_db=float(prominence),
        )
        for index, prominence in zip(peak_indices, prominences)
    ]


def filter_voiced_nuclei(
    nuclei: list[Nucleus], f0: Contour, voicing_half_window_s, min_voiced_fraction
) -> list[Nucleus]:

    voiced = []

    for nucleus in nuclei:
        mask = np.abs(f0.times - nucleus.time) <= voicing_half_window_s

        local_f0 = f0.values[mask]

        if len(local_f0) == 0:
            continue

        voiced_fraction = np.mean(np.isfinite(local_f0))

        if voiced_fraction >= min_voiced_fraction:
            voiced.append(nucleus)

    return voiced


def _lowest_intensity_time(
    intensity: Contour,
    start: float,
    end: float,
) -> float:

    mask = (intensity.times >= start) & (intensity.times <= end)

    indices = np.flatnonzero(mask)

    if len(indices) == 0:
        return (start + end) / 2.0

    local_values = intensity.values[indices]

    minimum_index = indices[np.nanargmin(local_values)]

    return float(intensity.times[minimum_index])


def _find_pause_between(
    intensity: Contour,
    start: float,
    end: float,
    silence_threshold_db: float,
    min_pause_duration_s: float,
) -> tuple[float, float] | None:
    """
    Find a pause between 2 nucleus "start" and "end".

    Params:
        silence_threshold_db: the intensity of background noise
        min_pause_duration_s: the minimum length of pause to be valid

    Returns:
        (pause_start, pause_end):

    or None if no qualifying pause exists.
    """

    mask = (intensity.times >= start) & (intensity.times <= end)

    times = intensity.times[mask]
    values = intensity.values[mask]

    if len(times) == 0:
        return None

    silent_indices = np.flatnonzero(values <= silence_threshold_db)

    if len(silent_indices) == 0:
        return None

    pause_start = float(times[silent_indices[0]])

    pause_end = float(times[silent_indices[-1]])

    pause_duration = pause_end - pause_start

    if pause_duration < min_pause_duration_s:
        return None

    return pause_start, pause_end


def estimate_syllable_intervals(
    nuclei: list[Nucleus],
    intensity: Contour,
    audio_duration_s: float,
    min_pause_duration_s: float,
    silence_threshold_db: float,
) -> list[SyllableInterval]:
    """
    Return:
        [(start, nucleus, end), ...]
    """

    if not nuclei:
        return []

    nuclei = sorted(nuclei, key=lambda n: n.time)

    nucleus_times = [nucleus.time for nucleus in nuclei]

    starts = np.zeros(len(nucleus_times), dtype=float)
    ends = np.zeros(len(nucleus_times), dtype=float)

    starts[0] = _lowest_intensity_time(
        intensity,
        0.0,
        nucleus_times[0],
    )

    ends[-1] = _lowest_intensity_time(
        intensity,
        nucleus_times[-1],
        audio_duration_s,
    )

    for i in range(len(nucleus_times) - 1):
        left_nucleus = nucleus_times[i]
        right_nucleus = nucleus_times[i + 1]

        pause = _find_pause_between(
            intensity,
            left_nucleus,
            right_nucleus,
            silence_threshold_db,
            min_pause_duration_s,
        )

        if pause is not None:
            pause_start, pause_end = pause

            ends[i] = pause_start
            starts[i + 1] = pause_end

        else:
            # Normal case:
            # adjacent syllables share the intensity valley.
            boundary = _lowest_intensity_time(
                intensity,
                left_nucleus,
                right_nucleus,
            )

            ends[i] = boundary
            starts[i + 1] = boundary

    return [
        (
            float(starts[i]),
            float(nucleus_times[i]),
            float(ends[i]),
        )
        for i in range(len(nucleus_times))
    ]


def _measure_pitch(
    start: float,
    end: float,
    f0: Contour,
    percentile: float,
) -> float:

    mask = (f0.times >= start) & (f0.times <= end)

    values = f0.values[mask]
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, percentile))


def _measure_rms_db(
    start: float,
    end: float,
    samples: np.ndarray,
    sample_rate: int,
) -> float:

    start_index = max(0, round(start * sample_rate))

    end_index = min(len(samples), round(end * sample_rate))

    segment = samples[start_index:end_index]

    if len(segment) == 0:
        return np.nan

    rms = np.sqrt(np.mean(segment**2))

    if rms <= 0:
        return np.nan

    return float(20.0 * np.log10(rms))


def measure_syllables(
    intervals: list[SyllableInterval],
    samples: np.ndarray,
    sample_rate: int,
    f0: Contour,
    pitch_percentile,
) -> list[Syllable]:

    syllables = []

    for start, nucleus, end in intervals:
        duration = end - start

        pitch = _measure_pitch(
            start,
            end,
            f0,
            pitch_percentile,
        )

        intensity = _measure_rms_db(
            start,
            end,
            samples,
            sample_rate,
        )

        syllables.append(
            Syllable(
                start=start,
                nucleus=nucleus,
                end=end,
                pitch_hz=pitch,
                intensity_db=intensity,
                duration_s=duration,
            )
        )

    return syllables


def _local_prominence(
    values: np.ndarray,
    radius: int,
    baseline_quantile: float,
) -> np.ndarray:

    prominences = np.full(
        len(values),
        np.nan,
        dtype=np.float64,
    )

    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue

        left = max(0, i - radius)

        right = min(len(values), i + radius + 1)

        neighbors = np.concatenate(
            [
                values[left:i],
                values[i + 1 : right],
            ]
        )

        neighbors = neighbors[np.isfinite(neighbors)]

        if len(neighbors) == 0:
            continue

        baseline = np.quantile(
            neighbors,
            baseline_quantile,
        )

        prominences[i] = value - baseline

    return prominences


def calculate_local_prominences(
    syllables: list[Syllable],
    prominence_neighbor_radius,
    prominence_baseline_quantile,
) -> list[Prominence]:

    pitch = np.asarray([syllable.pitch_hz for syllable in syllables])

    intensity = np.asarray([syllable.intensity_db for syllable in syllables])

    duration = np.asarray([syllable.duration_s for syllable in syllables])

    pitch_prominence = _local_prominence(
        pitch,
        prominence_neighbor_radius,
        prominence_baseline_quantile,
    )

    intensity_prominence = _local_prominence(
        intensity,
        prominence_neighbor_radius,
        prominence_baseline_quantile,
    )

    duration_prominence = _local_prominence(
        duration,
        prominence_neighbor_radius,
        prominence_baseline_quantile,
    )

    return list(
        zip(
            pitch_prominence,
            intensity_prominence,
            duration_prominence,
        )
    )


def _robust_zscore(
    values: np.ndarray,
) -> np.ndarray:
    """Std can be distorted by many outliers in audio"""

    values = np.asarray(values, dtype=np.float64)

    result = np.full_like(values, np.nan)

    valid = np.isfinite(values)

    if not np.any(valid):
        return result

    valid_values = values[valid]

    median = np.median(valid_values)

    mad = np.median(np.abs(valid_values - median))

    # Convert MAD to an estimate comparable to standard deviation.
    scale = 1.4826 * mad

    # Fallback if every prominence is nearly identical.
    if scale <= np.finfo(float).eps:
        scale = np.std(valid_values)

    if scale <= np.finfo(float).eps:
        result[valid] = 0.0
        return result

    result[valid] = (valid_values - median) / scale

    return result


def normalize_prominences(
    prominences: list[Prominence],
) -> list[Prominence]:

    if not prominences:
        return []

    values = np.asarray(prominences, dtype=np.float64)

    pitch_z = _robust_zscore(values[:, 0])

    intensity_z = _robust_zscore(values[:, 1])

    duration_z = _robust_zscore(values[:, 2])

    return list(
        zip(
            pitch_z,
            intensity_z,
            duration_z,
        )
    )


def _normalize_weights(
    pitch_weight: float,
    intensity_weight: float,
    duration_weight: float,
):

    weights = np.asarray(
        [
            pitch_weight,
            intensity_weight,
            duration_weight,
        ],
        dtype=np.float64,
    )

    if np.any(weights < 0):
        raise ValueError("Stress weights cannot be negative.")

    total = weights.sum()

    if total <= 0:
        raise ValueError("At least one stress weight must be greater than zero.")

    weights /= total

    return weights[0], weights[1], weights[2]


def calculate_stress_scores(
    normalized_prominences: list[Prominence],
    weights: tuple[float, float, float],
) -> list[float]:
    """
    Missing features are ignored and the remaining
    weights are renormalized for that syllable.
    """

    weights_array = np.asarray(weights, dtype=np.float64)

    scores = []

    for features in normalized_prominences:
        features = np.asarray(features, dtype=np.float64)

        valid = np.isfinite(features)

        if not np.any(valid):
            scores.append(np.nan)
            continue

        available_weights = weights_array[valid]

        available_weights = available_weights / available_weights.sum()

        score = np.sum(features[valid] * available_weights)

        scores.append(float(score))

    return scores


def classify_stresses(
    prominences: list[Prominence],
    scores: list[float],
    stress_score_threshold,
    require_pitch_intensity_cooccurrence,
) -> list[bool]:

    stressed = []

    for prominence, score in zip(prominences, scores):
        pitch_prominence, intensity_prominence, _ = prominence

        if not np.isfinite(score):
            stressed.append(False)
            continue

        is_stressed = score >= stress_score_threshold

        if require_pitch_intensity_cooccurrence:
            is_stressed = (
                is_stressed
                and np.isfinite(pitch_prominence)
                and np.isfinite(intensity_prominence)
                and pitch_prominence > 0
                and intensity_prominence > 0
            )

        stressed.append(bool(is_stressed))

    return stressed


def detect_stressed_syllables(
    audio_input: str | Path | AudioSegment,
    params: AnalysisParams = DEFAULT_PARAMS,
) -> list[Syllable]:
    """
    Detect stressed syllables from an audio recording.

    Ultimate output:
        list[Syllable]

    Only syllables classified as stressed are returned.
    """

    sound, samples, sample_rate = load_audio(audio_input)

    intensity = get_intensity_contour(
        sound,
        params.intensity_floor_hz,
    )

    f0 = get_f0_contour(
        sound,
        pitch_time_step_s=params.pitch_time_step_s,
        pitch_floor_hz=params.pitch_floor_hz,
        pitch_ceiling_hz=params.pitch_ceiling_hz,
    )

    silence_threshold_db = get_silence_threshold_db(
        intensity, params.silence_threshold_relative_db
    )

    nucleus_candidates = detect_nucleus_candidates(
        intensity,
        silence_threshold_db,
        params.min_nucleus_prominence_db,
    )

    nuclei = filter_voiced_nuclei(
        nucleus_candidates, f0, params.voicing_half_window_s, params.min_voiced_fraction
    )

    intervals = estimate_syllable_intervals(
        nuclei,
        intensity,
        sound.duration,
        params.min_pause_duration_s,
        silence_threshold_db,
    )

    syllables = measure_syllables(
        intervals, samples, sample_rate, f0, params.pitch_percentile
    )

    prominences = calculate_local_prominences(
        syllables,
        params.prominence_neighbor_radius,
        params.prominence_baseline_quantile,
    )

    normalized_prominences = normalize_prominences(prominences)

    weights = _normalize_weights(
        params.pitch_weight,
        params.intensity_weight,
        params.duration_weight,
    )

    scores = calculate_stress_scores(
        normalized_prominences,
        weights,
    )

    stress_flags = classify_stresses(
        prominences,
        scores,
        params.stress_score_threshold,
        params.require_pitch_intensity_cooccurrence,
    )

    return [syllable for syllable, stressed in zip(syllables, stress_flags) if stressed]
