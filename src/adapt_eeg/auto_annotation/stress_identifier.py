import numpy as np

from adapt_eeg.auto_annotation.classes import (
    Contour,
    StressEvidence,
    StressIdentifyParams,
    Syllable,
    SyllablizedAudio,
)
from adapt_eeg.auto_annotation.exceptions import InsufficientEligibleSyllablesError

DEFAULT_PARAMS = StressIdentifyParams()


def _line_slices(
    syllables: list[Syllable],
    line_boundaries: tuple[float, ...],
) -> list[slice]:
    if any(
        right <= left for left, right in zip(line_boundaries, line_boundaries[1:])
    ):
        raise ValueError("Line boundaries must be strictly increasing.")
    if any(
        right.nucleus < left.nucleus for left, right in zip(syllables, syllables[1:])
    ):
        raise ValueError("Syllables must be ordered by nucleus time.")

    nuclei = np.asarray([syllable.nucleus for syllable in syllables])
    indices = np.searchsorted(nuclei, line_boundaries, side="left")
    stops = [*indices.tolist(), len(syllables)]
    starts = [0, *indices.tolist()]
    return [slice(start, stop) for start, stop in zip(starts, stops)]


def _neighbor_values(
    contour: Contour,
    syllable: Syllable,
    neighbor_half_window_s: float,
    min_nucleus_distance_s: float,
) -> np.ndarray:
    around_start = np.abs(contour.times - syllable.start) <= neighbor_half_window_s
    around_end = np.abs(contour.times - syllable.end) <= neighbor_half_window_s
    far_from_nucleus = (
        np.abs(contour.times - syllable.nucleus) >= min_nucleus_distance_s
    )

    mask = (around_start | around_end) & far_from_nucleus

    values = contour.values[mask]

    return values[np.isfinite(values)]


def _measure_intensity_prominence(
    syllable: Syllable,
    intensity: Contour,
    neighbor_half_window_s: float,
    min_nucleus_distance_s: float,
    silence_threshold_db: float,
) -> float:
    neighbor_values = _neighbor_values(
        intensity,
        syllable,
        neighbor_half_window_s,
        min_nucleus_distance_s,
    )

    if len(neighbor_values) == 0:
        return float(syllable._nucleus.intensity_db - silence_threshold_db)

    return float(syllable._nucleus.intensity_db - np.nanmedian(neighbor_values))


def _measure_pitch_prominence(
    syllable: Syllable,
    f0: Contour,
    neighbor_half_window_s: float,
    min_nucleus_distance_s: float,
    pitch_floor_hz: float,
) -> float:
    if not np.isfinite(syllable.pitch_hz):
        return np.nan

    neighbor_values = _neighbor_values(
        f0,
        syllable,
        neighbor_half_window_s,
        min_nucleus_distance_s,
    )

    if len(neighbor_values) == 0:
        return float(syllable.pitch_hz - pitch_floor_hz)

    return float(syllable.pitch_hz - np.nanmedian(neighbor_values))


def _measure_duration_prominences(
    syllables: list[Syllable],
    duration_neighbor_radius: int,
) -> np.ndarray:
    durations = np.asarray(
        [syllable.duration_s for syllable in syllables],
        dtype=np.float64,
    )

    prominences = np.full(
        len(durations),
        np.nan,
        dtype=np.float64,
    )

    for i, duration in enumerate(durations):
        left = max(0, i - duration_neighbor_radius)
        right = min(len(durations), i + duration_neighbor_radius + 1)

        neighbors = np.concatenate(
            [
                durations[left:i],
                durations[i + 1 : right],
            ]
        )

        neighbors = neighbors[np.isfinite(neighbors)]

        if len(neighbors) == 0:
            continue

        prominences[i] = duration - np.median(neighbors)

    return prominences


def calculate_stress_evidence(
    syllablized_audio: SyllablizedAudio,
    prominence_neighbor_half_window_s: float,
    prominence_min_nucleus_distance_s: float,
    duration_neighbor_radius: int,
    pitch_floor_hz: float,
) -> list[StressEvidence]:
    duration_prominences = np.full(len(syllablized_audio.syllables), np.nan)
    for line_slice in _line_slices(
        syllablized_audio.syllables,
        syllablized_audio.line_boundaries,
    ):
        duration_prominences[line_slice] = _measure_duration_prominences(
            syllablized_audio.syllables[line_slice],
            duration_neighbor_radius,
        )

    return [
        (
            _measure_pitch_prominence(
                syllable,
                syllablized_audio.f0,
                prominence_neighbor_half_window_s,
                prominence_min_nucleus_distance_s,
                pitch_floor_hz,
            ),
            _measure_intensity_prominence(
                syllable,
                syllablized_audio.intensity,
                prominence_neighbor_half_window_s,
                prominence_min_nucleus_distance_s,
                syllablized_audio.silence_threshold_db,
            ),
            float(duration_prominence),
        )
        for syllable, duration_prominence in zip(
            syllablized_audio.syllables,
            duration_prominences,
        )
    ]


def _robust_zscore(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    result = np.full_like(values, np.nan)

    valid = np.isfinite(values)

    if not np.any(valid):
        return result

    valid_values = values[valid]

    median = np.median(valid_values)
    mad = np.median(np.abs(valid_values - median))

    scale = 1.4826 * mad

    if scale <= np.finfo(float).eps:
        scale = np.std(valid_values)

    if scale <= np.finfo(float).eps:
        result[valid] = 0.0
        return result

    result[valid] = (valid_values - median) / scale

    return result


def normalize_stress_evidence(
    evidence: list[StressEvidence],
) -> list[StressEvidence]:
    if not evidence:
        return []

    values = np.asarray(evidence, dtype=np.float64)

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
    weights_raw: tuple[float, float, float],
) -> tuple[float, float, float]:
    weights = np.asarray(weights_raw, dtype=np.float64)

    if np.any(weights < 0):
        raise ValueError("Stress weights cannot be negative.")

    total = weights.sum()

    if total <= 0:
        raise ValueError("At least one stress weight must be greater than zero.")

    weights /= total

    return float(weights[0]), float(weights[1]), float(weights[2])


def calculate_stress_scores(
    normalized_evidence: list[StressEvidence],
    weights: tuple[float, float, float],
) -> list[float]:
    weights_array = np.asarray(weights, dtype=np.float64)

    scores = []

    for features in normalized_evidence:
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


def classify_top_stresses_per_line(
    syllables: list[Syllable],
    evidence: list[StressEvidence],
    weights: tuple[float, float, float],
    line_boundaries: tuple[float, ...],
    stressed_syllables_per_line: int,
) -> list[bool]:
    if stressed_syllables_per_line < 0:
        raise ValueError("stressed_syllables_per_line cannot be negative.")
    if len(syllables) != len(evidence):
        raise ValueError("Syllables and stress evidence must have the same length.")

    stressed = [False] * len(evidence)

    for line_index, line_slice in enumerate(_line_slices(syllables, line_boundaries)):
        start = line_slice.start or 0
        end = line_slice.stop or 0
        line_syllables = syllables[start:end]
        line_evidence = evidence[start:end]
        normalized_evidence = normalize_stress_evidence(line_evidence)
        scores = calculate_stress_scores(normalized_evidence, weights)
        scoreable_indices = [
            index for index, score in enumerate(scores) if np.isfinite(score)
        ]

        if len(scoreable_indices) < stressed_syllables_per_line:
            raise InsufficientEligibleSyllablesError(
                line_index=line_index,
                required_count=stressed_syllables_per_line,
                syllables=line_syllables,
                evidence=line_evidence,
                syllable_offset=start,
            )

        # The fifth-highest score is the dynamic threshold. Stable index
        # ordering breaks ties so that every line has exactly the requested
        # number of stresses.
        selected_indices = sorted(
            scoreable_indices,
            key=lambda index: (-scores[index], index),
        )[:stressed_syllables_per_line]

        for index in selected_indices:
            stressed[start + index] = True

    return stressed


def flag_stressed_syllables(
    syllablized_audio: SyllablizedAudio,
    params: StressIdentifyParams = DEFAULT_PARAMS,
) -> list[bool]:
    evidence = calculate_stress_evidence(
        syllablized_audio,
        params.prominence_neighbor_half_window_s,
        params.prominence_min_nucleus_distance_s,
        params.duration_neighbor_radius,
        params.pitch_range_hz[0],
    )

    weights = _normalize_weights(params.stress_weights)

    stress_flags = classify_top_stresses_per_line(
        syllablized_audio.syllables,
        evidence,
        weights,
        syllablized_audio.line_boundaries,
        params.stressed_syllables_per_line,
    )

    return stress_flags


def detect_stressed_syllables(
    syllablized_audio: SyllablizedAudio,
    params: StressIdentifyParams = DEFAULT_PARAMS,
) -> list[Syllable]:
    stress_flags = flag_stressed_syllables(
        syllablized_audio,
        params,
    )

    return [
        syllable
        for syllable, stressed in zip(syllablized_audio.syllables, stress_flags)
        if stressed
    ]
