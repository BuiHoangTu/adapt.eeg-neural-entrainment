import numpy as np

from adapt_eeg.auto_annotation.classes import (
    Contour,
    StressEvidence,
    StressIdentifyParams,
    Syllable,
    SyllablizedAudio,
)

DEFAULT_PARAMS = StressIdentifyParams()


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
    duration_prominences = _measure_duration_prominences(
        syllablized_audio.syllables,
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


def classify_stresses(
    evidence: list[StressEvidence],
    scores: list[float],
    min_pitch_prominence_hz: float,
    min_intensity_prominence_db: float,
    stress_score_threshold: float,
) -> list[bool]:

    stressed = []

    for (pitch_prominence, intensity_prominence, _), score in zip(evidence, scores):
        stressed.append(
            bool(
                np.isfinite(score)
                and np.isfinite(pitch_prominence)
                and np.isfinite(intensity_prominence)
                and pitch_prominence > min_pitch_prominence_hz
                and intensity_prominence > min_intensity_prominence_db
                and score >= stress_score_threshold
            )
        )

    return stressed


def identify_stressed_syllables(
    syllablized_audio: SyllablizedAudio,
    params: StressIdentifyParams = DEFAULT_PARAMS,
) -> list[Syllable]:
    evidence = calculate_stress_evidence(
        syllablized_audio,
        params.prominence_neighbor_half_window_s,
        params.prominence_min_nucleus_distance_s,
        params.duration_neighbor_radius,
        params.pitch_range_hz[0],
    )

    normalized_evidence = normalize_stress_evidence(evidence)

    weights = _normalize_weights(params.stress_weights)

    scores = calculate_stress_scores(
        normalized_evidence,
        weights,
    )

    stress_flags = classify_stresses(
        evidence,
        scores,
        params.min_pitch_prominence_hz,
        params.min_intensity_prominence_db,
        params.stress_score_threshold,
    )

    return [
        syllable
        for syllable, stressed in zip(syllablized_audio.syllables, stress_flags)
        if stressed
    ]


def detect_stressed_syllables(
    syllablized_audio: SyllablizedAudio,
    params: StressIdentifyParams = DEFAULT_PARAMS,
) -> list[Syllable]:
    return identify_stressed_syllables(
        syllablized_audio,
        params,
    )
