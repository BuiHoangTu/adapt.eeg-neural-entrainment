from pathlib import Path

import numpy as np
import parselmouth
from pydub import AudioSegment
from scipy.signal import find_peaks

from adapt_eeg.auto_annotation.classes import (
    Contour,
    Nucleus,
    Syllable,
    SyllableIdentifyParams,
    SyllableInterval,
    SyllablizedAudio,
)

DEFAULT_PARAMS = SyllableIdentifyParams()





def _load_audio(
    audio_input: str | Path | AudioSegment,
) -> tuple[parselmouth.Sound, AudioSegment, np.ndarray, int]:
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

    return sound, audio, samples, sample_rate


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
    pitch_time_step_s: float,
    pitch_range_hz: tuple[float, float],
) -> Contour:

    pitch_floor_hz, pitch_ceiling_hz = pitch_range_hz

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


def get_silence_threshold_db(
    intensity: Contour,
    silence_threshold_relative_db: float,
) -> float:
    reference_db = np.nanquantile(intensity.values, 0.99)

    return reference_db + silence_threshold_relative_db  # type: ignore


def detect_nucleus_candidates(
    intensity: Contour,
    silence_threshold_db: float,
    min_nucleus_prominence_db: float,
) -> list[Nucleus]:

    if len(intensity.values) == 0:
        return []

    peak_indices, properties = find_peaks(
        intensity.values,
        height=silence_threshold_db,
        prominence=min_nucleus_prominence_db,
    )

    return [
        Nucleus(
            time=float(intensity.times[index]),
            intensity_db=float(intensity.values[index]),
            prominence_db=float(prominence),
        )
        for index, prominence in zip(
            peak_indices,
            properties["prominences"],
        )
    ]


def filter_voiced_nuclei(
    nuclei: list[Nucleus],
    f0: Contour,
    voicing_half_window_s: float,
    min_voiced_fraction: float,
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

    for i in range(len(nucleus_times) + 1):
        if i == 0:
            left_nucleus = 0.0
            right_nucleus = nucleus_times[0]
        elif i == len(nucleus_times):
            left_nucleus = nucleus_times[-1]
            right_nucleus = audio_duration_s
        else:
            left_nucleus = nucleus_times[i - 1]
            right_nucleus = nucleus_times[i]

        pause = _find_pause_between(
            intensity,
            left_nucleus,
            right_nucleus,
            silence_threshold_db,
            min_pause_duration_s,
        )

        if pause is not None:
            pause_start, pause_end = pause

            if i == 0:
                starts[0] = pause_end
            elif i == len(nucleus_times):
                ends[-1] = pause_start
            else:
                ends[i - 1] = pause_start
                starts[i] = pause_end

        else:
            boundary = _lowest_intensity_time(
                intensity,
                left_nucleus,
                right_nucleus,
            )

            if i == 0:
                starts[0] = boundary
            elif i == len(nucleus_times):
                ends[-1] = boundary
            else:
                # Normal case:
                # adjacent syllables share the intensity valley.
                ends[i - 1] = boundary
                starts[i] = boundary

    return [
        (
            float(starts[i]),
            nuclei[i],
            float(ends[i]),
        )
        for i in range(len(nuclei))
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
    audio: AudioSegment,
    samples: np.ndarray,
    sample_rate: int,
    f0: Contour,
    pitch_percentile: float,
) -> list[Syllable]:

    syllables = []

    for start, nucleus, end in intervals:
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
                _nucleus=nucleus,
                end=end,
                _audio=audio[round(start * 1000) : round(end * 1000)],  # type: ignore  # only return generator if slicing has step
                pitch_hz=pitch,
                intensity_db=intensity,
            )
        )

    return syllables


def syllablize_audio(
    audio_input: str | Path | AudioSegment,
    params: SyllableIdentifyParams = DEFAULT_PARAMS,
) -> SyllablizedAudio:
    sound, audio, samples, sample_rate = _load_audio(audio_input)

    intensity = get_intensity_contour(
        sound,
        params.intensity_floor_hz,
    )

    f0 = get_f0_contour(
        sound,
        params.pitch_time_step_s,
        params.pitch_range_hz,
    )

    silence_threshold_db = get_silence_threshold_db(
        intensity,
        params.silence_threshold_relative_db,
    )

    nucleus_candidates = detect_nucleus_candidates(
        intensity,
        silence_threshold_db,
        params.min_nucleus_prominence_db,
    )

    nuclei = filter_voiced_nuclei(
        nucleus_candidates,
        f0,
        params.voicing_half_window_s,
        params.min_voiced_fraction,
    )

    intervals = estimate_syllable_intervals(
        nuclei,
        intensity,
        sound.duration,
        params.min_pause_duration_s,
        silence_threshold_db,
    )

    syllables = measure_syllables(
        intervals,
        audio,
        samples,
        sample_rate,
        f0,
        params.pitch_percentile,
    )

    return SyllablizedAudio(
        sound=sound,
        audio=audio,
        samples=samples,
        sample_rate=sample_rate,
        intensity=intensity,
        f0=f0,
        silence_threshold_db=silence_threshold_db,
        syllables=syllables,
    )
