from praatio import textgrid
from praatio.data_classes.textgrid import Textgrid

from adapt_eeg.auto_annotation.classes import (
    Syllable,
    SyllablizedAudio,
    TranscriptedSyllable,
)
from adapt_eeg.auto_annotation.syllable_identifier import (
    _load_audio,
    detect_nucleus_candidates,
    get_f0_contour,
    get_intensity_contour,
    get_silence_threshold_db,
    measure_syllables,
)

PAUSE_ANNO = ["", "#"]


def _load_textgrid(tg):
    if isinstance(tg, Textgrid):
        return tg
    else:
        return textgrid.openTextgrid(tg, True)


def _nucleus_from_interval(start, end, intensity, nucleus_candidates):
    candidates = [
        nucleus for nucleus in nucleus_candidates if start <= nucleus.time <= end
    ]
    if candidates:
        return max(candidates, key=lambda nucleus: nucleus.intensity_db)

    raise ValueError(
        f"No detected nucleus candidate found in TextGrid interval {start:.3f}-{end:.3f}s"
    )

    # Fallback: choose the local intensity peak inside the TextGrid interval.
    # Keep this disabled until we decide whether non-detected nuclei should be allowed.
    # mask = (intensity.times >= start) & (intensity.times <= end)
    # indices = np.flatnonzero(mask)
    #
    # if len(indices) == 0:
    #     nearest_index = int(np.argmin(np.abs(intensity.times - ((start + end) / 2.0))))
    #     return Nucleus(
    #         time=float(intensity.times[nearest_index]),
    #         intensity_db=float(intensity.values[nearest_index]),
    #         prominence_db=np.nan,
    #     )
    #
    # local_values = intensity.values[indices]
    # finite_values = local_values[np.isfinite(local_values)]
    #
    # if len(finite_values) == 0:
    #     nearest_index = indices[len(indices) // 2]
    #     return Nucleus(
    #         time=float(intensity.times[nearest_index]),
    #         intensity_db=np.nan,
    #         prominence_db=np.nan,
    #     )
    #
    # peak_local_index = int(np.nanargmax(local_values))
    # peak_index = indices[peak_local_index]
    # peak_intensity = float(intensity.values[peak_index])
    #
    # return Nucleus(
    #     time=float(intensity.times[peak_index]),
    #     intensity_db=peak_intensity,
    #     prominence_db=peak_intensity - float(np.nanmin(local_values)),
    # )


def load_text_grid_syllable(
    audio,
    tg,
    intensity_floor_hz,
    pitch_timestep_s,
    pitch_range_hz,
    silence_threshold_relative_db,
    syllable_pitch_percentile,
    min_nucleus_prominence_db=0.0,
):
    sound, audio, samples, sample_rate = _load_audio(audio)
    intensity = get_intensity_contour(
        sound,
        intensity_floor_hz,
    )
    f0 = get_f0_contour(
        sound,
        pitch_timestep_s,
        pitch_range_hz,
    )
    silence_threshold_db = get_silence_threshold_db(
        intensity,
        silence_threshold_relative_db,
    )

    nucleus_candidates = detect_nucleus_candidates(
        intensity,
        silence_threshold_db,
        min_nucleus_prominence_db,
    )

    tg = _load_textgrid(tg)
    tier = tg.getTier(tg.tierNames[0])

    intervals = []
    labels = []
    for start, end, label in tier.entries:  # type: ignore
        if label in PAUSE_ANNO:
            continue

        nucleus = _nucleus_from_interval(start, end, intensity, nucleus_candidates)
        intervals.append((float(start), nucleus, float(end)))
        labels.append(label)

    syllables = measure_syllables(
        intervals,
        audio,
        samples,
        sample_rate,
        f0,
        syllable_pitch_percentile,
    )

    syllables: list[Syllable] = [
        TranscriptedSyllable.from_syllable(syl, label)
        for syl, label in zip(syllables, labels)
    ]

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
