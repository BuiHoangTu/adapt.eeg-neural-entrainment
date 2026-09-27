import logging
from collections.abc import Sequence

from adapt_eeg.auto_annotation.classes import (
    LinedAudio,
    Syllable,
    SyllableIdentifyParams,
    SyllablizedAudio,
    TranscriptedSyllable,
)
from adapt_eeg.auto_annotation.syllable_from_textgrid import load_textgrid_syllable
from adapt_eeg.auto_annotation.syllable_identifier import syllablize_audio
from adapt_eeg.constants import POEMS_CONFIG

logger = logging.getLogger(__name__)


def _alignment_warning(
    *,
    start: float,
    mid: float,
    end: float,
    left_transcription: str,
    right_transcription: str,
    candidate_count: int,
) -> str:
    return (
        "Could not align the manual line boundary "
        f"{left_transcription!r} | {right_transcription!r}: found "
        f"{candidate_count} automatic syllables in "
        f"start={start:.6f}s, mid={mid:.6f}s, end={end:.6f}s; need at least 2."
    )


def _overlaps(syllable: Syllable, start: float, end: float) -> bool:
    return syllable.end > start and syllable.start < end


def _nearest_pair(
    candidates: list[Syllable],
    mid: float,
) -> tuple[Syllable, Syllable]:
    candidates = sorted(candidates, key=lambda syllable: syllable.nucleus)
    if len(candidates) == 2:
        return candidates[0], candidates[1]

    left = [syllable for syllable in candidates if syllable.nucleus <= mid]
    right = [syllable for syllable in candidates if syllable.nucleus > mid]
    if left and right:
        return left[-1], right[0]

    # A rough manual boundary can fall wholly to one side of the automatic
    # nuclei. In that case, retain the two closest nuclei and restore time order.
    nearest = sorted(
        candidates,
        key=lambda syllable: (abs(syllable.nucleus - mid), syllable.nucleus),
    )[:2]
    nearest.sort(key=lambda syllable: syllable.nucleus)
    return nearest[0], nearest[1]


def derive_eol_times(
    manual_syllables: Sequence[TranscriptedSyllable],
    automatic_syllables: Sequence[Syllable],
    textgrid_eol_indices: Sequence[int],
) -> list[float]:
    """Align annotated TextGrid line ends to automatic syllable boundaries."""
    eol_times = []

    for index in textgrid_eol_indices:
        if index < 0 or index >= len(manual_syllables):
            raise IndexError(
                f"TextGrid EOL index {index} is outside the manual syllable list "
                f"of length {len(manual_syllables)}."
            )
        if index == len(manual_syllables) - 1:
            continue

        manual_left = manual_syllables[index]
        manual_right = manual_syllables[index + 1]
        start = manual_left.start
        mid = manual_left.end
        end = manual_right.end
        candidates = [
            syllable
            for syllable in automatic_syllables
            if _overlaps(syllable, start, end)
        ]

        if len(candidates) < 2:
            logger.warning(
                _alignment_warning(
                    start=start,
                    mid=mid,
                    end=end,
                    left_transcription=manual_left.transcription,
                    right_transcription=manual_right.transcription,
                    candidate_count=len(candidates),
                )
            )
            eol_times.append(mid)
            continue

        automatic_left, _ = _nearest_pair(candidates, mid)
        eol_times.append(automatic_left.end)

    return eol_times


def derive_eol_times_from_audio(
    audio_input,
    textgrid_input,
    textgrid_eol_indices: Sequence[int],
    *,
    intro_end_ms: int,
    params: SyllableIdentifyParams = SyllableIdentifyParams(),
) -> list[float]:
    """Load both alignments, remove the intro, and derive automatic EOL times."""
    return derive_lined_audio(
        audio_input,
        textgrid_input,
        textgrid_eol_indices,
        intro_end_ms=intro_end_ms,
        params=params,
    ).line_boundaries


def derive_lined_audio(
    audio_input,
    textgrid_input,
    textgrid_eol_indices: Sequence[int],
    *,
    intro_end_ms: int,
    params: SyllableIdentifyParams = SyllableIdentifyParams(),
) -> LinedAudio:
    """Load both alignments and return automatic audio with derived line boundaries."""
    manual = load_textgrid_syllable(
        audio_input,
        textgrid_input,
        intensity_floor_hz=params.intensity_floor_hz,
        pitch_timestep_s=params.pitch_time_step_s,
        pitch_range_hz=params.pitch_range_hz,
        silence_threshold_relative_db=params.silence_threshold_relative_db,
        syllable_pitch_percentile=params.pitch_percentile,
        min_nucleus_prominence_db=params.min_nucleus_prominence_db,
    )[intro_end_ms:]
    automatic = syllablize_audio(audio_input, params)[intro_end_ms:]
    return with_line_boundaries(
        automatic,
        derive_eol_times(
            manual.syllables,  # type: ignore[arg-type]
            automatic.syllables,
            textgrid_eol_indices,
        ),
    )


def with_line_boundaries(
    syllablized_audio: SyllablizedAudio,
    eol_times: Sequence[float],
) -> LinedAudio:
    """Return an existing automatic alignment carrying the derived boundaries."""
    return LinedAudio.from_syllablized_audio(
        syllablized_audio,
        list(eol_times),
    )


def print_line_transcription(id, textgrid_eol_indices):
    poem_config = POEMS_CONFIG[id]
    eol_times = derive_eol_times_from_audio(
        poem_config["audio_url"],
        poem_config["textgrid_url"],
        textgrid_eol_indices,
        intro_end_ms=poem_config["intro_end"],
    )

    syl_audio = load_textgrid_syllable(
        poem_config["audio_url"],
        poem_config["textgrid_url"],
        50,
        0.01,
        (75, 500),
        -25,
        90,
        1,
    )
    syl_audio = syl_audio[poem_config["intro_end"] :]
    transcriptions = [s.transcription for s in syl_audio.syllables]

    previous_index = 0
    for line_index, (eol_time, eol_idx) in enumerate(
        zip(eol_times, textgrid_eol_indices)
    ):
        print(
            f"line {line_index}: {eol_time:.6f}s: {transcriptions[previous_index: (eol_idx + 1)]}"
        )
        previous_index = eol_idx + 1

    # The final TextGrid EOL index is not included in ``eol_times`` because
    # there is no following syllable against which to derive a boundary.
    # Still print the final line for inspection.
    if previous_index < len(syl_audio.syllables):
        print(
            f"line {len(eol_times)}: {syl_audio.syllables[-1].end:.6f}s: "
            f"{transcriptions[previous_index:]}"
        )
