from adapt_eeg.auto_annotation.syllable_from_textgrid import load_textgrid_syllable
from adapt_eeg.auto_annotation.textgrid_annotation_fix.common import (
    derive_eol_times_from_audio,
)
from adapt_eeg.constants import POEMS_CONFIG


textgrid_eol_indices = [
    8,
    19,
    28,
    37,
    47,
    56,
    66,
    76,
    85,
    95,
    105,
    116,
    126,
    136,
    146,
    156,
    167,
    176,
    188,
    198,
    209,
    218,
    228,
    239,
    249,
    260,
    270,
    279,
    289,
    299,
    309,
    319,
]


def main() -> list[float]:
    poem_config = POEMS_CONFIG[0]
    eol_times = derive_eol_times_from_audio(
        poem_config["audio_url"],
        poem_config["textgrid_url"],
        textgrid_eol_indices,
        intro_end_ms=poem_config["intro_end"],
    )

    syl_audio = load_textgrid_syllable(poem_config["audio_url"],
            poem_config["textgrid_url"], 50, 0.01, (75, 500), -25, 90, 1)
    syl_audio = syl_audio[poem_config["intro_end"]:]
    transcriptions = [s.transcription for s in syl_audio.syllables]

    previous_index = 0
    for line_index, (eol_time, eol_idx) in enumerate(zip(eol_times, textgrid_eol_indices)):
        print(f"line {line_index}: {eol_time:.6f}s: {transcriptions[previous_index: (eol_idx + 1)]}")
        previous_index = eol_idx + 1
    return eol_times


if __name__ == "__main__":
    main()
