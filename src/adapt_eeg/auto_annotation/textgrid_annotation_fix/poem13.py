from adapt_eeg.auto_annotation.syllable_from_textgrid import load_textgrid_syllable
from adapt_eeg.auto_annotation.textgrid_annotation_fix.common import (
    derive_eol_times_from_audio,
)
from adapt_eeg.constants import POEMS_CONFIG

textgrid_eol_indices = [
    9,
    19,
    29,
    39,
    49,
    59,
    68,
    78,
    87,
    97,
    109,
    119,
    129,
    139,
    149,
    159,
    169,
    180,
    190,
    200,
    209,
    219,
    228,
    238,
    248,
    256,
    266,
    276,
    284,
    294,
    304,
    316,
    327,
    337,
    347,
    357,
    367,
    377,
    388,
    399,
    409,
    419,
    429,
    439,
    449,
    459,
    469,
    479,
    489,
    499,
    509,
    519,
    530,
    540,
    551,
    561,
    571,
    581,
    591,
    601,
    611,
    621,
    633,
    643,
]


def main() -> list[float]:
    poem_config = POEMS_CONFIG[12]
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
    return eol_times


if __name__ == "__main__":
    main()
