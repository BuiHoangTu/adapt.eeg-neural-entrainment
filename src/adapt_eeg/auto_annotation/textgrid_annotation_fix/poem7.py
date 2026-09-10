from adapt_eeg.auto_annotation.textgrid_annotation_fix.common import (
    derive_eol_times_from_audio,
)
from adapt_eeg.constants import POEMS_CONFIG

textgrid_eol_indices = [
    7,
    18,
    28,
    38,
    48,
    58,
    68,
]

def main() -> list[float]:
    poem_config = POEMS_CONFIG[6]
    eol_times = derive_eol_times_from_audio(
        poem_config["audio_url"],
        poem_config["textgrid_url"],
        textgrid_eol_indices,
        intro_end_ms=poem_config["intro_end"],
    )
    for line_index, eol_time in enumerate(eol_times):
        print(f"line {line_index}: {eol_time:.6f}s")
    return eol_times


if __name__ == "__main__":
    main()
