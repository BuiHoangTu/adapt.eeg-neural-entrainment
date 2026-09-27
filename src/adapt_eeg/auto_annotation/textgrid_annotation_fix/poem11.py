from adapt_eeg.auto_annotation.textgrid_annotation_fix.common import (
    print_line_transcription,
)

textgrid_eol_indices = [
    8,
    19,
    29,
    39,
    49,
    60,
    70,
    80,
    90,
    101,
    112,
    122,
    133,
    145,
]


def main():
    print_line_transcription(10, textgrid_eol_indices)


if __name__ == "__main__":
    main()
