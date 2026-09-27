from adapt_eeg.auto_annotation.textgrid_annotation_fix.common import (
    print_line_transcription,
)

# first line missing from audio 
textgrid_eol_indices = [
    7,
    18,
    28,
    38,
    48,
    58,
    68,
]


def main():
    print_line_transcription(6, textgrid_eol_indices)


if __name__ == "__main__":
    main()
