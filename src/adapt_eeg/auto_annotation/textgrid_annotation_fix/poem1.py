from adapt_eeg.auto_annotation.textgrid_annotation_fix.common import (
    print_line_transcription,
)

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


def main():
    print_line_transcription(0, textgrid_eol_indices)


if __name__ == "__main__":
    main()
