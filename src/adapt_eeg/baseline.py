"""Legacy Poem 1 configuration used by the original ADAPT analysis."""

REGULAR_LINES = frozenset({3, 4, 10, 15, 23, 27, 30})
IRREGULAR_LINES = frozenset({1, 2, 5, 6, 7, 8, 9})
SELECTED_LINES = REGULAR_LINES | IRREGULAR_LINES

FREQUENCY_HZ = 2.0
WINDOW_DURATION_S = 0.3
N_LINE_INTERVALS = 30
CHANNELS_OF_INTEREST = ("Pz", "C4", "T7")
