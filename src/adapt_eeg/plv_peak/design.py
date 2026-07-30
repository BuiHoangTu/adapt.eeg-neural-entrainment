from __future__ import annotations

from adapt_eeg.replicate.design import (
    ANALYSIS_LINES,
    CHANNELS_OF_INTEREST,
    EXPECTED_CHANNELS,
    LanguageGroup,
    PaperSample,
    RhythmCondition,
    discover_samples,
    read_eeglab_sample,
    validate_design_coverage,
)

FREQUENCIES_HZ = (2.0,)

__all__ = [
    "ANALYSIS_LINES",
    "CHANNELS_OF_INTEREST",
    "EXPECTED_CHANNELS",
    "FREQUENCIES_HZ",
    "LanguageGroup",
    "PaperSample",
    "RhythmCondition",
    "discover_samples",
    "read_eeglab_sample",
    "validate_design_coverage",
]
