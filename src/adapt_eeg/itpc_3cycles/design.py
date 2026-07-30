from __future__ import annotations

from adapt_eeg.replicate.design import (
    ANALYSIS_LINES,
    CHANNELS_OF_INTEREST,
    EXPECTED_CHANNELS,
    FREQUENCIES_HZ,
    LanguageGroup,
    PaperSample,
    RhythmCondition,
    discover_samples,
    incomplete_participant_lines,
    missing_analysis_lines,
    participants_by_line,
    read_eeglab_sample,
    validate_design_coverage,
)

__all__ = [
    "ANALYSIS_LINES",
    "CHANNELS_OF_INTEREST",
    "EXPECTED_CHANNELS",
    "FREQUENCIES_HZ",
    "LanguageGroup",
    "PaperSample",
    "RhythmCondition",
    "discover_samples",
    "incomplete_participant_lines",
    "missing_analysis_lines",
    "participants_by_line",
    "read_eeglab_sample",
    "validate_design_coverage",
]
