import numpy as np

from adapt_eeg.auto_annotation.classes import StressEvidence, Syllable


class InsufficientEligibleSyllablesError(ValueError):
    def __init__(
        self,
        line_index: int,
        required_count: int,
        syllables: list[Syllable],
        evidence: list[StressEvidence],
        min_pitch_prominence_hz: float,
        min_intensity_prominence_db: float,
        syllable_offset: int = 0,
    ):
        eligible_count = sum(
            1
            for pitch_prominence, intensity_prominence, _ in evidence
            if np.isfinite(pitch_prominence)
            and np.isfinite(intensity_prominence)
            and pitch_prominence > min_pitch_prominence_hz
            and intensity_prominence > min_intensity_prominence_db
        )

        super().__init__(
            f"Line {line_index} has {eligible_count} syllables passing the minimum "
            f"pitch/intensity prominence filters; need at least {required_count}."
        )
        self.line_index = line_index
        self.required_count = required_count
        self.eligible_count = eligible_count
        self.syllables = syllables
        self.evidence = evidence
        self.syllable_data = [
            {
                "syllable_index": syllable_offset + line_syllable_index,
                "line_syllable_index": line_syllable_index,
                "start": syllable.start,
                "nucleus": syllable.nucleus,
                "end": syllable.end,
                "pitch_hz": syllable.pitch_hz,
                "intensity_db": syllable.intensity_db,
                "duration_s": syllable.duration_s,
                "pitch_prominence_hz": pitch_prominence,
                "intensity_prominence_db": intensity_prominence,
                "duration_prominence_s": duration_prominence,
                "passes_pitch_filter": bool(
                    np.isfinite(pitch_prominence)
                    and pitch_prominence > min_pitch_prominence_hz
                ),
                "passes_intensity_filter": bool(
                    np.isfinite(intensity_prominence)
                    and intensity_prominence > min_intensity_prominence_db
                ),
            }
            for line_syllable_index, (
                syllable,
                (
                    pitch_prominence,
                    intensity_prominence,
                    duration_prominence,
                ),
            ) in enumerate(zip(syllables, evidence))
        ]