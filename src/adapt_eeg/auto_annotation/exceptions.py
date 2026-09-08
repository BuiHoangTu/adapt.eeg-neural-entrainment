import numpy as np

from adapt_eeg.auto_annotation.classes import StressEvidence, Syllable


class InsufficientEligibleSyllablesError(ValueError):
    def __init__(
        self,
        line_index: int,
        required_count: int,
        syllables: list[Syllable],
        evidence: list[StressEvidence],
        syllable_offset: int = 0,
    ):
        eligible_count = sum(1 for item in evidence if np.any(np.isfinite(item)))

        super().__init__(
            f"Line {line_index} has {eligible_count} scoreable syllables; "
            f"need at least {required_count}."
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
                "has_pitch_evidence": bool(np.isfinite(pitch_prominence)),
                "has_intensity_evidence": bool(np.isfinite(intensity_prominence)),
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
