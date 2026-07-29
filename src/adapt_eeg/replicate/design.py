from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
import re

import mne

from adapt_eeg.constants import (
    ENGLISH_COMPETENCE_PARTICIPANTS,
    IRREGULAR_LINES,
    N_CHANNELS,
    REGULAR_LINES,
)

ANALYSIS_LINES = REGULAR_LINES | IRREGULAR_LINES
FREQUENCIES_HZ = (2.0, 4.0)
CHANNELS_OF_INTEREST = ("Pz", "C4", "T7")
EXPECTED_CHANNELS = N_CHANNELS
DATASET_FILENAME_RE = re.compile(r"p(?P<participant>\d+)\.set$")
VERSE_DIR_RE = re.compile(r"v(?P<line>\d+)$")


class RhythmCondition(str, Enum):
    REGULAR = "regular"
    IRREGULAR = "irregular"

    @classmethod
    def from_line(cls, line: int) -> RhythmCondition:
        if line in REGULAR_LINES:
            return cls.REGULAR
        if line in IRREGULAR_LINES:
            return cls.IRREGULAR
        raise ValueError(f"Line {line} is not part of the paper analysis subset")


class LanguageGroup(str, Enum):
    ENGLISH = "english_competence"
    NON_ENGLISH = "non_english_competence"

    @classmethod
    def from_participant(cls, participant: int) -> LanguageGroup:
        if participant in ENGLISH_COMPETENCE_PARTICIPANTS:
            return cls.ENGLISH
        return cls.NON_ENGLISH


@dataclass(frozen=True)
class PaperSample:
    participant: int
    line: int
    set_path: Path
    rhythm_condition: RhythmCondition
    language_group: LanguageGroup


def discover_samples(data_root: Path) -> list[PaperSample]:
    samples: list[PaperSample] = []
    for verse_dir in sorted(data_root.glob("v*")):
        verse_match = VERSE_DIR_RE.fullmatch(verse_dir.name)
        if verse_match is None:
            logging.info("Skipping non-verse directory: %s", verse_dir)
            continue

        line = int(verse_match.group("line"))
        if line not in ANALYSIS_LINES:
            continue

        dataset_dir = verse_dir / "4hz_datasets"
        for set_path in sorted(dataset_dir.glob("p*.set")):
            participant_match = DATASET_FILENAME_RE.fullmatch(set_path.name)
            if participant_match is None:
                logging.info("Skipping non-participant EEGLAB file name: %s", set_path)
                continue

            participant = int(participant_match.group("participant"))
            samples.append(
                PaperSample(
                    participant=participant,
                    line=line,
                    set_path=set_path,
                    rhythm_condition=RhythmCondition.from_line(line),
                    language_group=LanguageGroup.from_participant(participant),
                )
            )
    return samples


def missing_analysis_lines(samples: list[PaperSample]) -> tuple[int, ...]:
    present_lines = {sample.line for sample in samples}
    return tuple(sorted(ANALYSIS_LINES - present_lines))


def participants_by_line(samples: list[PaperSample]) -> dict[int, set[int]]:
    participants: dict[int, set[int]] = defaultdict(set)
    for sample in samples:
        participants[sample.line].add(sample.participant)
    return dict(participants)


def incomplete_participant_lines(samples: list[PaperSample]) -> dict[int, list[int]]:
    by_line = participants_by_line(samples)
    all_participants = set().union(*by_line.values()) if by_line else set()
    return {
        line: sorted(all_participants - participants)
        for line, participants in sorted(by_line.items())
        if participants != all_participants
    }


def validate_design_coverage(samples: list[PaperSample]) -> None:
    missing_lines = missing_analysis_lines(samples)
    if missing_lines:
        raise ValueError(
            "Missing source data for paper analysis lines: "
            + ", ".join(str(line) for line in missing_lines)
        )

    incomplete_lines = incomplete_participant_lines(samples)
    if incomplete_lines:
        details = "; ".join(
            f"line {line} missing participants {participants}"
            for line, participants in incomplete_lines.items()
        )
        raise ValueError(f"Incomplete participant coverage across analysis lines: {details}")


def read_eeglab_sample(sample: PaperSample) -> mne.io.BaseRaw:
    return mne.io.read_raw_eeglab(sample.set_path, preload=True, verbose="ERROR")
