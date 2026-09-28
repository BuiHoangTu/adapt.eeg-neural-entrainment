from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import parselmouth

from adapt_eeg.constants import POEMS_CONFIG

TRIGGER_DESCRIPTION = "Trigger#90"
MAX_DURATION_ERROR_S = 0.1


class InvalidPoemRecordingError(ValueError):
    """Raised when a continuous EEG file cannot be aligned to its poem audio."""


@dataclass(frozen=True)
class PoemRecording:
    poem: int
    participant: int
    path: Path
    raw: mne.io.BaseRaw
    trigger_start_s: float
    trigger_end_s: float
    usable_start_s: float
    usable_end_s: float
    expected_audio_duration_s: float
    duration_error_s: float


def aligned_line_end_s(recording: PoemRecording, line_end_s: float) -> float:
    """Map an audio-relative line end to EEG, preserving the true poem end."""
    if np.isclose(
        line_end_s,
        recording.expected_audio_duration_s,
        atol=MAX_DURATION_ERROR_S,
        rtol=0.0,
    ):
        return recording.usable_end_s
    return recording.usable_start_s + line_end_s


def discover_ica_cleaned_files(data_root: Path) -> list[Path]:
    """Find component-cleaned ``ICA.set`` and ``ICA final.set`` recordings."""
    return sorted(
        path
        for path in data_root.rglob("*.set")
        if re.search(r"\bica(?: final)?\.set$", path.name, re.IGNORECASE)
    )


def ids_from_path(path: Path) -> tuple[int, int]:
    poem_match = re.search(r"poem(\d+)", str(path), re.IGNORECASE)
    participant_match = re.search(r"part(?:icipant)?(\d+)", str(path), re.IGNORECASE)
    if poem_match is None or participant_match is None:
        raise InvalidPoemRecordingError(
            f"Cannot parse poem and participant IDs from {path}"
        )
    return int(poem_match.group(1)), int(participant_match.group(1))


def unique_recording_paths(paths: list[Path]) -> tuple[list[Path], dict[tuple[int, int], list[Path]]]:
    grouped: dict[tuple[int, int], list[Path]] = {}
    for path in paths:
        grouped.setdefault(ids_from_path(path), []).append(path)
    duplicates = {key: values for key, values in grouped.items() if len(values) > 1}
    unique = [values[0] for values in grouped.values() if len(values) == 1]
    return sorted(unique), duplicates


def load_poem_recording(path: Path, *, preload: bool) -> PoemRecording:
    poem, participant = ids_from_path(path)
    if poem < 1 or poem > len(POEMS_CONFIG):
        raise InvalidPoemRecordingError(f"{path}: poem {poem} has no configuration")

    raw = mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    trigger_onsets = sorted(
        float(onset)
        for onset, description in zip(
            raw.annotations.onset,
            raw.annotations.description,
            strict=True,
        )
        if description == TRIGGER_DESCRIPTION
    )
    if len(trigger_onsets) != 2:
        raise InvalidPoemRecordingError(
            f"{path}: expected exactly two {TRIGGER_DESCRIPTION!r} annotations, "
            f"found {len(trigger_onsets)}"
        )

    trigger_start_s, trigger_end_s = trigger_onsets
    config = POEMS_CONFIG[poem - 1]
    intro_s = config["intro_end"] / 1000.0
    usable_start_s = trigger_start_s + intro_s
    if usable_start_s >= trigger_end_s:
        raise InvalidPoemRecordingError(
            f"{path}: intro-adjusted start {usable_start_s:.6f}s is not before "
            f"end trigger {trigger_end_s:.6f}s"
        )

    expected_audio_duration_s = (
        parselmouth.Sound(str(config["audio_url"])).duration - intro_s
    )
    usable_duration_s = trigger_end_s - usable_start_s
    duration_error_s = usable_duration_s - expected_audio_duration_s
    if abs(duration_error_s) > MAX_DURATION_ERROR_S:
        raise InvalidPoemRecordingError(
            f"{path}: usable trigger duration {usable_duration_s:.6f}s differs "
            f"from intro-trimmed audio duration {expected_audio_duration_s:.6f}s "
            f"by {duration_error_s:.6f}s"
        )

    return PoemRecording(
        poem=poem,
        participant=participant,
        path=path,
        raw=raw,
        trigger_start_s=trigger_start_s,
        trigger_end_s=trigger_end_s,
        usable_start_s=usable_start_s,
        usable_end_s=trigger_end_s,
        expected_audio_duration_s=expected_audio_duration_s,
        duration_error_s=duration_error_s,
    )
