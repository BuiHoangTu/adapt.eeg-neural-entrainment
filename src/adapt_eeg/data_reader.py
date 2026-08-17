from __future__ import annotations

import logging
import re
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any

import mne
import numpy as np
from pydantic import BaseModel, ConfigDict

from adapt_eeg.constants import (
    ENGLISH_COMPETENCE_PARTICIPANTS,
    IRREGULAR_LINES,
    N_CHANNELS,
    REGULAR_LINES,
)


class RhythmType(str, Enum):
    REGULAR = "regular"
    IRREGULAR = "irregular"

    @classmethod
    def from_line(cls, line: int) -> RhythmType:
        if line in REGULAR_LINES:
            return cls.REGULAR
        if line in IRREGULAR_LINES:
            return cls.IRREGULAR
        raise ValueError(f"Line {line} is not assigned to a rhythm condition")


class LanguageUnderstanding(str, Enum):
    YES = "english_competence"
    NO = "non_english_competence"

    @classmethod
    def from_participant(cls, participant: int) -> LanguageUnderstanding:
        if participant in ENGLISH_COMPETENCE_PARTICIPANTS:
            return cls.YES
        return cls.NO


class SampleMetadata(BaseModel):
    participant: int
    line: int
    set_path: str
    fdt_path: str
    rhythm_type: RhythmType
    language_understanding: LanguageUnderstanding


def build_metadata(data_root: Path):
    FILE_RE = re.compile(r"part(?P<participant>\d+)v(?P<line>\d+)\.set$")

    samples: list[SampleMetadata] = []
    for set_path in data_root.glob("line*/*.set"):
        match = FILE_RE.search(set_path.name)
        if not match:
            logging.info("Skipping file %s. Does not match format", set_path.name)
            continue

        participant = int(match.group("participant"))
        line = int(match.group("line"))
        fdt_path = set_path.with_suffix(".fdt")

        samples.append(
            SampleMetadata(
                participant=participant,
                line=line,
                set_path=str(set_path),
                fdt_path=str(fdt_path),
                rhythm_type=RhythmType.from_line(line),
                language_understanding=LanguageUnderstanding.from_participant(
                    participant
                ),
            )
        )
    return samples


def merge_data(
    epochs: mne.Epochs,
    min_shift: int = 15,
    max_shift: int = 40,
    absolute_tolerance: float = 1e-12,
    normalized_tolerance: float = 1e-5,
    verbose: bool = False,
) -> tuple[mne.EpochsArray, dict]:
    """
    Merge sequential overlapping epochs into one reconstructed epoch.

    For each adjacent pair, integer shifts from `min_shift` through
    `max_shift` are tested. A shift is accepted only when the overlapping
    samples match up to floating-point numerical error.

    A constant offset is estimated separately for every channel because the
    original EEGLAB epochs were independently baseline-corrected.

    Parameters
    ----------
    epochs
        Sequential MNE epochs with shape:
        (n_epochs, n_channels, n_times).

    min_shift
        Smallest candidate shift in samples.

    max_shift
        Largest candidate shift in samples, inclusive.

    absolute_tolerance
        Absolute residual below which differences are treated as numerical
        noise.

    normalized_tolerance
        Maximum permitted residual relative to the robust signal amplitude.
        A value such as 1e-5 permits tiny floating-point discrepancies but
        rejects genuinely mismatched signals.

    verbose
        Print the selected shift and matching error for each pair.

    Returns
    -------
    merged_epochs
        An MNE EpochsArray containing one reconstructed epoch with shape:
        (1, n_channels, reconstructed_n_times).

    diagnostics
        Dictionary containing pairwise shifts, errors, offsets, and epoch
        start positions.

    Raises
    ------
    RuntimeError
        If any adjacent pair has no perfect matching shift in the requested
        range.
    """
    data = epochs.get_data(copy=True)

    if data.ndim != 3:
        raise ValueError("Expected data with shape " "(n_epochs, n_channels, n_times).")

    n_epochs, n_channels, n_times = data.shape
    sfreq = float(epochs.info["sfreq"])

    if n_epochs < 2:
        raise ValueError("At least two epochs are required.")

    if min_shift < 1:
        raise ValueError("min_shift must be at least 1.")

    if max_shift < min_shift:
        raise ValueError("max_shift must be >= min_shift.")

    if max_shift >= n_times:
        raise ValueError(
            f"max_shift must be less than the epoch length " f"({n_times} samples)."
        )

    aligned_epochs = data.copy()

    selected_shifts: list[int] = []
    selected_errors: list[float] = []
    selected_absolute_errors: list[float] = []
    selected_offsets: list[np.ndarray] = []

    for pair_index in range(n_epochs - 1):
        reference = aligned_epochs[pair_index]
        current_original = data[pair_index + 1]

        candidates: list[dict] = []

        for shift in range(min_shift, max_shift + 1):
            overlap_length = n_times - shift

            reference_overlap = reference[:, shift:]
            current_overlap = current_original[:, :overlap_length]

            # Correct the constant difference introduced by independent
            # baseline correction.
            offset = np.median(
                reference_overlap - current_overlap,
                axis=1,
            )

            current_corrected = current_overlap + offset[:, None]

            residual = reference_overlap - current_corrected

            max_absolute_error = float(np.max(np.abs(residual)))

            # Robust channel scale based on median absolute deviation.
            channel_scale = np.median(
                np.abs(
                    reference_overlap
                    - np.median(
                        reference_overlap,
                        axis=1,
                        keepdims=True,
                    )
                ),
                axis=1,
            )

            # Avoid division by zero for flat channels.
            channel_scale = np.maximum(
                channel_scale,
                absolute_tolerance,
            )

            channel_normalized_error = np.max(np.abs(residual), axis=1) / channel_scale

            normalized_error = float(np.median(channel_normalized_error))

            # Accept either truly tiny absolute residuals or tiny residuals
            # relative to the EEG amplitude.
            is_perfect_match = (
                max_absolute_error <= absolute_tolerance
                or normalized_error <= normalized_tolerance
            )

            candidates.append(
                {
                    "shift": shift,
                    "offset": offset,
                    "absolute_error": max_absolute_error,
                    "normalized_error": normalized_error,
                    "is_perfect": is_perfect_match,
                }
            )

        perfect_candidates = [
            candidate for candidate in candidates if candidate["is_perfect"]
        ]

        if not perfect_candidates:
            candidate_summary = "\n".join(
                (
                    f"  shift={candidate['shift']:2d}: "
                    f"absolute_error="
                    f"{candidate['absolute_error']:.6e}, "
                    f"normalized_error="
                    f"{candidate['normalized_error']:.6e}"
                )
                for candidate in candidates
            )

            raise RuntimeError(
                f"No perfect overlap found for epochs "
                f"{pair_index}->{pair_index + 1}.\n"
                f"Tested shifts {min_shift} through {max_shift}.\n"
                f"Candidate errors:\n{candidate_summary}"
            )

        # Among acceptable matches, select the smallest normalized residual.
        best = min(
            perfect_candidates,
            key=lambda candidate: (
                candidate["normalized_error"],
                candidate["absolute_error"],
            ),
        )

        shift = int(best["shift"])
        offset = np.asarray(best["offset"])

        # Apply the channel-wise offset to the complete next epoch.
        aligned_epochs[pair_index + 1] = current_original + offset[:, None]

        selected_shifts.append(shift)
        selected_offsets.append(offset)
        selected_errors.append(float(best["normalized_error"]))
        selected_absolute_errors.append(float(best["absolute_error"]))

        if verbose:
            print(
                f"{pair_index:02d}->{pair_index + 1:02d}: "
                f"shift={shift:2d} samples "
                f"({shift / sfreq * 1000:.3f} ms), "
                f"normalized_error="
                f"{best['normalized_error']:.3e}, "
                f"absolute_error="
                f"{best['absolute_error']:.3e}"
            )

    # Construct the cumulative start sample for every epoch.
    starts = np.zeros(n_epochs, dtype=int)
    starts[1:] = np.cumsum(selected_shifts)

    reconstructed_length = int(starts[-1] + n_times)

    accumulated = np.zeros(
        (n_channels, reconstructed_length),
        dtype=float,
    )

    weights = np.zeros(
        reconstructed_length,
        dtype=float,
    )

    for epoch_index, start in enumerate(starts):
        stop = start + n_times

        accumulated[:, start:stop] += aligned_epochs[epoch_index]
        weights[start:stop] += 1.0

    if np.any(weights == 0):
        missing_samples = np.flatnonzero(weights == 0)

        raise RuntimeError(
            "The reconstructed sequence contains gaps at samples "
            f"{missing_samples.tolist()}."
        )

    reconstructed = accumulated / weights[None, :]

    # Create one merged epoch. Preserve the original epoch start time.
    merged_data = reconstructed[None, :, :]

    merged_events = np.array(
        [[0, 0, 1]],
        dtype=int,
    )

    merged_epochs = mne.EpochsArray(
        merged_data,
        epochs.info.copy(),
        events=merged_events,
        event_id={"reconstructed_sequence": 1},
        tmin=float(epochs.tmin),
        baseline=None,
        verbose=False,
    )

    diagnostics = {
        "pairwise_shifts": np.asarray(
            selected_shifts,
            dtype=int,
        ),
        "epoch_starts": starts,
        "normalized_errors": np.asarray(
            selected_errors,
            dtype=float,
        ),
        "absolute_errors": np.asarray(
            selected_absolute_errors,
            dtype=float,
        ),
        "channel_offsets": np.asarray(
            selected_offsets,
            dtype=float,
        ),
        "weights": weights,
        "sampling_frequency": sfreq,
    }

    if verbose:
        shifts = diagnostics["pairwise_shifts"]

        print("\nReconstruction complete")
        print(f"Original epochs: {n_epochs}")
        print(f"Original epoch length: {n_times} samples")
        print(f"Reconstructed length: " f"{reconstructed_length} samples")
        print(f"Reconstructed duration: " f"{(reconstructed_length - 1) / sfreq:.6f} s")
        print(f"Shift sequence: {shifts.tolist()}")
        print(f"Mean shift: {shifts.mean():.4f} samples")
        print(
            "Maximum normalized error: " f"{diagnostics['normalized_errors'].max():.3e}"
        )

    return merged_epochs, diagnostics


class Sample(SampleMetadata, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Any
    
    @classmethod
    def from_auto_label(cls, participant: int, line: int, data: Any):
        return cls(
            participant=participant,
            line=line,
            set_path="",
            fdt_path="",
            rhythm_type=RhythmType.from_line(line),
            language_understanding=LanguageUnderstanding.from_participant(participant),
            data=data,
        )


def read_unepoch_sample(sample: SampleMetadata):
    """Read a single sample from disk and return an MNE Epochs object."""
    try:
        epochs = mne.read_epochs_eeglab(sample.set_path, verbose="ERROR")
        merged_epochs, _ = merge_data(epochs, verbose=False)  # type: ignore
        return Sample(
            **sample.model_dump(),
            data=merged_epochs,
        )
    except Exception as _:
        raw = mne.io.read_raw_eeglab(sample.set_path, verbose="ERROR", preload=True)
        return Sample(
            **sample.model_dump(),
            data=raw,
        )

def read_merge_epoched_samples(path: Path | str):
    path = Path(path)
    
    metadatas = build_metadata(path)
    samples = [read_unepoch_sample(metadata) for metadata in metadatas]
    
    return samples

def read_raw_data(data_root: str | PathLike):
    data_root = Path(data_root)

    samples: list[Sample] = []

    for verse_dir in data_root.glob("v*"):
        verse_match = re.fullmatch(r"v(\d+)", verse_dir.name)
        if not verse_match:
            logging.warning(
                "The dir `%s` is not verse_id formatted but is in the verse_ids dir",
                str(verse_dir),
            )
            continue

        verse_id = int(verse_match.group(1))
        dataset_dir = verse_dir / "4hz_datasets"

        for participant_file in dataset_dir.glob("p*.set"):
            participant_match = re.fullmatch(r"^p(\d+)\.set$", participant_file.name)
            if not participant_match:
                logging.warning(
                    "The file `%s` is not participant_id formatted but is in the participant_ids dir",
                    str(verse_dir),
                )
                continue

            participant_id = int(participant_match.group(1))

            data = mne.io.read_raw_eeglab(participant_file, verbose="ERROR", preload=True)

            n_channels = len(data.ch_names)
            if n_channels != N_CHANNELS:
                logging.info(
                    "Participant %d at line %d has %d channels. Expecting %d",
                    participant_id,
                    verse_id,
                    n_channels,
                    N_CHANNELS,
                )

            samples.append(
                Sample(
                    participant=participant_id,
                    line=verse_id,
                    set_path=str(participant_file),
                    fdt_path=str(participant_file.with_suffix(".fdt")),
                    rhythm_type=RhythmType.from_line(verse_id),
                    language_understanding=LanguageUnderstanding.from_participant(
                        participant_id
                    ),
                    data=data,
                )
            )

    return samples
