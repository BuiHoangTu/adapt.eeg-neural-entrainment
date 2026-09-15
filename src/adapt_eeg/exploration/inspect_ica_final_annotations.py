from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from adapt_eeg.poem_eeg import (
    InvalidPoemRecordingError,
    discover_ica_cleaned_files,
    load_poem_recording,
    unique_recording_paths,
)

EXPECTED_FILE_COUNT = 200


def inspect_file(path: Path) -> dict:
    recording = load_poem_recording(path, preload=False)
    raw = recording.raw

    return {
        "poem": recording.poem,
        "participant": recording.participant,
        "set_path": str(path),
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": len(raw.ch_names),
        "recording_duration_s": raw.n_times / float(raw.info["sfreq"]),
        "trigger_count": 2,
        "trigger_start_s": recording.trigger_start_s,
        "trigger_end_s": recording.trigger_end_s,
        "intro_end_s": recording.usable_start_s - recording.trigger_start_s,
        "usable_start_s": recording.usable_start_s,
        "usable_end_s": recording.usable_end_s,
        "usable_duration_s": recording.usable_end_s - recording.usable_start_s,
        "expected_audio_duration_s": recording.expected_audio_duration_s,
        "duration_error_s": recording.duration_error_s,
        "status": "ok",
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw-eeg"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/exploration/eeg_annotations.csv")
    )
    parser.add_argument("--expected-file-count", type=int, default=EXPECTED_FILE_COUNT)
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write the audit report and return success when invalid files are found.",
    )
    args = parser.parse_args()

    discovered_files = discover_ica_cleaned_files(args.data_root)
    files, duplicate_paths = unique_recording_paths(discovered_files)
    rows = []
    errors = [
        f"poem {poem}, participant {participant}: duplicate ICA-cleaned files: "
        + " | ".join(str(path) for path in paths)
        for (poem, participant), paths in sorted(duplicate_paths.items())
    ]
    for path in files:
        try:
            rows.append(inspect_file(path))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            rows.append(
                {
                    "set_path": str(path),
                    "status": "error",
                    "error": str(exc),
                }
            )

    if len(discovered_files) != args.expected_file_count:
        errors.append(
            f"Expected {args.expected_file_count} ICA-cleaned files, "
            f"found {len(discovered_files)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(
        f"Discovered {len(discovered_files)} files and inspected {len(files)} "
        f"unique recordings; {len(errors)} errors"
    )
    if errors and not args.allow_errors:
        raise InvalidPoemRecordingError("\n".join(errors))


if __name__ == "__main__":
    main()
