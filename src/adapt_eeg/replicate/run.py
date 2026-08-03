from __future__ import annotations

import argparse
from pathlib import Path

import mne
import pandas as pd

from adapt_eeg.constants import FREQUENCIES as FREQUENCY_BY_NAME, N_CHANNELS
from adapt_eeg.data_reader import Sample, read_raw_data
from adapt_eeg.replicate.itpc import fixed_length_itpc

CSV_FLOAT_FORMAT = "%.6f"


def _frequency_name_for_hz(frequency_hz: float) -> str:
    for frequency_name, candidate_hz in FREQUENCY_BY_NAME.items():
        if candidate_hz == frequency_hz:
            return frequency_name
    return str(frequency_hz)


def _pick_eeg_channels(inst: mne.Epochs | mne.io.BaseRaw) -> list[str]:
    picks = mne.pick_types(inst.info, eeg=True, exclude=[])
    channel_names = [inst.ch_names[pick] for pick in picks]
    if len(channel_names) != N_CHANNELS:
        print(
            f"Warning: expected {N_CHANNELS} EEG channels, found "
            f"{len(channel_names)} in {getattr(inst, 'filename', '<memory>')}"
        )
    return channel_names



def _itpc_from_raw(
    raw: mne.io.BaseRaw,
    frequency_hz: float,
    window_seconds: float,
    step_seconds: float,
) -> list[float]:
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw_eeg = raw.copy().pick(picks)
    return fixed_length_itpc(
        raw_eeg,
        frequency_hz,
        window_seconds,
        step_seconds,
    ).tolist()


def analyze_sample(
    sample: Sample,
    window_seconds: float,
    step_seconds: float,
) -> tuple[list[dict], dict | None]:
    try:
        raw = sample.data
        channel_names = _pick_eeg_channels(raw)
        rows: list[dict] = []
        for frequency_hz in FREQUENCY_BY_NAME.values():
            values = _itpc_from_raw(raw, frequency_hz, window_seconds, step_seconds)

            for channel, itpc in zip(channel_names, values, strict=False):
                rows.append(
                    {
                        "participant": sample.participant,
                        "line": sample.line,
                        "rhythm_condition": sample.rhythm_type.value,
                        "language_group": sample.language_understanding.value,
                        "frequency_name": _frequency_name_for_hz(frequency_hz),
                        "frequency_hz": frequency_hz,
                        "channel": channel,
                        "itpc": float(itpc),
                    }
                )
        return rows, None
    except Exception as exc:
        return [], {
            "participant": sample.participant,
            "line": sample.line,
            "error": repr(exc),
        }


def summarize_outputs(channel_itpc: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"itpc_by_channel_line": channel_itpc}


def run_replication(
    data_root: Path,
    output_dir: Path,
    window_seconds: float,
    step_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_raw_data(data_root)

    all_rows: list[dict] = []
    errors: list[dict] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] participant {sample.participant}, line {sample.line}")
        rows, error = analyze_sample(
            sample,
            window_seconds,
            step_seconds,
        )
        all_rows.extend(rows)
        if error is not None:
            errors.append(error)

    # metadata = pd.DataFrame(
    #     [
    #         {
    #             "participant": sample.participant,
    #             "line": sample.line,
    #             "rhythm_condition": sample.rhythm_type.value,
    #             "language_group": sample.language_understanding.value,
    #         }
    #         for sample in samples
    #     ]
    # )
    # metadata.to_csv(output_dir / "metadata.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "load_errors.csv", index=False)
    if errors:
        raise RuntimeError(
            f"Failed to load or analyze {len(errors)} required samples; see load_errors.csv"
        )

    channel_itpc = pd.DataFrame(all_rows)
    if channel_itpc.empty:
        raise RuntimeError("No ITPC rows were produced; see load_errors.csv")
    channel_itpc.to_csv(
        output_dir / "itpc_by_channel_line.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )

    print(f"Wrote replication outputs to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replicate the ADAPT EEG poetry-rhythm ITPC experiment."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/datasets_4Hz"),
        help="Root containing v*/4hz_datasets/p*.set Raw line segments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/replicate"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.3,
        help="Fixed window length from the paper autocorrelation analysis.",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=0.05,
        help="Step size when slicing continuous line segments into ITPC trials.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_replication(
        args.data_root,
        args.output_dir,
        args.window_seconds,
        args.step_seconds,
    )


if __name__ == "__main__":
    main()
