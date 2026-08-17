from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adapt_eeg.fully_replicate.itpc import compute_itpc_table
from adapt_eeg.utils.xlsx_generation import export_xlsx


def write_matlab_style_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        raise ValueError("Cannot export an empty ITPC table")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "itpc_long.csv", index=False)
    export_xlsx(frame, output_dir / "poem_itpc_by_channel.xlsx")

    for frequency_hz, frequency_frame in frame.groupby("frequency_hz"):
        frequency_dir = output_dir / f"{frequency_hz:g}Hz"
        frequency_dir.mkdir(parents=True, exist_ok=True)
        _write_line_tables(frequency_frame, frequency_dir, frequency_hz)
        _write_channel_mean_table(frequency_frame, output_dir, frequency_hz)


def _write_line_tables(
    frequency_frame: pd.DataFrame,
    frequency_dir: Path,
    frequency_hz: float,
) -> None:
    line_dir = frequency_dir / "lines"
    line_dir.mkdir(parents=True, exist_ok=True)

    for line, line_frame in frequency_frame.groupby("line"):
        table = line_frame.pivot_table(
            index="participant",
            columns="channel",
            values="itpc",
            aggfunc="mean",
        ).sort_index()
        table.to_excel(line_dir / f"Line{int(line)}_{frequency_hz:g}Hz.xlsx")


def _write_channel_mean_table(
    frequency_frame: pd.DataFrame,
    output_dir: Path,
    frequency_hz: float,
) -> None:
    ordered_lines = line_order_for_export(frequency_frame)
    table = frequency_frame.pivot_table(
        index="participant",
        columns="line",
        values="itpc",
        aggfunc="mean",
    ).reindex(columns=ordered_lines).sort_index()
    table.columns = [f"line{int(column)}" for column in table.columns]
    table.to_excel(output_dir / f"ITPC_results_{frequency_hz:g}Hz.xlsx")


def line_order_for_export(frame: pd.DataFrame) -> list[int]:
    line_info = (
        frame[["line", "rhythm_condition"]]
        .drop_duplicates()
        .assign(
            rhythm_rank=lambda df: df["rhythm_condition"]
            .map({"irregular": 0, "regular": 1})
            .fillna(2)
        )
        .sort_values(["rhythm_rank", "line"])
    )
    return line_info["line"].astype(int).tolist()


def main(args: argparse.Namespace) -> None:
    frame, errors = compute_itpc_table(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not errors.empty:
        errors.to_csv(args.output_dir / "itpc_errors.csv", index=False)

    if frame.empty:
        raise RuntimeError(
            "No ITPC rows were computed; all discovered files failed. "
            f"See {args.output_dir / 'itpc_errors.csv'}"
        )

    write_matlab_style_outputs(frame, args.output_dir)
    print(f"Wrote {len(frame)} ITPC rows to {args.output_dir}")
    if not errors.empty:
        print(f"Skipped {len(errors)} files; see {args.output_dir / 'itpc_errors.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replicate the recovered MATLAB ITPC extraction pipeline in Python."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/itpc-30event-good"),
        help="Folder containing line*/part*v*.set files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fully_replicate/itpc_results"),
        help="Folder for long CSV and MATLAB-style Excel outputs.",
    )
    main(parser.parse_args())
