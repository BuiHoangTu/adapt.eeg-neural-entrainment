from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adapt_eeg.replicate.analysis import (
    included_itpc_from_channel_lines,
    paper_table_1,
    paper_table_2,
    paper_table_3,
)
from adapt_eeg.utils.visualization import write_itpc_visualizations

CSV_FLOAT_FORMAT = "%.6f"


def write_paper_tables(results_dir: Path) -> None:
    channel_lines_path = results_dir / "itpc_by_channel_line.csv"
    if not channel_lines_path.exists():
        raise FileNotFoundError(
            f"Missing {channel_lines_path}. Run adapt_eeg.itpc_3cycles.run before table analysis."
        )

    included_itpc = included_itpc_from_channel_lines(pd.read_csv(channel_lines_path))
    outputs = {
        "paper_table_1_beat_distances.csv": paper_table_1(),
        "paper_table_2_mixed_anova.csv": paper_table_2(included_itpc),
        "paper_table_3_group_differences.csv": paper_table_3(included_itpc),
    }
    for filename, frame in outputs.items():
        frame.to_csv(results_dir / filename, index=False, float_format=CSV_FLOAT_FORMAT)
    write_itpc_visualizations(results_dir, excluded_lines={8, 9})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate paper-style tables from 3-cycle ITPC outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/itpc_3cycles"),
        help="Directory containing outputs from adapt_eeg.itpc_3cycles.run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    write_paper_tables(args.results_dir)


if __name__ == "__main__":
    main()
