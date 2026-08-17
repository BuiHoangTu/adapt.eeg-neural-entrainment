from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from adapt_eeg.replicate.analysis import included_itpc_from_channel_lines, paper_table_2

CSV_FLOAT_FORMAT = "%.6f"
PAPER_CHANNEL_ORDER = (
    "P7",
    "P4",
    "Pz",
    "P3",
    "P8",
    "O1",
    "O2",
    "T8",
    "F8",
    "C4",
    "F4",
    "Fp2",
    "Fz",
    "C3",
    "F3",
    "Fp1",
    "T7",
    "F7",
)
STAT_COLUMNS = ("F", "p", "part η²")
BLOCK_LABELS = (
    (
        "Poetic Rhythm (2Hz)",
        "Effect of Regular vs Irregular Rhythm on entrainment",
        "Effect of group (bilinguals, monolinguals)",
    ),
    (
        "Syllabic Rhythm (4Hz)",
        "Effect of Regular vs Irregular Poetic Rhythm on entrainment",
        "Effect of group (bilinguals, monolinguals)",
    ),
)


def table_2_from_fully_replicated_itpc(frame: pd.DataFrame) -> pd.DataFrame:
    included_itpc = included_itpc_from_channel_lines(frame)
    table = paper_table_2(included_itpc)
    table = table.loc[table["eeg_channel"].isin(PAPER_CHANNEL_ORDER)].copy()
    table["eeg_channel"] = pd.Categorical(
        table["eeg_channel"], categories=PAPER_CHANNEL_ORDER, ordered=True
    )
    return table.sort_values(["eeg_channel", "frequency_hz"]).reset_index(drop=True)


def paper_layout_table_2(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for channel in PAPER_CHANNEL_ORDER:
        row = {"EEG Channel": channel}
        channel_table = table.loc[table["eeg_channel"] == channel]
        for frequency_hz, rhythm_label, rhythm_effect_label, group_effect_label in (
            (2.0, *BLOCK_LABELS[0]),
            (4.0, *BLOCK_LABELS[1]),
        ):
            frequency_table = channel_table.loc[channel_table["frequency_hz"] == frequency_hz]
            rhythm_columns = [
                f"{rhythm_label} {rhythm_effect_label} {stat}" for stat in STAT_COLUMNS
            ]
            group_columns = [
                f"{rhythm_label} {group_effect_label} {stat}" for stat in STAT_COLUMNS
            ]
            if frequency_table.empty:
                for column in rhythm_columns + group_columns:
                    row[column] = pd.NA
                continue

            values = frequency_table.iloc[0]
            row[rhythm_columns[0]] = values["rhythm_F"]
            row[rhythm_columns[1]] = values["rhythm_p"]
            row[rhythm_columns[2]] = values["rhythm_partial_eta_squared"]
            row[group_columns[0]] = values["group_F"]
            row[group_columns[1]] = values["group_p"]
            row[group_columns[2]] = values["group_partial_eta_squared"]
        rows.append(row)
    return pd.DataFrame(rows)


def _write_formatted_table_2_xlsx(
    writer: pd.ExcelWriter,
    paper_layout: pd.DataFrame,
    table: pd.DataFrame,
) -> None:
    workbook = writer.book
    worksheet = workbook.add_worksheet("Table 2")
    writer.sheets["Table 2"] = worksheet

    header_format = workbook.add_format(
        {"bold": True, "align": "center", "valign": "vcenter", "border": 1, "text_wrap": True}
    )
    cell_format = workbook.add_format({"align": "center", "border": 1})
    number_format = workbook.add_format({"align": "center", "border": 1, "num_format": "0.000"})

    worksheet.merge_range(0, 0, 2, 0, "EEG\nChannel", header_format)
    worksheet.merge_range(0, 1, 0, 6, BLOCK_LABELS[0][0], header_format)
    worksheet.merge_range(0, 7, 0, 12, BLOCK_LABELS[1][0], header_format)
    worksheet.merge_range(1, 1, 1, 3, BLOCK_LABELS[0][1], header_format)
    worksheet.merge_range(1, 4, 1, 6, BLOCK_LABELS[0][2], header_format)
    worksheet.merge_range(1, 7, 1, 9, BLOCK_LABELS[1][1], header_format)
    worksheet.merge_range(1, 10, 1, 12, BLOCK_LABELS[1][2], header_format)

    for offset in (1, 4, 7, 10):
        for stat_index, stat_label in enumerate(STAT_COLUMNS):
            worksheet.write(2, offset + stat_index, stat_label, header_format)

    for row_index, row in enumerate(paper_layout.itertuples(index=False), start=3):
        worksheet.write(row_index, 0, row[0], cell_format)
        for column_index, value in enumerate(row[1:], start=1):
            if pd.isna(value):
                worksheet.write_blank(row_index, column_index, None, cell_format)
            else:
                worksheet.write_number(row_index, column_index, float(value), number_format)

    worksheet.set_column(0, 0, 12)
    worksheet.set_column(1, 12, 12)
    worksheet.set_row(0, 34)
    worksheet.set_row(1, 58)
    worksheet.set_row(2, 22)
    table.to_excel(writer, sheet_name="Long", index=False)


def write_table_2(input_csv: Path, output_dir: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing fully replicated ITPC output: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table = table_2_from_fully_replicated_itpc(pd.read_csv(input_csv))
    paper_layout = paper_layout_table_2(table)

    table.to_csv(
        output_dir / "paper_table_2_mixed_anova_long.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    paper_layout.to_csv(
        output_dir / "paper_table_2_mixed_anova.csv",
        index=False,
        float_format=CSV_FLOAT_FORMAT,
    )
    with pd.ExcelWriter(output_dir / "paper_table_2_mixed_anova.xlsx", engine="xlsxwriter") as writer:
        _write_formatted_table_2_xlsx(writer, paper_layout, table)


def main(args: argparse.Namespace) -> None:
    write_table_2(args.input_csv, args.output_dir)
    print(f"Wrote paper Table 2 outputs to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate paper Table 2 from fully replicated ITPC outputs."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("results/fully_replicate/itpc_results/itpc_long.csv"),
        help="Long ITPC CSV produced by adapt_eeg.fully_replicate.export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fully_replicate/itpc_results"),
        help="Directory for paper Table 2 CSV/XLSX outputs.",
    )
    main(parser.parse_args())
