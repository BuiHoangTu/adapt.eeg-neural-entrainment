from os import PathLike
from pathlib import Path
import re

import pandas as pd


def export_xlsx(
    df,
    output_file="poem_itpc_by_channel.xlsx",
    participant_col="participant",
    line_col="line",
    rhythm_col="rhythm_condition",
    language_col="language_group",
    frequency_name_col="frequency_name",
    frequency_hz_col="frequency_hz",
    channel_col="channel",
    value_col="itpc",
    poem_value="POEM",
    poem_frequency_hz=2.0,
    english_value="english_competence",
    regular_value="regular",
):
    """
    Export POEM 2 Hz ITPC data to an Excel workbook.

    Each EEG channel is written to a separate worksheet.

    Layout:
        - Rows: participants
        - Columns: line numbers
        - Cell values: ITPC

    Ordering:
        - English-competent participants first
        - Regular lines first
        - IDs and line numbers sorted within each group

    Coloring:
        - English-competent participant IDs: red
        - Regular line-number headers: green
        - ITPC content cells: no background color

    Returns
    -------
    str
        Output Excel file path.
    """

    required_columns = {
        participant_col,
        line_col,
        rhythm_col,
        language_col,
        frequency_name_col,
        frequency_hz_col,
        channel_col,
        value_col,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError("The input dataframe is empty.")

    # Keep only POEM observations at 2 Hz.
    filtered_df = df.loc[
        df[frequency_name_col].astype(str).str.upper().eq(str(poem_value).upper())
        & pd.to_numeric(
            df[frequency_hz_col],
            errors="coerce",
        ).eq(float(poem_frequency_hz))
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No rows found where "
            f"{frequency_name_col} == {poem_value!r} and "
            f"{frequency_hz_col} == {poem_frequency_hz}."
        )

    def safe_sheet_name(value, used_names):
        """
        Make a valid and unique Excel worksheet name.
        """

        name = str(value)

        # Excel forbids these characters in worksheet names.
        name = re.sub(r"[\[\]:*?/\\]", "_", name)
        name = name.strip() or "Unnamed channel"
        name = name[:31]

        original_name = name
        counter = 2

        while name.lower() in used_names:
            suffix = f"_{counter}"
            name = original_name[: 31 - len(suffix)] + suffix
            counter += 1

        used_names.add(name.lower())
        return name

    with pd.ExcelWriter(
        output_file,
        engine="xlsxwriter",
    ) as writer:
        workbook = writer.book
        used_sheet_names = set()

        corner_format = workbook.add_format(
            {
                "bold": True,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }
        )

        regular_line_format = workbook.add_format(
            {
                "bold": True,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "bg_color": "#92D050",
            }
        )

        irregular_line_format = workbook.add_format(
            {
                "bold": True,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }
        )

        english_participant_format = workbook.add_format(
            {
                "bold": True,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "bg_color": "#FF6666",
            }
        )

        other_participant_format = workbook.add_format(
            {
                "bold": True,
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }
        )

        # No background color is assigned to data cells.
        value_format = workbook.add_format(
            {
                "num_format": "0.0000",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }
        )

        empty_format = workbook.add_format(
            {
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }
        )

        channels = filtered_df[channel_col].dropna().unique()

        for channel in channels:
            channel_df = filtered_df.loc[filtered_df[channel_col] == channel].copy()

            matrix = channel_df.pivot_table(
                index=participant_col,
                columns=line_col,
                values=value_col,
                aggfunc="mean",
            )

            if matrix.empty:
                continue

            participant_languages = (
                channel_df[[participant_col, language_col]]
                .dropna(subset=[participant_col])
                .drop_duplicates(
                    subset=[participant_col],
                    keep="first",
                )
                .set_index(participant_col)[language_col]
                .to_dict()
            )

            line_conditions = (
                channel_df[[line_col, rhythm_col]]
                .dropna(subset=[line_col])
                .drop_duplicates(
                    subset=[line_col],
                    keep="first",
                )
                .set_index(line_col)[rhythm_col]
                .to_dict()
            )

            # English-competent participants first.
            participants = sorted(
                matrix.index.tolist(),
                key=lambda participant: (
                    participant_languages.get(participant) != english_value,
                    participant,
                ),
            )

            # Regular lines first.
            line_numbers = sorted(
                matrix.columns.tolist(),
                key=lambda line_number: (
                    line_conditions.get(line_number) != regular_value,
                    line_number,
                ),
            )

            matrix = matrix.loc[
                participants,
                line_numbers,
            ]

            sheet_name = safe_sheet_name(
                channel,
                used_sheet_names,
            )

            worksheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = worksheet

            worksheet.write(
                0,
                0,
                "Participant \\ line",
                corner_format,
            )

            # Write line-number headers.
            for column_number, line_number in enumerate(
                line_numbers,
                start=1,
            ):
                condition = line_conditions.get(line_number)

                if condition == regular_value:
                    header_format = regular_line_format
                else:
                    header_format = irregular_line_format

                worksheet.write(
                    0,
                    column_number,
                    line_number,
                    header_format,
                )

            # Write participant IDs and ITPC values.
            for row_number, participant_id in enumerate(
                participants,
                start=1,
            ):
                language_group = participant_languages.get(participant_id)

                if language_group == english_value:
                    participant_format = english_participant_format
                else:
                    participant_format = other_participant_format

                worksheet.write(
                    row_number,
                    0,
                    participant_id,
                    participant_format,
                )

                for column_number, line_number in enumerate(
                    line_numbers,
                    start=1,
                ):
                    value = matrix.loc[
                        participant_id,
                        line_number,
                    ]

                    if pd.isna(value):
                        worksheet.write_blank(
                            row_number,
                            column_number,
                            None,
                            empty_format,
                        )
                    else:
                        worksheet.write_number(
                            row_number,
                            column_number,
                            float(value),
                            value_format,
                        )

            worksheet.freeze_panes(1, 1)
            worksheet.set_column(0, 0, 20)

            if line_numbers:
                worksheet.set_column(
                    1,
                    len(line_numbers),
                    10,
                )

            worksheet.set_row(0, 26)
            worksheet.hide_gridlines(2)

    return output_file


def _format(
    df,
    excluded_lines=(8, 9),
    participant_col="participant",
    line_col="line",
    rhythm_col="rhythm_condition",
    language_col="language_group",
    frequency_name_col="frequency_name",
    frequency_hz_col="frequency_hz",
    channel_col="channel",
    value_col="itpc",
):
    """
    Calculate average ITPC for three categories:

    1. POEM_REGULAR:
       - frequency_name == "POEM"
       - frequency_hz == 2
       - rhythm_condition == "regular"

    2. POEM_IRREGULAR:
       - frequency_name == "POEM"
       - frequency_hz == 2
       - rhythm_condition == "irregular"

    3. SPEAKING:
       - frequency_name == "SPEAKING"
       - frequency_hz == 4

    Lines 8 and 9 are excluded from all categories.

    Returns one row per:
        participant × language_group × channel × category
    """

    required_columns = {
        participant_col,
        line_col,
        rhythm_col,
        language_col,
        frequency_name_col,
        frequency_hz_col,
        channel_col,
        value_col,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing_columns))
        )

    if df.empty:
        raise ValueError("The input dataframe is empty.")

    working_df = df.copy()

    working_df[frequency_hz_col] = pd.to_numeric(
        working_df[frequency_hz_col],
        errors="coerce",
    )

    working_df[value_col] = pd.to_numeric(
        working_df[value_col],
        errors="coerce",
    )

    # Exclude lines 8 and 9 from every category.
    working_df = working_df.loc[~working_df[line_col].isin(excluded_lines)].copy()

    frequency_names = working_df[frequency_name_col].astype(str).str.strip().str.upper()

    rhythm_conditions = working_df[rhythm_col].astype(str).str.strip().str.lower()

    # POEM, 2 Hz, regular.
    poem_regular_df = working_df.loc[
        frequency_names.eq("POEM")
        & working_df[frequency_hz_col].eq(2.0)
        & rhythm_conditions.eq("regular")
    ].copy()

    poem_regular_df["category"] = "POEM_REGULAR"
    poem_regular_df["category_frequency_hz"] = 2.0

    # POEM, 2 Hz, irregular.
    poem_irregular_df = working_df.loc[
        frequency_names.eq("POEM")
        & working_df[frequency_hz_col].eq(2.0)
        & rhythm_conditions.eq("irregular")
    ].copy()

    poem_irregular_df["category"] = "POEM_IRREGULAR"
    poem_irregular_df["category_frequency_hz"] = 2.0

    # SPEAKING, 4 Hz.
    speaking_df = working_df.loc[
        frequency_names.eq("SPEAKING") & working_df[frequency_hz_col].eq(4.0)
    ].copy()

    speaking_df["category"] = "SPEAKING"
    speaking_df["category_frequency_hz"] = 4.0

    category_df = pd.concat(
        [
            poem_regular_df,
            poem_irregular_df,
            speaking_df,
        ],
        ignore_index=True,
    )

    if category_df.empty:
        raise ValueError(
            "No observations matched the POEM regular, "
            "POEM irregular, or SPEAKING criteria."
        )

    result = (
        category_df.groupby(
            [
                participant_col,
                language_col,
                channel_col,
                "category",
                "category_frequency_hz",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            average_itpc=(value_col, "mean"),
            n_observations=(value_col, "count"),
        )
        .reset_index()
        .rename(
            columns={
                "category_frequency_hz": "frequency_hz",
            }
        )
    )

    # English-competent participants first.
    result["_language_order"] = (result[language_col] != "english_competence").astype(
        int
    )

    result["_category_order"] = result["category"].map(
        {
            "POEM_REGULAR": 0,
            "POEM_IRREGULAR": 1,
            "SPEAKING": 2,
        }
    )

    result = (
        result.sort_values(
            by=[
                "_language_order",
                participant_col,
                channel_col,
                "_category_order",
            ]
        )
        .drop(
            columns=[
                "_language_order",
                "_category_order",
            ]
        )
        .reset_index(drop=True)
    )

    return result


def itpc_by_line_channel2xlsx(
    itpc_by_line_channel_file: PathLike | Path,
    output_file: PathLike | Path = "results/poem_itpc_by_channel.xlsx",  # type: ignore
):
    itpc_by_line_channel_file = Path(itpc_by_line_channel_file)
    output_file = Path(output_file)

    df = pd.read_csv(itpc_by_line_channel_file)
    export_xlsx(df, str(output_file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate an Excel workbook of POEM 2 Hz ITPC data by channel."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="results/datasets_4Hz/itpc_by_line_channel.csv",
        help="Path to the CSV file containing ITPC data by line and channel. Example: itpc_by_line_channel.csv",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/datasets_4Hz/poem_itpc_by_channel.xlsx",
        help="Path to the output Excel file.",
    )
    args = parser.parse_args()

    itpc_by_line_channel2xlsx(
        itpc_by_line_channel_file=args.input_file,
        output_file=args.output_file,
    )
