import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import stats

from adapt_eeg.constants import FREQUENCIES
from adapt_eeg.data_reader import SampleMetadata, build_metadata


CHANNELS_OF_INTEREST = ("Pz", "C4", "T7")


def group_difference_tests_by_channel(
    participant_condition_channel: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (
        frequency_name,
        freq,
        channel,
    ), freq_df in participant_condition_channel.groupby(
        ["frequency_name", "frequency_hz", "channel"]
    ):
        wide = freq_df.pivot_table(
            index=["participant", "language_group"],
            columns="rhythm_condition",
            values="itpc_mean",
        )
        if not {"regular", "irregular"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["regular", "irregular"])
        if wide.empty:
            continue
        wide = wide.reset_index()
        wide["regular_minus_irregular"] = wide["regular"] - wide["irregular"]
        groups = {
            group: group_df["regular_minus_irregular"].to_numpy()
            for group, group_df in wide.groupby("language_group")
        }
        if set(groups) != {"english_competence", "non_english_competence"}:
            continue
        ttest = stats.ttest_ind(
            groups["english_competence"],
            groups["non_english_competence"],
            equal_var=False,
        )
        mannwhitney = stats.mannwhitneyu(
            groups["english_competence"],
            groups["non_english_competence"],
            alternative="two-sided",
        )
        rows.append(
            {
                "frequency_name": frequency_name,
                "frequency_hz": freq,
                "channel": channel,
                "english_competence_n": int(len(groups["english_competence"])),
                "non_english_competence_n": int(len(groups["non_english_competence"])),
                "english_competence_mean_difference": float(
                    groups["english_competence"].mean()
                ),
                "non_english_competence_mean_difference": float(
                    groups["non_english_competence"].mean()
                ),
                "welch_t_statistic": float(ttest.statistic),  # type: ignore
                "welch_t_pvalue": float(ttest.pvalue),  # type: ignore
                "mannwhitney_statistic": float(mannwhitney.statistic),
                "mannwhitney_pvalue": float(mannwhitney.pvalue),
            }
        )
    return pd.DataFrame(rows)


def language_summary_by_channel(
    participant_condition_channel: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (
        frequency_name,
        freq,
        channel,
        group,
    ), group_df in participant_condition_channel.groupby(
        ["frequency_name", "frequency_hz", "channel", "language_group"]
    ):
        wide = group_df.pivot(
            index="participant", columns="rhythm_condition", values="itpc_mean"
        )
        if not {"regular", "irregular"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["regular", "irregular"])
        if wide.empty:
            continue
        diff = wide["regular"] - wide["irregular"]
        rows.append(
            {
                "frequency_name": frequency_name,
                "frequency_hz": freq,
                "channel": channel,
                "language_group": group,
                "n_participants": int(len(wide)),
                "regular_mean": float(wide["regular"].mean()),
                "irregular_mean": float(wide["irregular"].mean()),
                "mean_difference_regular_minus_irregular": float(diff.mean()),
                "sd_difference": float(diff.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def paired_tests_by_channel(
    participant_condition_channel: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (
        frequency_name,
        freq,
        channel,
    ), freq_df in participant_condition_channel.groupby(
        ["frequency_name", "frequency_hz", "channel"]
    ):
        wide = freq_df.pivot(
            index="participant", columns="rhythm_condition", values="itpc_mean"
        )
        if not {"regular", "irregular"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["regular", "irregular"])
        if wide.empty:
            continue
        diff = wide["regular"] - wide["irregular"]
        ttest = stats.ttest_rel(wide["regular"], wide["irregular"])
        try:
            wilcoxon = stats.wilcoxon(
                wide["regular"], wide["irregular"], zero_method="wilcox"
            )
            wilcoxon_stat = float(wilcoxon.statistic)  # type: ignore
            wilcoxon_p = float(wilcoxon.pvalue)  # type: ignore
        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan
        rows.append(
            {
                "frequency_name": frequency_name,
                "frequency_hz": freq,
                "channel": channel,
                "n_participants": int(len(wide)),
                "regular_mean": float(wide["regular"].mean()),
                "irregular_mean": float(wide["irregular"].mean()),
                "mean_difference_regular_minus_irregular": float(diff.mean()),
                "sd_difference": float(diff.std(ddof=1)),
                "paired_t_statistic": float(ttest.statistic),
                "paired_t_pvalue": float(ttest.pvalue),
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_pvalue": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows)


def tfr_itpc(
    epochs: mne.Epochs, freqs: dict[str, float], n_cycles: float
) -> dict[str, np.ndarray]:
    """Compute channel-wise ITPC using MNE Epochs.compute_tfr return_itc."""
    frequency_names = list(freqs)
    _, itpc = epochs.compute_tfr(
        method="morlet",
        freqs=np.array(list(freqs.values())),
        n_cycles=n_cycles,
        average=True,
        return_itc=True,
        n_jobs=-1,
        verbose="ERROR",
    )
    return {
        frequency_name: itpc.data[:, frequency_index, :].mean(axis=-1)
        for frequency_index, frequency_name in enumerate(frequency_names)
    }


def fourier_itpc(
    data: np.ndarray, times: np.ndarray, freqs: dict[str, float]
) -> dict[str, np.ndarray]:
    """Compute phase locking across epochs using exact-frequency Fourier projection.

    Parameters
    ----------
    data
        Epoch data with shape n_epochs x n_channels x n_times.
    times
        Epoch sample times in seconds.
    freqs
        Mapping from frequency name to frequency in Hz.

    Returns
    -------
    dict
        Frequency name to channel-wise ITPC vector.
    """
    data = data - data.mean(axis=-1, keepdims=True)
    window = np.hanning(data.shape[-1])
    out = {}
    for frequency_name, freq in freqs.items():
        kernel = window * np.exp(-2j * np.pi * freq * times)
        coef = np.sum(data * kernel[None, None, :], axis=-1)
        valid = np.abs(coef) > np.finfo(float).eps
        unit_phase = np.zeros_like(coef, dtype=np.complex128)
        unit_phase[valid] = coef[valid] / np.abs(coef[valid])
        out[frequency_name] = np.abs(unit_phase.mean(axis=0))
    return out


def analyze_sample(
    metadata: SampleMetadata, method: str = "fourier", tfr_n_cycles: float = 0.25
) -> tuple[list[dict], list[dict]]:
    epochs = mne.read_epochs_eeglab(metadata.set_path, verbose="ERROR")
    data = epochs.get_data(copy=True)
    duration = float(epochs.times[-1] - epochs.times[0] + 1.0 / epochs.info["sfreq"])
    cycles = {
        frequency_name: duration * freq for frequency_name, freq in FREQUENCIES.items()
    }
    if method == "fourier":
        itpc_by_freq = fourier_itpc(data, epochs.times, FREQUENCIES)
    elif method == "compute_tfr":
        itpc_by_freq = tfr_itpc(epochs, FREQUENCIES, tfr_n_cycles)
    else:
        raise ValueError(f"Unsupported ITPC method: {method}")

    file_rows = []
    channel_rows = []
    for frequency_name, channel_values in itpc_by_freq.items():
        freq = FREQUENCIES[frequency_name]
        short_window = cycles[frequency_name] < 1.0
        file_rows.append(
            {
                **metadata.model_dump(),
                "rhythm_condition": metadata.rhythm_type.value,
                "language_group": metadata.language_understanding.value,
                "frequency_name": frequency_name,
                "frequency_hz": freq,
                "n_epochs": len(epochs),
                "n_channels": len(epochs.ch_names),
                "sfreq": epochs.info["sfreq"],
                "n_times": len(epochs.times),
                "tmin": float(epochs.times[0]),
                "tmax": float(epochs.times[-1]),
                "duration_s": duration,
                "cycles_in_epoch": cycles[frequency_name],
                "short_window_lt_1_cycle": short_window,
                "itpc_mean_all_channels": float(channel_values.mean()),
                "itpc_median_all_channels": float(np.median(channel_values)),
                "itpc_std_all_channels": float(channel_values.std(ddof=1)),
            }
        )
        for ch_name, value in zip(epochs.ch_names, channel_values, strict=True):
            channel_rows.append(
                {
                    "participant": metadata.participant,
                    "line": metadata.line,
                    "rhythm_condition": metadata.rhythm_type.value,
                    "language_group": metadata.language_understanding.value,
                    "frequency_name": frequency_name,
                    "frequency_hz": freq,
                    "channel": ch_name,
                    "itpc": float(value),
                }
            )
    return file_rows, channel_rows


def language_summary(participant_condition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (frequency_name, freq, group), group_df in participant_condition.groupby(
        ["frequency_name", "frequency_hz", "language_group"]
    ):
        wide = group_df.pivot(
            index="participant", columns="rhythm_condition", values="itpc_mean"
        )
        wide = wide.dropna(subset=["regular", "irregular"])
        if wide.empty:
            continue
        diff = wide["regular"] - wide["irregular"]
        rows.append(
            {
                "frequency_name": frequency_name,
                "frequency_hz": freq,
                "language_group": group,
                "n_participants": int(len(wide)),
                "regular_mean": float(wide["regular"].mean()),
                "irregular_mean": float(wide["irregular"].mean()),
                "mean_difference_regular_minus_irregular": float(diff.mean()),
                "sd_difference": float(diff.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def paired_tests(participant_condition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (frequency_name, freq), freq_df in participant_condition.groupby(
        ["frequency_name", "frequency_hz"]
    ):
        wide = freq_df.pivot(
            index="participant", columns="rhythm_condition", values="itpc_mean"
        )
        if not {"regular", "irregular"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["regular", "irregular"])
        diff = wide["regular"] - wide["irregular"]
        ttest = stats.ttest_rel(wide["regular"], wide["irregular"])
        try:
            wilcoxon = stats.wilcoxon(
                wide["regular"], wide["irregular"], zero_method="wilcox"
            )
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan
        rows.append(
            {
                "frequency_name": frequency_name,
                "frequency_hz": freq,
                "n_participants": int(len(wide)),
                "regular_mean": float(wide["regular"].mean()),
                "irregular_mean": float(wide["irregular"].mean()),
                "mean_difference_regular_minus_irregular": float(diff.mean()),
                "sd_difference": float(diff.std(ddof=1)),
                "paired_t_statistic": float(ttest.statistic),
                "paired_t_pvalue": float(ttest.pvalue),
                "wilcoxon_statistic": wilcoxon_stat,
                "wilcoxon_pvalue": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows)


def main(args) -> None:
    data_root = args.data_root
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(data_root)
    # metadata.to_csv(output_dir / "metadata.csv", index=False)

    file_rows = []
    channel_rows = []
    error_rows = []
    for row in metadata:
        try:
            one_file_rows, one_channel_rows = analyze_sample(
                row, method=args.method, tfr_n_cycles=args.tfr_n_cycles
            )
        except Exception as exc:
            error_rows.append(
                {
                    "participant": row.participant,
                    "line": row.line,
                    "rhythm_condition": row.rhythm_type.value,
                    "language_group": row.language_understanding.value,
                    "set_path": row.set_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        file_rows.extend(one_file_rows)
        channel_rows.extend(one_channel_rows)

    by_file = pd.DataFrame(file_rows)
    by_channel = pd.DataFrame(channel_rows)
    pd.DataFrame(error_rows).to_csv(output_dir / "load_errors.csv", index=False)
    if by_file.empty:
        raise RuntimeError("No files could be analyzed; see load_errors.csv")

    by_file.to_csv(output_dir / "itpc_by_file.csv", index=False)
    by_channel.to_csv(output_dir / "itpc_by_channel.csv", index=False)

    participant_condition_channel = by_channel.groupby(
        [
            "participant",
            "language_group",
            "rhythm_condition",
            "frequency_name",
            "frequency_hz",
            "channel",
        ],
        as_index=False,
    ).agg(
        itpc_mean=("itpc", "mean"),
        itpc_sd_across_lines=("itpc", "std"),
        n_lines=("line", "nunique"),
    )
    participant_condition_channel.to_csv(
        output_dir / "itpc_by_participant_condition_channel.csv", index=False
    )

    channels_of_interest = by_channel[
        by_channel["channel"].isin(CHANNELS_OF_INTEREST)
    ].copy()
    channels_of_interest.to_csv(
        output_dir / "itpc_channels_of_interest.csv", index=False
    )

    stats_by_channel = paired_tests_by_channel(participant_condition_channel)
    stats_by_channel.to_csv(output_dir / "condition_stats_by_channel.csv", index=False)
    stats_by_channel[stats_by_channel["channel"].isin(CHANNELS_OF_INTEREST)].to_csv(
        output_dir / "condition_stats_channels_of_interest.csv", index=False
    )

    language_summary_by_channel(participant_condition_channel).to_csv(
        output_dir / "language_group_summary_by_channel.csv", index=False
    )
    group_difference_tests_by_channel(participant_condition_channel).to_csv(
        output_dir / "group_difference_stats_by_channel.csv", index=False
    )

    participant_condition = by_file.groupby(
        [
            "participant",
            "language_group",
            "rhythm_condition",
            "frequency_name",
            "frequency_hz",
        ],
        as_index=False,
    ).agg(
        itpc_mean=("itpc_mean_all_channels", "mean"),
        itpc_sd_across_lines=("itpc_mean_all_channels", "std"),
        n_lines=("line", "nunique"),
        mean_epochs_per_line=("n_epochs", "mean"),
        mean_cycles_in_epoch=("cycles_in_epoch", "mean"),
        any_short_window_lt_1_cycle=("short_window_lt_1_cycle", "any"),
    )
    participant_condition.to_csv(
        output_dir / "itpc_by_participant_condition.csv", index=False
    )

    stats_df = paired_tests(participant_condition)
    stats_df.to_csv(output_dir / "condition_stats.csv", index=False)
    language_summary(participant_condition).to_csv(
        output_dir / "language_group_summary.csv", index=False
    )

    print(f"Wrote {len(metadata)} metadata rows to {output_dir / 'metadata.csv'}")
    print(f"Skipped {len(error_rows)} files; see {output_dir / 'load_errors.csv'}")
    print(
        f"Wrote {len(by_file)} file-frequency rows to {output_dir / 'itpc_by_file.csv'}"
    )
    print(
        f"Wrote {len(by_channel)} channel rows to {output_dir / 'itpc_by_channel.csv'}"
    )
    print(stats_df.to_string(index=False))
    if by_file["short_window_lt_1_cycle"].any():
        print(
            "WARNING: Each epoch contains less than one cycle at one or more requested frequencies; interpret 2/4 Hz phase-locking cautiously."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Repeat no-narrow-filter ITPC analysis for EEGLAB epochs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/itpc-30event-good"),
        help="Root directory containing line*/part*v*.set files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/itpc-30event-good"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--method",
        choices=("fourier", "compute_tfr"),
        default="fourier",
        help="ITPC phase extraction method.",
    )
    parser.add_argument(
        "--tfr-n-cycles",
        type=float,
        default=0.25,
        help="Morlet cycles for --method compute_tfr. Short epochs require a very small value.",
    )
    args = parser.parse_args()

    main(args)
