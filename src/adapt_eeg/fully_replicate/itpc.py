from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from adapt_eeg.constants import FREQUENCIES
from adapt_eeg.data_reader import SampleMetadata, build_metadata

TARGET_BANDS = {
    FREQUENCIES["POEM"]: (1.0, 3.0),
    FREQUENCIES["SPEAKING"]: (3.0, 5.0),
}
WAVELET_FREQUENCIES = np.logspace(np.log10(2), np.log10(30), 40)
WAVELET_CYCLES = np.logspace(np.log10(4), np.log10(10), 40)
WAVELET_TIME_SECONDS = 2.0


@dataclass(frozen=True)
class LoadedEpochs:
    metadata: SampleMetadata
    data: np.ndarray
    sfreq: float
    channel_names: tuple[str, ...]


def discover_samples(data_root: Path) -> list[SampleMetadata]:
    return sorted(
        build_metadata(data_root),
        key=lambda sample: (sample.line, sample.participant),
    )


def load_epochs(sample: SampleMetadata) -> LoadedEpochs:
    epochs = mne.read_epochs_eeglab(sample.set_path, verbose="ERROR")
    data = epochs.get_data(copy=True)
    picks = mne.pick_types(epochs.info, eeg=True, exclude=[])
    if len(picks) != data.shape[1]:
        data = data[:, picks, :]
    channel_names = tuple(epochs.ch_names[pick] for pick in picks)
    return LoadedEpochs(
        metadata=sample,
        data=np.transpose(data, (1, 2, 0)),
        sfreq=float(epochs.info["sfreq"]),
        channel_names=channel_names,
    )


def bandpass_epochs(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=1)


def wavelet_itpc(
    data: np.ndarray,
    sfreq: float,
    *,
    freqs: np.ndarray = WAVELET_FREQUENCIES,
    cycles: np.ndarray = WAVELET_CYCLES,
    wavelet_time_seconds: float = WAVELET_TIME_SECONDS,
) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError("data must have shape channels x times x trials")
    if data.shape[2] < 2:
        raise ValueError("ITPC requires at least two trials")

    n_channels, n_times, n_trials = data.shape
    wavelet_time = np.arange(
        -wavelet_time_seconds,
        wavelet_time_seconds + 1 / sfreq,
        1 / sfreq,
    )
    half_wave = (wavelet_time.size - 1) // 2
    n_data = n_times * n_trials
    n_conv = wavelet_time.size + n_data - 1

    flattened = data.reshape(n_channels, n_data, order="F")
    data_fft = np.fft.fft(flattened, n_conv, axis=1)
    itpc = np.empty((n_channels, len(freqs)), dtype=float)

    for frequency_index, (frequency, n_cycles) in enumerate(zip(freqs, cycles, strict=True)):
        sigma = n_cycles / (2 * np.pi * frequency)
        wavelet = np.exp(2j * np.pi * frequency * wavelet_time) * np.exp(
            -(wavelet_time**2) / (2 * sigma**2)
        )
        wavelet_fft = np.fft.fft(wavelet, n_conv)
        analytic = np.fft.ifft(data_fft * wavelet_fft[None, :], n_conv, axis=1)
        analytic = analytic[:, half_wave : analytic.shape[1] - half_wave]
        analytic = analytic.reshape(n_channels, n_times, n_trials, order="F")
        phase_vectors = np.exp(1j * np.angle(analytic))
        itpc[:, frequency_index] = np.abs(phase_vectors.mean(axis=2)).mean(axis=1)

    return itpc


def sample_itpc_rows(
    loaded: LoadedEpochs,
    *,
    target_bands: dict[float, tuple[float, float]] = TARGET_BANDS,
) -> list[dict]:
    rows: list[dict] = []
    for target_frequency, band in sorted(target_bands.items()):
        filtered = bandpass_epochs(loaded.data, loaded.sfreq, band)
        spectrum = wavelet_itpc(filtered, loaded.sfreq)
        frequency_index = int(np.argmin(np.abs(WAVELET_FREQUENCIES - target_frequency)))
        values = spectrum[:, frequency_index]
        frequency_name = _frequency_name(target_frequency)
        for channel, value in zip(loaded.channel_names, values, strict=True):
            rows.append(
                {
                    "participant": loaded.metadata.participant,
                    "line": loaded.metadata.line,
                    "rhythm_condition": loaded.metadata.rhythm_type.value,
                    "language_group": loaded.metadata.language_understanding.value,
                    "frequency_name": frequency_name,
                    "frequency_hz": target_frequency,
                    "filter_low_hz": band[0],
                    "filter_high_hz": band[1],
                    "channel": channel,
                    "itpc": float(value),
                }
            )
    return rows


def compute_itpc_table(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not data_root.exists():
        raise FileNotFoundError(f"ITPC data root does not exist: {data_root}")

    samples = discover_samples(data_root)
    if not samples:
        raise ValueError(f"No ITPC .set files found under: {data_root}")

    rows: list[dict] = []
    errors: list[dict] = []
    for sample in samples:
        try:
            loaded = load_epochs(sample)
            rows.extend(sample_itpc_rows(loaded))
        except Exception as exc:
            errors.append(
                {
                    "participant": sample.participant,
                    "line": sample.line,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(errors)


def line_order(frame: pd.DataFrame) -> list[int]:
    line_info = (
        frame[["line", "rhythm_condition"]]
        .drop_duplicates()
        .assign(
            rhythm_rank=lambda df: df["rhythm_condition"].map(
                {"irregular": 0, "regular": 1}
            ).fillna(2)
        )
        .sort_values(["rhythm_rank", "line"])
    )
    return line_info["line"].astype(int).tolist()


def _frequency_name(frequency_hz: float) -> str:
    for name, candidate in FREQUENCIES.items():
        if float(candidate) == float(frequency_hz):
            return name
    return f"{frequency_hz:g}Hz"
