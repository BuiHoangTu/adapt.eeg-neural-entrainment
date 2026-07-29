from __future__ import annotations

import numpy as np
from mne.time_frequency import tfr_array_morlet


def mne_itpc(
    data: np.ndarray,
    sampling_frequency: float,
    frequency_hz: float,
    n_cycles: float | None = None,
) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError("Expected data with shape (trials, channels, times)")
    if data.shape[0] < 2:
        raise ValueError("ITPC requires at least two trials/windows")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")

    centered = data - data.mean(axis=-1, keepdims=True)
    duration = data.shape[-1] / sampling_frequency
    cycles = n_cycles if n_cycles is not None else max(frequency_hz * duration / 3, 0.1)
    itc = tfr_array_morlet(
        centered,
        sfreq=sampling_frequency,
        freqs=np.array([frequency_hz], dtype=float),
        n_cycles=cycles,
        output="itc",
        zero_mean=True,
        verbose="ERROR",
    )
    return itc[:, 0, :].mean(axis=-1)


def fixed_length_itpc(
    raw_data: np.ndarray,
    sampling_frequency: float,
    frequency_hz: float,
    window_seconds: float,
    step_seconds: float,
) -> np.ndarray:
    if raw_data.ndim != 2:
        raise ValueError("Expected raw data with shape (channels, times)")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")

    window_samples = int(round(window_seconds * sampling_frequency))
    step_samples = int(round(step_seconds * sampling_frequency))
    if window_samples < 1:
        raise ValueError("window_seconds is too short for the sampling frequency")
    if step_samples < 1:
        raise ValueError("step_seconds is too short for the sampling frequency")
    if window_samples > raw_data.shape[-1]:
        raise ValueError("ITPC window is longer than the raw segment")

    starts = range(0, raw_data.shape[-1] - window_samples + 1, step_samples)
    windows = np.stack(
        [raw_data[:, start : start + window_samples] for start in starts],
        axis=0,
    )
    return mne_itpc(windows, sampling_frequency, frequency_hz)
