from __future__ import annotations

import numpy as np
import mne


def raw_to_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    window_seconds: float,
    step_seconds: float,
) -> mne.Epochs:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if step_seconds > window_seconds:
        raise ValueError("step_seconds must be less than or equal to window_seconds")

    duration = raw.n_times / float(raw.info["sfreq"])
    if duration < window_seconds:
        raise ValueError("ITPC window is longer than the raw segment")

    return mne.make_fixed_length_epochs(
        raw,
        duration=window_seconds,
        overlap=window_seconds - step_seconds,
        preload=True,
        verbose="ERROR",
    )


def mne_itpc(
    epochs: mne.Epochs,
    frequency_hz: float,
    n_cycles: float | None = None,
) -> np.ndarray:
    if len(epochs) < 2:
        raise ValueError("ITPC requires at least two trials/windows")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")

    cycles = n_cycles if n_cycles is not None else max(frequency_hz * epochs.times.size / epochs.info["sfreq"] / 3, 0.1)
    _, itc = epochs.compute_tfr(
        method="morlet",
        freqs=np.array([frequency_hz], dtype=float),
        n_cycles=cycles,
        average=True,
        return_itc=True,
        verbose="ERROR",
    )
    return itc.data[:, 0, :].mean(axis=-1)


def fixed_length_itpc(
    raw: mne.io.BaseRaw,
    frequency_hz: float,
    window_seconds: float,
    step_seconds: float,
    n_cycles: float | None = None,
) -> np.ndarray:
    epochs = raw_to_fixed_length_epochs(raw, window_seconds, step_seconds)
    return mne_itpc(epochs, frequency_hz, n_cycles=n_cycles)
