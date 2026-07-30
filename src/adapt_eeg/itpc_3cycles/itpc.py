from __future__ import annotations

import mne
import numpy as np


def epoch_timing_for_frequency(frequency_hz: float, cycles_per_epoch: float = 3.0, shift_cycles: float = 1.0) -> tuple[float, float]:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if cycles_per_epoch < 1:
        raise ValueError("cycles_per_epoch must be at least 1")
    if shift_cycles < 1:
        raise ValueError("shift_cycles must be at least 1")
    return cycles_per_epoch / frequency_hz, shift_cycles / frequency_hz


def raw_to_frequency_epochs(
    raw: mne.io.BaseRaw,
    frequency_hz: float,
    cycles_per_epoch: float = 3.0,
    shift_cycles: float = 1.0,
) -> mne.Epochs:
    duration, shift = epoch_timing_for_frequency(
        frequency_hz,
        cycles_per_epoch=cycles_per_epoch,
        shift_cycles=shift_cycles,
    )
    segment_duration = raw.n_times / float(raw.info["sfreq"])
    if segment_duration < duration:
        raise ValueError(
            f"Raw segment is {segment_duration:.3f}s, shorter than the required "
            f"{duration:.3f}s ({cycles_per_epoch:g} cycles at {frequency_hz:g} Hz)"
        )

    return mne.make_fixed_length_epochs(
        raw,
        duration=duration,
        overlap=duration - shift,
        preload=True,
        verbose="ERROR",
    )


def fourier_itpc(epochs: mne.Epochs, frequency_hz: float) -> np.ndarray:
    if len(epochs) < 2:
        raise ValueError("ITPC requires at least two epochs")

    data = epochs.get_data(copy=True)
    data = data - data.mean(axis=-1, keepdims=True)
    window = np.hanning(data.shape[-1])
    kernel = window * np.exp(-2j * np.pi * frequency_hz * epochs.times)
    coef = np.sum(data * kernel[None, None, :], axis=-1)
    valid = np.abs(coef) > np.finfo(float).eps
    unit_phase = np.zeros_like(coef, dtype=np.complex128)
    unit_phase[valid] = coef[valid] / np.abs(coef[valid])
    return np.abs(unit_phase.mean(axis=0))


def three_cycle_itpc(
    raw: mne.io.BaseRaw,
    frequency_hz: float,
    cycles_per_epoch: float = 3.0,
    shift_cycles: float = 1.0,
) -> np.ndarray:
    epochs = raw_to_frequency_epochs(
        raw,
        frequency_hz,
        cycles_per_epoch=cycles_per_epoch,
        shift_cycles=shift_cycles,
    )
    return fourier_itpc(epochs, frequency_hz)
