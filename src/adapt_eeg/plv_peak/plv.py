from __future__ import annotations

import mne
import numpy as np

from adapt_eeg.plv_peak.peaks import POEM_FREQUENCY_HZ, PeakWindow


def channel_phase_signal(raw: mne.io.BaseRaw, frequency_hz: float = POEM_FREQUENCY_HZ) -> np.ndarray:
    filtered = raw.copy().filter(
        l_freq=frequency_hz - 0.5,
        h_freq=frequency_hz + 0.5,
        method="iir",
        verbose="ERROR",
    )
    analytic = filtered.apply_hilbert(envelope=False, verbose="ERROR")
    return np.angle(analytic.get_data())


def peak_window_plv(
    phases: np.ndarray,
    sfreq: float,
    window: PeakWindow,
) -> np.ndarray:
    start_sample = max(0, int(np.floor(window.window_start * sfreq)))
    end_sample = min(phases.shape[-1], int(np.ceil(window.window_end * sfreq)) + 1)
    if end_sample <= start_sample:
        raise ValueError(
            f"No samples for peak window {window.window_start:.3f}-{window.window_end:.3f}s"
        )
    return np.abs(np.exp(1j * phases[:, start_sample:end_sample]).mean(axis=-1))


def centered_plv(frame, value_column: str = "plv"):
    result = frame.copy()
    result["participant_channel_mean_plv"] = result.groupby(
        ["participant", "channel"], sort=False
    )[value_column].transform("mean")
    result["centered_plv"] = result[value_column] - result["participant_channel_mean_plv"]
    return result
