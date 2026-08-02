from __future__ import annotations

import mne
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from adapt_eeg.plv_peak.peaks import POEM_FREQUENCY_HZ, PeakWindow


def _bandpass_phase(
    data: np.ndarray,
    sfreq: float,
    frequency_hz: float,
    bandwidth_hz: float = 1.0,
    padding_seconds: float = 3.0,
) -> np.ndarray:
    """Band-pass and return phase after reflected padding to reduce edge artefacts."""
    values = np.asarray(data, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("Phase extraction expects a 1D or 2D array")
    if values.shape[-1] < 8:
        raise ValueError("Signal is too short for phase extraction")

    nyquist = sfreq / 2.0
    half_band = bandwidth_hz / 2.0
    low_hz = frequency_hz - half_band
    high_hz = frequency_hz + half_band
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"Invalid band {low_hz:.3f}-{high_hz:.3f} Hz for sfreq={sfreq:.3f}"
        )

    sos = butter(4, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    requested_pad = int(round(padding_seconds * sfreq))
    pad_samples = min(requested_pad, values.shape[-1] - 1)
    if pad_samples < 1:
        raise ValueError("Signal is too short to reflect-pad")

    padded = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(pad_samples, pad_samples)], mode="reflect")
    filtered = sosfiltfilt(sos, padded, axis=-1)
    analytic = hilbert(filtered, axis=-1)
    return np.angle(analytic[..., pad_samples:-pad_samples])


def channel_phase_signal(
    raw: mne.io.BaseRaw,
    frequency_hz: float = POEM_FREQUENCY_HZ,
    bandwidth_hz: float = 1.0,
    padding_seconds: float = 3.0,
) -> np.ndarray:
    return _bandpass_phase(
        raw.get_data(),
        sfreq=float(raw.info["sfreq"]),
        frequency_hz=frequency_hz,
        bandwidth_hz=bandwidth_hz,
        padding_seconds=padding_seconds,
    )


def audio_envelope_phase(
    audio: np.ndarray,
    sample_rate: float,
    eeg_times: np.ndarray,
    frequency_hz: float = POEM_FREQUENCY_HZ,
    bandwidth_hz: float = 1.0,
    padding_seconds: float = 3.0,
) -> np.ndarray:
    """Return the low-frequency phase of the audio envelope on the EEG time grid."""
    values = np.asarray(audio, dtype=float)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Audio must be a non-empty mono or stereo signal")

    envelope = np.abs(hilbert(values))
    phase = _bandpass_phase(
        envelope,
        sfreq=float(sample_rate),
        frequency_hz=frequency_hz,
        bandwidth_hz=bandwidth_hz,
        padding_seconds=padding_seconds,
    )
    audio_times = np.arange(phase.size, dtype=float) / float(sample_rate)
    if eeg_times[-1] > audio_times[-1]:
        raise ValueError(
            f"EEG duration ({eeg_times[-1]:.3f}s) exceeds audio duration "
            f"({audio_times[-1]:.3f}s)"
        )
    unwrapped = np.unwrap(phase)
    resampled = np.interp(eeg_times, audio_times, unwrapped)
    return np.angle(np.exp(1j * resampled))


def peak_window_stimulus_plv(
    eeg_phases: np.ndarray,
    audio_phase: np.ndarray,
    sfreq: float,
    window: PeakWindow,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return channel-wise EEG–audio PLV, circular mean lag, and sample count."""
    start_sample = max(0, int(np.floor(window.window_start * sfreq)))
    end_sample = min(
        eeg_phases.shape[-1],
        audio_phase.shape[-1],
        int(np.ceil(window.window_end * sfreq)) + 1,
    )
    if end_sample <= start_sample:
        raise ValueError(
            f"No samples for window {window.window_start:.3f}-{window.window_end:.3f}s"
        )

    phase_difference = (
        eeg_phases[:, start_sample:end_sample]
        - audio_phase[start_sample:end_sample][None, :]
    )
    vectors = np.exp(1j * phase_difference).mean(axis=-1)
    return np.abs(vectors), np.angle(vectors), end_sample - start_sample
