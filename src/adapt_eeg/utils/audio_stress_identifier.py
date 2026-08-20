from pathlib import Path

import numpy as np
from pydub import AudioSegment
from scipy.signal import hilbert, butter, sosfiltfilt, find_peaks


def stress_identify(
    audio,
    lowpass_hz=8.0,
    min_peak_distance_s=0.20,
    prominence=0.05,
):

    # ============================================================
    # INPUT
    # Can be either:
    #   - a file path
    #   - a pydub.AudioSegment
    # ============================================================

    audio_input = audio

    # Example alternative:
    #
    # audio_input = AudioSegment.from_file("poem.wav")

    # ============================================================
    # LOAD AUDIO
    # ============================================================

    if isinstance(audio_input, AudioSegment):
        audio = audio_input

    elif isinstance(audio_input, (str, Path)):
        audio = AudioSegment.from_file(str(audio_input))

    else:
        raise TypeError("audio_input must be a file path or pydub.AudioSegment")

    # Convert to mono
    audio = audio.set_channels(1)

    fs = audio.frame_rate

    # pydub -> NumPy
    waveform = np.array(audio.get_array_of_samples(), dtype=np.float64)

    # Normalize waveform
    max_abs = np.max(np.abs(waveform))

    if max_abs > 0:
        waveform /= max_abs

    # ============================================================
    # HILBERT ENVELOPE
    # ============================================================

    analytic_signal = hilbert(waveform)

    envelope = np.abs(analytic_signal)

    # ============================================================
    # LOW-PASS FILTER THE ENVELOPE
    # ============================================================

    sos = butter(N=4, Wn=lowpass_hz, btype="lowpass", fs=fs, output="sos")

    envelope_filtered = sosfiltfilt(sos, envelope)

    # Remove possible tiny negative numerical values
    envelope_filtered = np.maximum(envelope_filtered, 0)

    # Normalize to 0–1
    env_min = envelope_filtered.min()
    env_max = envelope_filtered.max()

    if env_max > env_min:
        envelope_filtered = (envelope_filtered - env_min) / (env_max - env_min)

    # ============================================================
    # FIND CANDIDATE STRESS PEAKS
    # ============================================================

    min_distance_samples = int(min_peak_distance_s * fs)

    peaks, properties = find_peaks(
        envelope_filtered, distance=max(1, min_distance_samples), prominence=prominence
    )

    # ============================================================
    # CONVERT PEAK LOCATIONS TO TIME
    # ============================================================

    time = np.arange(len(waveform)) / fs

    stress_times = peaks / fs

    stress_strength = envelope_filtered[peaks]

    return stress_times, stress_strength
