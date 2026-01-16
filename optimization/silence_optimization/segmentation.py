import numpy as np
from scipy import signal as scipy_signal

from app.src.audio.signal import Signal
from app.src.audio_features.builders import SilenceCurveBuilder
from app.src.audio_features.features import SilenceCurve


def run_silence_segmentation(
    audio_data: np.ndarray,
    sampling_rate: float,
    silence_type: str = "spectral",
    frame_length: int = 44100,
    hop_length: int = 22050,
    threshold: float = 0.7,
    min_silence_duration_sec: float = 0.0,
    min_distance_sec: float = 10.0,
) -> tuple[SilenceCurve, np.ndarray]:
    """Run silence detection on audio data.

    Args:
        audio_data: Audio signal data
        sampling_rate: Sampling rate of the audio
        silence_type: Type of silence detection ("amplitude" or "spectral")
        frame_length: Length of analysis frame in samples
        hop_length: Hop length between frames in samples
        threshold: Threshold for silence detection (0.0 to 1.0)
        min_silence_duration_sec: Minimum duration of silence to consider (seconds)
        min_distance_sec: Minimum distance between detected silences (seconds)

    Returns:
        silence_curve: The computed silence curve
        peaks_sec: Array of detected silence timestamps in seconds
    """

    # Create Signal object from audio data
    signal_obj = Signal(audio_data, sampling_rate, "")

    # Build silence curve - exactly like segmenter_silence_m.py
    builder = SilenceCurveBuilder(
        silence_type=silence_type,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    sc = builder.build(signal_obj)

    # Get initial peaks from the silence curve using basic peak detection
    silence_data = sc.data()[0]
    min_distance_samples = max(1, int(min_distance_sec * sc.sampling_rate()))

    # Find peaks in silence curve (silence curve already handles inversion internally)
    peaks_frames, _ = scipy_signal.find_peaks(
        silence_data, distance=min_distance_samples
    )

    min_silence_duration_seconds = min_silence_duration_sec

    # Filter peaks based on minimum silence duration - exactly like segmenter_silence_m.py
    if len(peaks_frames) > 0 and min_silence_duration_seconds > 0.0 and sc is not None:
        # Get the silence curve data
        silence_data = sc.data()[0]
        sampling_rate = sc.sampling_rate()
        min_duration_samples = int(min_silence_duration_seconds * sampling_rate)

        filtered_peaks = []
        for peak in peaks_frames:
            # Find the start and end of the silence region around this peak
            # Look backwards from peak to find where silence starts
            start_idx = peak
            while start_idx > 0 and silence_data[start_idx] >= threshold * np.max(
                silence_data
            ):
                start_idx -= 1

            # Look forwards from peak to find where silence ends
            end_idx = peak
            while end_idx < len(silence_data) - 1 and silence_data[
                end_idx
            ] >= threshold * np.max(silence_data):
                end_idx += 1

            # Check if the silence duration meets the minimum requirement
            silence_duration_samples = end_idx - start_idx
            if silence_duration_samples >= min_duration_samples:
                filtered_peaks.append(peak)

        peaks_frames = np.array(filtered_peaks)

    # Convert frames to seconds
    peak_times = peaks_frames / sampling_rate

    return sc, peak_times


def run_optimized_silence_segmentation(
    audio_data: np.ndarray,
    sampling_rate: float,
    silence_type: str,
    frame_length: int,
    hop_length: int,
    threshold: float,
    min_silence_duration_sec: float,
    min_distance_sec: float,
) -> tuple[SilenceCurve, np.ndarray]:
    """Wrapper function for optimization that matches the expected signature."""

    return run_silence_segmentation(
        audio_data=audio_data,
        sampling_rate=sampling_rate,
        silence_type=silence_type,
        frame_length=frame_length,
        hop_length=hop_length,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_distance_sec=min_distance_sec,
    )
