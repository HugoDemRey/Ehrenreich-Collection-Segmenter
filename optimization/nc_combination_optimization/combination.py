import numpy as np

from app.src.audio_features.features import NoveltyCurve


def run_combination(
    method: str,
    chroma_nc: NoveltyCurve,
    mfcc_nc: NoveltyCurve,
    tempo_nc: NoveltyCurve,
    sr: int,
    w_chroma: float,
    w_mfcc: float,
    w_tempo: float,
    peak_threshold: float,
):
    # Combine novelty curves with given weights
    combined_nc = NoveltyCurve.combine(
        [chroma_nc, mfcc_nc, tempo_nc], [w_chroma, w_mfcc, w_tempo], method=method
    )

    from scipy.signal import find_peaks as scipy_find_peaks

    height_threshold = np.max(combined_nc.data()) * peak_threshold
    distance_samples = int(15 * sr)
    peaks, _ = scipy_find_peaks(
        combined_nc.data(), height=height_threshold, distance=distance_samples
    )
    peaks_sec = peaks / sr

    return combined_nc, peaks_sec
