"""Model for silence-based segmentation analysis.

Provides segmentation analysis based on silence detection
and audio energy thresholds for pause-based structure analysis.

Author: Hugo Demule
Date: January 2026
"""

from typing import override

import numpy as np
from constants.parameters import SilenceParameters
from constants.segmenter_configurations import SegmenterConfig
from models.analysis_page.components.interactable_plot_m import InteractablePlotModel
from src.audio.signal import Signal
from src.audio_features.builders import SilenceCurveBuilder
from src.audio_features.features import SilenceCurve
from src.interfaces.feature import PeakFinder
from src.io.session_cache import SessionCache


class SegmenterSilenceModel(InteractablePlotModel):
    """Model for silence-based segmentation analysis.

    Detects audio segments based on silence thresholds and energy
    levels for pause-based structural audio analysis.
    """

    def __init__(self, signal: Signal, id: str):
        super().__init__(signal=signal, id=id)
        self.config = SegmenterConfig.SILENCE

    def compute_silence_curve(self, params: SilenceParameters) -> SilenceCurve:
        """Compute the base feature using the builder."""

        builder = SilenceCurveBuilder(
            silence_type=params.silence_type,
            frame_length=params.frame_length,
            hop_length=params.hop_length,
        )

        sc = builder.build(self.signal)
        return sc

    def silence_curve_to_time_values(
        self, sc: SilenceCurve
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert base feature to time-values for plotting."""

        v = sc.data()[0]
        t = np.linspace(0, self.signal.duration_seconds(), len(v))

        session_id = SessionCache.save_module_state(
            page_id=self.id,
            module_type=self.config,
            nc_x=t,
            nc_y=v,
        )

        print("page_id:", self.id, "config:", self.config, " -> MODULE ID:", session_id)

        return t, v

    @override
    def find_peaks_seconds(
        self,
        pf: PeakFinder,
        params,
        sampling_rate: float,
        sc: SilenceCurve | None = None,
    ) -> np.ndarray:
        """Find peaks in the silence curve based on parameters."""

        min_silence_duration_seconds = (
            params.min_silence_duration_sec
        )  # Minimum duration of silence to consider (in seconds)

        # Get initial peaks from the silence curve
        peaks_frames = super().find_peaks_frames(pf, params)

        # Filter peaks based on minimum silence duration
        if (
            len(peaks_frames) > 0
            and min_silence_duration_seconds > 0.0
            and sc is not None
        ):
            # Get the silence curve data
            silence_data = sc.data()[0]
            sampling_rate = sc.sampling_rate()
            min_duration_samples = int(min_silence_duration_seconds * sampling_rate)

            filtered_peaks = []
            for peak in peaks_frames:
                # Find the start and end of the silence region around this peak
                # Look backwards from peak to find where silence starts
                start_idx = peak
                while start_idx > 0 and silence_data[
                    start_idx
                ] >= params.threshold * np.max(silence_data):
                    start_idx -= 1

                # Look forwards from peak to find where silence ends
                end_idx = peak
                while end_idx < len(silence_data) - 1 and silence_data[
                    end_idx
                ] >= params.threshold * np.max(silence_data):
                    end_idx += 1

                # Check if the silence duration meets the minimum requirement
                silence_duration_samples = end_idx - start_idx
                if silence_duration_samples >= min_duration_samples:
                    filtered_peaks.append(peak)

            peaks_frames = np.array(filtered_peaks)

        peak_times = peaks_frames / sampling_rate

        SessionCache.save_module_state(
            page_id=self.id,
            module_type=self.config,
            transitions_sec=peak_times.tolist(),
            params=params.__dict__,
        )

        return peak_times
