"""Model for interactive plot components with transition detection.

Provides data management and analysis functionality for interactive
plots with transition point detection and management capabilities.

Author: Hugo Demule
Date: January 2026
"""

import numpy as np
from constants.novelty_curve_code import NoveltyCurveCode
from constants.segmenter_configurations import SegmenterConfig
from src.audio.signal import Signal
from src.audio_features.features import NoveltyCurve
from src.interfaces.feature import Feature, PeakFinder
from src.io.session_cache import SessionCache


class InteractablePlotModel:
    """Model for interactive plot components with transition detection.

    Manages audio signal analysis, transition point detection, and user
    interaction state for interactive analysis plot widgets.
    """

    def __init__(self, signal: Signal, id: str):
        self.signal = signal
        self.tolerance = (
            0.005 * self.signal.duration_seconds()
        )  # 0.5% of the signal duration
        self.config = SegmenterConfig.UNDEFINED
        self.id = id

    def get_transitions_pos_and_index(
        self, transitions_list, position
    ) -> tuple[float | None, int | None]:
        """Get the transition positions and their indices."""
        diff_list = np.array(transitions_list) - position

        if np.all(np.abs(diff_list) > self.tolerance):
            return None, None

        index = int(np.argmin(np.abs(diff_list)))
        pos = transitions_list[index]

        return pos, index

    def is_on_transition(self, transitions_list, position) -> bool:
        """Check if the given position is on a transition line."""
        for transition_pos in transitions_list:
            if abs(position - transition_pos) <= self.tolerance:
                return True
        return False

    def find_peaks_frames(self, pf: PeakFinder, params) -> np.ndarray:
        """Placeholder method to be overridden in subclasses."""
        peaks = pf.find_peaks(
            params.threshold, distance_seconds=int(params.min_distance_sec)
        )

        return peaks

    def find_peaks_seconds(
        self, pf: PeakFinder, params, sampling_rate: float
    ) -> np.ndarray:
        """Find peaks and convert to seconds."""
        peaks_frames = self.find_peaks_frames(pf, params)
        peak_times = peaks_frames / sampling_rate

        SessionCache.save_module_state(
            page_id=self.id,
            module_type=self.config,
            transitions_sec=peak_times.tolist(),
            params=params.__dict__,
        )

        return peak_times

    def peaks_frames_to_seconds(self, peaks_frames: np.ndarray, feature: Feature):
        """Convert peak frame indices to time in seconds."""
        peak_times = peaks_frames / feature.sampling_rate()
        return peak_times

    def novelty_curve_to_time_values(
        self, nc: NoveltyCurve | NoveltyCurveCode
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert novelty curve to time-values for plotting."""
        if isinstance(nc, NoveltyCurveCode):
            return np.linspace(0, self.signal.duration_seconds(), 0), np.array([])
        t = np.linspace(0, self.signal.duration_seconds(), len(nc.data()))
        v = nc.data()

        SessionCache.save_module_state(
            page_id=self.id,
            module_type=self.config,
            nc_x=t,
            nc_y=v,
        )

        return t, v

    def empty_output(self) -> tuple[np.ndarray, np.ndarray]:
        """Return empty time-values for plotting."""
        return np.linspace(0, self.signal.duration_seconds(), 0), np.array([])
