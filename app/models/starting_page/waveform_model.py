"""Waveform model for audio visualization data management.

Manages waveform data, zoom levels, and selection state
for interactive audio waveform display widgets.

Author: Hugo Demule
Date: January 2026
"""

from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal


class WaveformModel(QObject):
    """Model for managing waveform data and visualization state.

    Handles audio data loading, zoom operations, and selection tracking
    for interactive waveform display components.

    Signals:
        data_loaded (object, int): Audio data and sample rate loaded
    """

    # Signals for data updates
    data_loaded = pyqtSignal(object, int)  # audio_data, sample_rate
    range_updated = pyqtSignal(float, float)  # start, end
    timestamps_loaded = pyqtSignal(list)  # timestamps

    def __init__(self):
        """Initialize the waveform model."""
        super().__init__()
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: Optional[int] = None
        self._duration_seconds: float = 0.0
        self._range_start: Optional[float] = None
        self._range_end: Optional[float] = None
        self._segmentation_timestamps: List[float] = []

    def load_audio_data(self, audio_data: np.ndarray, sample_rate: int):
        """Load audio data into the model."""
        self._audio_data = audio_data
        self._sample_rate = sample_rate
        self._duration_seconds = len(audio_data) / sample_rate

        # Clear any existing range selection
        self.clear_range_selection()

        # Emit signal for view to update
        self.data_loaded.emit(audio_data, sample_rate)

    def get_audio_data(self) -> Optional[np.ndarray]:
        """Get the current audio data."""
        return self._audio_data

    def get_sample_rate(self) -> Optional[int]:
        """Get the current sample rate."""
        return self._sample_rate

    def get_duration_seconds(self) -> float:
        """Get the audio duration in seconds."""
        return self._duration_seconds

    def set_range_selection(self, start: Optional[float], end: Optional[float]):
        """Set the range selection."""
        self._range_start = start
        self._range_end = end
        if start is not None and end is not None:
            self.range_updated.emit(start, end)

    def clear_range_selection(self):
        """Clear the range selection."""
        self._range_start = None
        self._range_end = None

    def get_range_selection(self) -> Tuple[Optional[float], Optional[float]]:
        """Get the current range selection."""
        return self._range_start, self._range_end

    def load_segmentation_timestamps(self, timestamps: List[float]):
        """Load segmentation timestamps."""
        self._segmentation_timestamps = timestamps.copy()
        self.timestamps_loaded.emit(timestamps)

    def get_segmentation_timestamps(self) -> List[float]:
        """Get the segmentation timestamps."""
        return self._segmentation_timestamps.copy()

    def has_audio_data(self) -> bool:
        """Check if audio data is loaded."""
        return self._audio_data is not None

    def get_time_bounds(self) -> Tuple[float, float]:
        """Get the time bounds of the audio."""
        return 0.0, self._duration_seconds

    def clamp_position(self, position: float) -> float:
        """Clamp a position to valid audio bounds."""
        return max(0.0, min(position, self._duration_seconds))

    def clear_all_data(self):
        """Clear all data to free memory."""
        self._audio_data = None
        self._sample_rate = None
        self._duration_seconds = 0.0
        self._range_start = None
        self._range_end = None
        self._segmentation_timestamps = []
