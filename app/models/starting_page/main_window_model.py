"""Model for starting page main window state management.

Handles application initialization state, file selection,
and startup page business logic.

Author: Hugo Demule
Date: January 2026
"""

import os
from typing import Optional

from PyQt6.QtCore import QObject


class MainWindowModel(QObject):
    """Model for managing starting page application state.

    Handles file paths, validation, and startup configuration
    for the audio analysis application initialization.
    """

    def __init__(self):
        """Initialize the main window model."""
        super().__init__()
        self._current_file_path: Optional[str] = None
        self._current_project_path: Optional[str] = None
        self._range_start: Optional[float] = None
        self._range_end: Optional[float] = None
        self._range_selection_mode: bool = False
        self._audio_duration: Optional[float] = None

    @property
    def current_file_path(self) -> Optional[str]:
        """Get the current audio file path."""
        return self._current_file_path

    @current_file_path.setter
    def current_file_path(self, path: Optional[str]):
        """Set the current audio file path."""
        if path is not None and not os.path.exists(path):
            raise ValueError(f"Audio file does not exist: {path}")
        self._current_file_path = path

    @property
    def current_project_path(self) -> Optional[str]:
        """Get the current project file path."""
        return self._current_project_path

    @current_project_path.setter
    def current_project_path(self, path: Optional[str]):
        """Set the current project file path."""
        self._current_project_path = path

    @property
    def audio_duration(self) -> Optional[float]:
        """Get the audio duration in seconds."""
        return self._audio_duration

    @audio_duration.setter
    def audio_duration(self, duration: Optional[float]):
        """Set the audio duration in seconds."""
        if duration is not None and duration < 0:
            raise ValueError("Audio duration cannot be negative")
        self._audio_duration = duration

    @property
    def range_selection_mode(self) -> bool:
        """Get the range selection mode state."""
        return self._range_selection_mode

    @range_selection_mode.setter
    def range_selection_mode(self, enabled: bool):
        """Set the range selection mode state."""
        self._range_selection_mode = enabled

    @property
    def range_start(self) -> Optional[float]:
        """Get the range start position."""
        return self._range_start

    @range_start.setter
    def range_start(self, position: Optional[float]):
        """Set the range start position."""
        if position is not None:
            # Clamp position to valid audio bounds
            clamped_position = max(position, 0.0)
            if self._audio_duration is not None:
                clamped_position = min(clamped_position, self._audio_duration)
            self._range_start = clamped_position
        else:
            self._range_start = None

    @property
    def range_end(self) -> Optional[float]:
        """Get the range end position."""
        return self._range_end

    @range_end.setter
    def range_end(self, position: Optional[float]):
        """Set the range end position."""
        if position is not None:
            # Clamp position to valid audio bounds
            clamped_position = max(position, 0.0)
            if self._audio_duration is not None:
                clamped_position = min(clamped_position, self._audio_duration)
            self._range_end = clamped_position
        else:
            self._range_end = None

    def clear_range_selection(self):
        """Clear the current range selection."""
        self._range_start = None
        self._range_end = None
        self._range_selection_mode = False
        # Note: We don't clear audio_duration as it's still valid for the loaded audio

    def has_valid_range(self) -> bool:
        """Check if there's a valid range selection."""
        return (
            self._range_start is not None
            and self._range_end is not None
            and self._range_start != self._range_end
        )

    def get_range_duration(self) -> Optional[float]:
        """Get the duration of the selected range."""
        if (
            self.has_valid_range()
            and self._range_start is not None
            and self._range_end is not None
        ):
            return abs(self._range_end - self._range_start)
        return None

    def get_ordered_range(self) -> tuple[Optional[float], Optional[float]]:
        """Get the range with start <= end."""
        if (
            not self.has_valid_range()
            or self._range_start is None
            or self._range_end is None
        ):
            return None, None

        start = min(self._range_start, self._range_end)
        end = max(self._range_start, self._range_end)
        return start, end
