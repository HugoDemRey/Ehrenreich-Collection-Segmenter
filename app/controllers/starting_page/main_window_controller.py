"""Controller for starting page main window functionality.

Coordinates file operations, audio loading, and navigation between
the starting page and analysis page interfaces.

Author: Hugo Demule
Date: January 2026
"""

from typing import Optional

from controllers.audio_c import AudioController
from models.audio_m import AudioModel
from models.starting_page.file_service import FileService
from models.starting_page.main_window_model import MainWindowModel
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication
from src.audio.audio_file import AudioFile
from views.analysis_page.components.audio_control_bar import AudioControlBar


class MainWindowController(QObject):
    """Controller for managing starting page interactions."""

    # Signals
    audio_loaded = pyqtSignal(object, int, str)  # audio_data, sample_rate, file_path
    range_selected = pyqtSignal(float, float)  # start, end
    navigate_to_analysis = pyqtSignal(
        str, float, float, str
    )  # file_path, start, end, project_path
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    loading_progress = pyqtSignal(int)  # progress percentage

    def __init__(self, model: MainWindowModel, file_service: FileService):
        """Initialize the main window controller."""
        super().__init__()
        self.model = model
        self.file_service = file_service
        self.audio_model: Optional[AudioModel] = None
        self.audio_controller: Optional[AudioController] = None
        self.audio_control_bar: Optional[AudioControlBar] = None

    def select_import_file(self, parent=None):
        """Handle file selection (audio or project file)."""
        audio_path, project_path, is_project = (
            self.file_service.select_audio_or_project_file(parent)
        )
        if audio_path:
            # Store both paths in model
            self.model.current_project_path = project_path
            self.model.current_file_path = audio_path

            # Load the audio file
            self.load_audio_file(audio_path)

            return audio_path, project_path, is_project
        return None, None, False

    def load_audio_file(self, file_path: str):
        """Load the audio file and set up audio components."""
        if not file_path:
            return

        self.loading_started.emit()

        # Process events to update UI
        QApplication.processEvents()

        try:
            # Load audio using AudioFile
            self.loading_progress.emit(25)
            audio_file = AudioFile(file_path)
            signal = audio_file.load()

            self.loading_progress.emit(50)

            # Create audio components
            self.audio_control_bar = AudioControlBar.init(
                file_path, orientation=Qt.Orientation.Horizontal
            )
            self.audio_model = self.audio_control_bar._controller.model
            self.audio_controller = self.audio_control_bar._controller

            self.loading_progress.emit(75)

            # Store file path in model
            self.model.current_file_path = file_path

            # Store audio duration for proper range clamping
            self.model.audio_duration = signal.duration_seconds()

            # Emit signal for view to update (only pass file path, view gets info from controller)
            self.audio_loaded.emit(signal.samples, signal.sample_rate, file_path)

            self.loading_progress.emit(100)
            self.loading_finished.emit()

        except Exception as e:
            print(f"Error loading audio file: {e}")
            self.loading_finished.emit()

    def handle_waveform_position_clicked(self, position_seconds: float):
        """Handle position clicks on the waveform."""
        if self.model.range_selection_mode:
            self.handle_range_selection(position_seconds)
        elif self.audio_controller:
            self.audio_controller.seek(position_seconds)

    def handle_range_selection(self, position_seconds: float):
        """Handle range selection logic."""
        if self.model.range_start is None:
            # First click - set start
            self.model.range_start = position_seconds
        elif self.model.range_end is None:
            # Second click - set end
            self.model.range_end = position_seconds
            # Check if range is valid (different positions)
            start, end = self.model.get_ordered_range()
            if start is not None and end is not None:
                # Valid range - emit signal
                self.range_selected.emit(start, end)
            else:
                # Invalid range (same position clicked twice) - emit with zero duration
                # The view will handle the validation and show warning
                self.range_selected.emit(position_seconds, position_seconds)
        else:
            # Third click - reset and start new range
            self.model.clear_range_selection()
            self.model.range_start = position_seconds

    def toggle_range_selection_mode(self):
        """Toggle range selection mode on/off."""
        self.model.range_selection_mode = not self.model.range_selection_mode
        if not self.model.range_selection_mode:
            self.model.clear_range_selection()

    def validate_and_proceed(self):
        """Validate selection and proceed to analysis page."""
        file_path = self.model.current_file_path
        project_path = self.model.current_project_path

        if not file_path:
            return False

        if self.model.has_valid_range():
            start, end = self.model.get_ordered_range()
            self.navigate_to_analysis.emit(file_path, start, end, project_path or "")
        else:
            # No range selected, use full audio
            self.navigate_to_analysis.emit(file_path, 0.0, -1.0, project_path or "")

        return True

    def get_audio_controller(self) -> Optional[AudioController]:
        """Get the current audio controller."""
        return self.audio_controller

    def get_audio_control_bar(self) -> Optional[AudioControlBar]:
        """Get the current audio control bar."""
        return self.audio_control_bar

    def get_audio_info(self) -> Optional[str]:
        """Get formatted audio information."""
        if not self.model.current_file_path:
            return None

        # Since AudioModel uses QMediaPlayer, we can't get detailed audio info from it
        # Instead, we'll calculate it from the original audio data from the controller
        from src.audio.audio_file import AudioFile

        try:
            audio_file = AudioFile(self.model.current_file_path)
            signal = audio_file.load()
            duration = signal.duration()
            sample_rate = signal.sample_rate
            num_samples = len(signal.samples)

            return f"🎵 Audio loaded: {duration:.2f}s, {sample_rate} Hz, {num_samples} samples"
        except Exception as e:
            print(f"Error getting audio info: {e}")
            return "🎵 Audio loaded successfully"

    def get_file_info(self) -> Optional[str]:
        """Get formatted file information."""
        if not self.model.current_file_path or not self.file_service:
            return None

        file_info = self.file_service.get_file_info(self.model.current_file_path)
        if file_info:
            return f"Selected: {file_info['name']} ({file_info['size_mb']} MB)"
        return None

    def get_range_display_text(self) -> Optional[str]:
        """Get formatted text for range selection button."""
        duration = self.model.get_range_duration()
        if duration is not None:
            return f"🎯 Range: {duration:.2f}s"
        return None

    def get_range_info_text(self) -> Optional[str]:
        """Get formatted text for range info display."""
        if self.model.has_valid_range():
            start, end = self.model.get_ordered_range()
            duration = self.model.get_range_duration()
            if start is not None and end is not None and duration is not None:
                return f"🎵 Audio loaded | 🎯 Selected: {duration:.2f}s ({start:.1f}s - {end:.1f}s)"
        return None

    def has_loaded_file(self) -> bool:
        """Check if a file has been loaded."""
        return self.model.current_file_path is not None

    def cleanup_resources(self):
        """Clean up resources before closing."""
        if self.audio_model and hasattr(self.audio_model, "cleanup"):
            self.audio_model.cleanup()
        if self.audio_model:
            self.audio_model.stop()
