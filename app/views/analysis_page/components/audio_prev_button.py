"""Audio preview button widget with playback controls.

Provides a widget for previewing audio files with play/stop functionality
and visual feedback. Integrates with MVC pattern through controller.

Author: Hugo Demule
Date: January 2026
"""

from constants.svg import SVG
from controllers.analysis_page.components.audio_prev_button_c import (
    AudioPrevButtonController,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QPushButton, QWidget


class AudioPrevButton(QWidget):
    """Widget for audio file preview with playback controls.

    Provides play/stop button with visual feedback for audio preview.
    Uses MVC pattern with dedicated controller for playback logic.

    Signals:
        toggle_button_clicked: Emitted when play/stop is toggled
        stop_requested: Emitted when stop is explicitly requested
    """

    # Initialize Signals
    toggle_button_clicked = pyqtSignal()
    stop_requested = pyqtSignal()

    # Reference to controller to avoid garbage collection
    _controller: AudioPrevButtonController

    @staticmethod
    def init(audio_file_path: str) -> "AudioPrevButton":
        """Create AudioPrevButton widget with audio model and controller.

        Args:
            audio_file_path: Path to audio file for preview

        Returns:
            Configured AudioPrevButton widget
        """
        from models.audio_m import AudioModel

        model = AudioModel(audio_file_path=audio_file_path, create_temp_file=False)
        widget = AudioPrevButton()
        controller = AudioPrevButtonController(widget, model)
        widget._controller = controller
        return widget

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create toggle button
        self.toggle_button = QPushButton()
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.clicked.connect(self.on_toggle_button_clicked)

        # Set initial state to play (stopped)
        self.is_playing = False
        self.update_button_icon()

        # Simple layout - just the button
        self.toggle_button.setParent(self)

    def create_play_icon(self) -> QIcon:
        """Create play icon from SVG."""
        play_svg_data = SVG.PLAY_ICON
        renderer = QSvgRenderer(play_svg_data)
        size = renderer.defaultSize() * 1.5
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return QIcon(pixmap)

    def create_stop_icon(self) -> QIcon:
        """Create stop icon from SVG."""
        stop_svg_data = SVG.STOP_ICON
        renderer = QSvgRenderer(stop_svg_data)
        size = renderer.defaultSize() * 1.5
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return QIcon(pixmap)

    def update_button_icon(self):
        """Update button icon based on current state."""
        if self.is_playing:
            self.toggle_button.setIcon(self.create_stop_icon())
        else:
            self.toggle_button.setIcon(self.create_play_icon())

        # Set icon size
        icon_size = self.toggle_button.size()
        if icon_size.width() > 0:
            self.toggle_button.setIconSize(icon_size)

    def reset_button_icon(self):
        """Reset button icon to play state."""
        self.is_playing = False
        self.update_button_icon()

    def on_toggle_button_clicked(self):
        """Handle button click and toggle state."""
        self.is_playing = not self.is_playing
        self.update_button_icon()
        self.toggle_button_clicked.emit()

    def request_stop(self):
        """Handle external stop request."""
        self.is_playing = False
        self.update_button_icon()
        self.stop_requested.emit()

    def set_playing_state(self, is_playing: bool):
        """Set the playing state from external controller."""
        self.is_playing = is_playing
        self.update_button_icon()

    def resizeEvent(self, a0):
        """Handle resize to fit button to widget."""
        super().resizeEvent(a0)
        size = min(self.width(), self.height())
        self.toggle_button.setFixedSize(size, size)
        self.toggle_button.move((self.width() - size) // 2, (self.height() - size) // 2)
        self.update_button_icon()
