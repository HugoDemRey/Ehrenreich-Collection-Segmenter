"""Audio playback control bar widget for the analysis interface.

Provides play/pause controls with keyboard shortcuts and audio position control
for synchronized audio playback during analysis.

Author: Hugo Demule
Date: January 2026
"""

from constants.svg import SVG
from controllers.audio_c import AudioController
from PyQt6.QtCore import QKeyCombination, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class AudioControlBar(QWidget):
    """Audio playback control widget with play/pause functionality.

    Provides audio control buttons with keyboard shortcuts for controlling
    audio playback during analysis sessions.

    Signals:
        play_button_clicked: User clicked play button
        pause_button_clicked: User clicked pause button
        spacebar_pressed: User pressed spacebar (play/pause toggle)
        Various arrow key signals for seeking
    """

    # Initializing Signals
    play_button_clicked = pyqtSignal()
    pause_button_clicked = pyqtSignal()
    spacebar_pressed = pyqtSignal()
    right_arrow_pressed = pyqtSignal()
    left_arrow_pressed = pyqtSignal()
    ctrl_right_arrow_pressed = pyqtSignal()
    ctrl_left_arrow_pressed = pyqtSignal()

    # Initialize variables

    # Reference to controller to avoid garbage collection
    _controller: AudioController

    @staticmethod
    def init(
        audio_file_path: str, orientation: Qt.Orientation = Qt.Orientation.Vertical
    ) -> "AudioControlBar":
        """Initialize audio control bar with MVC pattern setup.

        Args:
            audio_file_path (str): Path to audio file for playback.
            orientation (Qt.Orientation): Widget orientation. Default: Vertical.

        Returns:
            AudioControlBar: Configured control bar with attached controller.
        """
        from models.audio_m import AudioModel

        model = AudioModel(audio_file_path=audio_file_path)
        widget = AudioControlBar(orientation=orientation)
        controller = AudioController(widget, model)
        widget._controller = controller
        return widget

    def _resize_buttons(self, event):
        """Resize buttons based on widget dimensions."""
        parent_size = (
            self.height()
            if self.orientation == Qt.Orientation.Vertical
            else self.width()
        )
        button_size = int(parent_size * 0.5)
        self.play_button.setFixedSize(button_size, button_size)
        self.pause_button.setFixedSize(button_size, button_size)

    def __init__(
        self, parent=None, orientation: Qt.Orientation = Qt.Orientation.Vertical
    ):
        """Initialize the audio control bar widget.

        Args:
            parent: Parent widget.
            orientation (Qt.Orientation): Control bar orientation. Default: Vertical.
        """
        super().__init__(parent)

        self.orientation = orientation

        # Create layout based on orientation
        if orientation == Qt.Orientation.Vertical:
            ctrl_layout = QVBoxLayout(self)
        else:
            ctrl_layout = QHBoxLayout(self)

        ctrl_layout.setContentsMargins(0, 0, 0, 0)

        # Create play button with SVG icon
        self.play_button = QPushButton()
        play_svg_data = SVG.PLAY_ICON
        play_renderer = QSvgRenderer(play_svg_data)
        play_size = play_renderer.defaultSize() * 1.5  # Make icon 2x bigger
        play_pixmap = QPixmap(play_size)
        play_pixmap.fill(Qt.GlobalColor.transparent)
        from PyQt6.QtGui import QPainter

        painter = QPainter(play_pixmap)
        play_renderer.render(painter)
        painter.end()
        self.play_button.setIcon(QIcon(play_pixmap))
        self.play_button.setIconSize(play_size)  # Set the icon size
        self.play_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.play_button.clicked.connect(self.on_play_button_clicked)

        # Create pause button with SVG icon
        self.pause_button = QPushButton()
        pause_svg_data = SVG.PAUSE_ICON
        pause_renderer = QSvgRenderer(pause_svg_data)
        pause_size = pause_renderer.defaultSize() * 1.5  # Make icon 2x bigger
        pause_pixmap = QPixmap(pause_size)
        pause_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pause_pixmap)
        pause_renderer.render(painter)
        painter.end()
        self.pause_button.setIcon(QIcon(pause_pixmap))
        self.pause_button.setIconSize(pause_size)  # Set the icon size
        self.pause_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_button.clicked.connect(self.on_pause_button_clicked)

        # create space bar listener
        self.space_bar_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.space_bar_shortcut.activated.connect(self.spacebar_pressed.emit)

        self.right_arrow_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.right_arrow_shortcut.activated.connect(self.right_arrow_pressed.emit)

        self.left_arrow_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.left_arrow_shortcut.activated.connect(self.left_arrow_pressed.emit)

        self.ctrl_right_arrow_shortcut = QShortcut(
            QKeySequence(
                QKeyCombination(Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Right)
            ),
            self,
        )
        self.ctrl_right_arrow_shortcut.activated.connect(
            self.ctrl_right_arrow_pressed.emit
        )

        self.ctrl_left_arrow_shortcut = QShortcut(
            QKeySequence(
                QKeyCombination(Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Left)
            ),
            self,
        )
        self.ctrl_left_arrow_shortcut.activated.connect(
            self.ctrl_left_arrow_pressed.emit
        )

        # Add buttons to layout
        ctrl_layout.addWidget(self.play_button)
        ctrl_layout.addWidget(self.pause_button)
        self.setLayout(ctrl_layout)

        # Resize buttons to 50% of parent height and width = height
        self.play_button.resizeEvent = self._resize_buttons
        self.pause_button.resizeEvent = self._resize_buttons

    def on_play_button_clicked(self):
        self.play_button_clicked.emit()

    def on_pause_button_clicked(self):
        self.pause_button_clicked.emit()

    def highlight_playing(self):
        self.play_button.setStyleSheet("background-color: #52b788")
        self.pause_button.setStyleSheet("background-color: ")

    def highlight_paused(self):
        self.pause_button.setStyleSheet("background-color: #ce4257")
        self.play_button.setStyleSheet("background-color: ")
