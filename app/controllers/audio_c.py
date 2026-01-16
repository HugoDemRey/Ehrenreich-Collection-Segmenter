"""Audio controller for playback control coordination.

Coordinates between audio control views and audio models to handle
playback requests, position updates, and media control operations.

Author: Hugo Demule
Date: January 2026
"""

from PyQt6.QtCore import QObject, pyqtSignal


class AudioController(QObject):
    """Controller for audio playback control coordination.

    Manages communication between audio control bar view and audio model,
    handling play/pause requests and position synchronization.

    Signals:
        position_updated (float): Current playback position in seconds
    """

    # Initialize Signals
    position_updated = pyqtSignal(float)  # Emits current position in seconds

    def __init__(self, view, model):
        super().__init__()
        from models.audio_m import AudioModel
        from views.analysis_page.components.audio_control_bar import AudioControlBar

        self.view: AudioControlBar = view
        self.model: AudioModel = model

        self.init_signal_connections()

    def init_signal_connections(self):
        self.view.play_button_clicked.connect(self.request_play)
        self.view.pause_button_clicked.connect(self.request_pause)
        self.view.spacebar_pressed.connect(self.request_toggle_play_pause)
        self.view.right_arrow_pressed.connect(lambda: self.model.seek_forward())
        self.view.left_arrow_pressed.connect(lambda: self.model.seek_backward())
        self.view.ctrl_right_arrow_pressed.connect(
            lambda: self.model.seek_forward(30.0)
        )
        self.view.ctrl_left_arrow_pressed.connect(
            lambda: self.model.seek_backward(30.0)
        )
        self.model.position_updated.connect(lambda pos: self.position_updated.emit(pos))

    def request_pause(self):
        self.model.pause()
        self.view.highlight_paused()

    def request_play(self):
        self.model.play()
        self.view.highlight_playing()

    def request_toggle_play_pause(self):
        mode = self.model.toggle_play_pause()
        if mode == 2:
            self.view.highlight_playing()
        else:
            self.view.highlight_paused()

    def seek(self, position_seconds: float):
        """Seek to a specific position in seconds."""
        self.model.set_current_position(position_seconds)
