"""Controller for audio preview button functionality.

Coordinates between audio preview button view and audio model to handle
preview playback, position tracking, and button state management.

Author: Hugo Demule
Date: January 2026
"""

from PyQt6.QtCore import QObject, pyqtSignal


class AudioPrevButtonController(QObject):
    """Controller for audio preview button operations.

    Manages preview playback control and coordinates between
    the preview button view and audio model.

    Signals:
        position_updated (float): Current playback position in seconds
    """

    # Initialize Signals
    position_updated = pyqtSignal(float)  # Emits current position in seconds

    def __init__(self, view, model):
        super().__init__()
        from models.audio_m import AudioModel
        from views.analysis_page.components.audio_prev_button import AudioPrevButton

        self.view: AudioPrevButton = view
        self.model: AudioModel = model

        self.init_signal_connections()

    def init_signal_connections(self):
        self.view.toggle_button_clicked.connect(self.request_toggle_play_stop)
        self.view.stop_requested.connect(self.model.stop)

    def request_pause(self):
        self.model.pause()
        self.view.set_playing_state(False)

    def request_play(self):
        self.model.play()
        self.view.set_playing_state(True)

    def request_toggle_play_stop(self):
        mode = self.model.toggle_play_stop()
        if mode == 2:
            self.view.set_playing_state(True)
        else:
            self.view.set_playing_state(False)
