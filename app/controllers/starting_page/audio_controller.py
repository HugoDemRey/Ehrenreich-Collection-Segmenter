"""Audio controller for starting page playback management.

Manages audio playback state and coordinates playback operations
for the starting page audio preview functionality.

Author: Hugo Demule
Date: January 2026
"""

from PyQt6.QtCore import QObject, pyqtSignal


class AudioController(QObject):
    """Controller for starting page audio playback operations.

    Handles play/pause/stop/seek requests and coordinates between
    audio controls and playback services.

    Signals:
        play_requested: Request to start playback
        pause_requested: Request to pause playback
        stop_requested: Request to stop playback
        seek_requested (float): Request to seek to position in seconds
    """

    # Signals
    play_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    seek_requested = pyqtSignal(float)  # position in seconds

    def __init__(self):
        """Initialize the audio controller."""
        super().__init__()
        self.is_playing = False
        self.is_paused = False

    def play(self):
        """Request audio playback."""
        self.play_requested.emit()
        self.is_playing = True
        self.is_paused = False

    def pause(self):
        """Request audio pause."""
        self.pause_requested.emit()
        self.is_playing = False
        self.is_paused = True

    def stop(self):
        """Request audio stop."""
        self.stop_requested.emit()
        self.is_playing = False
        self.is_paused = False

    def seek(self, position_seconds: float):
        """Request seek to position."""
        self.seek_requested.emit(position_seconds)

    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()
