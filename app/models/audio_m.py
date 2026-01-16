"""Audio model for media playback and file management.

Provides audio file loading, playback control, and caching functionality
with Qt multimedia integration for the audio analysis application.

Author: Hugo Demule
Date: January 2026
"""

import logging

from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

# Import the singleton audio cache manager
from src.io.audio_cache import AudioCache

logger = logging.getLogger(__name__)


class AudioModel(QObject):
    """Model for audio file handling and playback control.

    Manages audio file loading, caching, and media player integration
    with position tracking and playback state management.

    Signals:
        position_updated (float): Current playback position in seconds
    """

    # Initialize Signals
    position_updated = pyqtSignal(float)  # Emits current position in seconds

    def __init__(self, audio_file_path: str, create_temp_file: bool = True):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.temp_audio_file = None
        self.using_cache = create_temp_file

        # Get temp file from cache manager or use original file
        if create_temp_file:
            try:
                print(f"Loading audio file: {audio_file_path}")
                self.temp_audio_file = AudioCache.create_temp_file(audio_file_path)

                if not self.temp_audio_file:
                    raise RuntimeError(
                        f"Failed to create temp file for: {audio_file_path}"
                    )

                print(f"Using temp file: {self.temp_audio_file}")

            except Exception as e:
                print(f"Failed to load audio file {audio_file_path}: {e}")
                raise RuntimeError(f"Failed to load audio file: {audio_file_path}")
        else:
            self.temp_audio_file = audio_file_path

        # Initialize media player with temp file
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        if self.temp_audio_file:
            self.media_player.setSource(QUrl.fromLocalFile(self.temp_audio_file))

        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.media_player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.media_player.errorOccurred.connect(self._on_error_occurred)

        # Timer to update playback position - use longer interval to reduce CPU load
        self.position_timer = QTimer()
        self.position_timer.setInterval(
            200
        )  # 200ms instead of 100ms to reduce conflicts
        self.position_timer.timeout.connect(self._update_position)

        self.current_state = 0  # 0: stop, 1: paused, 2: playing
        self._cleanup_done = False  # Track if cleanup has been performed

    def _on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            print(f"Invalid media detected for: {self.audio_file_path}")
            raise RuntimeError(f"Failed to load audio file: {self.audio_file_path}")
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            print("End of media reached")
            self.stop()
        elif status == QMediaPlayer.MediaStatus.StalledMedia:
            print("Media is stalled - potential playback issue")

    def _on_playback_state_changed(self, state):
        """Handle playback state changes"""
        state_names = {
            QMediaPlayer.PlaybackState.StoppedState: "Stopped",
            QMediaPlayer.PlaybackState.PlayingState: "Playing",
            QMediaPlayer.PlaybackState.PausedState: "Paused",
        }
        print(f"Playback state changed to: {state_names.get(state, 'Unknown')}")

    def _on_error_occurred(self, error, error_string):
        """Handle media player errors"""
        print(f"Media player error: {error} - {error_string}")

    def play(self):
        self.media_player.play()
        self.position_timer.start()
        self.current_state = 2

    def pause(self):
        self.media_player.pause()
        self.position_timer.stop()
        self.current_state = 1

    def stop(self):
        self.media_player.stop()
        self.position_timer.stop()
        self.current_state = 0

    def toggle_play_pause(self) -> int:
        if self.current_state == 2:
            self.pause()
        else:
            self.play()
        return self.current_state

    def toggle_play_stop(self) -> int:
        if self.current_state == 2:
            self.stop()
        else:
            self.play()
        return self.current_state

    def seek_forward(self, seconds: float = 5.0):
        new_position = self.media_player.position() + int(seconds * 1000)
        self.set_current_position(new_position / 1000.0)
        if self.current_state == 1:
            self._update_position()

    def seek_backward(self, seconds: float = 5.0):
        new_position = self.media_player.position() - int(seconds * 1000)
        self.set_current_position(max(0, new_position / 1000.0))
        if self.current_state == 1:
            self._update_position()

    def _update_position(self):
        # Get position in milliseconds, convert to seconds
        position_ms = self.media_player.position()
        self.position_updated.emit(position_ms / 1000.0)

    def get_current_position(self) -> float:
        """Return current playback position in seconds."""
        return self.media_player.position() / 1000.0

    def set_current_position(self, position_seconds: float):
        """Set playback position in seconds."""
        position_ms = int(position_seconds * 1000)
        self.media_player.setPosition(position_ms)

    def cleanup(self):
        """Clean up resources using cache manager"""
        if self._cleanup_done:
            return

        self._cleanup_done = True

        # IMMEDIATE cleanup - don't just rely on deleteLater()
        if hasattr(self, "media_player") and self.media_player is not None:
            self.media_player.stop()

            # Disconnect all signals to break circular references
            try:
                self.media_player.mediaStatusChanged.disconnect()
                self.media_player.playbackStateChanged.disconnect()
                self.media_player.errorOccurred.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signals may not be connected

            # IMMEDIATELY clear source to release file handle
            self.media_player.setSource(QUrl())  # This releases the file handle NOW
            self.media_player.setAudioOutput(None)  # Disconnect audio output NOW

            # Then schedule for Qt deletion
            self.media_player.deleteLater()  # Schedule for Qt deletion

        if hasattr(self, "audio_output"):
            # Disconnect from media player first
            if hasattr(self, "media_player") and self.media_player is not None:
                self.media_player.setAudioOutput(None)

            # Then schedule for deletion
            self.audio_output.deleteLater()  # Schedule for Qt deletion

        if hasattr(self, "position_timer") and self.position_timer is not None:
            self.position_timer.stop()

            # Disconnect timer signals
            try:
                self.position_timer.timeout.disconnect()
            except (TypeError, RuntimeError):
                pass

            self.position_timer.deleteLater()

        # Now release temp file from cache (this decrements reference count)
        if self.using_cache and self.temp_audio_file:
            try:
                AudioCache.release_temp_file(self.temp_audio_file)
            except Exception as e:
                print(f"AudioModel: Error releasing temp file from cache: {e}")

        print("AudioModel: Cleanup completed")

    def __del__(self):
        """Destructor - emergency cleanup if needed"""
        if not getattr(self, "_cleanup_done", False):
            try:
                self.cleanup()
            except:
                pass
