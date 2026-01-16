"""Audio service model for playback and processing functionality.

Provides audio loading, playback control, and processing capabilities
for the starting page audio preview and analysis preparation.

Author: Hugo Demule
Date: January 2026
"""

import os
import tempfile
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from PyQt6.QtCore import QObject, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer


class AudioService(QObject):
    """Service for audio loading, processing, and playback."""

    # Signals
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()
    loading_progress = pyqtSignal(int)  # Progress percentage
    audio_loaded = pyqtSignal(
        np.ndarray, int, str
    )  # audio_data, sample_rate, file_path
    playback_position_changed = pyqtSignal(float)  # Position in seconds
    playback_state_changed = pyqtSignal(str)  # "playing", "paused", "stopped"

    def __init__(self):
        """Initialize the audio service."""
        super().__init__()

        # Audio data
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[float] = None
        self.duration: Optional[float] = None

        # Playback
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        # Position tracking timer
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self._update_position)
        self.position_timer.setInterval(100)  # Update every 100ms

        # Connect media player signals
        self.media_player.playbackStateChanged.connect(self._on_playback_state_changed)

        # Temporary file for playback
        self.temp_audio_file: Optional[str] = None

    def load_audio_file(self, file_path: str) -> bool:
        """
        Load an audio file asynchronously.

        Args:
            file_path: Path to the audio file

        Returns:
            True if loading started successfully, False otherwise
        """
        try:
            self.loading_started.emit()

            # Load audio using librosa
            self.loading_progress.emit(50)
            audio_data, sample_rate = librosa.load(file_path, sr=None)

            # Store audio data
            self.audio_data = audio_data
            self.sample_rate = sample_rate
            self.duration = len(audio_data) / sample_rate

            self.loading_progress.emit(75)

            # Create temporary file for playback
            self._create_temp_playback_file()

            self.loading_progress.emit(100)
            self.loading_finished.emit()
            self.audio_loaded.emit(self.audio_data, self.sample_rate, file_path)

            return True

        except Exception as e:
            print(f"Error loading audio file: {e}")
            self.loading_finished.emit()
            return False

    def _create_temp_playback_file(self):
        """Create a temporary audio file for Qt media player."""
        try:
            # Stop and clear media player first to release file handles
            self.media_player.stop()
            self.position_timer.stop()
            self.media_player.setSource(QUrl())  # Clear the source

            # Clean up previous temp file
            if self.temp_audio_file and os.path.exists(self.temp_audio_file):
                try:
                    os.remove(self.temp_audio_file)
                except PermissionError:
                    # If file is still locked, try a few times with delay
                    import time

                    for i in range(3):
                        time.sleep(0.1)
                        try:
                            os.remove(self.temp_audio_file)
                            break
                        except PermissionError:
                            if i == 2:  # Last attempt
                                print(
                                    f"Warning: Could not remove old temp file: {self.temp_audio_file}"
                                )

            # Create new temp file
            fd, self.temp_audio_file = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            # Write audio data to temp file
            sf.write(self.temp_audio_file, self.audio_data, self.sample_rate)

            # Set media source
            self.media_player.setSource(QUrl.fromLocalFile(self.temp_audio_file))

        except Exception as e:
            print(f"Error creating temp playback file: {e}")

    def play(self):
        """Start audio playback."""
        if self.temp_audio_file:
            self.media_player.play()
            self.position_timer.start()

    def pause(self):
        """Pause audio playback."""
        self.media_player.pause()
        self.position_timer.stop()

    def stop(self):
        """Stop audio playback."""
        self.media_player.stop()
        self.position_timer.stop()
        self.playback_position_changed.emit(0.0)

    def seek(self, position_seconds: float):
        """
        Seek to a specific position in the audio.

        Args:
            position_seconds: Position in seconds
        """
        if self.duration:
            position_ms = int(position_seconds * 1000)
            self.media_player.setPosition(position_ms)

    def get_duration(self) -> float:
        """Get audio duration in seconds."""
        return self.duration or 0.0

    def get_audio_data(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get the loaded audio data and sample rate."""
        return self.audio_data, self.sample_rate

    def _update_position(self):
        """Update current playback position."""
        if self.media_player.duration() > 0:
            position_ms = self.media_player.position()
            position_seconds = position_ms / 1000.0
            self.playback_position_changed.emit(position_seconds)

    def _on_playback_state_changed(self, state):
        """Handle playback state changes."""
        state_map = {
            QMediaPlayer.PlaybackState.PlayingState: "playing",
            QMediaPlayer.PlaybackState.PausedState: "paused",
            QMediaPlayer.PlaybackState.StoppedState: "stopped",
        }
        self.playback_state_changed.emit(state_map.get(state, "unknown"))

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
            except:
                pass
