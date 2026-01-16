"""File service model for file operations and management.

Handles file selection dialogs, file validation, and persistence
operations for audio files and annotation data.

Author: Hugo Demule
Date: January 2026
"""

import os
import pickle
from typing import List, Optional

from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QFileDialog
from src.io.ts_annotation import TSAnnotations
from pathlib import Path


class FileService(QObject):
    """Service for file operations and dialog management.

    Provides file selection dialogs, validation, and persistence
    for audio files and time-stamped annotation data.
    """

    def __init__(self):
        """Initialize the file service."""
        super().__init__()

        # Supported audio formats
        self.supported_formats = ["*.wav", "*.mp3"]

    def select_audio_file(self, parent=None) -> Optional[str]:
        """
        Open a file dialog to select an audio file.

        Args:
            parent: Parent widget for the dialog

        Returns:
            Selected file path or None if cancelled
        """
        filter_string = f"Audio Files ({' '.join(self.supported_formats)})"

        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Select Audio File", "", filter_string
        )

        return file_path if file_path else None

    def select_timestamp_file(self, parent=None) -> Optional[str]:
        """
        Open a file dialog to select a timestamp file.

        Args:
            parent: Parent widget for the dialog

        Returns:
            Selected file path or None if cancelled
        """
        filter_string = "Timestamp Files (*.txt);;All Files (*)"

        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Select Timestamp File", "", filter_string
        )

        return file_path if file_path else None

    def select_project_file(self, parent=None) -> Optional[str]:
        """
        Open a file dialog to select a project file (.ehra).

        Args:
            parent: Parent widget for the dialog

        Returns:
            Selected file path or None if cancelled
        """
        filter_string = "Project Files (*.ehra);;All Files (*)"

        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Select Project File", "", filter_string
        )

        return file_path if file_path else None

    def extract_audio_path_from_project(self, project_path: str) -> Optional[str]:
        """
        Extract audio file path from a .ehra project file.

        Args:
            project_path: Path to the .ehra project file

        Returns:
            Audio file path or None if extraction fails
        """
        try:
            with open(project_path, "rb") as f:
                session_data = pickle.load(f)

            audio_path = session_data.get("session_metadata", {}).get("audio_file_path")

            if audio_path and os.path.exists(audio_path):
                return audio_path
            else:
                print(f"Audio file not found at: {audio_path}")
                return None

        except Exception as e:
            print(f"Failed to extract audio path from project file: {e}")
            return None
        
    def _get_default_audio_folder(self) -> str:
        """Get the default home folder for the current OS."""
        try:
            return str(Path.home())
        except:
            # Final fallback to current directory
            return ""

    def select_audio_or_project_file(
        self, parent=None
    ) -> tuple[Optional[str], Optional[str], bool]:
        """
        Open a file dialog to select either an audio file or project file.

        Args:
            parent: Parent widget for the dialog

        Returns:
            Tuple of (audio_path, project_path, is_project_file)
        """
        audio_formats = " ".join(self.supported_formats)
        filter_string = (
            f"Audio Files ({audio_formats});;Project Files (*.ehra);;All Files (*)"
        )

        default_folder = self._get_default_audio_folder()

        file_path, selected_filter = QFileDialog.getOpenFileName(
            parent, "Select Audio or Project File", default_folder, filter_string
        )

        if not file_path:
            return None, None, False

        # Check if it's a project file
        if file_path.lower().endswith(".ehra"):
            audio_path = self.extract_audio_path_from_project(file_path)
            return audio_path, file_path, True
        else:
            return file_path, None, False

    def extract_timestamps(self, file_path: str) -> List[float]:
        """
        Extract timestamps from a given file.

        Args:
            file_path: Path to the timestamp file

        Returns:
            List of timestamps in seconds
        """
        return TSAnnotations.load_transitions_txt(file_path)

    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate if the file is a supported audio file.

        Args:
            file_path: Path to the file

        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(file_path):
            return False

        _, ext = os.path.splitext(file_path.lower())
        return f"*{ext}" in self.supported_formats

    def get_file_info(self, file_path: str) -> dict:
        """
        Get basic file information.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            return {}

        stat = os.stat(file_path)
        return {
            "name": os.path.basename(file_path),
            "path": file_path,
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": os.path.splitext(file_path)[1].lower(),
        }
