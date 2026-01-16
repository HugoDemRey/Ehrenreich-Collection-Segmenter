"""Controller for navigation bar functionality.

Handles menu actions, file operations, navigation commands,
and coordinates between navbar view and application services.

Author: Hugo Demule
Date: January 2026
"""

import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

from models.analysis_page.components.navbar_m import NavBarModel
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from src.audio.audio_file import AudioFile
from src.audio.signal import Signal
from src.io.audio_cache import AudioCache


class NavBarController(QObject):
    """Controller for navigation bar menu actions and operations.

    Handles file operations, export functionality, navigation commands,
    and coordinates between navbar view and application models.
    """

    s_project_save_requested = pyqtSignal(str)  # file_path
    s_project_load_requested = pyqtSignal(str)  # file_path

    def __init__(self, view, model: NavBarModel):
        from views.analysis_page.components.navbar import Navbar

        super().__init__()

        self.view: Navbar = view
        self.model = model
        self.connect_signals()

    def connect_signals(self):
        """Connect signals from the view to controller methods."""
        self.view.s_load_project.connect(self.on_load_project_requested)
        self.view.s_save_project.connect(self.on_save_project_requested)
        self.view.s_load_transitions.connect(self.on_load_transitions)
        self.view.s_save_transitions.connect(self.on_save_transitions)
        self.view.s_export_audio_segments.connect(self.on_export_audio_segments)
        self.view.s_clear_naxos_cache.connect(self.on_clear_cache)

    def _select_transitions_file(self, parent=None, directory=None) -> Optional[str]:
        """Show file dialog to select transitions file for loading."""
        filter_string = "Transitions Files (*.json)"
        start_directory = directory or ""

        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Select Transitions File", start_directory, filter_string
        )

        return file_path if file_path else None

    def _select_save_file(
        self, default_name="transitions.json", parent=None, directory=None
    ) -> Optional[str]:
        """Show file dialog to select file path for saving transitions."""
        filter_string = "Transitions Files (*.json)"

        if directory and default_name:
            default_path = os.path.join(directory, default_name)
        else:
            default_path = default_name or ""

        file_path, _ = QFileDialog.getSaveFileName(
            parent, "Save Transitions File", default_path, filter_string
        )

        return file_path if file_path else None

    def _select_project_save_file(
        self, default_name="project.ehra", parent=None, directory=None
    ) -> Optional[str]:
        """Show file dialog to select file path for saving project."""
        filter_string = "Ehrenreich Analysis Files (*.ehra)"

        if directory and default_name:
            default_path = os.path.join(directory, default_name)
        else:
            default_path = default_name or ""

        file_path, _ = QFileDialog.getSaveFileName(
            parent, "Save Project", default_path, filter_string
        )

        return file_path if file_path else None

    def _select_project_load_file(self, parent=None, directory=None) -> Optional[str]:
        """Show file dialog to select file path for loading project."""
        filter_string = "Ehrenreich Analysis Files (*.ehra)"
        start_directory = directory or ""

        file_path, _ = QFileDialog.getOpenFileName(
            parent, "Load Project", start_directory, filter_string
        )

        return file_path if file_path else None

    def on_load_transitions(self):
        print("Loading transitions...")
        parent_dir = os.path.dirname(
            self.view.out_controller.view.signal.origin_filename
        )

        # Use file dialog to get file path
        filepath = self._select_transitions_file(directory=parent_dir)
        if not filepath:
            return

        # Load transitions using model
        transitions_data = self.model.load_transitions_from_file(filepath)
        if transitions_data is None:
            print("No transitions loaded.")
            return

        # Validate data using model
        if not self.model.validate_transitions_data(transitions_data):
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox(
                QMessageBox.Icon.Warning,
                "Invalid Data",
                "The selected file contains invalid transition data.",
                QMessageBox.StandardButton.Ok,
            )
            msg_box.exec()
            return

        success = True

        success = success and self.view.out_controller.remove_all_transitions()

        # Handle both old format (list of floats) and new format (list of dicts with timestamp and color)
        if transitions_data and len(transitions_data) > 0:
            first_item = transitions_data[0]
            if isinstance(first_item, dict) and "timestamp" in first_item:
                # New format with colors
                for item in transitions_data:
                    if isinstance(item, dict):
                        timestamp = item["timestamp"]
                        color = item.get("color", "white")
                        success = success and self.view.out_controller.add_transition(
                            timestamp, color=color
                        )

        if success:
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox(
                QMessageBox.Icon.Information,
                "Success",
                "Transitions loaded successfully.",
                QMessageBox.StandardButton.Ok,
            )
            msg_box.exec()
        else:
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox(
                QMessageBox.Icon.Warning,
                "Warning",
                "Some transitions could not be loaded properly.",
                QMessageBox.StandardButton.Ok,
            )
            msg_box.exec()

    def on_save_transitions(self):
        print("Saving transitions...")
        if not hasattr(self.view, "out_controller"):
            raise AttributeError("NavBarView is not linked to Timeline Controller.")

        transitions = self.view.out_controller.view.transitions
        transitions_lines = self.view.out_controller.view.transition_lines
        transitions_colors = [line.get_color() for line in transitions_lines]

        # Use model to format transitions data
        transitions_data = self.model.format_transitions_for_export(
            transitions, transitions_colors
        )

        default_name = (
            os.path.basename(
                self.view.out_controller.view.signal.origin_filename
            ).split(".")[0]
            + "_segmentation.json"
        )
        parent_dir = os.path.dirname(
            self.view.out_controller.view.signal.origin_filename
        )

        # Use file dialog to get save path
        filepath = self._select_save_file(
            default_name=default_name, directory=parent_dir
        )
        if not filepath:
            return

        # Save using model
        success = self.model.save_transitions_to_file(transitions_data, filepath)

        if success:
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox(
                QMessageBox.Icon.Information,
                "Success",
                "Transitions saved successfully.",
                QMessageBox.StandardButton.Ok,
            )
            msg_box.exec()
        else:
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox(
                QMessageBox.Icon.Warning,
                "Warning",
                "Failed to save transitions.",
                QMessageBox.StandardButton.Ok,
            )
            msg_box.exec()

    def on_export_audio_segments(self):
        print("Exporting audio segments...")
        if not hasattr(self.view, "out_controller"):
            raise AttributeError("NavBarView is not linked to Timeline Controller.")

        transitions = self.view.out_controller.view.transitions
        transitions = sorted(transitions)
        signal: Signal = self.view.out_controller.view.signal

        subsignals: list[Signal] = []

        # Create subsignals based on transition points
        print("Number of segments to export:", len(transitions) + 1)
        segment_timestamps = [0.0] + transitions + [signal.duration_seconds()]
        for i in range(len(segment_timestamps) - 1):
            start = segment_timestamps[i]
            end = segment_timestamps[i + 1]
            subsignals.append(signal.subsignal(start, end))
            print(f"Segment {i + 1}: {start:.2f}s to {end:.2f}s")

        # User Choose a Directory to Save Segments
        out_dir_parent = QFileDialog.getExistingDirectory(
            None,
            "Select Directory to Save Audio Segments",
            os.path.dirname(
                signal.origin_filename
            ),  # Default to audio file's directory
        )

        # Check if user cancelled the dialog
        if not out_dir_parent:
            print("Export cancelled by user.")
            return

        basename = os.path.basename(signal.origin_filename)
        out_dir_temp = os.path.join(
            out_dir_parent, f"{os.path.splitext(basename)[0]}_audio_segments"
        )
        out_dir = out_dir_temp
        counter = 1
        while os.path.exists(out_dir):
            out_dir = f"{out_dir_temp}_{counter}"
            counter += 1

        os.makedirs(out_dir, exist_ok=True)

        description = (
            f"Ehrenreich Collection Segmenter - Audio Segments Export\n"
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f" - Original Audio File: {os.path.basename(signal.origin_filename)}\n"
            f" - Original File Path: {os.path.abspath(signal.origin_filename)}\n"
            f" - Number of Segments: {len(subsignals)}\n\n"
            "All segments in this folder were extracted from the above audio file using transition points."
        )

        with open(os.path.join(out_dir, "README.txt"), "w") as f:
            f.write(description)

        for idx, subsignal in enumerate(subsignals):
            segment_filename = f"segment_{idx + 1}.wav"
            AudioFile.save(os.path.join(out_dir, segment_filename), signal=subsignal)

        # Open the output directory in file explorer
        self._open_folder(out_dir)
        print(f"Audio segments exported successfully to: {out_dir}")

    def _open_folder(self, folder_path):
        """Open the specified folder in the system's file explorer."""
        try:
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", folder_path])
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            print(f"Could not open folder: {e}")

    def on_save_project_requested(self):
        """Handle save project request - only get file path."""
        print("Selecting save path for project...")

        # Get default name based on audio file
        audio_filename = os.path.basename(
            self.view.out_controller.view.signal.origin_filename
        )
        default_name = f"{os.path.splitext(audio_filename)[0]}_project.ehra"
        parent_dir = os.path.dirname(
            self.view.out_controller.view.signal.origin_filename
        )

        file_path = self._select_project_save_file(
            default_name=default_name, parent=self.view, directory=parent_dir
        )

        if file_path:
            # Emit signal with file path - main window will handle actual saving
            self.s_project_save_requested.emit(file_path)
        else:
            print("Save cancelled by user.")

    def on_load_project_requested(self):
        """Handle load project request - only get file path."""
        print("Selecting project file to load...")

        file_path = self._select_project_load_file(parent=self.view)

        if file_path:
            # Emit signal with file path - main window will handle actual loading
            self.s_project_load_requested.emit(file_path)
        else:
            print("Load cancelled by user.")

    def on_clear_cache(self):
        from PyQt6.QtWidgets import QMessageBox

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText(
            "Are you sure you want to clear the Naxos cache?\n"
            "This cache allows you to paste a naxos URL and retrieve its content quickly if you already performed the operation once.\n\n"
            "This will delete all cached audio previews. "
            "The Naxos module will have to re-scrape naxos.com to re-download previews afterward, which may take some time."
        )
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )

        result = msg_box.exec()

        if result != QMessageBox.StandardButton.Ok:
            return

        AudioCache.clear_naxos_cache()
