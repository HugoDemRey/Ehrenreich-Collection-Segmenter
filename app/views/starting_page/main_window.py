"""Main window for the audio segmenter application startup interface.

Provides the initial user interface for loading audio files, configuring
processing parameters, and launching the audio analysis workflow.

Author: Hugo Demule
Date: January 2026
"""

import os
from typing import Optional

from constants.colors import (
    ACCENT_COLOR,
    ACCENT_COLOR_HOVER,
    VALIDATE_COLOR,
    VALIDATE_COLOR_HOVER,
)
from controllers.starting_page.main_window_controller import MainWindowController
from models.starting_page.file_service import FileService
from models.starting_page.main_window_model import MainWindowModel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from styles.theme import get_starting_page_style
from views.starting_page.waveform_widget import InteractiveWaveformWidget


class MainWindow(QMainWindow):
    """Main application window for audio file loading and configuration.

    Provides the startup interface with file selection, waveform preview,
    and configuration options before launching the analysis interface.
    """

    def __init__(self):
        """Initialize the main window with MVC components and UI setup."""
        super().__init__()

        # Create MVC components
        self.model = MainWindowModel()
        self.file_service = FileService()
        self.controller = MainWindowController(self.model, self.file_service)

        # UI components
        self.waveform_widget: Optional[InteractiveWaveformWidget] = None

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Setup the user interface with scrollable content and styling."""
        self.setWindowTitle("Ehrenreich's Collection Segmenter")
        # Set minimal dimensions and full screen mode
        self.setMinimumSize(1000, 800)
        self.showMaximized()
        self.setStyleSheet(get_starting_page_style())

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Scroll area for main content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        central_layout.addWidget(scroll_area)

        # Main content widget inside scroll area
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        main_layout = QVBoxLayout(content_widget)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Header section
        self.setup_header(main_layout)
        # File selection section
        self.setup_file_selection(main_layout)
        # Loading section
        self.setup_loading_section(main_layout)
        # Audio player section
        self.setup_audio_section(main_layout)

        # Initially hide audio section
        self.audio_section.setVisible(False)
        self.range_select_btn.setVisible(False)
        self.validate_btn.setVisible(False)

    def setup_header(self, layout):
        """Setup the header section."""
        header_frame = QFrame()
        header_frame.setObjectName("card")
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(30, 20, 30, 20)

        # Title
        title = QLabel("Ehrenreich's Collection Segmenter")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Load audio files and segment your musical collections")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(subtitle)

        layout.addWidget(header_frame, stretch=0)

    def setup_file_selection(self, layout):
        """Setup the file selection section."""
        file_frame = QFrame()
        file_frame.setObjectName("card")
        file_layout = QVBoxLayout(file_frame)
        file_layout.setSpacing(20)
        file_layout.setContentsMargins(30, 15, 30, 15)
        file_frame.setMaximumHeight(200)

        # File selection button
        self.select_file_btn = QPushButton("📁 Choose Audio or Project File")
        self.select_file_btn.setStyleSheet(self._get_file_selection_button_style())
        self.select_file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_file_btn.clicked.connect(self.select_import_file)

        # Proceed to Analysis button (initially hidden)
        self.validate_btn = QPushButton("✅ Proceed to Audio Analysis")
        self.validate_btn.setStyleSheet(self._get_validation_button_style())
        self.validate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.validate_btn.clicked.connect(self.validate_selection)
        self.validate_btn.setVisible(False)

        # Range selection button (initially hidden)
        self.range_select_btn = QPushButton("📐 Select Range")
        self.range_select_btn.setStyleSheet(self._get_range_button_style())
        self.range_select_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.range_select_btn.clicked.connect(self.toggle_range_selection)
        self.range_select_btn.setVisible(False)
        # Add tooltip explaining the range selection functionality
        self.range_select_btn.setToolTip(
            "Select a specific time range for analysis.\n"
            "Useful for the alignment module to improve precision\n"
            "by removing silences before and after the music.\n"
            "Note: Range selection has no impact on other analysis modules."
        )

        # Center the buttons
        btn_layout = QHBoxLayout()
        btn_layout.addItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        btn_layout.addWidget(self.select_file_btn)
        btn_layout.addWidget(self.range_select_btn)
        btn_layout.addWidget(self.validate_btn)
        btn_layout.addItem(
            QSpacerItem(
                40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
            )
        )
        file_layout.addLayout(btn_layout)

        # Selected file info
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet(
            "font-size: 14px; color: rgba(255,255,255,0.8); text-align: center;"
        )
        self.file_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_layout.addWidget(self.file_info_label)

        layout.addWidget(file_frame, stretch=0)

    def setup_loading_section(self, layout):
        """Setup the loading section."""
        self.loading_frame = QFrame()
        self.loading_frame.setObjectName("card")
        loading_layout = QVBoxLayout(self.loading_frame)
        loading_layout.setSpacing(15)
        loading_layout.setContentsMargins(30, 25, 30, 25)

        # Loading label
        self.loading_label = QLabel("Loading audio file...")
        self.loading_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(self.loading_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.1);
                text-align: center;
                color: white;
                font-weight: bold;
                min-height: 50px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT_COLOR};
                border-radius: 8px;
            }}
        """
        )
        loading_layout.addWidget(self.progress_bar)

        layout.addWidget(self.loading_frame, stretch=0)
        self.loading_frame.setVisible(False)

    def setup_audio_section(self, layout):
        """Setup the audio player section."""
        self.audio_section = QFrame()
        self.audio_section.setObjectName("card")
        audio_layout = QVBoxLayout(self.audio_section)
        audio_layout.setSpacing(25)
        audio_layout.setContentsMargins(30, 25, 30, 25)

        # Audio info
        self.audio_info_label = QLabel()
        self.audio_info_label.setStyleSheet(
            "font-size: 16px; font-weight: 600; margin-bottom: 15px;"
        )
        self.audio_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        audio_layout.addWidget(self.audio_info_label)

        # Interactive waveform display
        self.waveform_widget = InteractiveWaveformWidget()
        self.waveform_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        audio_layout.addWidget(self.waveform_widget, stretch=1)

        # Control buttons - will be set when audio is loaded
        self.controls_placeholder = QWidget()
        self.controls_placeholder.setVisible(False)
        audio_layout.addWidget(self.controls_placeholder)
        layout.addWidget(self.audio_section, stretch=1)

    def setup_connections(self):
        """Setup signal-slot connections with controller."""
        # Controller signals
        self.controller.audio_loaded.connect(self.on_audio_loaded)
        self.controller.range_selected.connect(self.on_range_selected)
        self.controller.loading_started.connect(self.show_loading)
        self.controller.loading_finished.connect(self.hide_loading)
        self.controller.loading_progress.connect(self.update_progress)
        self.controller.navigate_to_analysis.connect(self.launch_analysis_page)

    def center_on_screen(self):
        """Center the window on the screen."""
        screen = self.screen()
        if screen:
            center_point = screen.availableGeometry().center()
            window_rect = self.frameGeometry()
            window_rect.moveCenter(center_point)
            self.move(window_rect.topLeft())

    def cleanup_current_audio(self):
        """Clean up current audio state before loading new audio."""
        print("MainWindow: Cleaning up current audio state...")

        # Hide audio section and buttons
        self.audio_section.setVisible(False)
        self.range_select_btn.setVisible(False)
        self.validate_btn.setVisible(False)

        # Clean up waveform widget
        if hasattr(self, "waveform_widget") and self.waveform_widget:
            try:
                self.waveform_widget.position_clicked.disconnect()
            except (TypeError, RuntimeError):
                pass  # Signal may not be connected or already deleted

            # Clear waveform data
            if hasattr(self.waveform_widget, "clear_audio_data"):
                self.waveform_widget.clear_audio_data()
            if hasattr(self.waveform_widget, "clear_range_selection"):
                self.waveform_widget.clear_range_selection()

        # Clean up controller resources
        if hasattr(self, "controller") and self.controller:
            self.controller.cleanup_resources()

        # Reset audio controls by recreating the controls placeholder
        audio_layout = self.audio_section.layout()
        if audio_layout:
            # Remove all widgets except the first two (audio_info_label and waveform_widget)
            for i in reversed(range(audio_layout.count())):
                if i >= 2:  # Keep audio_info_label (0) and waveform_widget (1)
                    child = audio_layout.itemAt(i)
                    if child and child.widget() and isinstance(child.widget(), QWidget):
                        child.widget().deleteLater()  # type: ignore
                        audio_layout.removeItem(child)

            # Recreate controls placeholder
            self.controls_placeholder = QWidget()
            self.controls_placeholder.setVisible(False)
            audio_layout.addWidget(self.controls_placeholder)

        # Reset range selection mode
        if hasattr(self, "model") and self.model:
            self.model.range_selection_mode = False

        # Reset button styles
        self.range_select_btn.setText("📐 Select Range")
        self.range_select_btn.setStyleSheet(self._get_range_button_style())

        print("MainWindow: Audio cleanup completed")

    def select_import_file(self):
        """Handle file selection (audio or project file)."""
        # Clean up previous audio state if any exists
        if hasattr(self, "controller") and self.controller.has_loaded_file():
            self.cleanup_current_audio()

        audio_path, project_path, is_project = self.controller.select_import_file(self)
        if audio_path:
            # Update UI to show what type of file was loaded
            if is_project:
                project_name = (
                    os.path.basename(project_path) if project_path else "Unknown"
                )
                audio_name = os.path.basename(audio_path)
                self.file_info_label.setText(
                    f"Project loaded: {project_name}\nAudio file: {audio_name}"
                )
            else:
                audio_name = os.path.basename(audio_path)
                self.file_info_label.setText(f"Audio file selected: {audio_name}")

    def update_progress(self, value: int):
        """Update progress bar value."""
        self.progress_bar.setValue(value)

    def show_loading(self):
        """Show loading animation."""
        self.loading_frame.setVisible(True)
        self.audio_section.setVisible(False)
        self.progress_bar.setValue(0)

    def hide_loading(self):
        """Hide loading animation."""
        self.loading_frame.setVisible(False)

    def on_audio_loaded(self, audio_data, sample_rate, file_path: str):
        """Handle audio loaded signal from controller."""
        # Get formatted info from controller instead of calculating in view
        audio_info = self.controller.get_audio_info()
        file_info = self.controller.get_file_info()

        if file_info:
            self.file_info_label.setText(file_info)

        if audio_info:
            self.audio_info_label.setText(audio_info)

        # Load audio data into waveform widget
        if self.waveform_widget:
            self.waveform_widget.load_audio_data(audio_data, sample_rate)

        # Replace the controls placeholder with actual audio control bar
        audio_control_bar = self.controller.get_audio_control_bar()
        if audio_control_bar and isinstance(audio_control_bar, QWidget):
            audio_layout = self.audio_section.layout()
            if not audio_layout:
                return

            # Only remove placeholder if it still exists
            if hasattr(self, "controls_placeholder") and self.controls_placeholder:
                try:
                    audio_layout.removeWidget(self.controls_placeholder)
                    self.controls_placeholder.deleteLater()
                    self.controls_placeholder = None
                except RuntimeError:
                    # Widget may have already been deleted
                    pass

            # Add centered AudioControlBar with increased height
            controls_container = QWidget()
            controls_container.setMinimumHeight(80)  # Increase height as needed
            centered_controls = QHBoxLayout(controls_container)
            centered_controls.setContentsMargins(0, 0, 0, 0)
            centered_controls.addItem(
                QSpacerItem(
                    40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
                )
            )
            centered_controls.addWidget(audio_control_bar)
            centered_controls.addItem(
                QSpacerItem(
                    40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
                )
            )
            audio_layout.addWidget(controls_container)

            # Connect position updates to waveform
            audio_controller = self.controller.get_audio_controller()
            if audio_controller and self.waveform_widget:
                audio_controller.position_updated.connect(
                    self.waveform_widget.set_position
                )

        # Connect waveform position clicks to controller
        if self.waveform_widget:
            self.waveform_widget.position_clicked.connect(
                self.controller.handle_waveform_position_clicked
            )

        self.audio_section.setVisible(True)
        self.range_select_btn.setVisible(True)
        self.validate_btn.setVisible(True)

    def on_range_selected(self, start: float, end: float):
        """Handle completed range selection from controller."""
        # Validate minimum range duration (15 seconds)
        range_duration = abs(end - start)
        if range_duration < 15.0:
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Range Too Short")
            msg_box.setText(
                f"The selected range is {range_duration:.1f} seconds.\n\n"
                "Minimum required range is 15 seconds.\n\n"
                "Range selection has been cancelled."
            )
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()

            # Cancel the range selection
            self.controller.toggle_range_selection_mode()  # Exit selection mode
            self.model.clear_range_selection()  # Clear the range from model

            # Update UI to reflect cancelled state
            self.range_select_btn.setText("📐 Select Range")
            self.range_select_btn.setStyleSheet(self._get_range_button_style())

            # Clear range visualization
            if hasattr(self, "waveform_widget") and self.waveform_widget:
                self.waveform_widget.clear_range_selection()

            # Reset info label
            if self.controller.has_loaded_file():
                self.audio_info_label.setText("🎵 Audio loaded - ready for analysis")

            return

        # Get formatted range info from controller
        range_display_text = self.controller.get_range_display_text()
        range_info_text = self.controller.get_range_info_text()

        if self.waveform_widget:
            self.waveform_widget.set_range(start, end)

        if range_display_text:
            self.range_select_btn.setText(range_display_text)

        # Automatically exit range selection mode when complete
        self.range_select_btn.setStyleSheet(self._get_range_button_style())

        if range_info_text:
            self.audio_info_label.setText(range_info_text)

        print(f"Range selected: {start:.2f}s to {end:.2f}s")

    def update_range_info(self, message):
        """Update the audio info label with range selection information."""
        self.audio_info_label.setText(f"🎯 Range Selection: {message}")

    def validate_selection(self):
        """Validate and proceed to analysis page - delegate to controller."""
        return self.controller.validate_and_proceed()

    def launch_analysis_page(
        self,
        file_path: str,
        start_seconds: float,
        end_seconds: float,
        project_path: str,
    ):
        """Launch the analysis page with the specified parameters."""
        from views.analysis_page.main_window import AnalysisPageView
        from views.loading_dialog import LoadingWindow

        LoadingWindow.show("Loading Analysis Page", "Preparing analysis environment...")

        # Stop any playing audio before transitioning
        self.controller.cleanup_resources()

        # Create and show analysis page
        audio_start = start_seconds if start_seconds != 0.0 else None
        audio_end = end_seconds if end_seconds != -1.0 else None

        self.analysis_window = AnalysisPageView(
            audio_path=file_path,
            audio_start=audio_start,
            audio_end=audio_end,
            project_path=project_path if project_path else None,
        )

        audio_name = os.path.basename(file_path)
        self.analysis_window.setWindowTitle(f"Analysis Page - {audio_name}")

        # Show with the same size as the current window
        if self.isMaximized():
            self.analysis_window.showMaximized()
        else:
            self.analysis_window.resize(self.size())
            self.analysis_window.move(self.pos())
            self.analysis_window.show()

        # Close the starting page
        self.close()

    def cleanup_resources(self):
        """Clean up all resources, timers, and connections before closing."""
        print("MainWindow: Starting resource cleanup...")

        # Delegate audio cleanup to controller
        if hasattr(self, "controller") and self.controller:
            self.controller.cleanup_resources()

        # Cleanup waveform widget
        if hasattr(self, "waveform_widget") and self.waveform_widget:
            try:
                self.waveform_widget.position_clicked.disconnect()
            except TypeError:
                pass  # Signal may not be connected
            # Clear waveform data to free memory
            if hasattr(self.waveform_widget, "clear_audio_data"):
                self.waveform_widget.clear_audio_data()

        print("MainWindow: Resource cleanup completed")

    def closeEvent(self, a0):
        """Handle window closing - ensure proper cleanup before Qt exits"""
        print("MainWindow: Window closing - performing proper cleanup...")
        self.cleanup_resources()
        print("MainWindow: Cleanup completed - window can now close safely")
        a0.accept() if isinstance(a0, QCloseEvent) else None
        super().closeEvent(a0)

    def _get_file_selection_button_style(self) -> str:
        """Get styling for the file selection button."""
        return f"""
            QPushButton {{
                background: {ACCENT_COLOR};
                border: none;
                border-radius: 15px;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 15px 30px;
                min-width: 200px;
                min-height: 25px;
            }}
            QPushButton:hover {{
                background: {ACCENT_COLOR_HOVER};
            }}
            QPushButton:pressed {{
                background: {ACCENT_COLOR_HOVER};
                opacity: 0.8;
            }}
        """

    def _get_validation_button_style(self) -> str:
        """Get styling for the validation/proceed button with green color."""
        return f"""
            QPushButton {{
                background: {VALIDATE_COLOR};
                border: none;
                border-radius: 15px;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 15px 30px;
                min-width: 200px;
                min-height: 25px;
            }}
            QPushButton:hover {{
                background: {VALIDATE_COLOR_HOVER};
            }}
            QPushButton:pressed {{
                background: {VALIDATE_COLOR_HOVER};
                opacity: 0.8;
            }}
        """

    def _get_range_button_style(self) -> str:
        """Get styling for the range selection button."""
        return """
            QPushButton {
                background: #6c757d;
                border: none;
                border-radius: 15px;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 15px 30px;
                min-width: 120px;
                min-height: 25px;
            }
            QPushButton:hover {
                background: #5a6268;
            }
            QPushButton:pressed {
                background: #545b62;
            }
        """

    def toggle_range_selection(self):
        """Toggle range selection mode - delegate to controller."""
        # Safety check to ensure we have a loaded file
        if not (
            hasattr(self, "controller")
            and self.controller
            and self.controller.has_loaded_file()
        ):
            return

        self.controller.toggle_range_selection_mode()

        # Update UI based on model state
        if self.model.range_selection_mode:
            self.range_select_btn.setText("📐 Cancel Range")
            self.range_select_btn.setStyleSheet(
                """
                QPushButton {
                    background: #dc3545;
                    border: none;
                    border-radius: 15px;
                    color: white;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 15px 30px;
                    min-width: 120px;
                    min-height: 25px;
                }
                QPushButton:hover {
                    background: #c82333;
                }
                QPushButton:pressed {
                    background: #bd2130;
                }
            """
            )
            # Clear any existing range visualization
            if hasattr(self, "waveform_widget") and self.waveform_widget:
                self.waveform_widget.clear_range_selection()
            # Update info label
            self.update_range_info("Click on waveform to set start point")
        else:
            self.range_select_btn.setText("📐 Select Range")
            self.range_select_btn.setStyleSheet(self._get_range_button_style())
            # Clear range selection visualization
            if hasattr(self, "waveform_widget") and self.waveform_widget:
                self.waveform_widget.clear_range_selection()
            # Reset info label through controller
            if self.controller.has_loaded_file():
                self.audio_info_label.setText("🎵 Audio loaded - ready for analysis")

    def select_timestamps(self):
        """Select timestamp file and load timestamps."""
        if self.file_service:
            file_path = self.file_service.select_timestamp_file(self)
            if file_path:
                timestamps = self.file_service.extract_timestamps(file_path)
                if self.waveform_widget:
                    self.waveform_widget.load_segmentation_timestamps(timestamps)
