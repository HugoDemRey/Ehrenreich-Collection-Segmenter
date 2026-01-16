"""Main analysis interface for audio segmentation and structural analysis.

Provides the primary analysis window with interactive plots, parameter controls,
and real-time audio analysis visualization.

Author: Hugo Demule
Date: January 2026
"""

from typing import Optional

import numpy as np
from constants.parameters import *
from constants.segmenter_configurations import SegmenterConfig
from controllers.analysis_page.managers.plot_sync_manager import PlotSyncManager
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QKeyCombination, Qt, pyqtBoundSignal, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from src.audio.audio_file import AudioFile
from src.io.session_cache import SessionCache

# Import the singleton audio cache manager for cleanup
from styles.theme import get_analysis_page_style
from views.analysis_page.components.audio_control_bar import AudioControlBar
from views.analysis_page.components.expand_collapse_button import SectionHeader
from views.analysis_page.components.navbar import Navbar
from views.analysis_page.modules.combination_segmenter_with_config import (
    CombinationSegmenterWithConfig,
)
from views.analysis_page.modules.naxos_module import NaxosModule
from views.analysis_page.modules.segmenter_with_config import SegmenterWithConfig
from views.analysis_page.modules.timeline import Timeline


# Plot widget template
class RandomPlotWidget(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(title))
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.plot_random_data()

    def plot_random_data(self):
        ax = self.fig.add_subplot(111)
        data = np.random.randn(100).cumsum()
        ax.plot(data)
        # Placeholder for preview cursor
        self.preview_cursor = ax.axvline(0, color="red", linewidth=2)
        self.canvas.draw()

    # In a real app: add click handling, cursor updates, etc.


class SingleAnalysisPageWidget(QWidget):
    """A single analysis page widget for one segment of audio"""

    def __init__(
        self, signal, page_number: int, s_page_changed: pyqtBoundSignal, parent=None
    ):
        super().__init__(parent)
        self.signal = signal
        self.page_number = page_number
        self.s_page_changed = s_page_changed

        # Main vertical layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Scrollable area for feature modules and API module ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(10, 10, 10, 10)
        scroll_content_layout.setSpacing(5)  # Reduced spacing for tighter layout
        scroll_content_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop
        )  # Keep everything aligned to top

        # --- FeatureModules1 ---
        silence_seg, hrps_seg = self._setup_feature_module1(scroll_content_layout)

        # --- FeatureModules2 ---
        chromagram_seg, mfcc_seg, tempogram_seg, combination_plot_seg = (
            self._setup_feature_module2(scroll_content_layout)
        )

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, stretch=1)

        # Store controllers for sync manager
        self.controllers = {
            "mfcc": mfcc_seg._controller,
            "chromagram": chromagram_seg._controller,
            "tempogram": tempogram_seg._controller,
            "combination": combination_plot_seg._controller,
            "silence": silence_seg._controller,
            "hrps": hrps_seg._controller,
        }

        self.setLayout(layout)

    def _setup_feature_module1(self, scroll_content_layout: QVBoxLayout):
        # --- FeatureModules ---
        modules = QWidget()
        self._modules1 = modules  # Store reference for expansion control
        modules_layout1 = QHBoxLayout(modules)
        modules_layout1.setContentsMargins(0, 0, 0, 0)
        modules_layout1.setSpacing(20)

        silence_segmenter = SegmenterWithConfig.init(
            "Silence Curve Analysis",
            self.signal,
            SegmenterConfig.SILENCE,
            id=str(self.page_number),
        )

        hrps_segmenter = SegmenterWithConfig.init(
            "Applause Curve Analysis (HRPS)",
            self.signal,
            SegmenterConfig.HRPS,
            id=str(self.page_number),
        )

        modules_layout1.addWidget(silence_segmenter)
        modules_layout1.addWidget(hrps_segmenter)

        # Add section header with expand/collapse functionality
        self.section_header1 = SectionHeader(
            "Basic Segmentation Modules", initially_collapsed=True
        )
        modules.setVisible(False)  # Initially collapsed

        # Connect the toggle signal
        self.section_header1.section_toggled.connect(modules.setVisible)

        # Create a container for the section to maintain consistent spacing
        section_container = QWidget()
        section_container_layout = QVBoxLayout(section_container)
        section_container_layout.setContentsMargins(0, 0, 0, 0)
        section_container_layout.setSpacing(5)
        section_container_layout.addWidget(self.section_header1)
        section_container_layout.addWidget(modules)

        scroll_content_layout.addWidget(section_container)

        # Add spacing between sections
        scroll_content_layout.addSpacing(15)

        return silence_segmenter, hrps_segmenter

    def _setup_feature_module2(self, scroll_content_layout: QVBoxLayout):
        # --- FeatureModules Container ---
        modules_container = QWidget()
        self._modules2 = modules_container  # Store reference for expansion control
        modules_container_layout = QVBoxLayout(modules_container)
        modules_container_layout.setContentsMargins(0, 0, 0, 0)
        modules_container_layout.setSpacing(10)

        # Top row: Chromagram + MFCC
        top_row = QWidget()
        top_row_layout = QHBoxLayout(top_row)
        top_row_layout.setContentsMargins(0, 0, 0, 0)
        top_row_layout.setSpacing(20)

        chromagram_module = SegmenterWithConfig.init(
            "Chromagram Analysis",
            self.signal,
            SegmenterConfig.CHROMAGRAM,
            id=str(self.page_number),
        )

        mfcc_module = SegmenterWithConfig.init(
            "MFCC Analysis", self.signal, SegmenterConfig.MFCC, id=str(self.page_number)
        )

        top_row_layout.addWidget(chromagram_module)
        top_row_layout.addWidget(mfcc_module)

        # Bottom row: Tempogram + Combination (using Tempogram for testing)
        bottom_row = QWidget()
        bottom_row_layout = QHBoxLayout(bottom_row)
        bottom_row_layout.setContentsMargins(0, 0, 0, 0)
        bottom_row_layout.setSpacing(20)

        tempogram_module = SegmenterWithConfig.init(
            "Tempogram Analysis",
            self.signal,
            SegmenterConfig.TEMPOGRAM,
            id=str(self.page_number),
        )

        combination_module = SegmenterWithConfig.init(
            "Novelty Curves Combination",
            self.signal,
            SegmenterConfig.TEMPOGRAM,
            id=str(self.page_number),
        )

        combination_module = CombinationSegmenterWithConfig.init(
            "Novelty Curves Combination",
            self.signal,
            SegmenterConfig.NC_COMBINATION,
            [
                chromagram_module._controller,
                mfcc_module._controller,
                tempogram_module._controller,
            ],  # type: ignore
            id=str(self.page_number),
        )

        bottom_row_layout.addWidget(tempogram_module)
        bottom_row_layout.addWidget(combination_module)

        # Add rows to container
        modules_container_layout.addWidget(top_row)
        modules_container_layout.addWidget(bottom_row)

        # Add section header with expand/collapse functionality
        self.section_header2 = SectionHeader(
            "Novelty Curve Modules", initially_collapsed=True
        )
        modules_container.setVisible(False)  # Initially collapsed

        # Connect the toggle signal
        self.section_header2.section_toggled.connect(modules_container.setVisible)

        # Create a container for the section to maintain consistent spacing
        section_container = QWidget()
        section_container_layout = QVBoxLayout(section_container)
        section_container_layout.setContentsMargins(0, 0, 0, 0)
        section_container_layout.setSpacing(5)
        section_container_layout.addWidget(self.section_header2)
        section_container_layout.addWidget(modules_container)

        scroll_content_layout.addWidget(section_container)

        # Add spacing between sections
        scroll_content_layout.addSpacing(15)

        return chromagram_module, mfcc_module, tempogram_module, combination_module

    def set_modules_expansion(self, expand1, expand2):
        self.section_header1.set_expanded(expand1)
        self.section_header2.set_expanded(expand2)

        # Manually update content visibility to match header state
        self._modules1.setVisible(expand1)
        self._modules2.setVisible(expand2)

    def get_modules_expansion(self):
        return self.section_header1.is_expanded, self.section_header2.is_expanded


class AnalysisPageView(QWidget):
    # Signal emitted when page changes
    s_page_changed = pyqtSignal(
        int, float, float
    )  # current_page, start_time_seconds, end_time_seconds

    def __init__(
        self,
        audio_path: str,
        audio_start=None,
        audio_end=None,
        project_path: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet(get_analysis_page_style())

        # Flag to track if close is programmatic (restart) vs user-initiated
        self._programmatic_close = False

        # Store project path for later loading
        self.project_path = project_path

        # Constants for pagination
        self.PAGE_DURATION = 16 * 60  # 16 minutes in seconds
        self.OVERLAP_DURATION = 1 * 60  # 1 minute overlap in seconds

        # Emit initial page changed signal
        self.s_page_changed.emit(0, 0.0, self.PAGE_DURATION)

        self.signal = AudioFile(audio_path).load()
        print("signal samples", self.signal.samples.shape)
        print("type", type(self.signal.samples[0]))

        SessionCache.create_session(audio_file_path=audio_path, signal=self.signal)

        # Calculate pages
        self.pages_data = self._calculate_pages()
        self.current_page = 0

        # Main vertical layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Navigation bar at the top ---
        navbar = Navbar.init()
        main_layout.addWidget(navbar, stretch=0)

        navbar._controller.s_project_save_requested.connect(self.on_save_project)
        navbar._controller.s_project_load_requested.connect(self.on_load_project)

        # --- Page navigation controls ---
        page_nav_widget = self._setup_page_navigation()
        main_layout.addWidget(page_nav_widget, stretch=0)

        # --- Stacked widget for pages ---
        self.stacked_widget = QStackedWidget()
        self.analysis_pages = []
        self.sync_managers = []

        # Create all analysis pages
        for i, (start_time, end_time) in enumerate(self.pages_data):
            page_signal = self.signal.subsignal(start_time, end_time)
            analysis_page = SingleAnalysisPageWidget(
                page_signal, i + 1, self.s_page_changed
            )
            self.analysis_pages.append(analysis_page)
            self.stacked_widget.addWidget(analysis_page)

        main_layout.addWidget(self.stacked_widget, stretch=1)

        # --- Add separator line ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("QFrame { color: #cccccc; }")
        main_layout.addWidget(separator, stretch=0)

        # --- Naxos Module as separate section ---
        naxos_section = self._setup_naxos_section(audio_start, audio_end)
        main_layout.addWidget(naxos_section, stretch=0)

        # --- Sticky bottom bar: waveform timeline + audio controls ---
        audio_ctrl_bar = AudioControlBar.init(audio_path)
        timeline = self._setup_bottom_bar(audio_ctrl_bar, main_layout)
        navbar.link_out_controller(timeline._controller)  # type: ignore

        # Connect navbar home signal
        navbar.s_home.connect(self.restart_application)

        # Create sync managers for each page
        for analysis_page in self.analysis_pages:
            sync_manager = PlotSyncManager(
                audio_controller=audio_ctrl_bar._controller,
                out_controller=timeline._controller,  # type: ignore
                controllers=[
                    analysis_page.controllers["mfcc"],
                    analysis_page.controllers["chromagram"],
                    analysis_page.controllers["tempogram"],
                    analysis_page.controllers["combination"],
                    analysis_page.controllers["silence"],
                    analysis_page.controllers["hrps"],
                ],  # type: ignore
                naxos_controller=self.naxos_module._controller,  # type: ignore
            )
            self.sync_managers.append(sync_manager)

        # Show first page
        self.stacked_widget.setCurrentIndex(0)
        self._update_page_buttons()

        self.setLayout(main_layout)

        from views.loading_dialog import LoadingWindow

        LoadingWindow.hide()

        # Setup keyboard shortcuts for page navigation
        self.shift_left_shortcut = QShortcut(
            QKeySequence(
                QKeyCombination(Qt.KeyboardModifier.ShiftModifier, Qt.Key.Key_Left)
            ),
            self,
        )
        self.shift_left_shortcut.activated.connect(self._prev_page)

        self.shift_right_shortcut = QShortcut(
            QKeySequence(
                QKeyCombination(Qt.KeyboardModifier.ShiftModifier, Qt.Key.Key_Right)
            ),
            self,
        )
        self.shift_right_shortcut.activated.connect(self._next_page)

        # Load project session if project_path was provided
        if self.project_path:
            self.on_load_project(self.project_path)

    def closeEvent(self, a0):
        """Handle window closing - ensure proper cleanup before Qt exits"""
        print("AnalysisPageView: Window closing...")

        from PyQt6.QtGui import QCloseEvent
        from PyQt6.QtWidgets import QMessageBox

        # Only show confirmation dialog for user-initiated closes (X button)
        # Not for programmatic closes (restart button)
        if not self._programmatic_close:
            print("AnalysisPageView: User-initiated close - asking for confirmation...")

            # Show confirmation dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Confirm Close")
            msg_box.setText(
                "Are you sure you want to close the analysis page? "
                "All unsaved progress will be lost."
            )
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            msg_box.setDefaultButton(QMessageBox.StandardButton.No)

            result = msg_box.exec()

            if result != QMessageBox.StandardButton.Yes:
                print("AnalysisPageView: Close cancelled by user")
                a0.ignore() if isinstance(a0, QCloseEvent) else None
                return
        else:
            print(
                "AnalysisPageView: Programmatic close - proceeding without confirmation..."
            )

        print("AnalysisPageView: Proceeding with cleanup...")

        # CRITICAL: Cleanup all AudioModel instances BEFORE Qt event loop exits
        # This ensures deleteLater() gets processed while event loop is still running
        for analysis_page in self.analysis_pages:
            if hasattr(analysis_page, "controllers"):
                for controller_name, controller in analysis_page.controllers.items():
                    if hasattr(controller, "model") and hasattr(
                        controller.model, "cleanup"
                    ):
                        print(f"Cleaning up {controller_name} AudioModel...")
                        controller.model.cleanup()

        # Also cleanup any other AudioModel instances that might exist
        import gc

        for obj in gc.get_objects():
            if str(type(obj)) == "<class 'models.audio_m.AudioModel'>":
                if hasattr(obj, "cleanup"):
                    print("Found and cleaning up additional AudioModel instance")
                    obj.cleanup()

        # Force Qt to process all deleteLater() calls while event loop is still active
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            app.processEvents()

        print("AnalysisPageView: Cleanup completed - window can now close safely")
        a0.accept() if isinstance(a0, QCloseEvent) else None
        super().closeEvent(a0)

    def _calculate_pages(self):
        """Calculate page segments with overlaps"""
        total_duration = self.signal.duration_seconds()
        pages = []

        if total_duration <= self.PAGE_DURATION:
            # Single page for short audio
            pages.append((0, total_duration))
        else:
            # Multiple pages with overlaps
            start_time = 0
            while start_time < total_duration:
                end_time = min(start_time + self.PAGE_DURATION, total_duration)
                pages.append((start_time, end_time))

                # Calculate next start time with overlap
                next_start = start_time + self.PAGE_DURATION - self.OVERLAP_DURATION
                if next_start >= total_duration:
                    break
                start_time = next_start

        print(f"Created {len(pages)} pages for audio duration {total_duration:.2f}s")
        for i, (start, end) in enumerate(pages):
            print(
                f"  Page {i + 1}: {start:.2f}s - {end:.2f}s ({end - start:.2f}s duration)"
            )

        return pages

    def _setup_page_navigation(self):
        """Setup page navigation controls"""
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(10, 5, 10, 5)

        # Page info label
        self.page_info_label = QLabel()
        nav_layout.addWidget(self.page_info_label)

        nav_layout.addStretch()

        # Previous/Next buttons
        self.prev_button = QPushButton("← Previous")
        self.next_button = QPushButton("Next →")

        # Style the navigation buttons
        nav_button_style = """
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 5px;
                color: white;
                padding: 8px 15px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.3);
            }
            QPushButton:disabled {
                background: rgba(255, 255, 255, 0.05);
                border: none;
                color: rgba(255, 255, 255, 0.3);
            }
        """

        self.prev_button.setStyleSheet(nav_button_style)
        self.next_button.setStyleSheet(nav_button_style)

        # add a hand cursor on hover
        self.prev_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.prev_button.clicked.connect(self._prev_page)
        self.next_button.clicked.connect(self._next_page)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        # Page number buttons
        self.page_buttons = []
        self.page_button_group = QButtonGroup()

        # Style for page number buttons
        page_button_style = """
            QPushButton {
                background: rgba(255, 255, 255, 0.1);
                border: none;
                border-radius: 5px;
                color: white;
                padding: 8px 4px;
                min-width: 30px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.2);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.3);
            }
            QPushButton:checked {
                background: #4facfe;
                border: none;
                color: white;
            }
            QPushButton:checked:hover {
                background: #3a8bfd;
            }
        """

        for i in range(len(self.pages_data)):
            btn = QPushButton(str(i + 1))
            btn.setCheckable(True)
            btn.setStyleSheet(page_button_style)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, page=i: self._goto_page(page))
            self.page_buttons.append(btn)
            self.page_button_group.addButton(btn, i)
            nav_layout.addWidget(btn)

        return nav_widget

    def _update_page_buttons(self):
        """Update page navigation state"""
        # Update page info
        start_time, end_time = self.pages_data[self.current_page]
        self.page_info_label.setText(
            f"Page {self.current_page + 1} of {len(self.pages_data)} "
            f"({start_time / 60:.1f}m - {end_time / 60:.1f}m)"
        )

        # Update button states
        self.prev_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page < len(self.pages_data) - 1)

        # Update page number buttons
        for i, btn in enumerate(self.page_buttons):
            btn.setChecked(i == self.current_page)

        self.s_page_changed.emit(self.current_page, *self.pages_data[self.current_page])

    def _switch_to_page(self, new_page_number):
        """Helper method to switch pages while preserving expansion state"""
        if 0 <= new_page_number < len(self.pages_data):
            # Store current page's expansion state
            current_page_widget = self.analysis_pages[self.current_page]
            expansion_state = current_page_widget.get_modules_expansion()

            # Switch to new page
            self.current_page = new_page_number
            self.stacked_widget.setCurrentIndex(self.current_page)

            # Apply expansion state to new page
            new_page_widget = self.analysis_pages[self.current_page]
            new_page_widget.set_modules_expansion(*expansion_state)

            self._update_page_buttons()

    def _prev_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self._switch_to_page(self.current_page - 1)

    def _next_page(self):
        """Go to next page"""
        if self.current_page < len(self.pages_data) - 1:
            self._switch_to_page(self.current_page + 1)

    def _goto_page(self, page_number):
        """Go to specific page"""
        self._switch_to_page(page_number)

    def _setup_bottom_bar(
        self, audio_ctrl_bar: AudioControlBar, main_layout: QVBoxLayout
    ):
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(200)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(10, 5, 10, 5)
        bottom_layout.setSpacing(10)

        # Use the full original signal for the timeline (not the page subsignal)
        timeline = Timeline.init("", self.signal, self.s_page_changed)

        bottom_layout.addWidget(audio_ctrl_bar, stretch=0)
        bottom_layout.addWidget(timeline, stretch=1)

        main_layout.addWidget(bottom_bar, stretch=0)

        return timeline

    def _setup_naxos_section(self, audio_start=None, audio_end=None):
        """Setup the Naxos module as a separate section in the main layout"""
        # Create the naxos module
        self.naxos_module = NaxosModule.init(
            signal=self.signal,
            audio_start_seconds=audio_start,
            audio_end_seconds=audio_end,
        )

        title = "Naxos API Audio Alignment Module"
        if audio_start is not None and audio_end is not None:
            title += f" (Audio Segment: {audio_start:.1f}s - {audio_end:.1f}s)"

        # Add section header with expand/collapse functionality
        section_header = SectionHeader(title, initially_collapsed=True)
        self.naxos_module.setVisible(False)  # Initially collapsed

        # Connect the toggle signal
        section_header.section_toggled.connect(self.naxos_module.setVisible)

        # Create a container for the section to maintain consistent spacing
        section_container = QWidget()
        section_container_layout = QVBoxLayout(section_container)
        section_container_layout.setContentsMargins(10, 10, 10, 10)
        section_container_layout.setSpacing(5)
        section_container_layout.addWidget(section_header)
        section_container_layout.addWidget(self.naxos_module)

        return section_container

    def restart_application(self):
        """Restart the application by closing all windows and going back to starting page."""
        print("AnalysisPageView: Restarting application")

        from PyQt6.QtWidgets import QMessageBox

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Warning")
        msg_box.setText(
            "Are you sure you want to return to the home page? "
            "All unsaved progress will be lost."
        )
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )

        result = msg_box.exec()

        if result != QMessageBox.StandardButton.Ok:
            return

        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication

        SessionCache._clear_active_cache()

        # Prevent the application from quitting when we close this window
        app = QApplication.instance()
        if app and isinstance(app, QApplication):
            app.setQuitOnLastWindowClosed(False)

        # Use QTimer.singleShot to restart after a short delay
        QTimer.singleShot(100, self._do_restart)

    def _do_restart(self):
        """Actually perform the restart after closing current windows."""
        import os
        import sys

        from PyQt6.QtWidgets import QApplication

        try:
            from views.starting_page.main_window import MainWindow

            # Create and show main window first
            main_window = MainWindow()
            main_window.showMaximized()

            # Store reference to prevent garbage collection
            self._new_main_window = main_window

            # Re-enable quit on last window closed
            app = QApplication.instance()
            if app and isinstance(app, QApplication):
                app.setQuitOnLastWindowClosed(True)

            # Set flag to indicate this is a programmatic close, not user-initiated
            self._programmatic_close = True

            # Now close the current analysis window
            self.close()

        except Exception as e:
            print(f"Error restarting application: {e}")
            import traceback

            traceback.print_exc()
            # Fallback: restart the entire Python process
            os.execv(sys.executable, ["python"] + sys.argv)

            os.execv(sys.executable, ["python"] + sys.argv)

    def on_save_project(self, file_path: str):
        """Save the current session to the specified file path."""
        print(f"Saving project to: {file_path}")

        try:
            success = SessionCache.save_session_to_file(file_path)

            if success:
                print(f"Project saved successfully to: {file_path}")
                # Optional: Show success message
                from PyQt6.QtWidgets import QMessageBox

                msg_box = QMessageBox(
                    QMessageBox.Icon.Information,
                    "Success",
                    f"Project saved successfully!\n\nLocation: {file_path}",
                    QMessageBox.StandardButton.Ok,
                )
                msg_box.exec()
            else:
                print("Failed to save project")
                from PyQt6.QtWidgets import QMessageBox

                msg_box = QMessageBox(
                    QMessageBox.Icon.Warning,
                    "Error",
                    "Failed to save project. Please check the file path and try again.",
                    QMessageBox.StandardButton.Ok,
                )
                msg_box.exec()

        except Exception as e:
            print(f"Error saving project: {e}")
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox(
                QMessageBox.Icon.Critical,
                "Error",
                f"An error occurred while saving the project:\n\n{str(e)}",
                QMessageBox.StandardButton.Ok,
            )
            msg_box.exec()

    def on_load_project(self, file_path: str):
        """Load a session from the specified file path."""
        print(f"Loading project from: {file_path}")
        from PyQt6.QtWidgets import QMessageBox

        try:
            success, message = SessionCache.load_session_from_file(file_path)

            if success:
                # Show success message

                msg_box = QMessageBox(
                    QMessageBox.Icon.Information,
                    "Success",
                    message,
                    QMessageBox.StandardButton.Ok,
                )
                msg_box.exec()
            else:
                msg_box = QMessageBox(
                    QMessageBox.Icon.Critical,
                    "Error",
                    message,
                    QMessageBox.StandardButton.Ok,
                )
                msg_box.exec()

        except Exception as e:
            print(f"Error loading project: {e}")
            msg_box = QMessageBox(
                QMessageBox.Icon.Critical,
                "Error",
                f"An error occurred while loading the project:\n\n{str(e)}",
                QMessageBox.StandardButton.Ok,
            )
            msg_box.exec()
