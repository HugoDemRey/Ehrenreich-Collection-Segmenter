"""Naxos module for music database integration and alignment.

Provides interface for connecting to Naxos music database, performing
audio alignment, and managing alignment results.

Author: Hugo Demule
Date: January 2026
"""

import numpy as np
from constants.colors import ACCENT_COLOR
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from src.audio.signal import Signal
from views.analysis_page.components.context_menu import ConditionalContextMenu
from views.analysis_page.components.dynamic_button import DynamicButton


class NaxosModule(QWidget):
    """Naxos database integration widget for audio alignment.

    Provides interface for Naxos URL validation, audio alignment processing,
    and alignment result management.

    Signals:
        s_url_validated (str): Naxos URL validated
        s_start_alignment: Begin alignment process
        s_change_cursor_pos (float): Change playback position
        s_add_this_alignment (float): Add alignment at position
        s_add_all_alignments (object): Add all detected alignments
    """

    # Signals Initialization
    s_url_validated = pyqtSignal(str)
    s_start_alignment = pyqtSignal()
    s_change_cursor_pos = pyqtSignal(float)
    s_add_this_alignment = pyqtSignal(float)
    s_add_all_alignments = pyqtSignal(object)

    _controller = None

    def __init__(self, parent=None):
        """Initialize Naxos module widget.

        Args:
            parent: Parent widget. Default: None.
        """
        super().__init__(parent)
        super().__init__(parent)
        self._setup_ui()
        self._setup_context_menu()

    def reset_module(self):
        """
        Reset the module to its initial state (URL input overlay).
        Clears the URL input, resets all buttons, stops audio previews, and shows the URL overlay.
        """
        print("Resetting Naxos Module to initial state")

        # Stop all audio previews and reset preview buttons first
        if hasattr(self, "preview_buttons"):
            for preview_button in self.preview_buttons:
                preview_button.request_stop()
                preview_button.reset_button_icon()

        # Clear current preview index
        if hasattr(self, "current_preview_index"):
            delattr(self, "current_preview_index")

        # Clear table data
        if hasattr(self, "table"):
            self.table.setRowCount(0)
            self.table.setColumnCount(3)
            self.table.setHorizontalHeaderLabels(["Preview", "Title", "Duration (s)"])

        # Reset preview buttons list
        if hasattr(self, "preview_buttons"):
            self.preview_buttons.clear()

        # Reset alignment button state (before hiding content widget)
        if hasattr(self, "alignment_button"):
            self.alignment_button.reset()
            self.alignment_button.set_button_text("Start Alignment")
            self.alignment_button.setEnabled(True)
            # Ensure callback is restored
            self.alignment_button.callback = lambda: self.s_start_alignment.emit()

        # Show URL overlay and hide content
        self.show_url_overlay(True)

        # Clear URL input
        self.clear_url()
        
        # Reset validate button state with proper callback (after showing overlay)
        if hasattr(self, "validate_button"):
            self.validate_button.reset()
            self.validate_button.set_button_text("Validate")
            self.validate_button.setEnabled(True)
            # Ensure callback is restored
            self.validate_button.callback = self.on_validate_button_clicked

    def _setup_context_menu(self):
        """Setup the context menu configurations."""
        self.context_menu = ConditionalContextMenu(self)

        # Configuration for right-click on table rows
        self.context_menu.add_menu_config(
            "row_context",
            ["Add this alignment", "Add all alignments"],
            [self._add_this_alignment, self._add_all_alignments],
        )

    def _add_this_alignment(self, alignment_time: float):
        """
        Handle adding a specific alignment.

        Args:
            alignment_time (float): The alignment time to add.
        """
        print(f"Adding this alignment at time: {alignment_time}")
        self.s_add_this_alignment.emit(alignment_time)

    def _add_all_alignments(self, alignment_times: object):
        """Handle adding all alignments."""
        print("Adding all alignments")
        print(f"Alignment times: {alignment_times}")
        self.s_add_all_alignments.emit(
            alignment_times
        )  # Emit signal to add all alignments

    def _setup_ui(self):
        """Set up the main UI structure"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)

        # Create widgets and store as fields but don't add to layout yet
        self.url_overlay = self._create_url_overlay()
        self.content_widget = self._create_content_widget()

        # Show overlay by default
        self.main_layout.addWidget(self.url_overlay)

    def _create_url_overlay(self) -> QWidget:
        """Create the URL input overlay widget"""
        overlay = QFrame()
        overlay.setStyleSheet(
            "background: rgba(255, 255, 255, 0.05); border-radius: 10px;"
        )
        overlay.setMinimumHeight(200)  # Set specific height for URL overlay

        layout = QVBoxLayout(overlay)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(15)

        # Title label
        title_label = QLabel("Enter Naxos URL")
        title_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: white; background: none;"
        )
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # URL input field
        from constants.naxos import BASE_URL

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText(f"{BASE_URL}...")
        self.url_input.setMaximumWidth(800)
        self.url_input.setStyleSheet(
            f"""
            QLineEdit {{
                padding: 10px;
                font-size: 12px;
                border: 2px solid #666;
                border-radius: 5px;
                background: rgba(255, 255, 255, 0.1);
                color: white;
            }}
            QLineEdit:focus {{
                border-color: {ACCENT_COLOR};
            }}
        """
        )

        # Validate button
        self.validate_button = DynamicButton(
            "Validate",
            color=ACCENT_COLOR,
            callback=self.on_validate_button_clicked,
            display_mode="text",
        )

        # Button container for centering
        button_container = QWidget()
        button_container.setMaximumWidth(800)

        # Center the button container and URL input
        button_container.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )

        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.validate_button)

        # Add widgets to layout
        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(self.url_input)
        layout.addWidget(button_container)
        layout.addStretch()

        return overlay

    def on_validate_button_clicked(self):
        """Handle validate button click"""
        url = self.get_url()
        if self._controller and self._controller.check_url_validity(
            url
        ):
            self.s_url_validated.emit(url)
            return True  # Allow transform to loading state
        else:
            return False  # Prevent transform to loading state

    def _create_content_widget(self) -> QWidget:
        """Create the content widget with table on left and alignment section on right"""
        container = QWidget()
        container.setMinimumHeight(400)  # Set specific height for content widget
        main_layout = QHBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # Left side: Table container (50% of space)
        self.table_container = QWidget()
        table_layout = QVBoxLayout(self.table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(0)

        # Table widget
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setRowCount(0)  # Initially empty
        self.table.setHorizontalHeaderLabels(["Preview", "Title", "Duration (s)"])

        # Set up mouse event handling for click detection
        original_mouse_press = self.table.mousePressEvent

        def custom_mouse_press(event):
            item = self.table.itemAt(event.pos())
            if item:
                row = item.row()
                column = item.column()
                if event.button() == Qt.MouseButton.LeftButton:
                    print("Left click detected")
                    self._on_cell_clicked(row, column, "left")
                elif event.button() == Qt.MouseButton.RightButton:
                    print("Right click detected")
                    self._on_cell_clicked(row, column, "right")
            original_mouse_press(event)

        self.table.mousePressEvent = custom_mouse_press

        # Style the table
        self.table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid #666;
                border-radius: 5px;
                color: white;
                gridline-color: #666;
                outline: none;
            }}
            QHeaderView::section {{
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: 1px solid #666;
                padding: 8px;
                font-weight: bold;
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid #666;
                border: none;
                outline: none;
            }}
            QTableWidget::item:selected {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                outline: none;
            }}
            QTableWidget::item:focus {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                outline: none;
            }}
        """
        )

        # Set column widths and resize behavior
        header = self.table.horizontalHeader()
        if header:
            # Allow all columns to be resized freely within the container
            header.setStretchLastSection(False)  # Don't auto-stretch last column
            header.setSectionResizeMode(
                0, header.ResizeMode.Fixed
            )  # Preview column fixed width
            header.setSectionResizeMode(
                1, header.ResizeMode.Stretch
            )  # Title column resizable
            header.setSectionResizeMode(
                2, header.ResizeMode.Fixed
            )  # Duration column fixed

        # Set initial column widths proportional to container
        self.table.setColumnWidth(0, 80)  # Preview column width
        self.table.setColumnWidth(1, 180)  # Title column initial width
        self.table.setColumnWidth(2, 100)  # Duration column initial width

        table_layout.addWidget(self.table)

        # Right side: Alignment container (50% of space)
        self.alignment_container = QFrame()
        self.alignment_container.setStyleSheet(
            "background: rgba(255, 255, 255, 0.05); border: 1px solid #666; border-radius: 5px;"
        )
        alignment_layout = QVBoxLayout(self.alignment_container)
        alignment_layout.setContentsMargins(20, 0, 20, 0)
        alignment_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.alignment_button = DynamicButton(
            "Start Alignment",
            color=ACCENT_COLOR,
            display_mode="text",
            callback=lambda: self.s_start_alignment.emit(),
            max_height=50,
        )
        alignment_layout.addWidget(self.alignment_button)

        # Add both containers with equal stretch factors (50/50 split)
        main_layout.addWidget(self.table_container, stretch=1)  # 50% of space
        main_layout.addWidget(self.alignment_container, stretch=1)  # 50% of space

        return container

    def update_table(
        self, titles: np.ndarray, durations: np.ndarray, audio_paths: np.ndarray
    ):
        """
        Fill the table with titles and durations

        Args:
            titles (np.ndarray): Array of track titles
            durations (np.ndarray): Array of track durations
        """
        if len(titles) != len(durations):
            raise ValueError("Titles and durations arrays must have the same length")

        # Set the number of rows
        self.table.setRowCount(len(titles))

        # Import here to avoid circular imports
        from views.analysis_page.components.audio_prev_button import AudioPrevButton

        self.preview_buttons = []

        # Fill the table
        print("Filling table with audio paths:", audio_paths)
        for i, (title, duration, audio_path) in enumerate(
            zip(titles, durations, audio_paths)
        ):
            # Preview button (column 0)
            preview_button = AudioPrevButton.init(audio_path)  # Use actual audio path
            preview_button.setMaximumSize(60, 30)
            preview_button.setMinimumSize(60, 30)

            self.preview_buttons.append(preview_button)

            # Connect button signal - you can handle this in the controller
            # Use default parameter to capture current value of i
            preview_button.toggle_button_clicked.connect(
                lambda checked=False, idx=i: self._on_preview_clicked(idx)
            )
            self.table.setCellWidget(i, 0, preview_button)

            # Title item (column 1)
            title_item = QTableWidgetItem(str(title))
            title_item.setFlags(
                title_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )  # Read-only
            self.table.setItem(i, 1, title_item)

            # Duration item (column 2)
            duration_item = QTableWidgetItem(str(self._sec_to_hhmmss(duration)))
            duration_item.setFlags(
                duration_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )  # Read-only
            self.table.setItem(i, 2, duration_item)

        # Auto-resize rows to content
        self.table.resizeRowsToContents()

    def _on_preview_clicked(self, prev_button_index: int):
        """
        Handle preview button click for a specific row.

        Args:
            prev_button_index (int): The index of the clicked preview button
        """

        if (
            hasattr(self, "current_preview_index")
            and self.current_preview_index != prev_button_index
        ):
            # Reset previous button icon
            self.preview_buttons[self.current_preview_index].reset_button_icon()
            self.preview_buttons[self.current_preview_index].request_stop()

        self.current_preview_index = prev_button_index

    def set_alignment_verbose(self, msg: str, percent: int):
        """
        Update the verbose text label with a new message

        Args:
            verbose (tuple[str, int]): (message, percentage)
        """
        self.alignment_button.set_text(msg)
        self.alignment_button.set_loading(percent)

    def set_scraping_verbose(self, msg: str, percent: int):
        """
        Update the scraping verbose text label with a new message

        Args:
            verbose (tuple[str, int]): (message, percentage)
        """
        self.validate_button.set_text(msg)
        self.validate_button.set_loading(percent)

    def show_url_overlay(self, show: bool):
        """
        Show or hide the URL overlay

        Args:
            show (bool): True to show overlay, False to hide
        """
        if show:
            # Remove content widget if present
            if self.content_widget.parent() is not None:
                self.main_layout.removeWidget(self.content_widget)
                self.content_widget.setParent(None)
            # Add overlay widget if not present
            if self.url_overlay.parent() is None:
                self.main_layout.addWidget(self.url_overlay)
        else:
            # Remove overlay widget if present
            if self.url_overlay.parent() is not None:
                self.main_layout.removeWidget(self.url_overlay)
                self.url_overlay.setParent(None)
            # Add content widget if not present
            if self.content_widget.parent() is None:
                self.main_layout.addWidget(self.content_widget)

    def show_content(self, show: bool):
        """
        Show or hide the content widget (table + alignment)

        Args:
            show (bool): True to show content, False to hide
        """
        if show:
            # Remove overlay widget if present
            if self.url_overlay.parent() is not None:
                self.main_layout.removeWidget(self.url_overlay)
                self.url_overlay.setParent(None)
            # Add content widget if not present
            if self.content_widget.parent() is None:
                self.main_layout.addWidget(self.content_widget)
        else:
            # Remove content widget if present
            if self.content_widget.parent() is not None:
                self.main_layout.removeWidget(self.content_widget)
                self.content_widget.setParent(None)
            # Add overlay widget if not present
            if self.url_overlay.parent() is None:
                self.main_layout.addWidget(self.url_overlay)

    def show_final_table_view(self, alignment_starts=None, alignment_ends=None):
        """
        Switch to the final view: remove alignment section, expand table, add alignment columns.
        Optionally fill alignment columns with provided data.
        """
        # Remove alignment container from layout
        main_layout = self.content_widget.layout()
        if (
            main_layout
            and hasattr(self, "alignment_container")
            and self.alignment_container.parent() is not None
        ):
            main_layout.removeWidget(self.alignment_container)
            self.alignment_container.setParent(None)

        # Make table container take full width (100% instead of 50%)
        if main_layout and hasattr(main_layout, "setStretchFactor"):
            # Try to cast to QBoxLayout which has setStretch method
            from PyQt6.QtWidgets import QBoxLayout

            if isinstance(main_layout, QBoxLayout):
                main_layout.setStretch(0, 1)  # Table container takes all space

        # Add alignment columns if not already present
        if self.table.columnCount() < 5:
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(
                ["Preview", "Title", "Duration (s)", "Alignment Start", "Alignment End"]
            )

            # Update column resize behavior for 5 columns
            header = self.table.horizontalHeader()
            if header:
                # Set all columns to stretch proportionally to fill the entire table width
                header.setSectionResizeMode(
                    0, header.ResizeMode.Fixed
                )  # Preview button fixed width
                header.setSectionResizeMode(
                    1, header.ResizeMode.Stretch
                )  # Title takes more space
                header.setSectionResizeMode(
                    2, header.ResizeMode.ResizeToContents
                )  # Duration fixed but resizable
                header.setSectionResizeMode(
                    3, header.ResizeMode.ResizeToContents
                )  # Start fixed but resizable
                header.setSectionResizeMode(
                    4, header.ResizeMode.ResizeToContents
                )  # End fixed but resizable

            # Optionally fill alignment columns
            if alignment_starts is not None and alignment_ends is not None:
                for i in range(self.table.rowCount()):
                    start_item = QTableWidgetItem(
                        str(self._sec_to_hhmmss(alignment_starts[i]))
                        if i < len(alignment_starts)
                        else ""
                    )
                    end_item = QTableWidgetItem(
                        str(self._sec_to_hhmmss(alignment_ends[i]))
                        if i < len(alignment_ends)
                        else ""
                    )
                    start_item.setFlags(
                        start_item.flags() & ~Qt.ItemFlag.ItemIsEditable
                    )
                    end_item.setFlags(end_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(i, 3, start_item)
                    self.table.setItem(i, 4, end_item)

    def update_alignment_columns(self, alignment_starts, alignment_ends):
        """
        Update the alignment start/end columns in the table.
        """
        for i in range(self.table.rowCount()):
            start_item = QTableWidgetItem(
                str(self._sec_to_hhmmss(alignment_starts[i]))
                if i < len(alignment_starts)
                else ""
            )
            end_item = QTableWidgetItem(
                str(self._sec_to_hhmmss(alignment_ends[i]))
                if i < len(alignment_ends)
                else ""
            )
            start_item.setFlags(start_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            end_item.setFlags(end_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 3, start_item)
            self.table.setItem(i, 4, end_item)

    def get_url(self) -> str:
        """Get the current URL from the input field"""
        return self.url_input.text()

    def clear_url(self):
        """Clear the URL input field"""
        self.url_input.clear()

    def set_url(self, url: str):
        """Set the URL in the input field"""
        self.url_input.setText(url)

    def _on_cell_clicked(self, row: int, column: int = 0, button: str = "left"):
        """
        Handle table cell click events with mouse button detection.

        Args:
            row (int): Row index of the clicked cell
            column (int): Column index of the clicked cell
            button (str): Mouse button used ("left" or "right")
        """
        print(f"Cell clicked at row {row}, column {column} with {button} mouse button")

        is_full_table = self.table.columnCount() >= 5

        # Only process left clicks for cursor position changes
        if is_full_table:
            # Get alignment start time (column 3)
            all_start_items_float = [
                self._hhmmss_to_sec(self.table.item(r, 3).text())
                for r in range(self.table.rowCount())
            ]

            alignment_start_item = self.table.item(row, 3)
            if alignment_start_item and alignment_start_item.text():
                trans_start_at_row = alignment_start_item.text()
                print(
                    f"Emitting cursor position change: {self._hhmmss_to_sec(trans_start_at_row)}"
                )
                self.s_change_cursor_pos.emit(self._hhmmss_to_sec(trans_start_at_row))

                if button == "right":
                    # Show context menu for right clicks
                    self.context_menu.show_conditional_menu(
                        "row_context",
                        [
                            [self._hhmmss_to_sec(trans_start_at_row)],
                            [all_start_items_float],
                        ],
                    )

            elif button == "right":
                print(f"No alignment start data to emit for row {row}")
                # Show context menu for right clicks without alignment time
                self.context_menu.show_conditional_menu(
                    "row_context",
                    [[], []],  # Both buttons get no arguments
                )

            else:
                print(f"No alignment start data in row {row}")
        else:
            print("No alignment columns available yet")

    def _sec_to_hhmmss(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hrs == 0:
            return f"{mins:02}:{secs:02}"

        return f"{hrs:02}:{mins:02}:{secs:02}"

    def _hhmmss_to_sec(self, time_str: str) -> float:
        """Convert HH:MM:SS format to seconds"""
        parts = time_str.split(":")
        parts = [float(part) for part in parts]
        if len(parts) == 3:
            hrs, mins, secs = parts
        elif len(parts) == 2:
            hrs = 0
            mins, secs = parts
        elif len(parts) == 1:
            hrs = 0
            mins = 0
            secs = parts[0]
        else:
            raise ValueError("Invalid time format")

        return hrs * 3600 + mins * 60 + secs

    @classmethod
    def init(
        cls,
        signal: Signal,
        audio_start_seconds=None,
        audio_end_seconds=None,
        parent=None,
    ):
        """Factory method to create a NaxosModuleView instance"""

        from controllers.analysis_page.modules.naxos_module_c import (
            NaxosModuleController,
        )
        from models.analysis_page.modules.naxos_module_m import NaxosModuleModel

        view = cls(parent)

        # Cutting the signal if specified on the starting page
        if audio_start_seconds is not None and audio_end_seconds is not None:
            model = NaxosModuleModel(
                signal.subsignal(audio_start_seconds, audio_end_seconds)
            )
        else:
            model = NaxosModuleModel(signal)

        view._controller = NaxosModuleController(view, model)

        return view


if __name__ == "__main__":
    import sys

    from constants.paths import CACHE_PATH
    from PyQt6.QtWidgets import QApplication
    from src.audio.audio_file import AudioFile

    app = QApplication(sys.argv)

    # Test the module
    signal = AudioFile(CACHE_PATH + "bar103-t2-c2-0-900.wav").load()
    module = NaxosModule.init(signal)
    module.setWindowTitle("Naxos Module Test")
    module.setMinimumSize(1200, 600)
    module.show()

    sys.exit(app.exec())
    app = QApplication(sys.argv)

    # Test the module
    signal = AudioFile(CACHE_PATH + "bar103-t2-c2-0-900.wav").load()
    module = NaxosModule.init(signal)
    module.setWindowTitle("Naxos Module Test")
    module.setMinimumSize(1200, 600)
    module.show()

    sys.exit(app.exec())

    sys.exit(app.exec())
