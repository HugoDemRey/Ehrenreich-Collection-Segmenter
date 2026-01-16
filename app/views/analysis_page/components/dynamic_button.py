"""Dynamic button widget that transforms into a progress bar during processing.

Provides a button that morphs into a loading progress bar when clicked,
with callback support for long-running operations.

Author: Hugo Demule
Date: January 2026
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QProgressBar, QPushButton, QStackedLayout, QWidget


class DynamicButton(QWidget):
    """Button that transforms into progress bar during processing.

    Provides visual feedback by switching from button to progress bar
    during long-running operations with customizable styling.

    Signals:
        clicked: Button was clicked
    """

    clicked = pyqtSignal()

    def __init__(
        self,
        text="Click Me",
        color="#4CAF50",
        callback=None,
        display_mode="percentage",
        parent=None,
        max_width=None,
        max_height=None,
    ):
        """
        Initialize the dynamic button.

        Args:
            text (str): Initial button text
            color (str): Button and loading bar color (hex format)
            callback (callable): Function to call when button is clicked
            display_mode (str): "percentage" to show "X%" or "text" to show custom text
            parent: Parent widget
        """
        super().__init__(parent)
        self.color = color
        self.callback = callback
        self.display_mode = display_mode
        self.custom_text = ""
        self.loading_percentage = 0
        self.is_loading = False
        self.max_width = max_width
        self.max_height = max_height

        self._setup_ui(text)

    def resizeEvent(self, a0):
        """Ensure both widgets have the same size when the container is resized"""
        super().resizeEvent(a0)
        self._update_widget_sizes()

    def showEvent(self, a0):
        """Ensure proper sizing when widget is first shown"""
        super().showEvent(a0)
        self._update_widget_sizes()

    def _update_widget_sizes(self):
        """Update both widgets to fill the entire container"""
        size = self.size()
        self.button.resize(size)
        self.button.move(0, 0)
        self.loading_bar.resize(size)
        self.loading_bar.move(0, 0)

    def _setup_ui(self, text):
        """Set up the UI components"""
        # Create stacked layout for perfect overlay
        layout = QStackedLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create button
        self.button = QPushButton(text)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_button_style()
        self.button.clicked.connect(self._on_button_clicked)

        # Create loading bar
        self.loading_bar = QProgressBar()
        self.loading_bar.setMinimum(0)
        self.loading_bar.setMaximum(100)
        self.loading_bar.setValue(0)
        self.loading_bar.setTextVisible(True)
        self._apply_loading_bar_style()

        if self.max_width is not None:
            self.setMaximumWidth(self.max_width)
        if self.max_height is not None:
            self.setMaximumHeight(self.max_height)

        # Add widgets to stacked layout
        layout.addWidget(self.button)
        layout.addWidget(self.loading_bar)

        # Show button initially
        layout.setCurrentWidget(self.button)

    def _apply_button_style(self):
        """Apply styling to the button"""
        self.button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color};
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(self.color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(self.color, 0.8)};
            }}
        """)

    def _apply_loading_bar_style(self):
        """Apply styling to the loading bar"""
        self.loading_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 5px;
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                font-weight: bold;
                font-size: 14px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {self.color};
                border-radius: 5px;
            }}
        """)

    def _darken_color(self, color_hex, factor=0.85):
        """Darken a hex color by a given factor"""
        # Remove # if present
        color_hex = color_hex.lstrip("#")
        # Convert to RGB
        rgb = tuple(int(color_hex[i : i + 2], 16) for i in (0, 2, 4))
        # Darken by factor
        darkened_rgb = tuple(int(c * factor) for c in rgb)
        # Convert back to hex
        return f"#{darkened_rgb[0]:02x}{darkened_rgb[1]:02x}{darkened_rgb[2]:02x}"

    def _on_button_clicked(self):
        """Handle button click - transform to loading bar"""
        if not self.is_loading:
            # Call callback first and check return value
            should_load = True
            if self.callback:
                result = self.callback()
                should_load = result if result is not None else True

            # Only transform to loading state if callback returns True
            if should_load:
                self.is_loading = True
                # Switch to loading bar
                self.layout().setCurrentWidget(self.loading_bar)
                # Update loading bar display
                self._update_loading_display()
                # Emit clicked signal
                self.clicked.emit()

    def _update_loading_display(self):
        """Update the loading bar display text based on display_mode"""
        if self.display_mode == "percentage":
            self.loading_bar.setFormat(f"{self.loading_percentage}%")
        elif self.display_mode == "text":
            if self.custom_text:
                self.loading_bar.setFormat(self.custom_text)
            else:
                self.loading_bar.setFormat(f"{self.loading_percentage}%")

    def set_loading(self, percentage):
        """
        Set the loading percentage.

        Args:
            percentage (int): Loading percentage (0-100)
        """
        self.loading_percentage = max(0, min(percentage, 100))
        self.loading_bar.setValue(self.loading_percentage)
        self._update_loading_display()

    def set_text(self, text):
        """
        Set custom text for the loading bar.
        Only works when display_mode is 'text'.

        Args:
            text (str): Custom text to display
        """
        self.custom_text = text
        if self.is_loading:
            self._update_loading_display()

    def reset(self):
        """Reset button to initial state"""
        self.is_loading = False
        self.loading_percentage = 0
        self.custom_text = ""
        self.loading_bar.setValue(0)
        # Switch back to button
        self.layout().setCurrentWidget(self.button)
        # Force update the display
        self._update_loading_display()
        # Force repaint
        self.update()

    def set_button_text(self, text):
        """Set the button text"""
        self.button.setText(text)

    def set_color(self, color):
        """Change the button and loading bar color"""
        self.color = color
        self._apply_button_style()
        self._apply_loading_bar_style()
