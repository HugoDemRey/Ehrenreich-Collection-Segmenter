"""Simple loading dialog for displaying progress during long operations.

Provides a frameless loading window that stays on top during processing
operations in the audio analysis application.

Author: Hugo Demule
Date: January 2026
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class LoadingWindow:
    """Simple loading window for displaying progress messages.

    Provides a clean, frameless window that stays on top to show
    loading status during long-running operations.

    Example:
        >>> LoadingWindow.show("Processing", "Analyzing audio...")
        >>> # ... perform operation ...
        >>> LoadingWindow.hide()
    """

    _window = None

    @classmethod
    def show(cls, title="Loading", message="Please wait..."):
        """Show the loading window with specified title and message.

        Args:
            title (str): Window title. Default: "Loading".
            message (str): Loading message. Default: "Please wait...".
        """
        if cls._window:
            cls._window.close()

        cls._window = QWidget()
        cls._window.setWindowTitle(title)
        cls._window.setFixedSize(600, 300)  # Smaller height without progress bar
        cls._window.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint
        )

        # Layout
        layout = QVBoxLayout(cls._window)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 20, 30, 20)

        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 30px; font-weight: 600;")
        layout.addWidget(title_label)

        # Message
        message_label = QLabel(message)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setStyleSheet("font-size: 16px; opacity: 0.8;")
        layout.addWidget(message_label)

        # Style the window - simple and sober
        cls._window.setStyleSheet(
            """
            QWidget {
                background: black;
                border: 5px solid darkgray;
            }
            QLabel {
                color: #e2e8f0;
                background: transparent;
                border: none;
            }
        """
        )

        # Center on screen
        if QApplication.primaryScreen():
            screen = QApplication.primaryScreen().geometry()
            x = (screen.width() - cls._window.width()) // 2
            y = (screen.height() - cls._window.height()) // 2
            cls._window.move(x, y)

        cls._window.show()
        cls._window.raise_()

        # Set loading cursor
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

    @classmethod
    def hide(cls):
        """Hide the loading window and restore normal cursor."""
        if cls._window:
            # Restore normal cursor
            QApplication.restoreOverrideCursor()
            cls._window.close()
            cls._window = None
            QApplication.processEvents()
