#!/usr/bin/env python3
"""
Main entry point for Ehrenreich's Collection Segmenter application.
"""

import sys

from constants.paths import (
    ACTIVE_SESSION_CACHE_PATH,
    AUDIO_TEMP_CACHE_PATH,
    CACHE_PATH,
    NAXOS_CACHE_PATH,
)
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication
from views.loading_dialog import LoadingWindow
from views.starting_page.main_window import MainWindow


def setup_folders():
    """Ensure necessary cache folders exist."""
    import os

    for path in [
        CACHE_PATH,
        NAXOS_CACHE_PATH,
        ACTIVE_SESSION_CACHE_PATH,
        AUDIO_TEMP_CACHE_PATH,
    ]:
        os.makedirs(path, exist_ok=True)


def setup_application():
    """Setup application-wide configurations."""
    app = QApplication(sys.argv)
    app.setApplicationName("Ehrenreich's Collection Segmenter")
    app.setApplicationVersion("1.0.0")

    # Show loading window
    LoadingWindow.show("Ehrenreich's Collection Segmenter", "Initializing...")

    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # High DPI scaling is enabled by default in Qt6

    return app


def main():
    """Main application entry point."""

    setup_folders()

    # Simple cleanup
    from src.io.audio_cache import AudioCache

    try:
        AudioCache.initial_cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")

    app = setup_application()

    # Create and show main window
    main_window = MainWindow()
    main_window.showMaximized()

    LoadingWindow.hide()

    # Start event loop
    exit_code = app.exec()

    # Simple cleanup
    try:
        AudioCache.cleanup_all()
    except Exception as e:
        print(f"Error during cleanup: {e}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
