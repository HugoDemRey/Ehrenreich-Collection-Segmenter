"""Navigation controller for page transition management.

Handles navigation between different application pages and
coordinates page transition logic and routing.

Author: Hugo Demule
Date: January 2026
"""

from PyQt6.QtCore import QObject, pyqtSignal


class NavigationController(QObject):
    """Controller for managing navigation between application pages.

    Provides navigation signals and routing logic for transitions
    between different application interfaces.

    Signals:
        navigate_to_home: Navigate to home page
        navigate_to_additioner: Navigate to additional/analysis page
    """

    # Signals for navigation events
    navigate_to_home = pyqtSignal()
    navigate_to_additioner = pyqtSignal()

    def __init__(self):
        """Initialize the navigation controller."""
        super().__init__()

    def go_to_home(self):
        """Navigate to home page."""
        self.navigate_to_home.emit()

    def go_to_additioner(self):
        """Navigate to additioner page."""
        self.navigate_to_additioner.emit()
