"""Navigation bar widget for the analysis interface.

Provides menu-based navigation with file operations, project management,
and application navigation functionality.

Author: Hugo Demule
Date: January 2026
"""

from controllers.analysis_page.components.navbar_c import NavBarController
from controllers.analysis_page.modules.timeline_c import TimelineController
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QHBoxLayout, QMenu, QMenuBar, QWidget


class Navbar(QWidget):
    """Navigation bar widget with file and project management menus.

    Provides menu-based interface for project operations, file I/O,
    and application navigation.

    Signals:
        Various signals for menu actions (load, save, export, etc.)
    """

    # Defining signals can be added here if needed
    s_load_project = pyqtSignal()
    s_load_transitions = pyqtSignal()

    s_save_project = pyqtSignal()
    s_save_transitions = pyqtSignal()
    s_export_audio_segments = pyqtSignal()
    s_clear_naxos_cache = pyqtSignal()
    s_home = pyqtSignal()

    _controller: NavBarController

    def __init__(self, parent=None):
        """Initialize the navigation bar."""
        super().__init__(parent)
        self.setup_ui()

    @staticmethod
    def init() -> "Navbar":
        """Initialize navigation bar with MVC pattern setup."""
        from controllers.analysis_page.components.navbar_c import NavBarController
        from models.analysis_page.components.navbar_m import NavBarModel

        model = NavBarModel()
        widget = Navbar()
        controller = NavBarController(widget, model)
        widget._controller = controller
        return widget

    def link_out_controller(self, out_controller: TimelineController):
        """Link the NavBarView to the Timeline controller."""
        self.out_controller = out_controller

    def setup_ui(self):
        """Set up the navigation bar UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create menu bar
        self.menu_bar = QMenuBar(self)
        self.menu_bar.setSizePolicy(
            self.menu_bar.sizePolicy().horizontalPolicy(),
            self.menu_bar.sizePolicy().verticalPolicy(),
        )
        self.menu_bar.setMinimumWidth(0)
        self.menu_bar.setStyleSheet(
            """
            QMenuBar {
                background-color: #2b2b2b;
                color: white;
                border: none;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QMenuBar::item:selected {
                background-color: #4a4a4a;
            }
            QMenuBar::item:pressed {
                background-color: #5a5a5a;
            }
            QMenu {
                background-color: #3b3b3b;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px;
            }
            QMenu::item {
                padding: 6px 12px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #4a4a4a;
            }
        """
        )

        # Create home action
        self.app_menu = QMenu("Application", self)
        self.home_action = QAction("Restart Application", self)
        self.home_action.triggered.connect(self.s_home.emit)
        self.app_menu.addAction(self.home_action)

        self.clear_cache_action = QAction("Clear Naxos Cache", self)
        # add a tooltip to the action
        self.clear_cache_action.setToolTip(
            "The Naxos previews are cached locally to speed up loading times if you re-open the application. Clear the cache deletes all cached previews. "
            "The Naxos module will have to re-scrape naxos.com to re-download previews afterward, which may take some time."
        )
        self.clear_cache_action.setStatusTip("Clear all cached Naxos audio previews")

        self.clear_cache_action.triggered.connect(self.s_clear_naxos_cache.emit)
        self.app_menu.addAction(self.clear_cache_action)

        # Add File menu to menu bar
        self.menu_bar.addMenu(self.app_menu)

        # Create File menu
        self.file_menu = QMenu("File", self)

        # Create load transitions action
        self.load_project_action = QAction("Load Project", self)
        self.load_project_action.triggered.connect(self.s_load_project.emit)
        self.file_menu.addAction(self.load_project_action)

        # Create export project action
        self.save_project_action = QAction("Save Project", self)
        self.save_project_action.triggered.connect(self.s_save_project.emit)
        self.file_menu.addAction(self.save_project_action)

        self.file_menu.addSeparator()

        # Create load transitions action
        self.load_transitions_action = QAction("Load Segmentation", self)
        self.load_transitions_action.triggered.connect(self.s_load_transitions.emit)
        self.file_menu.addAction(self.load_transitions_action)

        # Create save transitions action
        self.save_transitions_action = QAction("Save Segmentation", self)
        self.save_transitions_action.triggered.connect(self.s_save_transitions.emit)
        self.file_menu.addAction(self.save_transitions_action)

        self.file_menu.addSeparator()

        self.export_audio_segments_action = QAction("Export Audio Segments", self)
        self.export_audio_segments_action.triggered.connect(
            self.s_export_audio_segments.emit
        )
        self.file_menu.addAction(self.export_audio_segments_action)

        self.menu_bar.addMenu(self.file_menu)

        layout.addWidget(self.menu_bar, stretch=1)  # Make menu bar expand to fill width

        # Set fixed height for the navbar
        self.setFixedHeight(30)
