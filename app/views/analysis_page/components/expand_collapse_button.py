"""Expand/collapse toggle button for collapsible UI sections.

Provides a button that toggles between expanded and collapsed states
with appropriate visual indicators.

Author: Hugo Demule
Date: January 2026
"""

from constants.svg import SVG
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class ExpandCollapseButton(QPushButton):
    """Toggle button for expand/collapse functionality.

    Displays appropriate icons (▼ for expanded, ▶ for collapsed)
    and emits toggle signals for collapsible content sections.

    Signals:
        toggled (bool): True when expanded, False when collapsed
    """

    # Signal emitted when the button is toggled
    toggled = pyqtSignal(bool)  # True = expanded, False = collapsed

    def __init__(self, parent=None):
        """Initialize expand/collapse button in collapsed state."""
        super().__init__(parent)

        # Initial state: collapsed (False)
        self._expanded = False

        # Set button properties
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(24, 24)
        self.setStyleSheet("""
            QPushButton {
                border: none;
                background: transparent;
                padding: 2px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
        """)

        # Update icon and connect signal
        self.update_icon()
        self.clicked.connect(self.toggle)

    def create_expand_icon(self) -> QIcon:
        """Create expand (▼) icon from SVG."""
        expand_svg_data = SVG.EXPAND_ICON
        renderer = QSvgRenderer(expand_svg_data)
        size = renderer.defaultSize()
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return QIcon(pixmap)

    def create_collapse_icon(self) -> QIcon:
        """Create collapse (▶) icon from SVG."""
        collapse_svg_data = SVG.COLLAPSE_ICON
        renderer = QSvgRenderer(collapse_svg_data)
        size = renderer.defaultSize()
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        return QIcon(pixmap)

    def update_icon(self):
        """Update button icon based on current state."""
        if self._expanded:
            self.setIcon(self.create_expand_icon())
            self.setToolTip("Collapse section")
        else:
            self.setIcon(self.create_collapse_icon())
            self.setToolTip("Expand section")

        # Set icon size
        self.setIconSize(self.size())

    def toggle(self):
        """Toggle the expanded state."""
        self._expanded = not self._expanded
        self.update_icon()
        self.toggled.emit(self._expanded)

    def set_expanded(self, expanded: bool):
        """Set the expanded state programmatically."""
        if self._expanded != expanded:
            self._expanded = expanded
            self.update_icon()

    @property
    def is_expanded(self) -> bool:
        """Get the current expanded state."""
        return self._expanded


class SectionHeader(QWidget):
    """A widget that combines a title label with an expand/collapse button"""

    # Signal emitted when the section is toggled
    section_toggled = pyqtSignal(bool)  # True = expanded, False = collapsed

    def __init__(self, title: str, initially_collapsed: bool = True, parent=None):
        super().__init__(parent)

        # Set fixed height and styling for consistent layout
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QWidget {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 5px;
                margin: 5px 0px;
            }
        """)

        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        # Create expand/collapse button
        self.expand_button = ExpandCollapseButton()
        self.expand_button.set_expanded(not initially_collapsed)

        # Create title label
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: white;"
        )

        # Add widgets to layout
        layout.addWidget(self.expand_button)
        layout.addWidget(self.title_label)
        layout.addStretch()  # Push everything to the left

        # Connect signals
        self.expand_button.toggled.connect(self.section_toggled.emit)

        # Make the title clickable too
        self.title_label.mousePressEvent = lambda ev: self.expand_button.toggle()
        self.title_label.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_expanded(self, expanded: bool):
        """Set the expanded state programmatically."""
        self.expand_button.set_expanded(expanded)

    @property
    def is_expanded(self) -> bool:
        """Get the current expanded state."""
        return self.expand_button.is_expanded
