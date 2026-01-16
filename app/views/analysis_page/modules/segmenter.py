"""Segmenter widget for audio structural analysis and boundary detection.

Provides interactive segmentation analysis with parameter controls,
visualization, and transition management capabilities.

Author: Hugo Demule
Date: January 2026
"""

from constants.parameters import Parameters, UndefinedParameters
from constants.segmenter_configurations import SegmenterConfig, SegmenterMVCBuilder
from constants.svg import SVG
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QLabel, QPushButton
from src.audio.signal import Signal
from views.analysis_page.components.context_menu import ConditionalContextMenu
from views.analysis_page.components.interactable_plot import InteractablePlot


class Segmenter(InteractablePlot):
    """Interactive segmentation analysis widget with context menu support.

    Provides segmentation analysis visualization with parameter controls,
    transition detection, and interactive editing capabilities.

    Signals:
        s_add_this_transition (float): Add transition at specific position
        s_add_all_transitions (object): Add all detected transitions
        s_compute_segmentation_curve: Recompute segmentation analysis
        s_compute_transitions: Recompute transition detection
    """

    # Signals for transition actions
    s_add_this_transition = pyqtSignal(float)
    s_add_all_transitions = pyqtSignal(
        object
    )  # Use object for complex types like lists
    s_compute_segmentation_curve = pyqtSignal()
    s_compute_transitions = pyqtSignal()

    # Create SVG icons for open/closed eye
    open_eye_svg = SVG.EYE_OPEN_ICON
    closed_eye_svg = SVG.EYE_CLOSED_ICON

    _controller: object
    _config: SegmenterConfig

    def __init__(self, name: str, signal: Signal):
        """Initialize segmenter widget with audio signal.

        Args:
            name (str): Segmenter display name.
            signal (Signal): Audio signal for analysis.
        """
        super().__init__(name, signal)
        self.setup_hide_transitions_button()
        self.loading_label = None
        self.setMinimumHeight(250)

        # Initialize generic context menu
        self._setup_context_menu()

        # Override the layer_container resize event to also position the hide button
        if hasattr(self, "layer_container"):
            original_resize = self.layer_container.resizeEvent

            def enhanced_resize(a0):
                # Call the original resize event first
                original_resize(a0)
                # Then position our hide button
                self.position_hide_button()
                # And reposition loading overlay if visible
                self.position_loading_overlay()

            self.layer_container.resizeEvent = enhanced_resize

    @staticmethod
    def init(
        name: str, signal: Signal, config: SegmenterConfig, id: str
    ) -> "Segmenter":
        """Initialize Segmenter with MVC pattern."""
        widget = Segmenter(name, signal)

        _, controller = SegmenterMVCBuilder.get_model_controller(
            widget, signal, config, id
        )  # Model is an internal object of the controller

        widget._config = config
        widget._controller = controller
        return widget

    def _setup_context_menu(self):
        """Setup the context menu configurations."""
        self.context_menu = ConditionalContextMenu(self)

        # Configuration for when a specific transition is right-clicked
        self.context_menu.add_menu_config(
            "transition_specific",
            ["Add this transition", "Add all transitions"],
            [self._add_this_transition, self._add_all_transitions],
            [self.stop_animation, lambda: self.show_cursor(True)],
        )

        # Configuration for when canvas is right-clicked (no specific transition)
        self.context_menu.add_menu_config(
            "general", ["Add all transitions"], [self._add_all_transitions]
        )

    def _add_this_transition(self, transition_pos: float):
        """Handle adding a specific transition."""
        self.s_add_this_transition.emit(transition_pos)

    def _add_all_transitions(self):
        """Handle adding all transitions."""
        if hasattr(self, "transitions"):
            self.s_add_all_transitions.emit(self.transitions)

    def show_context_menu(self, position: float, transition_pos: float | None):
        """
        Show context menu at the given position.

        Args:
            position (float): Position where the menu was requested.
            transition_pos (float | None): Position of the specific transition, if any.
        """
        if transition_pos is not None:
            # Right-clicked on a specific transition
            self.show_cursor(False)
            # Button 1: Add this transition (gets transition_pos)
            # Button 2: Add all transitions (gets no arguments)
            self.context_menu.show_conditional_menu(
                "transition_specific", [[transition_pos], []]
            )
        else:
            # Right-clicked on canvas (not on transition)
            # Button 1: Add all transitions (gets no arguments)
            self.context_menu.show_conditional_menu("general", [[]])

    def on_generate_button_clicked(self):
        """Handle generate button click."""
        # Delete the generate button after click
        if self.generate_button is not None:
            self.generate_button.hide()  # Hide immediately
            self.generate_button.deleteLater()
            self.generate_button = None

        print("(VIEW): Generate button clicked, computing base feature...")
        self.s_compute_segmentation_curve.emit()

        self.show_cursor(True)
        self.is_plot_generated = True
        self.hide_transitions_button.setVisible(True)
        self.transitions_hidden = False
        self.update_button_icon()

    def setup_hide_transitions_button(self):
        """Setup button to hide/show transitions."""

        # Add button to hide/show transitions that is at the bottom left of the dynamic layer
        self.hide_transitions_button = QPushButton(self.dynamic_layer)
        self.hide_transitions_button.setStyleSheet(
            """
            QPushButton {
                border-radius: 5px;
                padding: 2px;
                border: none;
            }
        """
        )
        self.hide_transitions_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hide_transitions_button.setVisible(False)
        self.hide_transitions_button.setFixedWidth(25)
        self.hide_transitions_button.setFixedHeight(25)
        self.hide_transitions_button.clicked.connect(self.on_transition_button_clicked)

        # Position will be set in resize event
        self.update_button_icon()

    def position_hide_button(self):
        """Position the hide button at the bottom left of the layer container."""
        if hasattr(self, "hide_transitions_button") and hasattr(
            self, "layer_container"
        ):
            # Position at bottom left with some margin
            x = 10
            y = (
                self.layer_container.height()
                - self.hide_transitions_button.height()
                - 5
            )
            self.hide_transitions_button.move(x, y)

        self.update_button_icon()

    def update_button_icon(self):
        """Update button icon based on transitions visibility."""

        svg_data = (
            self.open_eye_svg if not self.transitions_hidden else self.closed_eye_svg
        )
        renderer = QSvgRenderer()
        renderer.load(svg_data)

        pixmap = QPixmap(16, 16)
        pixmap.fill()  # Fill with transparentx
        painter = self.hide_transitions_button.palette().color(
            self.hide_transitions_button.backgroundRole()
        )
        pixmap.fill(painter)

        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()

        self.hide_transitions_button.setIcon(QIcon(pixmap))

    def on_transition_button_clicked(self):
        """Toggle visibility of transition lines."""
        self.transitions_hidden = not self.transitions_hidden
        self.hide_transition_lines(self.transitions_hidden)
        self.update_button_icon()

    def show_loading_overlay(self, message: str):
        """Show loading overlay with message in the center of layer container."""
        if self.loading_label is None:
            self.loading_label = QLabel(self.layer_container)
            self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.loading_label.setStyleSheet(
                """
                QLabel {
                    background-color: rgba(0, 0, 0, 0.7);
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    border-radius: 10px;
                    padding: 10px;
                }
            """
            )

        self.loading_label.setText(f"🔄 {message}")
        # make the text wrap if too long
        self.loading_label.setWordWrap(True)

        # Show first, then position (ensures parent has correct size)
        self.loading_label.show()
        self.position_loading_overlay()
        self.loading_label.raise_()

    def position_loading_overlay(self):
        """Position the loading overlay to take 80% of parent dimensions, centered."""
        if self.loading_label is None:
            return

        # Position the label to take 80% of parent dimensions, centered
        parent_size = self.layer_container.size()
        label_width = int(parent_size.width() * 0.8)
        label_height = int(parent_size.height() * 0.8)
        x = (parent_size.width() - label_width) // 2
        y = (parent_size.height() - label_height) // 2

        self.loading_label.setGeometry(x, y, label_width, label_height)

    def hide_loading_overlay(self):
        """Hide the loading overlay."""
        if self.loading_label is not None:
            self.loading_label.hide()

    def get_parameters(self) -> Parameters:
        """Get current parameters from the parameters panel."""
        return UndefinedParameters()
