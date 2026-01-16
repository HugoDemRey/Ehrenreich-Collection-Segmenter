"""Segmenter module with integrated configuration panel.

Combines segmentation analysis widget with parameter configuration
for complete segmentation workflow interface.

Author: Hugo Demule
Date: January 2026
"""

from typing import Any, List, override

from constants.colors import ACCENT_COLOR, ACCENT_COLOR_HOVER
from constants.parameters import Parameters, get_parameters_options
from constants.segmenter_configurations import SegmenterConfig, SegmenterMVCBuilder
from constants.svg import SVG
from numpy import dtype, ndarray
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSizePolicy
from src.audio.signal import Signal
from views.analysis_page.components.parameters_panel import (
    ParameterOption,
    ParametersPanel,
)
from views.analysis_page.modules.segmenter import Segmenter


class SegmenterWithConfig(Segmenter):
    """
    Extended Segmenter that includes a configuration panel.
    """

    _controller: object
    _config: SegmenterConfig

    def __init__(
        self,
        name: str,
        signal: Signal,
        config: SegmenterConfig,
        config_parameters_options: List[ParameterOption],
    ):
        # Store config parameters before calling super().__init__
        self.config_parameters_options = config_parameters_options
        self._config = config

        super().__init__(name, signal)

        # Add configuration panel after the main UI is set up
        self.setup_configuration_panel()

    @staticmethod
    def init(
        name: str, signal: Signal, config: SegmenterConfig, id: str, *args, **kwargs
    ) -> "SegmenterWithConfig":
        """Initialize SegmenterWithConfig with MVC pattern and configuration."""
        config_parameters: list[ParameterOption] = get_parameters_options(config)

        widget = SegmenterWithConfig(name, signal, config, config_parameters)

        _, controller = SegmenterMVCBuilder.get_model_controller(
            widget, signal, config, id
        )  # Model is an internal object of the controller
        widget._controller = controller

        return widget

    def setup_configuration_panel(self):
        """Add configuration panel to the module."""
        # Get the main layout from parent
        main_layout = self.layout()
        if main_layout is None:
            return

        # Create configuration panel
        self.config_panel = ParametersPanel(
            parameters_options=self.config_parameters_options,
            config=self._config,
            title="Parameters",
            parent=self,
        )

        # Style the configuration panel to match the module design
        self.config_panel.setStyleSheet(
            """
            ConfigurationPanel {
                background: transparent;
            }
        """
        )

        # Set a fixed height for the config panel to prevent it from growing too much
        self.config_panel.setFixedHeight(0)  # Start with 0 height
        self.config_panel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        # Connect configuration changes to update handler
        self.config_panel.s_button_clicked.connect(self.on_generate_button_clicked)

        # Add to main layout (addWidget doesn't take stretch parameter for QVBoxLayout)
        main_layout.addWidget(self.config_panel)

        # Initially hide the config panel
        self.config_panel.setVisible(False)
        self.config_panel_visible = False

        # Store the original height of the module to restore later
        self.original_height = self.minimumHeight()

        # Add toggle button for configuration panel
        self.setup_config_toggle_button()

    def setup_config_toggle_button(self):
        """Setup button to toggle configuration panel visibility."""
        from PyQt6.QtGui import QIcon, QPainter, QPixmap
        from PyQt6.QtSvg import QSvgRenderer
        from PyQt6.QtWidgets import QPushButton

        self.config_toggle_button = QPushButton(self.dynamic_layer)
        self.config_toggle_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 5px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.15);
            }
        """
        )
        self.config_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.config_toggle_button.setFixedWidth(25)
        self.config_toggle_button.setFixedHeight(25)
        self.config_toggle_button.clicked.connect(self.toggle_config_panel)

        # Create gear icon for settings
        gear_svg = SVG.GEAR_ICON

        renderer = QSvgRenderer()
        # Accept both str and bytes for SVG data
        if isinstance(gear_svg, bytes):
            renderer.load(gear_svg)
        else:
            renderer.load(gear_svg.encode())

        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()

        self.config_toggle_button.setIcon(QIcon(pixmap))

        # Position will be set in the enhanced resize event
        self.position_config_toggle_button()

    def position_config_toggle_button(self):
        """Position the config toggle button at the bottom right of the layer container."""
        if hasattr(self, "config_toggle_button") and hasattr(self, "layer_container"):
            # Position at bottom right with some margin
            x = self.layer_container.width() - self.config_toggle_button.width() - 5
            y = self.layer_container.height() - self.config_toggle_button.height() - 5
            self.config_toggle_button.move(x, y)

    def setup_ui(self):
        """Override setup_ui to enhance resize event for both buttons."""
        super().setup_ui()

        # Now enhance the resize event to handle both buttons
        if hasattr(self, "layer_container"):
            original_resize = self.layer_container.resizeEvent

            def enhanced_resize_with_config(a0):
                # Call the original enhanced resize (which handles generate and hide buttons)
                original_resize(a0)
                # Add positioning for config toggle button
                self.position_config_toggle_button()

            self.layer_container.resizeEvent = enhanced_resize_with_config

    def toggle_config_panel(self):
        """Toggle the visibility of the configuration panel."""
        self.config_panel_visible = not self.config_panel_visible

        if self.config_panel_visible:
            # Calculate the needed height for the configuration panel
            self.config_panel.show()  # Temporarily show to get size hint
            config_height = self.config_panel.sizeHint().height()

            # Set the config panel to its proper height
            self.config_panel.setFixedHeight(config_height)

            # Expand the module's minimum height to accommodate the config panel
            current_min_height = self.minimumHeight()
            self.setMinimumHeight(current_min_height + config_height)

        else:
            # Hide the config panel and restore original height
            self.config_panel.setVisible(False)
            self.config_panel.setFixedHeight(0)

            # Restore original minimum height
            if hasattr(self, "original_height"):
                self.setMinimumHeight(self.original_height)

        # Update button style to show active/inactive state
        if self.config_panel_visible:
            self.config_toggle_button.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {ACCENT_COLOR};
                    border-radius: 5px;
                    padding: 2px;
                }}
                QPushButton:hover {{
                    background-color: {ACCENT_COLOR_HOVER};
                }}
            """
            )
        else:
            self.config_toggle_button.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.1);
                    border-radius: 5px;
                    padding: 2px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.15);
                }
            """
            )

    @override
    def on_generate_button_clicked(self):
        """Override to also show the config toggle button."""
        self.config_panel.activate_compute_button(False)
        super().on_generate_button_clicked()

    @override
    def create_feature_plot(
        self,
        t: ndarray[tuple[Any, ...], dtype[Any]],
        values: ndarray[tuple[Any, ...], dtype[Any]],
        threshold: float | None = None,
        color: bool = False,
        opacity: float = 0.4,
    ):
        self.config_panel.activate_compute_button(True)
        return super().create_feature_plot(
            t, values, threshold=threshold, color=color, opacity=opacity
        )

    def get_parameters(self) -> Parameters:
        """Get the current value of a specific parameter."""
        return self.config_panel.get_parameters()

    @override
    def set_confirmed_state(self, confirmed: bool):
        """Set the confirmed state of the panel (visual indication)."""
        super().set_confirmed_state(confirmed)
        self.config_panel.set_confirmed_state(confirmed)
