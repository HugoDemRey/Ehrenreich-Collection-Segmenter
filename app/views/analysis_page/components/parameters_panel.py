"""Parameter configuration panel for analysis module settings.

Provides dynamic UI components for adjusting analysis parameters including
sliders, checkboxes, and dropdowns with real-time preview and tooltips.

Author: Hugo Demule
Date: January 2026
"""

from typing import Any, List

from constants.colors import ACCENT_COLOR, ACCENT_COLOR_HOVER
from constants.parameters import ParameterOption, Parameters, get_parameters_class
from constants.segmenter_configurations import SegmenterConfig
from PyQt6.QtCore import QEvent, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from src.io.session_cache import SessionCache
from src.utils.text_formatter import format_parameter_description


class CustomSlider(QSlider):
    """Enhanced slider widget with custom cursor behavior.

    Provides different cursor styles for handle and groove areas
    to improve user interaction feedback.
    """

    def __init__(self, orientation):
        """Initialize custom slider with mouse tracking."""
        super().__init__(orientation)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, ev: QMouseEvent | None):
        if ev is None:
            return

        # Get the handle position and size
        handle_rect = self.get_handle_rect()

        if handle_rect.contains(ev.position().toPoint()):
            self.setCursor(Qt.CursorShape.SizeHorCursor)  # Resize cursor for handle
        else:
            self.setCursor(Qt.CursorShape.PointingHandCursor)  # Hand cursor for groove

        super().mouseMoveEvent(ev)

    def get_handle_rect(self) -> QRect:
        """Calculate the handle rectangle based on slider value and geometry."""
        # Calculate handle position based on value and slider geometry
        value_range = self.maximum() - self.minimum()
        if value_range == 0:
            relative_pos = 0
        else:
            relative_pos = (self.value() - self.minimum()) / value_range

        # Account for handle margins
        usable_width = self.width() - 20  # Approximate margins
        handle_x = int(10 + relative_pos * usable_width)

        # Approximate handle size and position
        handle_width = 14
        handle_height = 14
        handle_y = (self.height() - handle_height) // 2

        return QRect(
            handle_x - handle_width // 2, handle_y, handle_width, handle_height
        )


class CustomComboBox(QComboBox):
    """Custom combo box that opens dropdown when clicked anywhere."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, e: QMouseEvent | None):
        if e is None:
            return
        # Always show popup when clicking anywhere on the combo box
        self.showPopup()

    def showEvent(self, e):
        """Override to install event filter on line edit after it's created."""
        super().showEvent(e)
        line_edit = self.lineEdit()
        if line_edit:
            line_edit.installEventFilter(self)
            line_edit.setCursor(Qt.CursorShape.PointingHandCursor)

    def eventFilter(self, a0, a1):
        """Filter events from the line edit to handle clicks."""
        line_edit = self.lineEdit()
        if (
            a0 == line_edit
            and isinstance(a1, QEvent)
            and line_edit
            and a1.type() == QEvent.Type.MouseButtonPress
        ):
            self.showPopup()
            return True
        return super().eventFilter(a0, a1)


class ParametersPanel(QWidget):
    """
    Flexible configuration panel for tweaking parameters.

    Supports:
    - Boolean parameters (toggle buttons)
    - Continuous parameters (sliders with value display)
    - Categorical parameters (dropdown menus)
    """

    # Signal emitted when any parameter value changes
    s_parameter_changed = pyqtSignal(str, object)  # parameter_name, new_value
    s_all_parameters_changed = pyqtSignal(dict)  # all current values
    s_button_clicked = pyqtSignal()

    def __init__(
        self,
        parameters_options: List[ParameterOption],
        config: SegmenterConfig,
        title: str = "Configuration",
        parent=None,
    ):
        super().__init__(parent)
        self.parameters = {param.name: param for param in parameters_options}
        self.parameter_widgets = {}  # Store references to widgets
        self.title = title
        self.original_title = (
            title  # Store original title for confirmed state management
        )
        self.config = config
        self.is_confirmed = True  # Track confirmed state
        self.title_label = None  # Reference to title label widget

        self.setup_ui()
        self._connect_signals()

    def setup_ui(self):
        """Set up the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Title
        if self.title:
            self.title_label = QLabel(self.title)
            self._update_title_style()
            layout.addWidget(self.title_label)

        # Main panel container
        panel_frame = QFrame()
        panel_frame.setStyleSheet(
            """
            QFrame {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 10px;
            }
        """
        )
        panel_layout = QGridLayout(panel_frame)
        panel_layout.setContentsMargins(15, 15, 15, 15)
        panel_layout.setSpacing(8)

        # Create widgets for each parameter in two columns
        row = 0
        col = 0
        for param_name, param_config in self.parameters.items():
            self._create_parameter_widget(panel_layout, param_config, row, col)
            col += 1
            if col >= 2:  # Move to next row after 2 columns
                col = 0
                row += 1

        layout.addWidget(panel_frame)

        # Button container
        button_layout = QHBoxLayout()

        # Reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_button.setStyleSheet(
            """
            QPushButton {
            background-color: rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.8);
            font-size: 11px;
            border-radius: 6px;
            padding: 6px 12px;
            }
            QPushButton:hover {
            background-color: rgba(255, 255, 255, 0.15);
            }
            QPushButton:pressed {
            background-color: rgba(255, 255, 255, 0.05);
            }
        """
        )
        reset_button.clicked.connect(self.reset_to_defaults)

        # Copy button
        copy_button = QPushButton("Copy")
        copy_button.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_button.setStyleSheet(
            """
            QPushButton {
            background-color: rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.8);
            font-size: 11px;
            border-radius: 6px;
            padding: 6px 12px;
            }
            QPushButton:hover {
            background-color: rgba(255, 255, 255, 0.15);
            }
            QPushButton:pressed {
            background-color: rgba(255, 255, 255, 0.05);
            }
        """
        )
        copy_button.clicked.connect(self.copy_parameters)

        # Paste button
        paste_button = QPushButton("Paste")
        paste_button.setCursor(Qt.CursorShape.PointingHandCursor)
        paste_button.setStyleSheet(
            """
            QPushButton {
            background-color: rgba(255, 255, 255, 0.1);
            color: rgba(255, 255, 255, 0.8);
            font-size: 11px;
            border-radius: 6px;
            padding: 6px 12px;
            }
            QPushButton:hover {
            background-color: rgba(255, 255, 255, 0.15);
            }
            QPushButton:pressed {
            background-color: rgba(255, 255, 255, 0.05);
            }
        """
        )
        paste_button.clicked.connect(self.paste_parameters)

        # Compute button
        self.compute_button = QPushButton("Compute")
        self.compute_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.compute_button.setStyleSheet(
            f"""
            QPushButton {{
            background-color: {ACCENT_COLOR};
            color: white;
            font-size: 11px;
            font-weight: bold;
            border-radius: 6px;
            padding: 6px 12px;
            border: none;
            }}
            QPushButton:hover {{
            background-color: {ACCENT_COLOR_HOVER};
            }}
            QPushButton:pressed {{
            background-color: {ACCENT_COLOR_HOVER};
            }}
        """
        )
        self.compute_button.clicked.connect(lambda: self.s_button_clicked.emit())

        button_layout.addWidget(reset_button)
        button_layout.addWidget(copy_button)
        button_layout.addWidget(paste_button)
        button_layout.addWidget(self.compute_button)
        layout.addLayout(button_layout)

    def _create_parameter_widget(
        self, layout: QGridLayout, param_config: ParameterOption, row: int, col: int
    ):
        """Create widget for a specific parameter type."""

        # Create a container for this parameter (label + widget)
        param_container = QWidget()
        param_layout = QVBoxLayout(param_container)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(4)

        # Parameter label (only for boolean and categorical)
        if param_config.param_type != "continuous":
            label = QLabel(param_config.display_name)
            label.setStyleSheet(
                """
                QLabel {
                    color: rgba(255, 255, 255, 0.8);
                    font-size: 11px;
                    border-radius: 0px;
                    font-weight: 700;
                }
            """
            )
            param_layout.addWidget(label)

        if param_config.param_type == "boolean":
            widget = self._create_boolean_widget(param_config)
            param_layout.addWidget(widget)

        elif param_config.param_type == "continuous":
            widget_container = self._create_continuous_widget(param_config)
            param_layout.addWidget(widget_container)

        elif param_config.param_type == "categorical":
            widget = self._create_categorical_widget(param_config)
            param_layout.addWidget(widget)

        # Add the parameter container to the grid
        layout.addWidget(param_container, row, col)

        # Add description tooltip if provided (only for boolean and categorical)
        if param_config.description and param_config.param_type != "continuous":
            label.setToolTip(format_parameter_description(param_config.description))  # type: ignore
            label.setCursor(Qt.CursorShape.WhatsThisCursor)  # type: ignore

    def _create_boolean_widget(self, param_config: ParameterOption) -> QPushButton:
        """Create a toggle button for boolean parameters."""
        button = QPushButton()
        button.setCheckable(True)
        button.setChecked(param_config.current_value)
        button.setMinimumHeight(25)
        button.setMinimumWidth(80)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setCursor(Qt.CursorShape.PointingHandCursor)

        def update_button_style():
            if button.isChecked():
                button.setText("ON")
                button.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: {ACCENT_COLOR};
                        color: white;
                        font-size: 10px;
                        font-weight: bold;
                        border-radius: 6px;
                        border: none;
                        text-align: center;
                    }}
                    QPushButton:hover {{
                        background-color: {ACCENT_COLOR_HOVER};
                    }}
                """
                )
            else:
                button.setText("OFF")
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: rgba(255, 255, 255, 0.15);
                        color: rgba(255, 255, 255, 0.7);
                        font-size: 10px;
                        font-weight: bold;
                        border-radius: 6px;
                        text-align: center;
                    }
                    QPushButton:hover {
                        background-color: rgba(255, 255, 255, 0.2);
                    }
                """
                )

        update_button_style()
        button.toggled.connect(update_button_style)

        self.parameter_widgets[param_config.name] = button
        return button

    def _create_continuous_widget(self, param_config: ParameterOption) -> QWidget:
        """Create a slider with value input for continuous parameters."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Top row: Label and Input field (50% each)
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        # Parameter label (50%)
        label = QLabel(param_config.display_name)
        label.setStyleSheet(
            """
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                font-size: 11px;
                border-radius: 0px;
                font-weight: 700;
            }
        """
        )

        # Value input field (50%)
        value_input = QLineEdit(
            f"{param_config.current_value:.{param_config.decimals}f}"
        )
        value_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_input.setStyleSheet(
            """
            QLineEdit {
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.9);
                font-size: 10px;
                font-weight: bold;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 4px;
                padding: 2px 6px;
                max-height: 20px;
            }
            QLineEdit:focus {
                border-color: rgba(255, 255, 255, 0.4);
                background: rgba(255, 255, 255, 0.15);
            }
        """
        )

        # Add to top layout with equal stretch
        top_layout.addWidget(label, stretch=1)
        top_layout.addWidget(value_input, stretch=1)

        # Slider (bottom row)
        slider = CustomSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(
            int((param_config.max_value - param_config.min_value) / param_config.step)
        )
        current_pos = int(
            (param_config.current_value - param_config.min_value) / param_config.step
        )
        slider.setValue(current_pos)
        slider.setStyleSheet(
            f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT_COLOR};
                border: 2px solid {ACCENT_COLOR};
                width: 12px;
                height: 12px;
                border-radius: 8px;
                margin: -4px 0;
            }}
            QSlider::handle:horizontal:hover {{
                background: {ACCENT_COLOR_HOVER};
                border-color: {ACCENT_COLOR_HOVER};
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT_COLOR};
                border-radius: 2px;
            }}
        """
        )

        # Update functions
        def update_from_slider():
            slider_value = slider.value()
            actual_value = param_config.min_value + (slider_value * param_config.step)
            actual_value = round(actual_value, param_config.decimals)
            value_input.setText(f"{actual_value:.{param_config.decimals}f}")

        def update_from_input():
            try:
                input_value = float(value_input.text())
                # Clamp to valid range
                clamped_value = max(
                    param_config.min_value, min(param_config.max_value, input_value)
                )
                # Round to nearest valid step
                steps_from_min = round(
                    (clamped_value - param_config.min_value) / param_config.step
                )
                actual_value = param_config.min_value + (
                    steps_from_min * param_config.step
                )
                actual_value = round(actual_value, param_config.decimals)

                # Update slider position
                slider_pos = int(
                    (actual_value - param_config.min_value) / param_config.step
                )
                slider.setValue(slider_pos)

                # Update input field to show corrected value
                value_input.setText(f"{actual_value:.{param_config.decimals}f}")
            except ValueError:
                # Invalid input, revert to current value
                value_input.setText(
                    f"{param_config.current_value:.{param_config.decimals}f}"
                )

        # Connect signals
        slider.valueChanged.connect(update_from_slider)
        value_input.editingFinished.connect(update_from_input)

        # Add to main layout
        layout.addWidget(top_row)
        layout.addWidget(slider)

        # Add tooltip if provided
        if param_config.description:
            label.setToolTip(format_parameter_description(param_config.description))
            label.setCursor(Qt.CursorShape.WhatsThisCursor)

        self.parameter_widgets[param_config.name] = {
            "slider": slider,
            "input": value_input,
            "config": param_config,
        }
        return container

    def _create_categorical_widget(
        self, param_config: ParameterOption
    ) -> CustomComboBox:
        """Create a dropdown for categorical parameters."""
        combo = CustomComboBox()
        # Add items in uppercase for better visual consistency
        combo.addItems([str(option).upper() for option in param_config.options])

        # Set current value (need to find uppercase version)
        try:
            uppercase_options = [str(option).upper() for option in param_config.options]
            current_uppercase = str(param_config.current_value).upper()
            current_index = uppercase_options.index(current_uppercase)
            combo.setCurrentIndex(current_index)
        except ValueError:
            combo.setCurrentIndex(0)

        combo.setCursor(Qt.CursorShape.PointingHandCursor)

        # Set up a custom style for centering text
        combo.setEditable(True)
        line_edit = combo.lineEdit()
        if line_edit:
            line_edit.setReadOnly(True)
            line_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        combo.setStyleSheet(
            f"""
            QComboBox {{
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.9);
                font-size: 10px;
                font-weight: bold;
                border-radius: 6px;
                padding: 4px 20px 4px 15px;
                min-width: 100px;
            }}
            QComboBox QLineEdit {{
                background: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.9);
                font-size: 10px;
                font-weight: bold;
                text-align: center;
                padding: 0px;
            }}
            QComboBox QAbstractItemView {{
                background: rgba(40, 40, 40, 0.95);
                color: white;
                border-radius: 6px;
                selection-background-color: {ACCENT_COLOR};
                outline: none;
                text-align: center;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 6px 15px;
                border: none;
                min-height: 20px;
                text-align: center;
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {ACCENT_COLOR};
                color: white;
                text-align: center;
            }}
            QComboBox:hover {{
                background: rgba(255, 255, 255, 0.15);
                border-color: rgba(255, 255, 255, 0.3);
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid rgba(255, 255, 255, 0.2);
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background: rgba(255, 255, 255, 0.05);
            }}
            QComboBox::down-arrow {{
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 5px solid rgba(255, 255, 255, 0.7);
                width: 0;
                height: 0;
                margin: 5px 3px;
            }}
            QComboBox::down-arrow:hover {{
                border-top-color: rgba(255, 255, 255, 0.9);
            }}

        """
        )

        self.parameter_widgets[param_config.name] = combo
        return combo

    def _update_title_style(self):
        """Update the title label based on confirmed state."""
        if self.title_label is None or not self.original_title:
            return

        self.title_label.setStyleSheet(
            """
                QLabel {
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 13px;
                    font-weight: bold;
                    padding: 0px 0px;
                    margin: 0px;
                }
            """
        )

        if self.is_confirmed:
            self.title_label.setText(self.original_title)

            self.title_label.setToolTip("")
            self.title_label.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.title_label.setText(f"{self.original_title} ⚠")
            self.title_label.setToolTip(
                "Parameters have changed since last computation. Please recompute to confirm."
            )
            self.title_label.setCursor(Qt.CursorShape.WhatsThisCursor)

        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

    def _connect_signals(self):
        """Connect all widget signals to parameter change handlers."""
        for param_name, widget_info in self.parameter_widgets.items():
            param_config = self.parameters[param_name]

            if param_config.param_type == "boolean":
                widget_info.toggled.connect(
                    lambda checked, name=param_name: self._on_toggle_changed(
                        name, checked
                    )
                )

            elif param_config.param_type == "continuous":
                slider = widget_info["slider"]
                input_field = widget_info["input"]
                slider.valueChanged.connect(
                    lambda value, name=param_name: self._on_continuous_changed(
                        name, value
                    )
                )
                input_field.editingFinished.connect(
                    lambda name=param_name: self._on_input_changed(name)
                )

            elif param_config.param_type == "categorical":
                widget_info.currentTextChanged.connect(
                    lambda text, name=param_name: self._on_categorical_changed(
                        name, text
                    )
                )

    def _on_toggle_changed(self, param_name: str, value: Any):
        """Handle parameter value changes."""
        self._set_value(param_name, value)

    def _on_continuous_changed(self, param_name: str, slider_value: int):
        """Handle continuous parameter changes."""
        param_config = self.parameters[param_name]
        actual_value = param_config.min_value + (slider_value * param_config.step)
        actual_value = round(actual_value, param_config.decimals)

        self._set_value(param_name, actual_value)

    def _on_categorical_changed(self, param_name: str, text: str):
        """Handle categorical parameter changes."""
        param_config = self.parameters[param_name]
        # Convert back to original type if needed, accounting for uppercase display
        for option in param_config.options:
            if str(option).upper() == text.upper():
                self._set_value(param_name, option)
                break

    def _on_input_changed(self, param_name: str):
        """Handle direct input changes for continuous parameters."""
        widget_info = self.parameter_widgets[param_name]
        param_config = self.parameters[param_name]
        input_field = widget_info["input"]

        try:
            input_value = float(input_field.text())
            # Clamp to valid range
            clamped_value = max(
                param_config.min_value, min(param_config.max_value, input_value)
            )
            # Round to nearest valid step
            steps_from_min = round(
                (clamped_value - param_config.min_value) / param_config.step
            )
            actual_value = param_config.min_value + (steps_from_min * param_config.step)
            actual_value = round(actual_value, param_config.decimals)
            if actual_value != param_config.current_value:
                self._set_value(param_name, actual_value)

        except ValueError:
            # Invalid input, keep current value
            pass

    def activate_compute_button(self, active: bool):
        """Enable or disable the compute button."""
        self.compute_button.setEnabled(active)
        if active:
            self.compute_button.setStyleSheet(
                f"""
                QPushButton {{
                background-color: {ACCENT_COLOR};
                color: white;
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px 12px;
                border: none;
                }}
                QPushButton:hover {{
                background-color: {ACCENT_COLOR_HOVER};
                }}
                QPushButton:pressed {{
                background-color: {ACCENT_COLOR_HOVER};
                }}
            """
            )
        else:
            self.compute_button.setStyleSheet(
                """
                QPushButton {
                background-color: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.5);
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
                padding: 6px 12px;
                border: none;
                }
            """
            )

    def get_parameters(self) -> Parameters:
        """Return a Parameters instance with current values."""
        current_values = {
            name: param.current_value for name, param in self.parameters.items()
        }
        print("Current parameter values:", current_values)
        param_cls = get_parameters_class(self.config)
        print("Parameter class:", param_cls)
        params = param_cls.from_dict(current_values)
        print("Constructed Parameters instance:", params)
        return params

    def update_content(self, param_name: str, value: Any):
        """Set the value of a specific parameter and update its widget UI."""
        if param_name not in self.parameters:
            print(f"(PARAMETERS PANEL): Parameter '{param_name}' not found.")
            return

        param_config = self.parameters[param_name]
        self._set_value(param_name, value)

        print("Updating widget for parameter", param_name)

        # Update the widget
        widget_info = self.parameter_widgets[param_name]

        if param_config.param_type == "boolean":
            widget_info.setChecked(value)

        elif param_config.param_type == "continuous":
            slider_value = int((value - param_config.min_value) / param_config.step)
            widget_info["slider"].setValue(slider_value)
            widget_info["input"].setText(f"{value:.{param_config.decimals}f}")

        elif param_config.param_type == "categorical":
            try:
                # Find the index by comparing uppercase versions
                uppercase_options = [
                    str(option).upper() for option in param_config.options
                ]
                target_uppercase = str(value).upper()
                index = uppercase_options.index(target_uppercase)
                widget_info.setCurrentIndex(index)
            except ValueError:
                pass

    def _set_value(self, param_name: str, value: Any):
        """Setting the value of a specific parameter without checks."""
        param_config = self.parameters[param_name]
        has_changed = param_config.current_value != value
        if has_changed:
            param_config.current_value = value

            self.s_parameter_changed.emit(param_name, value)

    def reset_to_defaults(self):
        """Reset all parameters to their default values."""
        for param_name, param_config in self.parameters.items():
            self.update_content(param_name, param_config.default_value)

    def copy_parameters(self):
        """Copy current parameter values."""
        current_values = {
            name: param.current_value for name, param in self.parameters.items()
        }
        print(f"copy {current_values}")
        SessionCache.save_parameters_copy(current_values)

    def paste_parameters(self):
        """Paste parameter values."""
        copied_values = SessionCache.get_parameters_copy()
        print(f"paste {copied_values}")
        if not copied_values:
            return

        # If not all parameters are present, open a QDialog to make the user confirm or not
        params_that_dont_exist = [
            name for name in copied_values.keys() if name not in self.parameters
        ]
        params_that_exist = [
            name for name in copied_values.keys() if name in self.parameters
        ]

        if params_that_dont_exist:
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Warning")
            msg_box.setText(
                f"The copied parameters contain {len(params_that_dont_exist)} "
                "parameters that do not exist in the current configuration. The parameters have been copied from another module type.\n\n"
                "Parameters not found: " + ", ".join(params_that_dont_exist) + "\n\n"
                f"Do you still want to paste the available parameters? "
                + ", ".join(params_that_exist)
            )
            msg_box.setStandardButtons(
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )

            result = msg_box.exec()

            if result != QMessageBox.StandardButton.Ok:
                return

        for param_name, value in copied_values.items():
            self.update_content(param_name, value)

    def set_confirmed_state(self, confirmed: bool):
        """Set the confirmed state of the panel (visual indication)."""

        # Update internal state
        self.is_confirmed = confirmed

        # Update visual elements
        self._update_title_style()
