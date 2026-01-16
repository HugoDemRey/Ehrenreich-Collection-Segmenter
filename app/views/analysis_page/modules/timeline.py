"""Timeline widget for audio waveform display with transition editing.

Provides an interactive timeline view of the audio signal with context menu
support for adding and removing structural transitions.

Author: Hugo Demule
Date: January 2026
"""

import numpy as np
from PyQt6.QtCore import Qt, pyqtBoundSignal, pyqtSignal
from PyQt6.QtWidgets import QLabel
from src.audio.signal import Signal
from views.analysis_page.components.context_menu import ConditionalContextMenu
from views.analysis_page.components.interactable_plot import InteractablePlot


class Timeline(InteractablePlot):
    """Interactive timeline widget for audio waveform display.

    Extends InteractablePlot with timeline-specific functionality including
    waveform visualization and transition management.

    Signals:
        remove_this_transition (float): Remove transition at position
        remove_all_transitions: Clear all transitions
        add_transition_here (float): Add transition at position
    """

    # Signals for transition actions
    remove_this_transition = pyqtSignal(float)
    remove_all_transitions = pyqtSignal()
    add_transition_here = pyqtSignal(float)

    MAX_SAMPLES_ON_PLOT = 100000

    def __init__(
        self, name: str, signal: Signal, s_page_changed: pyqtBoundSignal | None = None
    ):
        """Initialize timeline with audio signal and optional page change handler.

        Args:
            name (str): Timeline display name.
            signal (Signal): Audio signal to display.
            s_page_changed: Optional page change signal handler.
        """
        super().__init__(name, signal)

        # Add timer display
        self._setup_timer_display()

        # Downsample if too many samples
        if len(self.signal.samples) > self.MAX_SAMPLES_ON_PLOT:
            factor = len(self.signal.samples) // self.MAX_SAMPLES_ON_PLOT
            plot_samples = self.signal.samples[::factor]
        else:
            plot_samples = self.signal.samples

        self.create_feature_plot(
            np.linspace(0, self.signal.duration_seconds(), len(plot_samples)),
            plot_samples,
        )
        self.on_generate_button_clicked()

        # Setup context menu
        self._setup_context_menu()

        # Connect signal if provided
        if s_page_changed is not None:
            s_page_changed.connect(self.page_changed)

    @staticmethod
    def init(
        name: str, signal: Signal, s_page_changed: pyqtBoundSignal | None = None
    ) -> "Timeline":
        """Initialize Timeline with MVC pattern."""
        from controllers.analysis_page.modules.timeline_c import TimelineController
        from models.analysis_page.modules.timeline_m import TimelineModel

        model = TimelineModel(signal)
        widget = Timeline(name, signal, s_page_changed)
        controller = TimelineController(widget, model)
        widget._controller = controller
        return widget

    def page_changed(self, p: int, t1: float, t2: float):
        self.highlight_current_page(t1, t2)

    def highlight_current_page(self, start_time: float, end_time: float):
        """Highlight the current page on the timeline by highlighting the dynamic plot area."""
        self.page_highlight.set_x(start_time)
        self.page_highlight.set_width(end_time - start_time)
        self.page_highlight.set_visible(True)
        self.dynamic_layer.draw_idle()

    def _setup_context_menu(self):
        """Setup the context menu configurations."""
        self.context_menu = ConditionalContextMenu(self)

        # Configuration for when a specific transition is right-clicked
        self.context_menu.add_menu_config(
            "transition_specific",
            ["Remove this transition", "Remove all transitions"],
            [self._remove_this_transition, self._remove_all_transitions],
            [self.stop_animation, lambda: self.show_cursor(True)],
        )

        # Configuration for when canvas is right-clicked (no specific transition)
        self.context_menu.add_menu_config(
            "general",
            ["Add transition here", "Remove all transitions"],
            [self._add_transition_here, self._remove_all_transitions],
        )

    def _remove_this_transition(self, transition_pos: float):
        """
        Handle removing a specific transition.

        Args:
            transition_pos (float): Position of the transition to remove.
        """
        self.remove_this_transition.emit(transition_pos)

    def _remove_all_transitions(self):
        """Handle removing all transitions."""
        self.remove_all_transitions.emit()

    def _add_transition_here(self, position: float):
        """
        Handle adding a transition at a specific position.

        Args:
            position (float): Position where to add the transition.
        """
        print(f"(VIEW): Adding transition at position {position}")

        # Add to model cache via controller
        if hasattr(self, "_controller"):
            from controllers.analysis_page.modules.timeline_c import TimelineController
            from models.analysis_page.modules.timeline_m import TimelineModel

            assert isinstance(self._controller, TimelineController), (
                "Controller must be an instance of TimelineController"
            )
            assert isinstance(self._controller.model, TimelineModel), (
                "Model must be an instance of TimelineModel"
            )
            self._controller.model.add_transition(position, "white")

        # Add to view
        self.add_transition_line(position, color="white")
        self.add_transition_here.emit(position)

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
            # Button 1: Remove this transition (gets transition_pos)
            # Button 2: Remove all transitions (gets no arguments)
            self.context_menu.show_conditional_menu(
                "transition_specific", [[transition_pos], []]
            )
        else:
            # Right-clicked on canvas (not on transition)
            print("(VIEW): Right-clicked on canvas at position", position)
            # Button 1: Add transition here (gets position)
            # Button 2: Remove all transitions (gets no arguments)
            self.context_menu.show_conditional_menu("general", [[position], []])

    def _setup_timer_display(self):
        """Setup the timer display in the timeline."""
        # Create timer label
        self.timer_label = QLabel("00:00:00")
        self.timer_label.setParent(self)
        self.timer_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                font-weight: normal;
                background: transparent;
                border: none;
                border-radius: 3px;
                padding: 2px 6px;
            }
        """)

        # Position timer at bottom right corner as overlay
        self.timer_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.timer_label.raise_()  # Bring to front

        # Set initial position (will be updated in resizeEvent)
        self.timer_label.resize(self.timer_label.sizeHint())
        self._position_timer()

    def _position_timer(self):
        """Position the timer at the bottom right corner of the widget."""
        if hasattr(self, "timer_label"):
            # Position at bottom right with some margin
            margin = 10
            x = self.width() - self.timer_label.width() - margin
            y = self.height() - self.timer_label.height() - margin
            self.timer_label.move(x, y)

    def resizeEvent(self, a0):
        """Handle resize events to reposition timer."""
        super().resizeEvent(a0)
        self._position_timer()

    def createWidget(self):
        """Helper method to create a widget container."""
        from PyQt6.QtWidgets import QWidget

        return QWidget()

    def set_timer(self, seconds: float):
        """Set the timer display to the specified time in seconds."""
        # Convert seconds to hours:minutes:seconds format
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds_remainder = total_seconds % 60

        # Format as HH:MM:SS
        time_str = f"{hours:02d}:{minutes:02d}:{seconds_remainder:02d}"
        self.timer_label.setText(time_str)
