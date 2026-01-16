"""Controller for interactive plot component functionality.

Coordinates user interactions with analysis plots, handles transition
management, and synchronizes plot state with data models.

Author: Hugo Demule
Date: January 2026
"""

from models.analysis_page.components.interactable_plot_m import InteractablePlotModel
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from src.io.session_cache import SessionCache
from views.analysis_page.components.interactable_plot import InteractablePlot
from views.analysis_page.modules.segmenter_with_config import SegmenterWithConfig


class InteractablePlotController(QObject):
    """Controller for interactive plot user interactions.

    Manages click events, transition manipulation, cursor tracking,
    and coordinates between plot view and analysis model.

    Signals:
        change_cursor_pos (float): Cursor position change notification
    """

    change_cursor_pos = pyqtSignal(float)

    def __init__(self, view: InteractablePlot, model: InteractablePlotModel):
        # Import here to avoid circular import
        super().__init__()

        self.view: InteractablePlot = view
        self.model: InteractablePlotModel = model

        self.TRANSITION_LINE_COLOR = "white"

        self.init_signal_connections()

    def init_signal_connections(self):
        """Initialize signal connections between view and controller."""
        # Canvas interaction signals
        self.view.s_canvas_clicked_position.connect(self.on_canvas_clicked_position)
        self.view.s_canvas_hovered_position.connect(self.on_canvas_hovered_position)
        self.view.s_figure_exited.connect(self.on_canvas_exited)

        if isinstance(self.view, SegmenterWithConfig):
            self.view.config_panel.s_parameter_changed.connect(
                lambda _: self.view.set_confirmed_state(False)
            )
        SessionCache.s_session_loaded.connect(self.on_session_loaded)

    def on_session_loaded(self, session_id: str):
        """Handle session loaded signal - restore this controller's state."""
        try:
            page_id = self.model.id
            # Generate module ID for this controller
            module_id = SessionCache._generate_module_id(page_id, self.model.config)
            print(
                "page_id:",
                page_id,
                "config:",
                self.model.config,
                " -> MODULE ID:",
                module_id,
            )

            # DEBUG: Check what's actually in the PKL file
            raw_module_data = SessionCache._load_module_data(module_id)
            print("RAW MODULE DATA:", raw_module_data)

            # Try to load cached module state
            module_state = SessionCache.load_module_state(module_id)

            if module_state:
                print("MODULE STATE DATA:", module_state.get_data())
                print("NC_X:", module_state.get_nc_x())
                print("NC_Y:", module_state.get_nc_y())
                print("TRANSITIONS:", module_state.get_transitions_sec())
                print("PARAMETERS:", module_state.get_parameters())
            else:
                print("No module state found.")

            if not module_state:
                return

            print(f"Restoring {self.view.name} from session {session_id}...")

            self.view.activate_plot()

            # Get cached data
            nc_x = module_state.get_nc_x()
            nc_y = module_state.get_nc_y()
            transitions_sec = module_state.get_transitions_sec()
            parameters = module_state.get_parameters()

            # Restore view state
            if nc_x is not None and nc_y is not None:
                self.view.create_feature_plot(nc_x, nc_y)
                print(f"  ✓ Restored novelty curve for {self.view.name}")

            if transitions_sec:
                self.view.remove_all_transition_lines()
                for transition in transitions_sec:
                    self.view.add_transition_line(
                        transition, color=self.TRANSITION_LINE_COLOR
                    )
                print(
                    f"  ✓ Restored {len(transitions_sec)} transitions for {self.view.name}"
                )

            if isinstance(self.view, SegmenterWithConfig) and parameters:
                for param, value in parameters.items():
                    self.view.config_panel.update_content(param, value)
                print(f"  ✓ Restored parameters for {self.view.name}")

            self.view.set_confirmed_state(True)

        except Exception as e:
            print(f"Error restoring {self.view.name}: {e}")

    def on_canvas_clicked_position(self, position: float, button: int):
        """Handle mouse click events on canvas."""
        if button == 1:  # Left click
            self.on_canvas_left_clicked_position(position)
        elif button == 3:  # Right click
            self.on_canvas_right_clicked_position(position)

    def on_canvas_left_clicked_position(self, position: float):
        """Handle left-click on canvas."""
        self.view.stop_animation()

        # Check if click is on a transition and handle animation
        transition_pos, transition_index = None, None
        if not self.view.transitions_hidden:
            transition_pos, transition_index = self.model.get_transitions_pos_and_index(
                self.view.transitions, position
            )

        if transition_pos is not None and transition_index is not None:
            self.change_cursor_pos.emit(transition_pos)
        else:
            self.change_cursor_pos.emit(position)

    def on_canvas_right_clicked_position(self, position):
        """Handle right-click on canvas and show context menu."""
        # Check if right-click is near a transition line
        transition_pos, transition_index = None, None
        if not self.view.transitions_hidden:
            transition_pos, transition_index = self.model.get_transitions_pos_and_index(
                self.view.transitions, position
            )

        # If right-clicking on a transition
        if transition_pos is not None and transition_index is not None:
            self.view.start_transition_animation(transition_index)

        # Show Context Menu (placeholder - does nothing for now as requested)
        self.view.show_context_menu(position=position, transition_pos=transition_pos)

    def on_canvas_hovered_position(self, position: float):
        """Handle mouse hover events."""
        # Update preview cursor for feature widget functionality
        if self.view.is_plot_generated:
            self.view.set_preview_cursor_position(position)
            self.view.show_preview_cursor(True)

        # Change cursor based on whether we're over a transition line
        if not self.view.transitions_hidden and self.model.is_on_transition(
            self.view.transitions, position
        ):
            self.view.set_cursor_type(Qt.CursorShape.PointingHandCursor)
        else:
            self.view.set_cursor_type(Qt.CursorShape.ArrowCursor)

    def on_canvas_exited(self):
        """Handle mouse exit events."""
        if self.view.is_plot_generated:
            self.view.show_preview_cursor(False)
