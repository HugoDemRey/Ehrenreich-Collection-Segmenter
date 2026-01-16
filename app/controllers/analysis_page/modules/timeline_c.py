"""Controller for timeline visualization module.

Coordinates timeline display, transition management, and user
interactions for the timeline visualization component.

Author: Hugo Demule
Date: January 2026
"""

from controllers.analysis_page.components.interactable_plot_c import (
    InteractablePlotController,
)
from models.analysis_page.modules.timeline_m import TimelineModel
from src.io.session_cache import SessionCache
from views.analysis_page.modules.timeline import Timeline


class TimelineController(InteractablePlotController):
    """Controller for timeline visualization and interaction management.

    Manages timeline display, transition point interactions, and
    coordinates between timeline view and model components.
    """

    def __init__(self, view: Timeline, model: TimelineModel):
        super().__init__(view, model)
        self.setup_connections()

    def on_session_loaded(self, session_id: str):
        """Handle session loaded signal - restore this controller's state."""
        assert isinstance(self.model, TimelineModel), (
            "Model must be an instance of TimelineModel"
        )
        try:
            print(f"(TIMELINE CONTROLLER) Restoring state for {self.view.name}...")
            # Load Timeline data from session cache
            timeline_state = SessionCache.load_timeline_state()

            print("LOADED TIMELINE STATE:", timeline_state)

            if not timeline_state:
                return

            transitions_sec = timeline_state.get_transitions_sec()
            from pprint import pprint

            pprint(transitions_sec)

            self.remove_all_transitions()
            for item in transitions_sec:
                transition: float = item["time"]
                color: str = item.get("color", "white")
                self.model.add_transition(transition, color)
                self.view.add_transition_line(transition, color=color)

        except Exception as e:
            print(f"Error restoring {self.view.name}: {e}")

    def setup_connections(self):
        from views.analysis_page.modules.timeline import Timeline

        assert isinstance(self.view, Timeline), "View must be an instance of Timeline"
        self.view.remove_this_transition.connect(self.remove_transition_at_position)
        self.view.remove_all_transitions.connect(self.remove_all_transitions)

    def remove_transition_at_position(self, position: float) -> bool:
        """Remove transition at position and update cache."""
        # Remove from model cache first
        assert isinstance(self.model, TimelineModel), (
            "Model must be an instance of TimelineModel"
        )
        success = self.model.remove_transition(position)

        if success:
            # Remove from view
            self.view.remove_transition_line(position)
            print(f"(CONTROLLER): Removed transition at {position}")

        return success

    def remove_all_transitions(self) -> bool:
        """Remove all transition lines from canvas and cache."""
        # Clear model cache first
        assert isinstance(self.model, TimelineModel), (
            "Model must be an instance of TimelineModel"
        )
        success = self.model.remove_all_transitions()

        if success:
            # Clear view
            self.view.remove_all_transition_lines()
            print("(CONTROLLER): Removed all transitions")

        return success

    def add_transition(self, timestamp: float, color: str = "white") -> bool:
        """Add a transition at the specified timestamp with optional color."""
        # Add to model cache first
        assert isinstance(self.model, TimelineModel), (
            "Model must be an instance of TimelineModel"
        )
        success = self.model.add_transition(timestamp, color)

        if success:
            # Add to view
            self.view.add_transition_line(timestamp, color=color)
            print(f"(CONTROLLER): Added transition at {timestamp} with color {color}")

        return success
