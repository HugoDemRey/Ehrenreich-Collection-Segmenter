"""Model for timeline visualization with transition management.

Provides timeline data management, transition tracking, and
interaction handling for audio timeline visualization components.

Author: Hugo Demule
Date: January 2026
"""

from typing import Any, Dict, List

from models.analysis_page.components.interactable_plot_m import InteractablePlotModel
from src.audio.signal import Signal
from src.io.session_cache import SessionCache


class TimelineModel(InteractablePlotModel):
    """Model for timeline visualization with transition management.

    Manages timeline data, transition points, and user interactions
    for audio timeline display and analysis components.
    """

    def __init__(self, signal: Signal, id: str = "timeline"):
        super().__init__(signal=signal, id=id)
        self._transitions: List[Dict[str, Any]] = []  # Local cache for performance

    def add_transition(self, position: float, color: str = "white") -> bool:
        """Add a transition and update cache."""
        transition = {
            "time": position,
            "color": color,
            "id": f"transition_{position}_{color}",
        }

        # Add to local cache if not already present
        if not any(
            t["time"] == position and t["color"] == color for t in self._transitions
        ):
            self._transitions.append(transition)
            # Update session cache
            self._save_transitions_to_cache()
            return True

        return False

    def remove_transition(self, position: float) -> bool:
        """Remove a transition at specific position and update cache."""
        # Find and remove the transition at the given position
        if position in [t["time"] for t in self._transitions]:
            self._transitions = [t for t in self._transitions if t["time"] != position]
            self._save_transitions_to_cache()
            return True

        return False

    def remove_all_transitions(self) -> bool:
        """Remove all transitions and update cache."""
        self._transitions.clear()
        self._save_transitions_to_cache()
        return True

    def get_transitions(self) -> List[Dict[str, Any]]:
        """Get all current transitions."""
        return self._transitions.copy()

    def _save_transitions_to_cache(self) -> None:
        """Save current transitions to session cache."""
        SessionCache.save_timeline_state(self._transitions)
