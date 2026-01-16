"""Model for navigation bar component functionality.

Handles data operations for navigation menu actions including
file save/load operations and transition data management.

Author: Hugo Demule
Date: January 2026
"""

from src.io.ts_annotation import TSAnnotations


class NavBarModel:
    """Model for navigation bar data operations.

    Provides file I/O operations for transition data and
    annotation management through the navigation interface.
    """

    def save_transitions_to_file(self, transitions, filepath: str) -> bool:
        """Save transitions to the specified file path."""
        try:
            TSAnnotations.save_transitions(filepath, transitions)
            return True
        except Exception as e:
            print(f"Error saving transitions: {e}")
            return False

    def load_transitions_from_file(self, filepath: str):
        """Load transitions from the specified file path."""
        try:
            return TSAnnotations.load_transitions(filepath)
        except Exception as e:
            print(f"Error loading transitions: {e}")
            return None

    def validate_transitions_data(self, transitions) -> bool:
        """Validate transitions data format."""
        if not transitions:
            return False

        # Check if it's a list
        if not isinstance(transitions, list):
            return False

        # Check if all items have required structure
        for item in transitions:
            if isinstance(item, dict):
                if "timestamp" not in item:
                    return False
            elif not isinstance(item, (int, float)):
                return False

        return True

    def format_transitions_for_export(self, transitions, transition_colors=None):
        """Format transitions data for export with colors."""
        if not transitions:
            return []

        formatted_transitions = []

        for i, timestamp in enumerate(transitions):
            transition_data = {"timestamp": timestamp}

            # Add color if available
            if transition_colors and i < len(transition_colors):
                transition_data["color"] = transition_colors[i]
            else:
                transition_data["color"] = "#FF0000"  # Default red

            formatted_transitions.append(transition_data)

        return formatted_transitions
