"""Color utilities for consistent visualization across the audio analysis framework.

Provides predefined color palette and utility functions for consistent
color selection in plots, annotations, and user interface elements.

Author: Hugo Demule
Date: January 2026
"""

COLORS = ["#FFB274", "#79BEFF", "#77FFA6", "#FFEA00", "#D09BFF"]


def get_color(index: int) -> str:
    """Get a color from the predefined palette using cycling index.

    Args:
        index (int): Color index (automatically cycles through available colors).

    Returns:
        str: Hex color code from the predefined palette.

    Example:
        >>> color1 = get_color(0)  # "#FFB274" (orange)
        >>> color2 = get_color(5)  # "#FFB274" (cycles back to first)
    """
    return COLORS[index % len(COLORS)]
