"""Simple timer utility for performance measurement and duration tracking.

Provides a basic timer class for measuring elapsed time in audio processing
workflows and user interface operations.

Author: Hugo Demule
Date: January 2026
"""

import time


class Time:
    """Simple timer utility for measuring elapsed time and duration formatting.

    Provides start/stop timer functionality with automatic duration reporting
    and time formatting utilities for user-friendly display.

    Example:
        >>> timer = Time()
        >>> timer.start_timer()
        >>> # ... some processing ...
        >>> timer.stop_timer()  # Prints: "Duration: 2.34 seconds"
        >>>
        >>> # Format seconds to HH:MM:SS
        >>> formatted = Time.seconds_to_hms(3661)  # "01:01:01"
    """

    def __init__(self) -> None:
        """Initialize timer with reset state."""
        self.start_time = None
        self.end_time = None

    @staticmethod
    def seconds_to_hms(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format string.

        Args:
            seconds (float): Duration in seconds.

        Returns:
            str: Formatted time string in HH:MM:SS format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"

    def start_timer(self) -> None:
        """Start timing measurement."""
        self.start_time = time.time()

    def stop_timer(self) -> None:
        """Stop timer and print elapsed duration.

        Raises:
            ValueError: If timer was not started.
        """
        if self.start_time is None:
            raise ValueError(
                "Timer was not started. Please call start_timer() before stop_timer()."
            )
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        self.start_time = None  # Reset start time for future use
        print(f"Duration: {elapsed_time:.2f} seconds")
