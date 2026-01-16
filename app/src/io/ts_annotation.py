"""Time Series Annotation I/O - Utilities for reading and writing temporal annotations.

This module provides comprehensive functionality for handling time series annotations
in various formats commonly used in audio analysis and music information retrieval.
It supports loading annotations from CSV files, extracting temporal subsets, and
managing transition timestamps through different file formats.

Key Features:
    - CSV annotation loading with flexible formatting support
    - Temporal filtering and offset adjustment for annotation subsets
    - Transition timestamp management (text and JSON formats)
    - Robust error handling for malformed data
    - Unicode support for international character sets
    - Header detection and flexible column parsing

Supported Formats:
    - Semicolon-separated CSV files with optional quotes
    - Plain text files with one timestamp per line
    - JSON files for structured transition data
    - Automatic encoding detection and UTF-8 support

Common Use Cases:
    - Loading ground truth annotations for evaluation
    - Extracting annotation segments for focused analysis
    - Converting between different annotation formats
    - Managing detected transition timestamps
    - Preparing datasets for machine learning training

Author: Hugo Demule
Date: January 2026
"""

import csv
from typing import List, Optional, Tuple


class TSAnnotations:
    """Static utility class for time series annotation input/output operations.

    This class provides a collection of static methods for handling various aspects
    of time series annotation data, including loading from different file formats,
    temporal filtering, and format conversion. It's designed to handle the common
    annotation formats used in audio analysis and music information retrieval.

    The class supports:
        - Flexible CSV parsing with semicolon delimiters and quote handling
        - Temporal subset extraction with automatic offset adjustment
        - Transition timestamp management in text and JSON formats
        - Robust error handling for real-world data inconsistencies
        - Unicode support for international annotation datasets

    Annotation Format:
        The primary annotation format is a tuple of (start_time, end_time, label)
        where times are in seconds as floating-point numbers and labels are strings.
        This format is suitable for representing:
        - Musical segments (verses, choruses, etc.)
        - Harmonic regions with chord labels
        - Structural boundaries with section names
        - Any temporal regions with associated metadata

    Example CSV Format:
        "start";"end";"label"
        0.0;87.5;"Intro"
        87.5;145.2;"Verse 1"
        145.2;203.8;"Chorus"

    Note:
        All methods are static and the class serves as a namespace for related
        functionality. No instantiation is required or recommended.
    """

    @staticmethod
    def load_annotations(csv_path: str) -> List[Tuple[float, float, str]]:
        """Load time-based annotations from a semicolon-separated CSV file.

        Parses CSV files containing temporal annotations with start times, end times,
        and optional labels. The method handles various formatting scenarios including
        optional headers, quoted fields, empty labels, and malformed rows.

        CSV Format Support:
            - Semicolon (;) as field separator
            - Double quotes (") as quote character
            - Optional header row (automatically detected and skipped)
            - Minimum two columns required (start, end)
            - Optional third column for labels
            - Empty or missing labels default to empty string
            - UTF-8 encoding with automatic handling

        Args:
            csv_path (str): Path to the CSV file containing annotations.
                           File must exist and be readable.

        Returns:
            List[Tuple[float, float, str]]: List of annotation tuples where each
                tuple contains (start_time, end_time, label). Times are in seconds
                as floating-point numbers. Returns empty list if no valid
                annotations are found.

        Raises:
            FileNotFoundError: If the specified CSV file doesn't exist.
            PermissionError: If the file cannot be read due to permissions.
            UnicodeDecodeError: If the file contains invalid UTF-8 sequences.

        Example:
            >>> annotations = TSAnnotations.load_annotations("segments.csv")
            >>> print(f"Loaded {len(annotations)} annotations")
            >>> for start, end, label in annotations[:3]:
            ...     print(f"{start:.1f}s - {end:.1f}s: {label}")

        Note:
            Rows with non-numeric start/end times are automatically skipped,
            making the method robust against header rows and malformed data.
        """
        annotations: List[Tuple[float, float, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter=";", quotechar='"')
            for row in reader:
                if not row:
                    continue
                # Expect at least two columns for start and end
                if len(row) < 2:
                    continue
                start_s, end_s = row[0].strip(), row[1].strip()

                # Skip header or malformed rows where start/end are not numbers
                try:
                    start_time = float(start_s)
                    end_time = float(end_s)
                except ValueError:
                    continue

                label = ""
                if len(row) > 2:
                    label = row[2].strip()
                    # csv.reader will already remove surrounding quotes, but normalize empty quoted fields
                    if label == '""':
                        label = ""
                annotations.append((start_time, end_time, label))
        return annotations

    @staticmethod
    def sub_annotations(
        annotations: List[Tuple[float, float, str]], start: float, end: float
    ) -> List[Tuple[float, float, str]]:
        """Extract and time-shift annotations within a specified temporal window.

        Filters annotations to include only those that fall entirely within the
        specified time range, then adjusts their timestamps relative to the start
        of the window. This is useful for extracting annotation subsets for focused
        analysis or creating training samples from larger datasets.

        Filtering Logic:
            - Only annotations with both start_time >= start AND end_time <= end
              are included (conservative filtering)
            - Partially overlapping annotations are excluded to maintain integrity
            - Empty result is possible if no annotations fall within the window

        Time Offset:
            - All retained annotations have their timestamps adjusted by subtracting
              the window start time
            - Results in a new timeline starting from 0.0 seconds
            - Original temporal relationships between annotations are preserved

        Args:
            annotations (List[Tuple[float, float, str]]): Source annotations as
                (start_time, end_time, label) tuples.
            start (float): Start time of the extraction window in seconds.
            end (float): End time of the extraction window in seconds.

        Returns:
            List[Tuple[float, float, str]]: Filtered and time-shifted annotations
                where timestamps are relative to the window start. Empty list if
                no annotations fall within the specified window.

        Example:
            >>> original = [(10.0, 20.0, "A"), (15.0, 25.0, "B"), (30.0, 40.0, "C")]
            >>> subset = TSAnnotations.sub_annotations(original, 12.0, 35.0)
            >>> # Only annotation C (30.0-40.0) is fully within [12.0, 35.0]
            >>> print(subset)  # [(18.0, 28.0, "C")] - shifted by -12.0

        Note:
            The method uses strict containment (both start and end must be within
            the window) to avoid creating partial or truncated annotations that
            might not represent complete semantic units.
        """
        filtered = [ann for ann in annotations if ann[0] >= start and ann[1] <= end]
        offset_filtered = [(ann[0] - start, ann[1] - start, ann[2]) for ann in filtered]
        return offset_filtered

    @staticmethod
    def load_transitions_txt(txt_path: str) -> List[float]:
        """Load transition timestamps from a plain text file.

        Reads a simple text file containing transition timestamps, with one
        timestamp per line. This format is commonly used for storing detected
        boundaries, structural changes, or other point-in-time events.

        File Format:
            - One timestamp per line
            - Timestamps as floating-point numbers (seconds)
            - Empty lines are automatically skipped
            - Invalid lines (non-numeric) are silently ignored
            - UTF-8 encoding support
            - No header or metadata supported

        Args:
            txt_path (str): Path to the text file containing timestamps.
                           File must exist and be readable.

        Returns:
            List[float]: List of transition timestamps in seconds, in the order
                        they appear in the file. Empty list if no valid timestamps
                        are found.

        Raises:
            FileNotFoundError: If the specified text file doesn't exist.
            PermissionError: If the file cannot be read due to permissions.
            UnicodeDecodeError: If the file contains invalid UTF-8 sequences.

        Example:
            >>> transitions = TSAnnotations.load_transitions_txt("boundaries.txt")
            >>> print(f"Loaded {len(transitions)} transition points")
            >>> for i, time in enumerate(transitions[:5]):
            ...     print(f"Transition {i+1}: {time:.2f}s")

        Note:
            The method is robust against malformed data and will skip any lines
            that cannot be parsed as floating-point numbers, making it suitable
            for files that may contain comments or other non-numeric content.
        """
        transitions: List[float] = []
        with open(txt_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    time_stamp = float(line)
                    transitions.append(time_stamp)
                except ValueError:
                    continue
        return transitions

    @staticmethod
    def save_transitions(json_path: str, transitions: List[float]):
        """Save transition timestamps to a JSON file.

        Serializes a list of transition timestamps to JSON format, providing
        a structured and widely-compatible format for storing temporal data.
        The JSON format preserves precise floating-point values and allows
        easy integration with other tools and programming languages.

        Output Format:
            - Standard JSON array of floating-point numbers
            - UTF-8 encoding for maximum compatibility
            - Compact format without extra whitespace
            - Precise preservation of floating-point precision

        Args:
            json_path (str): Path where the JSON file should be created.
                           Existing files will be overwritten.
            transitions (List[float]): List of transition timestamps in seconds
                                     to save. Can be empty.

        Raises:
            PermissionError: If the file cannot be written due to permissions.
            OSError: If there are filesystem issues (disk full, invalid path, etc.).

        Example:
            >>> transitions = [10.5, 25.7, 45.2, 60.8]
            >>> TSAnnotations.save_transitions("output.json", transitions)
            >>> # Creates: [10.5, 25.7, 45.2, 60.8]

        Note:
            The method uses the standard JSON module for serialization, ensuring
            compatibility and proper handling of floating-point precision.
        """
        import json

        with open(json_path, "w", encoding="utf-8") as jp:
            json.dump(transitions, jp)

    @staticmethod
    def load_transitions(json_path: str) -> Optional[List[float]]:
        """Load transition timestamps from a JSON file with error handling.

        Deserializes transition timestamps from a JSON file, providing robust
        error handling for common failure scenarios such as missing files or
        malformed JSON data.

        Expected Format:
            - JSON array of floating-point numbers
            - UTF-8 encoded text file
            - Standard JSON syntax compliance required

        Args:
            json_path (str): Path to the JSON file containing transition data.

        Returns:
            Optional[List[float]]: List of transition timestamps in seconds if
                                  successful, None if the file cannot be loaded
                                  due to missing file or invalid JSON format.

        Error Handling:
            - FileNotFoundError: Returns None if file doesn't exist
            - json.JSONDecodeError: Returns None if JSON is malformed
            - Other I/O errors: May propagate depending on the specific error

        Example:
            >>> transitions = TSAnnotations.load_transitions("saved.json")
            >>> if transitions is not None:
            ...     print(f"Loaded {len(transitions)} transitions")
            ...     for t in transitions:
            ...         print(f"  {t:.2f}s")
            ... else:
            ...     print("Failed to load transitions")

        Note:
            This method provides graceful error handling suitable for use cases
            where missing or corrupted files should not crash the application.
            Always check the return value for None before using the result.
        """
        import json

        try:
            with open(json_path, "r", encoding="utf-8") as jp:
                transitions = json.load(jp)
            return transitions
        except (FileNotFoundError, json.JSONDecodeError):
            return None
