"""Audio Signal Processing - Core signal representation and manipulation classes.

This module provides the fundamental classes for representing and manipulating audio
signals in memory. It defines abstract base classes and concrete implementations for
audio signal processing, analysis, and visualization in digital audio applications.

Key Features:
    - Abstract Signal base class for polymorphic signal handling
    - SubSignal class for efficient temporal windowing and segmentation
    - In-memory audio sample manipulation with NumPy integration
    - Signal visualization with matplotlib integration
    - WAV file I/O operations with scipy integration
    - Temporal offset tracking for nested signal operations
    - Hash-based signal identification for caching and tracking
    - Comprehensive error handling and validation

Core Classes:
    - Signal: Abstract base class providing common audio signal operations
    - SubSignal: Concrete implementation for temporal signal segments

Common Use Cases:
    - Loading and representing audio data from various sources
    - Creating temporal segments for focused analysis
    - Signal visualization and waveform plotting
    - Audio format conversion and export operations
    - Caching and identification of processed audio segments
    - Multi-level temporal windowing for hierarchical analysis

Technical Details:
    - Uses NumPy arrays for efficient sample storage and manipulation
    - Supports arbitrary sample rates with floating-point precision
    - Implements proper temporal offset handling for nested segments
    - Provides hash-based identification for signal tracking
    - Includes comprehensive validation and error handling

Author: Hugo Demule
Date: January 2026
"""

import hashlib
from typing import override

import numpy as np


class Signal:
    """Class for digital audio signal representation and processing.

    Signal provides a comprehensive foundation for representing and manipulating
    digital audio signals, including functionality for signal analysis, visualization,
    temporal segmentation, and I/O operations. While it inherits from ABC to provide
    a consistent interface, Signal is a fully functional concrete class that can be
    instantiated directly, typically through the AudioFile class.

    The Signal class encapsulates:
        - Raw audio sample data as NumPy arrays
        - Sample rate information for temporal calculations
        - Original filename tracking for provenance
        - Unique identification system for caching and tracking
        - Duration calculations and temporal operations
        - Signal normalization and statistical operations

    Key Features:
        - Efficient NumPy-based sample storage and manipulation
        - Hash-based unique identification for signal tracking
        - Temporal offset support for nested signal operations
        - Comprehensive validation and error handling
        - Rich visualization capabilities with matplotlib integration
        - WAV file export functionality

    Creation Pattern:
        Signal instances are typically created through the AudioFile class,
        which handles file loading and provides the necessary audio data.
        Direct instantiation is possible but requires proper audio samples
        and metadata.
    Temporal Model:
        The class maintains precise temporal information through sample rates
        and sample counts, enabling accurate time-based operations. Temporal
        offsets are supported to handle nested signal segments correctly.

    Attributes:
        samples (np.ndarray): Array of audio samples (typically float32 or float64).
        sample_rate (float): Sample rate in Hz (samples per second).
        origin_filename (str): Original filename of the audio source for provenance.
        id (str): Unique MD5-based identifier derived from the filename.
        _duration (float): Cached duration of the signal in seconds.

    Example:
        >>> # Signal is abstract, so use a concrete subclass
        >>> signal = ConcreteSignal(samples, 44100, "audio.wav")
        >>> print(f"Duration: {signal.duration_seconds():.2f}s")
        >>> subsig = signal.subsignal(10.0, 20.0)  # Extract 10-second segment
        >>> signal.plot()  # Visualize the waveform
    """

    def __init__(
        self, samples: np.ndarray, sample_rate: float, origine_filename: str
    ) -> None:
        """Initialize a Signal instance with audio data and metadata.

        Creates a new Signal instance with the provided audio samples, sample rate,
        and filename information. Performs validation of input parameters and
        calculates derived properties such as duration and unique identifier.

        The initialization process:
            - Validates sample rate is positive
            - Ensures samples array is not empty
            - Calculates signal duration from sample count and rate
            - Generates unique MD5-based identifier from filename
            - Stores all metadata for future reference

        Args:
            samples (np.ndarray): Array of audio samples. Should be 1D array of
                                 floating-point values representing audio amplitudes.
                                 Typical range is [-1.0, 1.0] for normalized audio.
            sample_rate (float): Sample rate in Hz (samples per second). Must be
                               positive. Common values: 44100, 48000, 22050.
            origine_filename (str): Original filename of the audio source. Used for
                                  identification and provenance tracking. Can be
                                  empty string if no source file.

        Raises:
            ValueError: If sample_rate is not positive (sample_rate <= 0).
            ValueError: If samples array is empty (len(samples) == 0).

        Example:
            >>> samples = np.array([0.1, 0.2, -0.1, 0.0, 0.3])
            >>> signal = ConcreteSignal(samples, 44100, "test.wav")
            >>> print(signal.duration_seconds())  # Outputs duration in seconds

        Note:
            The samples array is stored by reference, not copied. Modifications
            to the original array will affect the Signal instance.
        """
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer.")
        if len(samples) == 0:
            raise ValueError("Samples list cannot be empty.")

        self.samples = samples
        self.sample_rate = sample_rate
        self._duration = len(samples) / sample_rate
        self.origin_filename = origine_filename
        self.id = hashlib.md5(origine_filename.encode()).hexdigest()

    def set_id(self, id: str) -> None:
        """Set a custom identifier for the signal.

        Updates the signal's unique identifier to a custom value. This can be
        useful for tracking signals through processing pipelines or assigning
        meaningful names to generated or modified signals.

        Args:
            id (str): The new identifier for the signal. Can be any string value.
                     Should be unique within the application context if used for
                     tracking or caching purposes.

        Example:
            >>> signal.set_id("processed_audio_v1")
            >>> print(signal.get_id())  # "processed_audio_v1"

        Note:
            This overwrites the automatically generated MD5-based identifier.
            Use with caution if the automatic ID is being used for caching
            or other identification purposes.
        """
        self.id = id

    def get_id(self) -> str:
        """Get the current identifier of the signal.

        Returns the unique identifier associated with this signal instance.
        By default, this is an MD5 hash of the original filename, but it
        can be customized using set_id().

        Returns:
            str: The current identifier of the signal. This could be either
                 the automatically generated MD5 hash or a custom identifier
                 set via set_id().

        Example:
            >>> signal_id = signal.get_id()
            >>> print(f"Signal ID: {signal_id}")

        Note:
            The ID is used internally for caching and tracking purposes.
            It should remain consistent for the same logical signal.
        """
        return self.id

    def duration_seconds(self) -> float:
        """Get the total duration of the signal in seconds.

        Calculates and returns the duration based on the number of samples
        and the sample rate. This is a fundamental property used throughout
        audio processing operations.

        The duration is calculated as: len(samples) / sample_rate

        Returns:
            float: Duration of the signal in seconds with floating-point precision.
                  Precision depends on the sample rate and total sample count.

        Example:
            >>> signal = ConcreteSignal(np.zeros(44100), 44100, "test.wav")
            >>> duration = signal.duration_seconds()
            >>> print(f"Duration: {duration:.3f} seconds")  # "Duration: 1.000 seconds"

        Note:
            This value is cached during initialization for performance but
            reflects the current state of the samples array.
        """
        return self._duration

    def subsignal(self, from_time: float, to_time: float) -> "SubSignal":
        """Extract a temporal segment as a new SubSignal instance.

        Creates a SubSignal representing a specific time range within this signal.
        The SubSignal maintains a reference to the original temporal context while
        providing access to only the requested segment of audio data.

        This method enables:
            - Temporal windowing for focused analysis
            - Creating training samples from larger audio files
            - Isolating specific events or regions of interest
            - Hierarchical signal processing workflows

        Args:
            from_time (float): Start time in seconds (inclusive). Must be >= 0
                              and < signal duration.
            to_time (float): End time in seconds (exclusive). Must be > from_time
                            and <= signal duration.

        Returns:
            SubSignal: A new SubSignal instance representing the specified temporal
                      segment. The SubSignal maintains proper offset information
                      for temporal context preservation.

        Raises:
            ValueError: If the audio signal is not loaded (samples is None).
            ValueError: If from_time < 0 or to_time > signal duration.
            ValueError: If from_time >= to_time (invalid time range).

        Example:
            >>> # Extract a 5-second segment starting at 10 seconds
            >>> segment = signal.subsignal(10.0, 15.0)
            >>> print(f"Segment duration: {segment.duration_seconds():.1f}s")  # 5.0s
            >>> print(f"Segment offset: {segment.offset_time():.1f}s")  # 10.0s

        Note:
            The returned SubSignal shares the same sample rate and maintains
            proper temporal offset information for accurate time-based operations.
        """
        if self.samples is None or self.sample_rate is None:
            raise ValueError("Signal not loaded. Please load the audio first.")

        start_sample = int(from_time * self.sample_rate)
        end_sample = int(to_time * self.sample_rate)

        # return Signal(self.samples[start_sample:end_sample], self.sample_rate, self.origin_filename, from_time=from_time, to_time=to_time)
        return SubSignal(self, from_time, to_time)

    def norm(self) -> float:
        """Compute the L-infinity norm (maximum absolute value) of the signal.

        Calculates the maximum absolute amplitude across all samples in the signal.
        This is useful for normalization, clipping detection, and dynamic range
        analysis. A small epsilon is added to prevent division by zero in
        downstream normalization operations.

        The calculation is: max(|samples|) + ε, where ε = 1e-12

        Returns:
            float: The L-infinity norm of the signal samples (maximum absolute
                  value) plus a small epsilon to avoid numerical issues.

        Example:
            >>> samples = np.array([0.1, -0.8, 0.3, -0.2])
            >>> signal = ConcreteSignal(samples, 44100, "test.wav")
            >>> norm_val = signal.norm()
            >>> print(f"Signal norm: {norm_val:.6f}")  # Approximately 0.800000

        Note:
            The epsilon addition (1e-12) prevents division by zero errors when
            this norm value is used for signal normalization operations.
        """
        return np.max(np.abs(self.samples)) + 1e-12  # Avoid division by zero

    def offset_time(self) -> float:
        """Get the temporal offset of this signal relative to its original context.

        For base Signal instances, this always returns 0.0 since they represent
        the complete original signal. SubSignal instances override this method
        to return their actual offset time within the parent signal.

        This method enables proper temporal context preservation when working
        with signal segments and nested subsignals.

        Returns:
            float: The temporal offset in seconds. Always 0.0 for base Signal
                  instances, actual offset for SubSignal instances.

        Example:
            >>> parent_signal = ConcreteSignal(samples, 44100, "audio.wav")
            >>> print(parent_signal.offset_time())  # 0.0
            >>>
            >>> sub = parent_signal.subsignal(10.0, 20.0)
            >>> print(sub.offset_time())  # 10.0

        Note:
            This method is part of the temporal context system that allows
            nested signal operations to maintain proper time references.
        """
        return 0.0

    def save_wav(self, filename: str) -> None:
        """Export the signal as a WAV audio file.

        Saves the current signal samples to a WAV file using the signal's
        sample rate. The output file will contain the raw audio data in
        standard WAV format compatible with most audio software.

        File Format:
            - Standard WAV format (RIFF container)
            - Sample rate matches the signal's sample_rate
            - Bit depth determined by sample data type
            - Mono audio (single channel)

        Args:
            filename (str): The complete file path (including extension) where
                          the WAV file should be saved. Parent directories
                          must exist.

        Raises:
            IOError: If the file cannot be written (permissions, disk space, etc.).
            ValueError: If the samples contain invalid audio data.

        Side Effects:
            - Creates a WAV file at the specified location
            - Overwrites existing files without warning
            - Prints a confirmation message to console

        Example:
            >>> signal.save_wav("/path/to/output.wav")
            Signal saved as WAV file: /path/to/output.wav

        Note:
            Requires scipy to be installed. The method uses scipy.io.wavfile
            for WAV file generation.
        """
        from scipy.io import wavfile

        wavfile.write(filename, self.sample_rate, self.samples)
        print(f"Signal saved as WAV file: {filename}")

    def plot(self, title=None) -> None:
        """Create a high-quality waveform visualization of the signal.

        Generates a matplotlib plot of the signal's waveform with professional
        styling, proper time axis labeling, and enhanced visual formatting.
        The plot includes temporal offset handling for SubSignal instances.

        Plot Features:
            - High-resolution figure (150 DPI) for crisp display
            - Professional color scheme with warm orange waveform
            - Proper time axis with offset support
            - Clean grid and axis styling
            - Automatic title generation from filename or ID
            - Responsive layout with tight margins

        Time Axis:
            The time axis correctly reflects the signal's temporal position,
            including any offset from parent signals. This ensures accurate
            time representation for SubSignal instances.

        Args:
            title (str, optional): Custom title for the plot. If None, the title
                                 is automatically generated from the original
                                 filename (if available) or the signal ID.

        Raises:
            ImportError: If matplotlib is not installed.

        Side Effects:
            - Opens a matplotlib plot window with the waveform visualization
            - Blocks execution until the plot window is closed
            - Uses matplotlib's default backend for display

        Example:
            >>> signal.plot()  # Uses automatic title
            >>> signal.plot("Custom Waveform Title")  # Uses custom title

        Styling Details:
            - Figure size: 12x6 inches at 150 DPI
            - Waveform color: Warm orange (#d98d1a)
            - Line width: 0.8 pixels with 90% opacity
            - Grid: Light gray with 30% transparency
            - Axes: Clean borders with minimal styling

        Note:
            Requires matplotlib to be installed. The plot window behavior
            depends on the matplotlib backend configuration.
        """
        import os

        import matplotlib.pyplot as plt

        # Create time axis with offset
        time_axis = np.arange(len(self.samples)) / self.sample_rate + self.offset_time()

        # Create high-quality figure
        plt.figure(figsize=(12, 6), dpi=150)

        # Plot with improved styling
        plt.plot(time_axis, self.samples, color="#d98d1a", linewidth=0.8, alpha=0.9)

        # Enhanced title and labels
        title = (
            title
            if title is not None
            else (
                os.path.basename(self.origin_filename)
                if self.origin_filename
                else self.id
            )
        )
        plt.title(
            f"Waveform Plot of Signal: {title}", fontsize=14, fontweight="bold", pad=20
        )
        plt.xlabel("Time (s)", fontsize=12, fontweight="medium")
        plt.ylabel("Amplitude", fontsize=12, fontweight="medium")

        # Improved grid and styling
        plt.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_linewidth(0.5)
        plt.gca().spines["bottom"].set_linewidth(0.5)

        # Adjust layout and show
        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """Return a detailed string representation of the Signal object.

        Creates a formatted, human-readable string containing comprehensive
        information about the signal including metadata, properties, and
        key characteristics. Useful for debugging, logging, and inspection.

        The output includes:
            - Class name (Signal or SubSignal)
            - Unique identifier
            - Original filename
            - Sample rate in Hz
            - Duration in seconds (3 decimal places)
            - Total number of samples

        Returns:
            str: Multi-line formatted string representation with indented
                 key-value pairs for easy readability.

        Example Output:
            Signal(
              id: 'a1b2c3d4e5f6...',
              origin_filename: 'audio_file.wav',
              sample_rate: 44100.0 Hz,
              duration: 145.230 s,
              samples number: 6398000
            )

        Note:
            This method is automatically called when using print() or str()
            on a Signal instance. The format is designed for human readability
            rather than machine parsing.
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"  id: '{self.id}',\n"
            f"  origin_filename: '{self.origin_filename}',\n"
            f"  sample_rate: {self.sample_rate} Hz,\n"
            f"  duration: {self._duration:.3f} s,\n"
            f"  samples number: {self.samples.shape[0]}\n"
            f")"
        )

    def duration(self) -> float:
        """Get the duration of the signal in seconds (alias for duration_seconds).

        Provides an alternative, shorter method name for accessing the signal
        duration. This method is functionally identical to duration_seconds()
        and is provided for convenience and API consistency.

        Returns:
            float: Duration of the signal in seconds with floating-point precision.
                  Same value as returned by duration_seconds().

        Example:
            >>> signal = ConcreteSignal(np.zeros(44100), 44100, "test.wav")
            >>> duration1 = signal.duration()          # Using short alias
            >>> duration2 = signal.duration_seconds()  # Using full method name
            >>> assert duration1 == duration2          # They return the same value

        Note:
            This is a convenience method that delegates to the cached _duration
            attribute. Both duration() and duration_seconds() return identical values.
        """
        return self._duration


class SubSignal(Signal):
    """Represents a temporal segment extracted from a parent Signal.

    SubSignal extends the Signal class to represent specific time ranges within
    a larger audio signal. It maintains all the functionality of its parent Signal
    while preserving crucial temporal context information including offset times
    and proper time-based referencing.

    Key Features:
        - Inherits all Signal functionality (plotting, analysis, I/O)
        - Maintains temporal context through offset tracking
        - Supports nested subsignal operations
        - Preserves parent signal metadata and provenance
        - Automatic ID generation with temporal suffixes
        - Efficient sample extraction without data duplication

    Temporal Model:
        SubSignal instances track their position within the original signal
        through the from_time attribute and offset_time() method. This enables:
        - Accurate time-axis labeling in plots
        - Proper temporal alignment in multi-signal analysis
        - Nested subsignal extraction with correct offsets
        - Preservation of original timing context

    ID Management:
        The SubSignal automatically generates a unique ID by appending temporal
        information to the parent signal's ID in the format:
        parent_id_startTime-endTime (e.g., "abc123_10-15" for 10s-15s segment)

    Attributes:
        from_time (float): Start time of the subsignal relative to the ultimate
                          parent signal (accounting for nested offsets).
        All other attributes inherited from Signal class.

    Example:
        >>> parent = ConcreteSignal(samples, 44100, "music.wav")
        >>> # Extract 10-second segment starting at 30 seconds
        >>> segment = SubSignal(parent, 30.0, 40.0)
        >>> print(segment.offset_time())    # 30.0
        >>> print(segment.duration())       # 10.0
        >>>
        >>> # Nested subsignal extraction
        >>> nested = segment.subsignal(2.0, 5.0)  # 3-second segment
        >>> print(nested.offset_time())     # 32.0 (30 + 2)

    Note:
        SubSignal shares sample data with its parent through array slicing,
        making it memory-efficient. However, modifications to the samples
        array will affect both the SubSignal and parent Signal.
    """

    def __init__(self, parent_signal: Signal, from_time: float, to_time: float) -> None:
        """Initialize a SubSignal from a parent Signal within specified time bounds.

        Creates a new SubSignal instance representing a temporal segment of the
        parent signal. Performs comprehensive validation of time bounds and
        extracts the appropriate sample range while preserving temporal context.

        The initialization process:
            - Validates time bounds against parent signal duration
            - Ensures from_time < to_time for valid temporal range
            - Calculates sample indices for extraction
            - Calls parent Signal.__init__ with extracted samples
            - Updates ID with temporal suffix for uniqueness
            - Calculates and stores offset time information

        Temporal Handling:
            The SubSignal correctly handles nested temporal contexts. If the
            parent is itself a SubSignal, the offset calculations account for
            the complete chain of temporal offsets.

        Args:
            parent_signal (Signal): The parent signal from which to extract
                                   the temporal segment. Can be any Signal
                                   instance including other SubSignals.
            from_time (float): Start time in seconds relative to the parent signal.
                              Must be >= 0 and < parent signal duration.
            to_time (float): End time in seconds relative to the parent signal.
                            Must be > from_time and <= parent signal duration.

        Raises:
            ValueError: If from_time < 0 or to_time > parent signal duration.
                       Includes detailed bounds information in error message.
            ValueError: If from_time >= to_time (invalid or zero-length range).
                       Includes actual values in error message.

        Example:
            >>> parent = ConcreteSignal(samples, 44100, "audio.wav")  # 60s duration
            >>> segment = SubSignal(parent, 10.0, 25.0)  # 15s segment at 10s offset
            >>> print(f"Segment: {segment.duration():.1f}s at {segment.offset_time():.1f}s")
            # Output: "Segment: 15.0s at 10.0s"

        Note:
            Sample extraction uses integer sample indices calculated from
            floating-point times, which may introduce minor quantization
            effects at very high precision requirements.
        """
        if from_time < 0 or to_time > parent_signal._duration:
            raise ValueError(
                f"Invalid from_time and to_time for subsignal. from_time and to_time must fall in the interval [0 - {parent_signal._duration}]. ---> from_time: {from_time}, to_time: {to_time}"
            )

        if from_time >= to_time:
            raise ValueError(
                f"from_time must be less than to_time for subsignal. from_time: {from_time}, to_time: {to_time}"
            )

        start_sample = int(from_time * parent_signal.sample_rate)
        end_sample = int(to_time * parent_signal.sample_rate)

        super().__init__(
            parent_signal.samples[start_sample:end_sample],
            parent_signal.sample_rate,
            parent_signal.origin_filename,
        )

        self.id += f"_{int(from_time + parent_signal.offset_time())}-{int(to_time + parent_signal.offset_time())}"
        self._duration = to_time - from_time
        self.from_time = from_time + parent_signal.offset_time()

    @override
    def offset_time(self) -> float:
        """Get the absolute temporal offset of this SubSignal.

        Returns the start time of this SubSignal relative to the ultimate parent
        signal, accounting for any nested SubSignal offsets in the chain. This
        enables proper temporal context preservation across multiple levels of
        signal segmentation.

        Offset Calculation:
            For direct SubSignals: parent.offset_time() + from_time
            For nested SubSignals: accumulates offsets through the entire chain

        This method enables:
            - Accurate time-axis labeling in plots
            - Proper temporal alignment across signal segments
            - Correct time references in nested subsignal operations
            - Preservation of original temporal context

        Returns:
            float: The absolute offset time in seconds from the beginning of
                  the ultimate parent signal. For a SubSignal of a SubSignal,
                  this represents the cumulative offset.

        Example:
            >>> parent = ConcreteSignal(samples, 44100, "audio.wav")
            >>> segment1 = SubSignal(parent, 10.0, 30.0)  # 10s-30s
            >>> segment2 = SubSignal(segment1, 5.0, 15.0)  # 5s-15s within segment1
            >>>
            >>> print(parent.offset_time())    # 0.0 (original signal)
            >>> print(segment1.offset_time())  # 10.0 (10s into original)
            >>> print(segment2.offset_time())  # 15.0 (10s + 5s into original)

        Note:
            This method overrides the base Signal.offset_time() which always
            returns 0.0. The @override decorator ensures type checker compliance
            and documents the inheritance relationship.
        """
        return self.from_time
