"""Abstract Feature Interfaces and Base Classes for Audio Analysis.

This module provides the foundational interface hierarchy for audio feature
representation, analysis, and visualization. It defines the core contracts
that all audio features must implement, along with sophisticated base
implementations that provide common functionality for spectral analysis,
structural analysis, and peak detection.

Key Components:
    - Feature: Abstract base interface for all audio features
    - PeakFinder: Interface for structural boundary and event detection
    - BaseFeature: Concrete base class with comprehensive analysis methods
    - SimilarityMatrix: Abstract base for self-similarity and structural analysis

Feature Hierarchy:
    The module establishes a clear inheritance hierarchy:

    Feature (ABC)
    ├── BaseFeature (concrete)
    │   ├── Spectrogram, Chromagram, MFCC, etc.
    │   └── Combined with PeakFinder for boundary detection
    └── SimilarityMatrix (ABC)
        ├── SelfSimilarityMatrix
        └── TimeLagMatrix

Core Capabilities:
    - Standardized data access and manipulation interfaces
    - Advanced normalization and preprocessing operations
    - Comprehensive visualization with temporal annotations
    - Peak detection for structural boundary analysis
    - Mathematical operations (log compression, smoothing, etc.)
    - Integration with FMP (Fundamentals of Music Processing) library

Design Philosophy:
    The interfaces follow object-oriented principles with:
    - Clear separation of concerns between feature types
    - Consistent method signatures across all implementations
    - Rich default implementations to minimize code duplication
    - Extensible design supporting new feature types
    - Integration with scientific Python ecosystem (NumPy, SciPy, Matplotlib)

Applications:
    - Music Information Retrieval (MIR) systems
    - Automatic music structure analysis and segmentation
    - Audio content analysis and classification
    - Beat tracking and tempo analysis
    - Cover song identification and music similarity
    - Real-time audio analysis and live performance systems

Author: Hugo Demule
Date: January 2026
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Self

import numpy as np
import src.libfmp.b
import src.libfmp.c3
from scipy import signal
from src.audio.signal import Signal
from src.utils.colors import get_color

if TYPE_CHECKING:
    from src.audio_features.features import NoveltyCurve


class Feature(ABC):
    """Abstract base class defining the core interface for all audio features.

    Feature provides the fundamental contract that all audio feature representations
    must implement, establishing consistent data access patterns, metadata management,
    and basic operations across the entire audio analysis framework.

    Core Responsibilities:
        - Standardized data storage and access through numpy arrays
        - Sampling rate management for temporal alignment
        - Feature metadata including names and processing flags
        - Deep copying capabilities for immutable operations
        - Foundation for specialized feature implementations

    Data Model:
        Features encapsulate:
        - Raw feature data as NumPy arrays (1D, 2D, or higher dimensions)
        - Sampling rate information for temporal interpretation
        - Processing state flags (e.g., dB scaling, normalization status)
        - Human-readable names for visualization and debugging

    Design Patterns:
        - Template Method: Provides structure for concrete implementations
        - Immutable Operations: Copy-based modifications preserve original data
        - Metadata Pattern: Rich feature descriptions for analysis workflows

    Integration:
        Feature serves as the base for specialized audio features including:
        - Spectral features (Spectrogram, MFCC, Chromagram)
        - Temporal features (Tempogram, Novelty curves)
        - Structural features (Self-similarity matrices)
        - Combined features (HRPS, multi-dimensional representations)

    Example Usage:
        >>> # Through concrete implementation
        >>> spectrogram = SpectrogramBuilder().build(audio_signal)
        >>> data = spectrogram.data()  # Access raw feature data
        >>> sr = spectrogram.sampling_rate()  # Get temporal sampling rate
        >>> copy = spectrogram.copy()  # Create independent copy
        >>> is_db = spectrogram.is_db_scaled()  # Check processing state

    Note:
        This is an abstract class requiring concrete implementation of the
        interface methods. Use BaseFeature for most practical implementations
        as it provides rich default functionality.
    """

    def __init__(self, data: np.ndarray, sampling_rate: float, name: str = "Undefined"):
        """Initialize Feature with data array, sampling rate, and metadata.

        Args:
            data (np.ndarray): Feature data array of any dimensionality. Common shapes:
                              - 1D: (time_frames,) for time series features
                              - 2D: (feature_bins, time_frames) for spectral features
                              - Square: (time_frames, time_frames) for similarity matrices
            sampling_rate (float): Feature sampling rate in Hz, determining temporal
                                  resolution and enabling time-based operations.
            name (str, optional): Human-readable feature name for visualization
                                and debugging. Default: "Undefined".
        """
        self._data = data
        self._sampling_rate = sampling_rate
        self._db_scaled = False  # To track if features are in dB scale
        self.feature_name = name

    def data(self) -> np.ndarray:
        """Access the raw feature data array.

        Returns:
            np.ndarray: The underlying feature data with original shape and values.
                       Modifications to this array will affect the feature instance.

        Example:
            >>> feature_data = spectrogram.data()
            >>> print(f"Shape: {feature_data.shape}")
            >>> # Direct access to spectral bins and time frames
            >>> spectral_slice = feature_data[:, 100]  # Frame 100 spectrum
        """
        return self._data

    def sampling_rate(self) -> float:
        """Get the feature sampling rate for temporal operations.

        Returns:
            float: Sampling rate in Hz, used for converting between frame indices
                  and absolute time values in seconds.

        Example:
            >>> sr = chromagram.sampling_rate()
            >>> frame_time = frame_index / sr  # Convert frame to seconds
            >>> duration = chromagram.data().shape[-1] / sr  # Total duration
        """
        return self._sampling_rate

    def is_db_scaled(self) -> bool:
        """Check if feature values are in decibel (logarithmic) scale.

        Returns:
            bool: True if feature data represents logarithmic/dB values,
                 False if data represents linear amplitude values.

        Note:
            This flag affects visualization (colorbar formatting) and determines
            the appropriateness of certain mathematical operations.
        """
        return self._db_scaled

    def copy(self) -> Self:
        """Create a deep copy of the feature instance.

        Creates an independent copy with separate data arrays, enabling
        immutable-style operations where modifications don't affect the original.

        Returns:
            Self: Independent copy of the feature with identical data and metadata.

        Example:
            >>> original = SpectrogramBuilder().build(audio_signal)
            >>> modified = original.copy()
            >>> # Apply modifications to copy without affecting original
            >>> modified = modified.log_compress(gamma=100)
            >>> # original remains unchanged

        Note:
            Uses shallow copy of object structure with deep copy semantics
            through dictionary update, providing efficient copying while
            maintaining data independence.
        """
        obj_copy = object.__new__(self.__class__)
        obj_copy.__dict__.update(self.__dict__)
        return obj_copy


class PeakFinder(ABC):
    """Abstract interface for peak detection and structural boundary analysis.

    PeakFinder defines the contract for detecting significant peaks in audio features,
    enabling automatic identification of structural boundaries, musical events, and
    temporal landmarks. This interface is crucial for music structure analysis,
    beat tracking, and automatic segmentation applications.

    Core Functionality:
        - Standardized peak detection across different feature types
        - Configurable threshold-based detection algorithms
        - Temporal distance constraints to prevent redundant detections
        - Integration with SciPy signal processing for robust analysis

    Detection Philosophy:
        Peak detection operates on the principle that significant changes or
        events in audio content manifest as local maxima in feature representations.
        Different features reveal different types of musical structure:
        - Spectral features: Detect timbral changes and instrument entrances
        - Harmonic features: Identify chord progressions and key changes
        - Novelty curves: Locate structural boundaries and section transitions
        - Energy features: Find dynamic changes and rhythmic patterns

    Parameter Control:
        The interface provides two primary control mechanisms:
        - Threshold: Relative or absolute minimum peak height
        - Distance: Minimum separation to avoid detecting multiple peaks for single events

    Applications:
        - Music structure analysis and automatic segmentation
        - Beat tracking and tempo analysis preprocessing
        - Audio event detection and content indexing
        - Real-time audio analysis for live performance systems
        - Music information retrieval and automatic annotation

    Example Usage:
        >>> # Through feature implementation
        >>> novelty = ssm.compute_novelty_curve()
        >>> boundaries = novelty.find_peaks(threshold=0.3, distance_seconds=10)
        >>> peak_times = boundaries / novelty.sampling_rate()
        >>> print(f"Found {len(boundaries)} structural boundaries")

    Note:
        Implementations should use scipy.signal.find_peaks for robust detection
        and provide clear documentation of their specific detection semantics.
    """

    @abstractmethod
    def find_peaks(self, threshold: float, distance_seconds: int) -> np.ndarray:
        """Detect significant peaks in the feature for boundary or event detection.

        Identifies local maxima that exceed the specified threshold and satisfy
        minimum distance constraints, returning their frame indices for further
        analysis or visualization.

        Args:
            threshold (float): Relative height threshold for peak detection as fraction
                              of maximum value (0 to 1). Higher values detect only
                              prominent peaks, lower values include subtle changes.
            distance_seconds (int): Minimum distance between peaks in seconds to avoid
                                   detecting multiple peaks for the same event or
                                   structural boundary.

        Returns:
            np.ndarray: Array of peak frame indices in ascending order. Convert to
                       time values by dividing by sampling_rate().

        Example:
            >>> peaks = feature.find_peaks(threshold=0.2, distance_seconds=5)
            >>> peak_times = peaks / feature.sampling_rate()
            >>> print(f"Detected {len(peaks)} peaks at times: {peak_times}")

        Implementation Notes:
            - Use scipy.signal.find_peaks for robust, well-tested detection
            - Convert distance_seconds to sample distance using sampling rate
            - Apply threshold as fraction of feature maximum for consistent behavior
            - Handle edge cases (no peaks, constant signals) gracefully

        Raises:
            NotImplementedError: If called on abstract base class.
        """
        pass


class BaseFeature(Feature):
    """Concrete base class providing comprehensive functionality for audio features.

    BaseFeature extends the abstract Feature interface with rich implementations
    of common audio analysis operations including normalization, filtering,
    mathematical transformations, and sophisticated visualization capabilities.
    This class serves as the foundation for most spectral and temporal audio features.

    Key Capabilities:
        - Advanced normalization (L1, L2, infinity norm) with FMP library integration
        - Temporal downsampling for computational efficiency and multi-resolution analysis
        - Signal smoothing using configurable windowing functions
        - Logarithmic compression for perceptual scaling and dynamic range management
        - Comprehensive visualization with temporal annotations and peak marking
        - Mathematical utilities (positive value enforcement, dB scaling)

    Processing Operations:
        All operations follow immutable semantics, returning new feature instances:
        - normalize(): Statistical normalization using FMP library methods
        - downsample(): Temporal resolution reduction with anti-aliasing
        - smooth(): Low-pass filtering using configurable windows
        - log_compress(): Logarithmic scaling for perceptual analysis
        - ensure_positive(): Value range adjustment for subsequent operations

    Visualization System:
        Provides sophisticated plotting capabilities supporting:
        - 1D features: Line plots with peak marking and temporal annotations
        - 2D features: Heatmaps with proper axis labeling and colorbars
        - Temporal alignment: Coordinate with original audio signals
        - Annotation support: Structural boundaries and section labeling
        - Export options: Publication-quality figures with customizable styling

    Mathematical Foundation:
        Integrates with scientific Python ecosystem:
        - NumPy: Efficient array operations and mathematical functions
        - SciPy: Signal processing, filtering, and statistical operations
        - Matplotlib: Professional visualization and publication graphics
        - FMP Library: Music-specific algorithms and proven implementations

    Applications:
        - Spectral feature analysis (Spectrograms, MFCCs, Chromagrams)
        - Temporal feature processing (Tempograms, Energy curves)
        - Multi-resolution analysis through downsampling
        - Perceptual analysis through logarithmic scaling
        - Interactive analysis through rich visualization

    Example Usage:
        >>> # Create and process features
        >>> spectrogram = SpectrogramBuilder().build(audio_signal)
        >>> # Apply processing chain
        >>> processed = (spectrogram
        ...              .ensure_positive()
        ...              .log_compress(gamma=100)
        ...              .normalize(norm='2')
        ...              .downsample(factor=4)
        ...              .smooth(filter_length=5))
        >>> # Visualize results
        >>> processed.plot(time_annotations=structure_annotations)

    Design Philosophy:
        - Immutable operations preserve data integrity
        - Method chaining enables fluent processing pipelines
        - Rich defaults minimize configuration complexity
        - Integration with established scientific libraries
        - Consistent interfaces across all feature types
    """

    def __init__(self, data: np.ndarray, sampling_rate: float, name: str = "Undefined"):
        """Initialize BaseFeature with comprehensive feature data and metadata.

        Args:
            data (np.ndarray): Feature data array. Common shapes:
                              - 1D: (time_frames,) for temporal features
                              - 2D: (feature_bins, time_frames) for spectral features
            sampling_rate (float): Feature sampling rate in Hz.
            name (str, optional): Feature name for identification. Default: "Undefined".
        """
        super().__init__(data, sampling_rate, name)

    def normalize(self, norm: str = "2", threshold=0.001, v=None) -> Self:
        """Apply statistical normalization using FMP library methods.

        Normalizes feature sequences using established music analysis techniques
        from the Fundamentals of Music Processing (FMP) library. This operation
        is crucial for consistent analysis across different audio sources.

        Args:
            norm (str, optional): Normalization type. Options:
                                - "1": L1 norm (sum of absolute values)
                                - "2": L2 norm (Euclidean norm) [Default]
                                - "max" or "inf": Maximum/infinity norm
                                Default: "2".
            threshold (float, optional): Minimum norm value to prevent division
                                       by zero for near-silent frames. Default: 0.001.
            v (Optional[float]): Target norm value. If None, normalizes to unit norm.

        Returns:
            Self: New BaseFeature instance with normalized data.

        Example:
            >>> # L2 normalization (most common)
            >>> normalized = chromagram.normalize(norm="2")
            >>> # L1 normalization for probability-like interpretation
            >>> l1_normalized = chromagram.normalize(norm="1")
            >>> # Max normalization for amplitude features
            >>> max_normalized = spectrogram.normalize(norm="max")

        Note:
            Uses src.libfmp.c3.normalize_feature_sequence for proven implementations.
            Normalization is applied per time frame, preserving temporal structure.
        """
        F_normalized = src.libfmp.c3.normalize_feature_sequence(
            self.data(), norm=norm, threshold=threshold, v=v
        )
        new_features = self.copy()
        new_features._data = F_normalized
        return new_features

    def downsample(self, factor: int = 10) -> Self:
        """Reduce temporal resolution by downsampling with specified factor.

        Performs temporal downsampling to reduce computational complexity and
        enable multi-resolution analysis. The operation preserves spectral
        content while reducing the number of time frames.

        Args:
            factor (int, optional): Downsampling factor. Must be >= 1. Higher values
                                   provide more aggressive downsampling. Default: 10.

        Returns:
            Self: New BaseFeature instance with reduced temporal resolution.
                 Sampling rate is adjusted to factor/original_rate.

        Example:
            >>> # Reduce by factor of 4 for computational efficiency
            >>> downsampled = spectrogram.downsample(factor=4)
            >>> print(f"Original: {spectrogram.data().shape[1]} frames")
            >>> print(f"Downsampled: {downsampled.data().shape[1]} frames")
            >>> print(f"New sampling rate: {downsampled.sampling_rate()} Hz")

        Raises:
            ValueError: If factor is less than 1.

        Note:
            This is a simple decimation operation. For audio signals requiring
            anti-aliasing, consider applying smoothing before downsampling.
        """
        if factor < 1:
            raise ValueError(
                "/!\\ Downsampling factor must be greater than 1. No downsampling applied."
            )

        new_features = self.copy()
        new_features._data = self.data()[:, ::factor]
        new_features._sampling_rate = self.sampling_rate() / factor
        return new_features

    def smooth(self, filter_length: int = 21, window_type="boxcar") -> Self:
        """Apply temporal smoothing using configurable windowing functions.

        Performs low-pass filtering along the time axis to reduce noise and
        emphasize temporal trends. This operation is particularly useful for
        novelty curves, energy features, and other time series analysis.

        Args:
            filter_length (int, optional): Length of smoothing window in samples.
                                          Must be positive and odd for symmetric filtering.
                                          Default: 21.
            window_type (str, optional): Type of smoothing window. Options include:
                                        - "boxcar": Uniform averaging (moving average)
                                        - "hamming": Hamming window (smooth tapering)
                                        - "hann": Hann window (cosine tapering)
                                        - "gaussian": Gaussian window (smooth, wide kernel)
                                        Default: "boxcar".

        Returns:
            Self: New BaseFeature instance with temporally smoothed data.

        Example:
            >>> # Light smoothing with uniform kernel
            >>> smooth_light = novelty.smooth(filter_length=5, window_type="boxcar")
            >>> # Heavy smoothing with Gaussian kernel
            >>> smooth_heavy = novelty.smooth(filter_length=41, window_type="gaussian")

        Raises:
            ValueError: If filter_length is not a positive odd integer.

        Note:
            Uses scipy.signal.convolve with 'same' mode to preserve array length.
            The kernel is normalized to preserve feature magnitude.
        """
        if filter_length < 1 or filter_length % 2 == 0:
            raise ValueError("Filter length must be a positive odd integer")

        filt_kernel = np.expand_dims(
            signal.get_window(window_type, filter_length), axis=0
        )

        new_features = self.copy()
        new_features._data = signal.convolve(
            self.data(), filt_kernel, mode="same"
        ) / float(filter_length)
        return new_features

    def ensure_positive(self, epsilon: float = 1e-8) -> "BaseFeature":
        """Ensure all feature values are positive for subsequent logarithmic operations.

        Adjusts feature values to be strictly positive, which is essential before
        applying logarithmic compression or other operations that require positive inputs.
        This preprocessing step is commonly needed for spectral features.

        Args:
            epsilon (float, optional): Small positive value added after shifting to
                                     avoid exact zeros, which can cause numerical
                                     issues in logarithmic operations. Default: 1e-8.

        Returns:
            BaseFeature: New feature instance with all positive values.

        Example:
            >>> # Prepare spectrogram for log compression
            >>> positive_spec = spectrogram.ensure_positive(epsilon=1e-10)
            >>> log_spec = positive_spec.log_compress(gamma=100)
            >>> # Chain operations fluently
            >>> processed = (spectrogram
            ...              .ensure_positive()
            ...              .log_compress(gamma=50)
            ...              .normalize())

        Note:
            Only applies offset if minimum value is <= 0. If all values are already
            positive, returns copy without modification. This preserves the original
            dynamic range when possible.
        """
        new_features = self.copy()
        data = self.data()

        if data.min() <= 0:
            # Shift all values to be positive
            offset = abs(data.min()) + epsilon
            new_features._data = data + offset

        return new_features

    def log_compress(self, gamma: float = 1) -> Self:
        """Apply logarithmic compression for perceptual scaling and dynamic range control.

        Transforms feature values using log(1 + gamma * x) to compress large values
        while preserving small values. This operation mimics human auditory perception
        and is essential for spectral features with wide dynamic ranges.

        Args:
            gamma (float, optional): Compression factor controlling the strength of
                                   logarithmic scaling. Higher values provide stronger
                                   compression. Must be > 0. Default: 1.

        Returns:
            Self: New BaseFeature instance with logarithmically compressed values.

        Example:
            >>> # Light compression for visualization
            >>> light_compressed = spectrogram.log_compress(gamma=10)
            >>> # Strong compression for analysis
            >>> strong_compressed = spectrogram.log_compress(gamma=1000)
            >>> # Prepare magnitude spectrogram for dB conversion
            >>> db_ready = magnitude_spec.ensure_positive().log_compress(gamma=1)

        Raises:
            ValueError: If any feature values are negative (use ensure_positive first).

        Note:
            Gamma parameter scales the input before logarithm: log(1 + gamma * x).
            Use ensure_positive() first if feature contains negative or zero values.
        """
        if gamma <= 0:
            return self

        new_features = self.copy()

        if self.data().min() < 0:
            raise ValueError(
                "Log compression requires all feature values to be non-negative."
            )

        new_features._data = np.log(1 + gamma * self.data())
        return new_features

    def _plot(
        self,
        features: np.ndarray,
        feature_sr: float,
        features_name: str,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_signal: Optional[Signal] = None,
        peaks: Optional[np.ndarray] = None,
        figsize=(12, 8),
    ):
        """Comprehensive visualization method for audio features with rich annotation support.

        Provides sophisticated plotting capabilities for both 1D and 2D audio features,
        with support for temporal annotations, peak marking, signal alignment, and
        publication-quality styling. This method serves as the foundation for all
        feature visualization throughout the audio analysis framework.

        Visualization Capabilities:
            - 1D Features: Line plots with peak detection and trend visualization
            - 2D Features: Heatmaps with proper axis labeling and colorbars
            - Signal Alignment: Dual-panel plots with original audio context
            - Temporal Annotations: Structural boundaries and section labeling
            - Peak Marking: Vertical lines for detected events and boundaries
            - Multiple Coordinate Systems: Time-based or frame-based x-axes

        Args:
            features (np.ndarray): Feature data to visualize. Shape determines plot type:
                                  - 1D or (1, N): Line plot
                                  - 2D (M, N): Heatmap with M frequency/feature bins
            feature_sr (float): Feature sampling rate for time axis conversion.
            features_name (str): Feature name for plot titles and labels.
            x_axis_type (str, optional): X-axis coordinate system:
                                        - "time": Display in seconds
                                        - "frame": Display in frame indices
                                        Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations for
                                                        structural visualization. Formats:
                                                        - [[start, end, label], ...] for regions
                                                        - [time1, time2, ...] for boundaries
                                                        Always in seconds regardless of x_axis_type.
            original_signal (Optional[Signal], optional): Reference audio signal for
                                                         dual-panel visualization showing
                                                         feature alignment with source audio.
            peaks (Optional[np.ndarray], optional): Peak frame indices to mark as
                                                   vertical dashed lines for boundary
                                                   or event visualization.
            figsize (tuple, optional): Figure size in inches (width, height).
                                     Default: (12, 8).

        Visualization Features:
            - Automatic plot type detection based on feature dimensionality
            - Professional colormaps optimized for audio analysis (FMP compressed gray)
            - Dynamic axis labeling with intelligent tick spacing for readability
            - Colorbar formatting with dB scale support for logarithmic features
            - Alpha-blended annotation overlays preserving feature visibility
            - Peak marking with legend support and customizable styling

        Layout Options:
            - Single Panel: Feature-only visualization for focused analysis
            - Dual Panel: Feature + original signal for temporal alignment context
            - Annotation Areas: Dedicated regions for structural boundary labeling

        Example Usage:
            >>> # Basic feature visualization
            >>> spectrogram._plot(
            ...     spectrogram.data(),
            ...     spectrogram.sampling_rate(),
            ...     "Magnitude Spectrogram"
            ... )
            >>>
            >>> # Advanced visualization with all features
            >>> novelty._plot(
            ...     novelty.data(),
            ...     novelty.sampling_rate(),
            ...     "Structural Novelty",
            ...     peaks=detected_boundaries,
            ...     time_annotations=[[0, 30, 'Verse'], [30, 60, 'Chorus']],
            ...     original_signal=audio_signal,
            ...     x_axis_type="time"
            ... )

        Technical Details:
            - Uses matplotlib for rendering with publication-quality defaults
            - Integrates with FMP library colormaps for audio-specific visualization
            - Supports both linear and logarithmic (dB) feature representations
            - Handles edge cases (empty peaks, missing annotations) gracefully
            - Memory-efficient rendering for large feature matrices

        Note:
            This is an internal method used by concrete feature implementations.
            End users should call the plot() method on specific feature instances.
        """
        import matplotlib.pyplot as plt

        if x_axis_type not in ["time", "frame"]:
            raise ValueError("x_axis_type must be 'time' or 'frame'")

        is_1d = features.ndim == 1 or features.shape[0] == 1
        n_frames = features.shape[0] if features.ndim == 1 else features.shape[1]
        peaks_x = None
        if peaks is not None:
            peaks_x = peaks if x_axis_type == "frame" else peaks / feature_sr

        MAX_BINS_FOR_TICKS = 5

        # Helper functions
        def get_x_coords():
            if x_axis_type == "time":
                return np.arange(n_frames) / feature_sr
            return np.arange(n_frames)

        def get_extent(max_coord):
            if is_1d:
                return None  # Not used for line plots
            return (0, max_coord, 0, features.shape[0])

        def convert_annotation_coords(start_time, end_time):
            if x_axis_type == "time":
                return start_time, end_time
            return start_time * feature_sr, end_time * feature_sr

        def plot_annotations(ax, y_pos=None, text_color="black"):
            if time_annotations is None:
                return
            for i, ann in enumerate(time_annotations):
                if len(ann) >= 2:
                    start_coord, end_coord = convert_annotation_coords(ann[0], ann[1])
                    ax.axvspan(start_coord, end_coord, alpha=0.2, color=get_color(i))
                    if y_pos is not None:
                        ax.text(
                            (start_coord + end_coord) / 2,
                            y_pos,
                            "",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=text_color,
                        )

        # Setup coordinates and labels
        x_coords = get_x_coords()
        max_coord = x_coords[-1] if x_axis_type == "time" else n_frames - 1
        xlabel = "Time (s)" if x_axis_type == "time" else "Frames"

        # Get colormap for 2D features
        try:
            cmap = src.libfmp.b.compressed_gray_cmap(alpha=-10)
        except (ImportError, AttributeError):
            cmap = "gray_r"

        # Plot with or without original signal
        if original_signal is not None and original_signal.sample_rate is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 2])

            # Plot original signal
            if x_axis_type == "time":
                audio_time = (
                    np.arange(len(original_signal.samples))
                    / original_signal.sample_rate
                )
                ax1.plot(
                    audio_time, original_signal.samples, color="gray", linewidth=0.8
                )
                ax1.set_xlim(0, max_coord)
            else:
                ax1.plot(original_signal.samples, color="gray", linewidth=0.8)
                ax1.set_xlim(0, len(original_signal.samples) - 1)

            ax1.set_xlabel(xlabel)
            ax1.set_ylabel("Amplitude")
            ax1.set_title("Original Audio Signal (aligned with features)")
            ax1.grid(True, alpha=0.3)

            # Plot features (1D as line, 2D as heatmap)
            if is_1d:
                f_plot = features.flatten()
                ax2.plot(x_coords, f_plot, color="black")

                # add peaks if provided (unit is assumed to be the same as x_axis_type so no conversion needed)
                # add the peaks as red vertical lines
                if peaks_x is not None and len(peaks_x) > 0:
                    for i, peak in enumerate(peaks_x):
                        ax2.axvline(
                            peak,
                            color="red",
                            linestyle="--",
                            label="Predictions" if i == 0 else None,
                            linewidth=1,
                        )
                    ax2.legend()

                ax2.set_ylabel(f"{self.feature_name} Intensity")
                plot_annotations(ax2, y_pos=np.mean(f_plot))
            else:
                im = ax2.imshow(
                    features,
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    cmap=cmap,
                    extent=get_extent(max_coord),
                )

                # Set up proper y-axis for bins
                n_bins = features.shape[0]
                print("N bins:", n_bins)
                if n_bins > MAX_BINS_FOR_TICKS:  # Too many bins, use dynamic spacing
                    step_size = max(2, n_bins // 10)  # Aim for ~10 ticks
                    tick_positions = np.arange(
                        0.5, n_bins, step_size
                    )  # Dynamic bin centers
                    tick_labels = np.arange(0, n_bins, step_size)  # Dynamic bin numbers
                else:
                    tick_positions = np.arange(0.5, n_bins, 1)  # All bin centers
                    tick_labels = np.arange(n_bins)  # All bin numbers
                ax2.set_yticks(tick_positions)
                ax2.set_yticklabels(tick_labels)
                ax2.set_ylabel(f"{self.feature_name} Bins")

                plt.colorbar(
                    im,
                    ax=ax2,
                    orientation="horizontal",
                    pad=0.15,
                    format="%+2.0f dB" if self.is_db_scaled() else None,
                    label="Intensity",
                )
                plot_annotations(ax2, y_pos=features.shape[0] * 0.9, text_color="white")

            ax2.set_xlabel(xlabel)
            ax2.set_xlim(0, max_coord)
            ax2.set_title(f"{features_name} Feature")

            # Plot annotations on signal
            if original_signal and time_annotations:
                signal_max = np.max(np.abs(original_signal.samples))
                plot_annotations(ax1, y_pos=signal_max * 0.8)

        else:
            # Single plot for features only
            fig, ax = plt.subplots(figsize=figsize)

            # Get feature name for labels
            feature_name = self.__class__.__name__
            feature_name = "".join(
                [" " + c if c.isupper() else c for c in feature_name]
            ).strip()

            if is_1d:
                ax.plot(x_coords, features, linewidth=1.5)
                ax.set_ylabel(f"{feature_name} Intensity")
                plot_annotations(ax, y_pos=np.mean(features))
            else:
                im = ax.imshow(
                    features,
                    aspect="auto",
                    origin="lower",
                    cmap=cmap,
                    extent=get_extent(max_coord),
                )

                # Set up proper y-axis for bins
                n_bins = features.shape[0]
                if n_bins > MAX_BINS_FOR_TICKS:  # Too many bins, use dynamic spacing
                    step_size = max(2, n_bins // 10)  # Aim for ~10 ticks
                    tick_positions = np.arange(
                        0.5, n_bins, step_size
                    )  # Dynamic bin centers
                    tick_labels = np.arange(0, n_bins, step_size)  # Dynamic bin numbers
                else:
                    tick_positions = np.arange(0.5, n_bins, 1)  # All bin centers
                    tick_labels = np.arange(n_bins)  # All bin numbers
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
                ax.set_ylabel(f"{feature_name} Bins")

                plt.colorbar(
                    im,
                    format="%+2.0f dB" if self.is_db_scaled() else None,
                    label="Intensity",
                )
                plot_annotations(ax, y_pos=features.shape[0] * 0.9, text_color="white")

            ax.set_xlabel(xlabel)
            ax.set_title(f"{features_name} Features")

        plt.tight_layout()
        plt.show()


class SimilarityMatrix(Feature):
    """Abstract base class for similarity matrix representations in structural music analysis.

    SimilarityMatrix defines the interface for matrix-based audio analysis techniques
    that capture relationships between different time points or feature vectors.
    These matrices are fundamental to music structure analysis, pattern detection,
    and automatic segmentation applications.

    Core Concepts:
        Similarity matrices encode pairwise relationships where entry (i,j) represents
        the similarity or dissimilarity between feature vectors at times i and j.
        Different matrix types reveal different aspects of musical structure:
        - Self-Similarity Matrix (SSM): Repetitions and musical patterns
        - Time-Lag Matrix (TLM): Temporal evolution and dynamic changes
        - Cross-Similarity Matrix: Relationships between different feature types

    Matrix Properties:
        - Square matrices: (time_frames, time_frames) for temporal analysis
        - Symmetric matrices: SSMs exhibit symmetry around main diagonal
        - Block structure: Homogeneous sections appear as uniform blocks
        - Diagonal patterns: Repetitive content creates parallel diagonal streaks

    Structural Analysis:
        Similarity matrices enable detection of:
        - Musical repetitions and recurring patterns (verse/chorus structure)
        - Section boundaries through novelty curve analysis
        - Temporal relationships and harmonic progressions
        - Multi-level structural hierarchy (phrases, sections, movements)

    Novelty Curve Interface:
        All similarity matrices must implement novelty curve computation,
        which transforms 2D similarity patterns into 1D boundary strength
        functions for automatic segmentation and peak detection.

    Applications:
        - Music structure analysis and automatic form identification
        - Cover song identification through cross-similarity analysis
        - Pattern discovery and motif detection in musical content
        - Thumbnail extraction and music summarization
        - Real-time structural analysis for live performance systems

    Example Implementation Pattern:
        >>> class CustomSimilarityMatrix(SimilarityMatrix):
        ...     def compute_novelty_curve(self, **params):
        ...         # Implement specific novelty detection algorithm
        ...         novelty_data = analyze_matrix_structure(self.data())
        ...         return NoveltyCurve(novelty_data, self.sampling_rate())
        ...
        ...     def plot_custom(self):
        ...         # Custom visualization for this matrix type
        ...         self._plot("Custom Matrix", annotations=self.structure)

    Integration:
        Works seamlessly with:
        - Feature extraction pipelines for matrix construction
        - Peak detection algorithms through novelty curves
        - Visualization systems for structural analysis
        - Music information retrieval applications

    Note:
        This is an abstract class requiring implementation of compute_novelty_curve().
        Use SelfSimilarityMatrix or TimeLagMatrix for concrete implementations.
    """

    def __init__(self, data: np.ndarray, sampling_rate: float, name: str = "Undefined"):
        """Initialize SimilarityMatrix with matrix data and metadata.

        Args:
            data (np.ndarray): Similarity matrix data, typically square with shape
                              (time_frames, time_frames) for temporal analysis.
            sampling_rate (float): Feature sampling rate in Hz for temporal operations.
            name (str, optional): Matrix type name for identification. Default: "Undefined".
        """
        super().__init__(data, sampling_rate, name)

    @abstractmethod
    def compute_novelty_curve(self, *args, **kwargs) -> "NoveltyCurve":
        """Compute novelty curve from similarity matrix for boundary detection.

        Transforms the 2D similarity matrix into a 1D novelty function that
        indicates the strength of structural boundaries or significant changes
        over time. Different matrix types use different analysis methods.

        Args:
            *args: Variable arguments specific to the novelty computation method.
            **kwargs: Keyword arguments for algorithm configuration and tuning.

        Returns:
            NoveltyCurve: 1D novelty function indicating boundary strength over time.
                         Higher values correspond to more significant structural changes.

        Implementation Notes:
            - Self-Similarity Matrices: Use checkerboard kernel analysis
            - Time-Lag Matrices: Use gradient-based temporal change detection
            - Custom matrices: Implement domain-specific novelty detection

        Example:
            >>> ssm = SelfSimilarityMatrixBuilder().build(chromagram)
            >>> novelty = ssm.compute_novelty_curve(kernel_size=16, variance=0.5)
            >>> boundaries = novelty.find_peaks(threshold=0.3, distance_seconds=10)

        Raises:
            NotImplementedError: If called on abstract base class.
        """
        pass

    def _plot(
        self,
        ssm_name: str,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_base_feature: Optional[BaseFeature] = None,
    ):
        """Sophisticated visualization system for similarity matrices with structural analysis.

        Provides comprehensive plotting capabilities specifically designed for similarity
        matrix visualization, with support for feature context, structural annotations,
        and publication-quality layouts. The visualization system mimics the FMP library
        layout standards for consistent, professional results.

        Visualization Modes:
            1. Matrix-Only: Simple square similarity matrix with optimized aspect ratio
            2. Feature Context: Dual-panel layout showing features above similarity matrix
            3. Annotated Analysis: Full 3x3 layout with dedicated annotation areas

        Key Features:
            - FMP-compatible layout system for professional structural analysis
            - Compressed gray colormap optimized for similarity visualization
            - Dynamic axis labeling with intelligent bin spacing
            - Structural annotation support with color-coded regions
            - Feature context integration for interpretation support
            - Multiple coordinate systems (time vs. frame based)

        Args:
            ssm_name (str): Matrix type name for titles and identification.
                          Examples: "Self-Similarity Matrix", "Time-Lag Matrix"
            x_axis_type (str, optional): Coordinate system for axes:
                                        - "time": Display in seconds for intuitive reading
                                        - "frame": Display in frame indices for technical analysis
                                        Default: "time".
            time_annotations (Optional[list], optional): Structural annotations for
                                                        boundary visualization. Format:
                                                        [[start_time, end_time, label], ...]
                                                        Times always in seconds regardless of x_axis_type.
            original_base_feature (Optional[BaseFeature], optional): Source feature used
                                                                    to compute the similarity matrix.
                                                                    Enables dual-panel visualization
                                                                    showing feature context above matrix.

        Layout Architecture:
            - Without Features: Single square matrix with equal aspect ratio
            - With Features: Multi-panel layout with features above matrix
            - With Annotations: Extended grid with dedicated annotation areas
            - Colorbar Integration: Professional colorbar placement and formatting

        Annotation System:
            When time_annotations are provided:
            - Bottom Panel: Horizontal structural segments with color coding
            - Left Panel: Vertical structural segments mirroring horizontal
            - Main Matrix: Clean visualization without cluttered labels
            - Color Coordination: Consistent colors across all annotation areas

        Technical Implementation:
            - Uses matplotlib gridspec for precise layout control
            - Integrates FMP library compressed_gray_cmap for optimal contrast
            - Supports both dB and linear intensity scales with appropriate formatting
            - Handles coordinate transformations between time and frame representations
            - Memory-efficient rendering for large similarity matrices

        Example Usage:
            >>> # Basic matrix visualization
            >>> ssm._plot("Self-Similarity Matrix")
            >>>
            >>> # Advanced structural analysis visualization
            >>> ssm._plot(
            ...     "Chromagram Self-Similarity",
            ...     x_axis_type="time",
            ...     time_annotations=[
            ...         [0, 30, "Verse 1"],
            ...         [30, 60, "Chorus 1"],
            ...         [60, 90, "Verse 2"]
            ...     ],
            ...     original_base_feature=chromagram
            ... )

        Visualization Standards:
            - Professional aspect ratios for structural analysis clarity
            - Consistent color schemes across all matrix visualizations
            - Publication-ready layouts with proper spacing and labeling
            - Integration with music analysis workflow standards

        Note:
            The layout system is designed to match src.libfmp.c4.plot_feature_ssm
            for consistency with established music analysis visualization standards.
            Large matrices are handled efficiently through optimized rendering.
        """
        import matplotlib.pyplot as plt

        if x_axis_type not in ["time", "frame"]:
            raise ValueError("x_axis_type must be 'time' or 'frame'")

        n_frames = self.data().shape[0]
        ssm_time = np.arange(n_frames) / self.sampling_rate()
        MAX_BINS_FOR_TICKS = 5

        # Use src.libfmp compressed gray colormap for SSM
        try:
            cmap_ssm = src.libfmp.b.compressed_gray_cmap(alpha=-10)
            # cmap_ssm = 'gray_r'
        except (ImportError, AttributeError):
            cmap_ssm = "gray_r"

        # Case 1: with features - mimic src.libfmp layout exactly ----------------------
        if original_base_feature is not None:
            max_time = ssm_time[-1]

            fig_width = 8
            fig_height = fig_width * 1.25  # Adjust height for better aspect ratio

            # Adjust layout based on whether annotations are provided
            if time_annotations is not None and len(time_annotations) > 0:
                # Full 3x3 layout with annotation areas
                fig, ax = plt.subplots(
                    3,
                    3,
                    gridspec_kw={
                        "width_ratios": [0.1, 1, 0.05],
                        "wspace": 0.2,
                        "height_ratios": [0.3, 1, 0.1],
                    },
                    figsize=(fig_width, fig_height),
                )

            else:
                # Simplified 2x3 layout without annotation areas
                fig, ax = plt.subplots(
                    2,
                    3,
                    gridspec_kw={
                        "width_ratios": [0.1, 1, 0.05],
                        "wspace": 0.2,
                        "height_ratios": [0.3, 1],
                    },
                    figsize=(fig_width, fig_height * 0.9),
                )

            # Features plot - top center position [0, 1] with colorbar [0, 2]
            if x_axis_type == "time":
                extent_feat = (0, max_time, 0, original_base_feature.data().shape[0])
                extent_ssm = (0, max_time, 0, max_time)
                xlabel, ylabel = "Time (s)", "Time (s)"
            else:
                extent_feat = (
                    0,
                    n_frames - 1,
                    0,
                    original_base_feature.data().shape[0],
                )
                extent_ssm = (0, n_frames - 1, 0, n_frames - 1)
                xlabel, ylabel = "Frames", "Frames"

            # Plot features using src.libfmp.b.plot_matrix style
            im_feat = ax[0, 1].imshow(
                original_base_feature.data(),
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=cmap_ssm,
                extent=extent_feat,
            )

            # Set up proper y-axis for bins
            n_bins = original_base_feature.data().shape[0]
            if n_bins > MAX_BINS_FOR_TICKS:  # Too many bins, use dynamic spacing
                step_size = max(2, n_bins // 10)  # Aim for ~10 ticks
                tick_positions = np.arange(
                    0.5, n_bins, step_size
                )  # Dynamic bin centers
                tick_labels = np.arange(0, n_bins, step_size)  # Dynamic bin numbers
            else:
                tick_positions = np.arange(0.5, n_bins, 1)  # All bin centers
                tick_labels = np.arange(n_bins)  # All bin numbers
            ax[0, 1].set_yticks(tick_positions)
            ax[0, 1].set_yticklabels(tick_labels)
            ax[0, 1].set_ylabel(f"{original_base_feature.feature_name} Bins")
            ax[0, 1].set_title(f"Features used for {ssm_name}")
            ax[0, 1].set_xlabel("")  # No xlabel for top plot

            # Features colorbar
            plt.colorbar(
                im_feat,
                cax=ax[0, 2],
                label="Intensity",
            )

            # Turn off corner axes like src.libfmp
            ax[0, 0].axis("off")

            # SSM plot - center position [1, 1] with colorbar [1, 2]
            im_ssm = ax[1, 1].imshow(
                self.data(),
                aspect="auto",
                origin="lower",
                cmap=cmap_ssm,
                extent=extent_ssm,
                interpolation="nearest",
            )

            # Handle SSM axis labels based on annotation presence
            if time_annotations is not None and len(time_annotations) > 0:
                # With annotations: labels go on annotation areas, main plot has no labels
                ax[1, 1].set_xlabel("")  # No labels on main SSM plot
                ax[1, 1].set_ylabel("")
                ax[1, 1].set_xticks([])  # Remove ticks like src.libfmp
                ax[1, 1].set_yticks([])
            else:
                # Without annotations: labels go directly on main SSM plot
                ax[1, 1].set_xlabel(xlabel)
                ax[1, 1].set_ylabel(ylabel)
                ax[1, 1].set_title(f"{ssm_name}")

            # SSM colorbar
            plt.colorbar(im_ssm, cax=ax[1, 2], label="Intensity")

            # Turn off left axis for SSM row
            ax[1, 0].axis("off")

            # Handle annotation areas only if annotations are provided
            if time_annotations is not None and len(time_annotations) > 0:
                # Bottom annotation area [2, 1] - populate with actual annotations
                ax[2, 1].set_xlim(extent_ssm[0], extent_ssm[1])
                ax[2, 1].set_ylim(-0.5, 0.5)  # Small height for annotation area
                ax[2, 1].set_xlabel(xlabel)
                ax[2, 1].set_ylabel("")
                ax[2, 1].tick_params(left=False, labelleft=False)

                # Left annotation area [1, 0] - populate with actual annotations
                ax[1, 0].set_ylim(extent_ssm[2], extent_ssm[3])
                ax[1, 0].set_xlim(-0.5, 0.5)  # Small width for annotation area
                ax[1, 0].set_ylabel(ylabel)
                ax[1, 0].set_xlabel("")
                ax[1, 0].tick_params(bottom=False, labelbottom=False)
                ax[1, 0].axis("on")  # Turn back on for annotations

                # Turn off remaining corner axes
                ax[2, 2].axis("off")
                ax[2, 0].axis("off")

                # Convert time annotations to frame indices if x_axis_type is 'frame'
                if x_axis_type == "frame":
                    time_annotations = [
                        (start * self.sampling_rate(), end * self.sampling_rate(), _)
                        for start, end, _ in time_annotations
                    ]

                # Manual annotation plotting with preserved axis settings
                for i, ann in enumerate(time_annotations):
                    if len(ann) >= 2:  # Expecting [start, end, label] format
                        start, end = ann[0], ann[1]

                        # Use coordinates directly as they are (frames or time based on x_axis_type)
                        start_coord = start
                        end_coord = end

                        # Bottom horizontal segments
                        ax[2, 1].barh(
                            0,
                            end_coord - start_coord,
                            left=start_coord,
                            height=1,
                            alpha=0.2,
                            color=get_color(i),
                        )  # Cycle through colors
                        ax[2, 1].text(
                            (start_coord + end_coord) / 2,
                            0,
                            "",
                            ha="center",
                            va="center",
                            fontsize=8,
                            rotation=(
                                0
                                if (end_coord - start_coord)
                                > (extent_ssm[1] - extent_ssm[0]) * 0.1
                                else 90
                            ),
                        )

                        # Left vertical segments
                        ax[1, 0].bar(
                            0,
                            end_coord - start_coord,
                            bottom=start_coord,
                            width=1,
                            alpha=0.2,
                            color=get_color(i),
                        )  # Same color as horizontal
                        ax[1, 0].text(
                            0,
                            (start_coord + end_coord) / 2,
                            "",
                            ha="center",
                            va="center",
                            fontsize=8,
                            rotation=90,
                        )

                # Ensure axis labels and ticks are properly set and visible
                # Bottom annotation area - preserve xlabel and show ticks
                ax[2, 1].set_xlabel(xlabel)
                ax[2, 1].set_ylabel("")
                ax[2, 1].tick_params(
                    bottom=True, labelbottom=True, left=False, labelleft=False
                )

                # Left annotation area - preserve ylabel and show ticks
                ax[1, 0].set_ylabel(ylabel)
                ax[1, 0].set_xlabel("")
                ax[1, 0].tick_params(
                    left=True, labelleft=True, bottom=False, labelbottom=False
                )

        # Case 2: SSM only - simple square layout ----------------------------------
        else:
            fig, ax_ssm = plt.subplots(figsize=(8, 8))

            if x_axis_type == "time":
                extent = (0, ssm_time[-1], 0, ssm_time[-1])
                xlabel = ylabel = "Time (s)"
            else:
                extent = (0, n_frames - 1, 0, n_frames - 1)
                xlabel = ylabel = "Frames"

            im_ssm = ax_ssm.imshow(
                self.data(),
                aspect="equal",
                origin="lower",
                cmap=cmap_ssm,
                extent=extent,
                interpolation="nearest",
            )
            ax_ssm.set_xlabel(xlabel)
            ax_ssm.set_ylabel(ylabel)
            ax_ssm.set_title(f"{ssm_name}")

            # Add colorbar
            plt.colorbar(
                im_ssm,
                format="%+2.0f dB" if self.is_db_scaled() else None,
                label="Intensity",
            )

        plt.tight_layout()
        plt.show()
        plt.tight_layout()
        plt.show()
