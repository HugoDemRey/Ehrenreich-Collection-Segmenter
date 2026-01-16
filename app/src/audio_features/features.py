"""Audio Feature Classes - Comprehensive feature representations for music analysis.

This module provides a complete suite of audio feature classes that encapsulate various
types of audio analysis data, from basic spectral representations to advanced structural
analysis matrices. These classes serve as data containers with rich functionality for
visualization, analysis, and processing of audio features in music information retrieval.

Key Features:
    - Object-oriented feature representations with consistent interfaces
    - Rich visualization capabilities with matplotlib integration
    - Peak detection functionality for structural analysis
    - Feature transformation and processing methods
    - Normalization and scaling operations
    - Multi-format plotting with annotation support
    - Integration with signal processing workflows
    - Memory-efficient data storage and manipulation

Feature Categories:
    1. Spectral Features: Time-frequency domain representations
       - Spectrogram: Power spectral density over time
       - Chromagram: 12-dimensional pitch class profiles
       - MFCC: Mel-Frequency Cepstral Coefficients for timbre
       - Tempogram: Tempo and rhythm analysis features

    2. Source Separation Features: Component-based analysis
       - HRPS: Harmonic-Residual-Percussive energy separation

    3. Activity Detection Features: Content analysis
       - SilenceCurve: Silence and activity detection over time

    4. Structural Analysis Features: Pattern and similarity analysis
       - SelfSimilarityMatrix: Pairwise similarity matrices
       - TimeLagMatrix: Time-lag representations for repetition analysis
       - NoveltyCurve: Novelty and change detection features

Core Functionality:
    - Inheritance from BaseFeature, Feature, and SimilarityMatrix interfaces
    - PeakFinder mixin for structural boundary detection
    - Comprehensive plotting with time/frame axis options
    - Feature combination and normalization methods
    - Integration with audio signal objects for context
    - Annotation support for ground truth visualization

Common Use Cases:
    - Music structure analysis and segmentation
    - Audio content classification and retrieval
    - Real-time audio feature visualization
    - Batch processing of audio collections
    - Research in music information retrieval
    - Audio similarity and recommendation systems
    - Automatic music transcription preprocessing
    - Performance analysis and comparison

Technical Dependencies:
    - librosa: Advanced audio analysis and processing
    - scipy: Signal processing and peak detection
    - numpy: Numerical operations and array manipulation
    - matplotlib: Rich visualization and plotting capabilities
    - Custom interfaces: BaseFeature, Feature, SimilarityMatrix, PeakFinder
    - Custom utilities: Color management and signal integration

Design Patterns:
    - Template Method: Common visualization and processing patterns
    - Strategy Pattern: Multiple algorithm implementations within features
    - Mixin Pattern: PeakFinder functionality across feature types
    - Factory Pattern: Consistent feature creation interfaces

Author: Hugo Demule
Date: January 2026
"""

from __future__ import annotations

from typing import Optional, Union, override

import librosa
import numpy as np
import src.libfmp
import src.libfmp.b
import src.libfmp.c4
from scipy.ndimage import gaussian_filter1d
from src.audio.signal import Signal
from src.interfaces.feature import BaseFeature, Feature, PeakFinder, SimilarityMatrix
from src.utils.colors import get_color


class Spectrogram(BaseFeature):
    """Time-frequency representation of audio signals using Short-Time Fourier Transform.

    Spectrogram provides a detailed view of how the frequency content of an audio signal
    varies over time, forming the foundation for many audio analysis tasks. It represents
    the power spectral density computed via STFT with configurable time and frequency
    resolution parameters.

    Key Features:
        - Power spectrogram representation (magnitude squared STFT)
        - Configurable time-frequency resolution trade-offs
        - Logarithmic (dB) scaling support for enhanced visualization
        - Rich plotting capabilities with annotation support
        - Integration with audio signals for temporal alignment
        - Memory-efficient storage of spectral data

    Mathematical Foundation:
        The spectrogram S[m,k] represents the power spectral density at time frame m
        and frequency bin k, computed as |STFT[m,k]|², where STFT is the Short-Time
        Fourier Transform of the input signal.

    Data Format:
        - Shape: (frequency_bins, time_frames)
        - Values: Power spectral density (linear scale by default)
        - Frequency axis: 0 to Nyquist frequency (sample_rate/2)
        - Time axis: Aligned with hop_length parameter from STFT

    Applications:
        - Time-frequency analysis and visualization
        - Foundation for other feature extractors (chromagrams, MFCCs)
        - Spectral content analysis over time
        - Audio classification preprocessing
        - Music transcription and source separation
        - Real-time audio visualization

    Example:
        >>> spectrogram = SpectrogramBuilder().build(audio_signal)
        >>> print(f"Spectrogram shape: {spectrogram.data().shape}")
        >>> spectrogram.plot()  # Linear scale visualization
        >>> db_spec = spectrogram.to_db()  # Convert to dB scale
        >>> db_spec.plot()  # Enhanced dynamic range visualization

    Note:
        The class supports conversion to dB scale for better visualization of
        audio content with wide dynamic ranges. DB conversion is performed
        using librosa.power_to_db() with reference to maximum value.
    """

    def __init__(self, S: np.ndarray, S_sr: float):
        """Initialize Spectrogram with power spectral data.

        Args:
            S (np.ndarray): Power spectrogram data with shape (frequency_bins, time_frames).
            S_sr (float): Feature sampling rate in Hz (related to hop_length in STFT).
        """
        super().__init__(S, S_sr, name="Spectrogram")

    def plot(
        self,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_signal: Optional[Signal] = None,
        figsize=(12, 8),
    ):
        """Plot the spectrogram with customizable visualization options.

        Creates a comprehensive visualization of the time-frequency representation
        with support for temporal annotations and signal context.

        Args:
            x_axis_type (str, optional): X-axis format. "time" for seconds,
                                       "frame" for frame indices. Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations for
                                                       structural visualization.
            original_signal (Optional[Signal], optional): Reference signal for
                                                         temporal alignment.
            figsize (tuple, optional): Figure size in inches. Default: (12, 8).
        """
        self._plot(
            self.data(),
            self.sampling_rate(),
            "Spectrogram",
            x_axis_type,
            time_annotations,
            original_signal,
            figsize=figsize,
        )

    def to_db(self) -> "Spectrogram":
        """Convert spectrogram to decibel (dB) scale for enhanced visualization.

        Transforms linear power values to logarithmic dB scale using librosa's
        power_to_db function with reference to the maximum value. This conversion
        enhances visualization of audio content with wide dynamic ranges.

        Returns:
            Spectrogram: New Spectrogram instance with dB-scaled data.
                        Original instance remains unchanged.

        Example:
            >>> linear_spec = SpectrogramBuilder().build(signal)
            >>> db_spec = linear_spec.to_db()
            >>> print(f"Linear range: {np.ptp(linear_spec.data()):.2f}")
            >>> print(f"dB range: {np.ptp(db_spec.data()):.2f}")

        Note:
            If the spectrogram is already in dB scale, returns self without
            modification and prints a warning message.
        """
        if not self._db_scaled:
            new_spec = self.copy()
            new_spec._data = librosa.power_to_db(self.data(), ref=np.max)
            new_spec._db_scaled = True
            return new_spec
        else:
            print("/!\\ Spectrogram is already in dB scale.")
            return self


class Chromagram(BaseFeature):
    """12-dimensional pitch class profile for harmonic and tonal analysis.

    Chromagram represents the relative intensity of each of the 12 pitch classes
    (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) over time by folding the frequency
    spectrum into a single octave. This feature is essential for music analysis
    tasks involving harmony, chord recognition, and key estimation.

    Key Features:
        - 12-bin representation for comprehensive pitch class analysis
        - Octave-folded frequency mapping for harmonic content focus
        - Robust to tempo, timbre, and octave variations
        - Essential for chord recognition and key estimation algorithms
        - Support for music structure analysis and pattern detection
        - Rich visualization with pitch class labeling

    Mathematical Foundation:
        Each chromagram frame C[n,p] represents the energy associated with pitch
        class p at time frame n, computed by mapping STFT frequency bins to
        12 pitch classes and aggregating energy across all octaves:
        C[n,p] = Σ |STFT[n,k]|² for all k where freq[k] maps to pitch class p

    Data Format:
        - Shape: (12, time_frames)
        - Pitch classes: [C, C#, D, D#, E, F, F#, G, G#, A, A#, B] (indices 0-11)
        - Values: Normalized energy for each pitch class
        - Time axis: Aligned with STFT hop_length parameter

    Applications:
        - Chord recognition and progression analysis
        - Key estimation and tonal center detection
        - Music structure analysis and segmentation
        - Audio similarity and cover song identification
        - Harmonic content analysis and music theory applications
        - Real-time chord tracking and music accompaniment

    Example:
        >>> chromagram = ChromagramBuilder().build(audio_signal)
        >>> print(f"Chroma shape: {chromagram.data().shape}")  # (12, time_frames)
        >>> # Find dominant pitch class at each time frame
        >>> dominant_pitches = np.argmax(chromagram.data(), axis=0)
        >>> pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        >>> print([pitch_names[p] for p in dominant_pitches[:10]])  # First 10 frames

    Note:
        Chromagrams are particularly effective for Western tonal music analysis.
        For atonal or microtonal music, other spectral representations may be
        more appropriate.
    """

    def __init__(self, C: np.ndarray, C_sr: float):
        """Initialize Chromagram with pitch class profile data.

        Args:
            C (np.ndarray): Chromagram data with shape (12, time_frames).
                          Each row represents one pitch class (C through B).
            C_sr (float): Feature sampling rate in Hz (time resolution).
        """
        super().__init__(C, C_sr, name="Chromagram")

    def plot(
        self,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_signal: Optional[Signal] = None,
        figsize=(12, 8),
        title_override: str = "Chromagram",
    ):
        """Plot the chromagram with pitch class visualization.

        Creates a comprehensive visualization of pitch class content over time
        with proper labeling of the 12 pitch classes and support for temporal
        annotations.

        Args:
            x_axis_type (str, optional): X-axis format. "time" for seconds,
                                       "frame" for frame indices. Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations for
                                                       harmonic analysis visualization.
            original_signal (Optional[Signal], optional): Reference signal for
                                                         temporal alignment.
            figsize (tuple, optional): Figure size in inches. Default: (12, 8).
            title_override (str, optional): Custom title for the plot.
                                          Default: "Chromagram".
        """
        self._plot(
            self.data(),
            self.sampling_rate(),
            title_override,
            x_axis_type,
            time_annotations,
            original_signal,
            figsize=figsize,
        )


class MFCC(BaseFeature):
    """Mel-Frequency Cepstral Coefficients for timbre and spectral shape analysis.

    MFCC features capture the spectral shape characteristics of audio signals by
    modeling human auditory perception through mel-scale frequency warping and
    cepstral analysis. These features are highly effective for audio classification,
    speaker recognition, and timbre analysis tasks.

    Key Features:
        - Mel-scale frequency warping for perceptual relevance
        - DCT-based decorrelation for compact representation
        - Configurable number of coefficients (typically 12-20)
        - Robust spectral shape characterization
        - Industry standard for audio classification and speech processing

    Mathematical Foundation:
        MFCCs are computed through: Signal → STFT → Mel Filterbank → Log → DCT
        Each coefficient captures different aspects of spectral shape:
        - C0: Log energy (often excluded for speech)
        - C1-C2: Broad spectral shape and tilt
        - C3-C12: Detailed spectral envelope characteristics

    Data Format:
        - Shape: (n_mfcc, time_frames)
        - Values: Cepstral coefficients (typically in range [-10, 10])
        - Coefficient order: C0 (energy) to Cn (fine spectral details)

    Applications:
        - Audio genre classification and music analysis
        - Speaker identification and verification systems
        - Timbre-based music similarity and recommendation
        - Audio content analysis and automatic tagging
        - Music instrument recognition and classification

    Example:
        >>> mfcc = MFCCBuilder(n_mfcc=13).build(audio_signal)
        >>> energy = mfcc.data()[0, :]  # C0: Energy coefficient
        >>> spectral_shape = mfcc.data()[1:, :]  # C1-C12: Shape coefficients
        >>> mfcc.plot()  # Visualize cepstral evolution over time
    """

    def __init__(self, mfcc: np.ndarray, mfcc_sr: float):
        """Initialize MFCC with cepstral coefficient data.

        Args:
            mfcc (np.ndarray): MFCC data with shape (n_mfcc, time_frames).
            mfcc_sr (float): Feature sampling rate in Hz.
        """
        super().__init__(mfcc, mfcc_sr, name="MFCC")

    def plot(
        self,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_signal: Optional[Signal] = None,
        figsize=(12, 8),
    ):
        """Plot MFCC coefficients over time with cepstral coefficient labeling.

        Args:
            x_axis_type (str, optional): X-axis format. Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations.
            original_signal (Optional[Signal], optional): Reference signal.
            figsize (tuple, optional): Figure size. Default: (12, 8).
        """
        self._plot(
            self.data(),
            self.sampling_rate(),
            "MFCC",
            x_axis_type,
            time_annotations,
            original_signal,
            figsize=figsize,
        )


class Tempogram(BaseFeature):
    """Tempo and rhythm analysis features for beat tracking and rhythm classification.

    Tempogram represents tempo information over time by analyzing onset strength
    patterns and computing Fourier-based tempo period detection. This feature is
    essential for rhythm analysis, beat tracking, and tempo-based music analysis.

    Key Features:
        - Tempo period analysis using Fourier-based detection
        - Onset strength-based computation for rhythmic emphasis
        - Normalization for consistent amplitude scaling
        - Support for tempo variation analysis over time
        - Integration with beat tracking algorithms

    Mathematical Foundation:
        Tempograms analyze the onset strength function O(n) by computing its
        autocorrelation or Fourier transform to detect periodic patterns
        corresponding to different tempo periods (beats per minute).

    Data Format:
        - Shape: (tempo_bins, time_frames)
        - Tempo axis: Different tempo periods (related to BPM)
        - Values: Normalized tempo strength for each period

    Applications:
        - Beat tracking and tempo estimation algorithms
        - Rhythm pattern analysis and dance music classification
        - Music structure analysis based on rhythmic content
        - Tempo variation detection in musical performances
        - Automatic DJ mixing and beat synchronization

    Example:
        >>> tempogram = TempogramBuilder().build(audio_signal)
        >>> # Find dominant tempo at each time frame
        >>> dominant_tempo_bins = np.argmax(tempogram.data(), axis=0)
        >>> tempogram.plot()  # Visualize tempo evolution over time
    """

    def __init__(self, T: np.ndarray, T_sr: float):
        """Initialize Tempogram with tempo analysis data.

        Args:
            T (np.ndarray): Tempogram data with shape (tempo_bins, time_frames).
            T_sr (float): Feature sampling rate in Hz.
        """
        super().__init__(T, T_sr, name="Tempogram")

    def plot(
        self,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_signal: Optional[Signal] = None,
        figsize=(12, 8),
    ):
        """Plot tempogram showing tempo periods over time.

        Args:
            x_axis_type (str, optional): X-axis format. Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations.
            original_signal (Optional[Signal], optional): Reference signal.
            figsize (tuple, optional): Figure size. Default: (12, 8).
        """
        self._plot(
            self.data(),
            self.sampling_rate(),
            "Tempogram",
            x_axis_type,
            time_annotations,
            original_signal,
            figsize=figsize,
        )


class HRPS(BaseFeature, PeakFinder):
    """Harmonic-Residual-Percussive Source separation features with peak detection.

    HRPS features represent the energy content of three separated audio components:
    harmonic (sustained tonal content), residual (unclassified content), and percussive
    (transient content). This advanced feature enables detailed analysis of musical
    texture and provides robust peak detection for structural analysis.

    Key Features:
        - Three-component energy analysis (Harmonic, Residual, Percussive)
        - Advanced median filtering-based source separation
        - Local energy computation for each component
        - Peak detection functionality for structural boundary analysis
        - Component-specific data access through properties
        - Normalization support for consistent analysis

    Algorithm Foundation:
        HRPS uses 2D median filtering on spectrograms to separate components:
        - Harmonic: Horizontal filtering preserves frequency consistency
        - Percussive: Vertical filtering preserves temporal precision
        - Residual: Complement of harmonic and percussive content
        - Energy: Local energy computation for each separated component

    Data Format:
        - Shape: (3, time_frames) where:
          - Row 0: Harmonic component energy
          - Row 1: Residual component energy
          - Row 2: Percussive component energy
        - Values: Local energy measures for each component

    Peak Detection:
        Implements PeakFinder interface with residual component analysis
        for detecting structural boundaries and significant events.

    Applications:
        - Music source separation and component analysis
        - Instrument-specific feature extraction and analysis
        - Rhythm vs. melody content analysis
        - Structural boundary detection in music
        - Audio texture analysis and classification
        - Music transcription preprocessing

    Example:
        >>> hrps = HRPSBuilder(L_h_frames=31, L_p_bins=31).build(audio_signal)
        >>> harmonic_energy = hrps.harmonic_data  # Access harmonic component
        >>> percussive_energy = hrps.percussive_data  # Access percussive component
        >>> peaks = hrps.find_peaks(threshold=0.1, distance_seconds=5)
        >>> hrps.plot()  # Visualize all three components

    Note:
        The residual component often captures room tone, noise, and content
        not clearly harmonic or percussive, making it useful for detecting
        transitions and structural boundaries.
    """

    def __init__(self, H: np.ndarray, H_sr: float):
        """Initialize HRPS with three-component energy data.

        Args:
            H (np.ndarray): HRPS energy data with shape (3, time_frames).
                           Row 0: Harmonic, Row 1: Residual, Row 2: Percussive.
            H_sr (float): Feature sampling rate in Hz.
        """
        super().__init__(H, H_sr, name="HRPS")

    def plot(
        self,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_signal: Optional[Signal] = None,
        figsize=(12, 8),
    ):
        """Plot HRPS features showing all three component energies.

        Args:
            x_axis_type (str, optional): X-axis format. Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations.
            original_signal (Optional[Signal], optional): Reference signal.
            figsize (tuple, optional): Figure size. Default: (12, 8).
        """
        self._plot(
            self.data(),
            self.sampling_rate(),
            "HRPS",
            x_axis_type,
            time_annotations,
            original_signal,
            figsize=figsize,
        )

    @property
    def harmonic_data(self) -> np.ndarray:
        """Access harmonic component energy data.

        Returns:
            np.ndarray: Harmonic component energy with shape (time_frames,).
        """
        return self.data()[0, :]

    @property
    def residual_data(self) -> np.ndarray:
        """Access residual component energy data.

        Returns:
            np.ndarray: Residual component energy with shape (time_frames,).
        """
        return self.data()[1, :]

    @property
    def percussive_data(self) -> np.ndarray:
        """Access percussive component energy data.

        Returns:
            np.ndarray: Percussive component energy with shape (time_frames,).
        """
        return self.data()[2, :]

    @staticmethod
    def _min_max_normalize(data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range using min-max scaling.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            np.ndarray: Normalized data in range [0, 1].
        """
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    @override
    def find_peaks(
        self, threshold: float = 0.1, distance_seconds: int = 10
    ) -> np.ndarray:
        """Find peaks in the residual component for structural boundary detection.

        Detects significant peaks in the residual component energy, which often
        correspond to structural boundaries, transitions, or significant events
        in the audio content.

        Args:
            threshold (float, optional): Relative height threshold for peak detection
                                       as fraction of maximum value (0 to 1).
                                       Default: 0.1.
            distance_seconds (int, optional): Minimum distance between peaks in seconds
                                            to avoid detecting multiple peaks for
                                            the same event. Default: 10.

        Returns:
            np.ndarray: Array of peak frame indices in the residual component.

        Example:
            >>> hrps = HRPSBuilder().build(audio_signal)
            >>> peaks = hrps.find_peaks(threshold=0.15, distance_seconds=8)
            >>> peak_times = peaks / hrps.sampling_rate()
            >>> print(f"Found {len(peaks)} structural boundaries")

        Note:
            Uses the residual component as it often contains transition artifacts
            and boundary-related content that makes it suitable for structural
            analysis.
        """
        from scipy.signal import find_peaks as scipy_find_peaks

        # Calculate absolute height threshold based on relative threshold
        height_threshold = np.max(self.residual_data) * threshold

        # Use scipy's find_peaks to identify peaks
        distance_samples = int(distance_seconds * self.sampling_rate())
        peaks, _ = scipy_find_peaks(
            self.residual_data, height=height_threshold, distance=distance_samples
        )
        return peaks


class SilenceCurve(BaseFeature, PeakFinder):
    """Silence detection and activity analysis feature with peak detection capabilities.

    SilenceCurve provides time-varying measures of audio activity and silence,
    enabling detailed analysis of audio content structure, voice activity detection,
    and automatic segmentation based on activity levels. The feature includes
    sophisticated peak detection for identifying silence boundaries.

    Key Features:
        - Time-varying silence and activity detection
        - Multiple detection algorithms (amplitude-based, spectral-based)
        - Automatic normalization to [0, 1] range for consistent analysis
        - Peak detection functionality for silence boundary detection
        - Support for enhanced contrast adjustment
        - Integration with structural analysis workflows

    Algorithm Foundation:
        Silence curves use inverted energy measures where:
        - Higher values indicate more likely silence regions
        - Lower values indicate higher activity regions
        - Min-max normalization ensures consistent range [0, 1]

    Data Format:
        - Shape: (1, time_frames) - single-channel feature
        - Values: Normalized silence probability in range [0, 1]
        - Interpretation: 0 = high activity, 1 = high silence probability

    Peak Detection:
        Implements PeakFinder interface for detecting silence boundaries,
        transitions between active and silent regions, and structural
        segmentation points.

    Applications:
        - Voice activity detection in speech processing
        - Music structure analysis and segmentation
        - Audio content analysis and automatic editing
        - Noise reduction preprocessing
        - Podcast and speech content analysis
        - Automatic silence removal and trimming

    Example:
        >>> silence = SilenceCurveBuilder("amplitude").build(audio_signal)
        >>> # Find silent regions (high values)
        >>> silence_threshold = 0.7
        >>> silent_frames = silence.data()[0, :] > silence_threshold
        >>> # Detect silence boundaries
        >>> boundaries = silence.find_peaks(threshold=0.5, distance_seconds=2)
        >>> silence.plot(peaks=boundaries)  # Visualize with detected boundaries

    Note:
        The silence curve uses inverted energy values for intuitive interpretation
        where peaks correspond to silent regions and valleys to active regions.
    """

    def __init__(self, silence_curve: np.ndarray, silence_sr: float):
        """Initialize SilenceCurve with normalized silence detection data.

        Processes raw silence detection data through min-max normalization
        and ensures proper 2D array format for consistent interface.

        Args:
            silence_curve (np.ndarray): Raw silence detection data.
            silence_sr (float): Feature sampling rate in Hz.
        """
        sc = self._min_max_normalize(silence_curve)

        # Enhance contrast Near 1
        # power_factor = 10
        # sc = np.power(sc, power_factor)

        if sc.ndim == 1:
            # transform to 2D array with shape (1, N)
            sc = sc[np.newaxis, :]
        super().__init__(sc, silence_sr, name="Silence Curve")

    def _min_max_normalize(self, data) -> np.ndarray:
        """Normalize data to [0, 1] range using min-max scaling.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            np.ndarray: Normalized data in range [0, 1].
        """
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def plot(
        self,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        peaks: Optional[np.ndarray] = None,
        original_signal: Optional[Signal] = None,
        figsize=(12, 8),
    ):
        """Plot silence curve with optional peak detection visualization.

        Creates a comprehensive visualization of silence detection over time
        with support for peak marking and temporal annotations.

        Args:
            x_axis_type (str, optional): X-axis format. "time" for seconds,
                                       "frame" for frame indices. Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations for
                                                       structural visualization.
            peaks (Optional[np.ndarray], optional): Peak frame indices to mark
                                                   as vertical lines on the plot.
            original_signal (Optional[Signal], optional): Reference signal for
                                                         temporal alignment.
            figsize (tuple, optional): Figure size in inches. Default: (12, 8).
        """
        self._plot(
            self.data(),
            self.sampling_rate(),
            "Silence Curve",
            x_axis_type,
            time_annotations,
            original_signal,
            peaks=peaks,
            figsize=figsize,
        )

    @override
    def find_peaks(
        self, threshold: float = 0.5, distance_seconds: int = 10
    ) -> np.ndarray:
        """Find peaks in the silence curve for boundary detection.

        Detects significant peaks in the silence curve that correspond to
        silent regions or transitions between active and silent content.

        Args:
            threshold (float, optional): Relative height threshold for peak detection
                                       as fraction of maximum value (0 to 1).
                                       Default: 0.5.
            distance_seconds (int, optional): Minimum distance between peaks in seconds
                                            to avoid detecting multiple peaks for
                                            the same silent region. Default: 10.

        Returns:
            np.ndarray: Array of peak frame indices corresponding to silence peaks.

        Example:
            >>> silence = SilenceCurveBuilder().build(audio_signal)
            >>> silence_peaks = silence.find_peaks(threshold=0.7, distance_seconds=5)
            >>> peak_times = silence_peaks / silence.sampling_rate()
            >>> print(f"Found {len(silence_peaks)} silence regions")

        Note:
            Higher threshold values detect only the most prominent silent regions,
            while lower values detect subtle activity changes.
        """
        from scipy.signal import find_peaks as scipy_find_peaks

        # Calculate absolute height threshold based on relative threshold
        height_threshold = np.max(self.data()) * threshold

        # Use scipy's find_peaks to identify peaks
        distance_samples = int(distance_seconds * self.sampling_rate())
        peaks, _ = scipy_find_peaks(
            self.data()[0, :], height=height_threshold, distance=distance_samples
        )
        return peaks


class SelfSimilarityMatrix(SimilarityMatrix):
    """Self-Similarity Matrix for structural music analysis and pattern detection.

    Self-Similarity Matrix (SSM) is a fundamental tool for music structure analysis
    that captures the pairwise similarity between different time points in a feature
    sequence. It reveals repetitive patterns, musical structure, and provides the
    foundation for automatic music segmentation and analysis.

    Key Features:
        - Symmetric matrix representation of temporal self-similarities
        - Support for thresholding and binarization operations
        - Novelty curve computation from checkerboard kernel analysis
        - Optimized implementations for large-scale analysis
        - Integration with structural analysis workflows

    Algorithm Foundation:
        SSM[i,j] represents the similarity between feature vectors at times i and j.
        Common similarity measures include cosine similarity, Euclidean distance,
        and normalized correlation. The matrix exhibits block structure corresponding
        to musical sections and diagonal stripes indicating repetitive patterns.

    Data Format:
        - Shape: (time_frames, time_frames) - square symmetric matrix
        - Values: Similarity scores (typically normalized to [0, 1])
        - Main diagonal: Always maximum similarity (self-comparison)
        - Off-diagonal blocks: Inter-segment similarities

    Structural Interpretation:
        - Main diagonal: Perfect self-similarity
        - Parallel diagonals: Repetitive patterns or sequences
        - Block structures: Homogeneous musical sections
        - Checkerboard patterns: Alternating musical content

    Applications:
        - Music structure analysis and automatic segmentation
        - Pattern discovery and motif detection in music
        - Cover song identification and music similarity
        - Repetition-based music analysis and visualization
        - Audio thumbnailing and music summarization
        - Automatic verse/chorus detection

    Example:
        >>> ssm = SelfSimilarityMatrixBuilder('cosine').build(chromagram)
        >>> # Apply threshold to enhance structure
        >>> ssm_filtered = ssm.threshold(thresh=0.7, binarize=True)
        >>> # Compute novelty for boundary detection
        >>> novelty = ssm.compute_novelty_curve(kernel_size=16)
        >>> ssm.plot()  # Visualize similarity structure

    Note:
        Large SSMs can be memory intensive. Consider feature downsampling
        or sliding window approaches for very long audio sequences.
    """

    def __init__(self, ssm: np.ndarray, ssm_sr: float):
        """Initialize Self-Similarity Matrix with similarity data.

        Args:
            ssm (np.ndarray): Square similarity matrix with shape (time_frames, time_frames).
            ssm_sr (float): Feature sampling rate in Hz.
        """
        super().__init__(ssm, ssm_sr, name="Self-Similarity Matrix")

    def plot(
        self,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_base_feature: Optional[BaseFeature] = None,
    ):
        """Plot Self-Similarity Matrix with structural visualization.

        Args:
            x_axis_type (str, optional): X-axis format. Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations.
            original_base_feature (Optional[BaseFeature], optional): Reference feature.
        """
        self._plot(
            "SelfSimilarityMatrix", x_axis_type, time_annotations, original_base_feature
        )

    def threshold(
        self,
        thresh: float = 0.5,
        binarize: bool = False,
    ) -> "SelfSimilarityMatrix":
        """Apply threshold-based filtering to enhance structural patterns.

        Filters the similarity matrix to emphasize significant similarities
        and suppress noise, making structural patterns more visible.

        Args:
            thresh (float, optional): Threshold value in range [0, 1]. Values below
                                     threshold are set to 0. Default: 0.5.
            binarize (bool, optional): If True, creates binary matrix (0/1 values).
                                     If False, preserves original values above threshold.
                                     Default: False.

        Returns:
            SelfSimilarityMatrix: Thresholded similarity matrix with same sampling rate.

        Example:
            >>> ssm = SelfSimilarityMatrixBuilder().build(features)
            >>> # Enhance strong similarities only
            >>> ssm_strong = ssm.threshold(thresh=0.8, binarize=False)
            >>> # Create binary structure matrix
            >>> ssm_binary = ssm.threshold(thresh=0.6, binarize=True)
        """
        # --- Thresholding (in-place, numpy) ---
        if binarize:
            ssm_thr = (self.data() >= thresh).astype(float)
        else:
            # Match the logic of class: set values below threshold to 0, keep others
            ssm_thr = np.where(self.data() >= thresh, self.data(), 0.0)

        return SelfSimilarityMatrix(ssm_thr, self.sampling_rate())

    def compute_novelty_curve(
        self, kernel_size: int = 16, variance: float = 0.5, exclude_borders: bool = True
    ) -> NoveltyCurve:
        """Compute novelty curve from SSM using checkerboard kernel analysis.

        Detects structural boundaries by analyzing local contrast patterns in the
        similarity matrix using a checkerboard kernel that emphasizes transitions.

        Args:
            kernel_size (int, optional): Half-size of checkerboard kernel. Total kernel
                                        size will be (2*L+1) x (2*L+1). Default: 16.
            variance (float, optional): Gaussian variance for kernel smoothing. Controls
                                      the kernel's sensitivity to local patterns. Default: 0.5.
            exclude_borders (bool, optional): If True, sets boundary regions to zero
                                            to avoid edge effects. Default: True.

        Returns:
            NoveltyCurve: Novelty curve indicating structural boundary strength over time.

        Example:
            >>> ssm = SelfSimilarityMatrixBuilder().build(chromagram)
            >>> novelty = ssm.compute_novelty_curve(kernel_size=20, variance=1.0)
            >>> boundaries = novelty.find_peaks(threshold=0.3, distance_seconds=10)

        Note:
            The checkerboard kernel detects changes in local similarity structure,
            making it effective for identifying musical section boundaries.
        """
        novelty_curve = src.libfmp.c4.compute_novelty_ssm(
            self.data(), L=kernel_size, var=variance, exclude=exclude_borders
        )
        novelty_sr = self.sampling_rate()
        return NoveltyCurve(novelty_curve, novelty_sr)

    def compute_novelty_curve_fast(
        self, kernel_size: int = 16, variance: float = 0.5, exclude_borders: bool = True
    ) -> NoveltyCurve:
        """Compute novelty curve using optimized sliding window implementation.

        Provides a faster alternative to standard novelty computation using
        vectorized operations and sliding window views for improved performance
        on large similarity matrices.

        Args:
            kernel_size (int, optional): Half-size of checkerboard kernel. Default: 16.
            variance (float, optional): Gaussian variance for kernel smoothing. Default: 0.5.
            exclude_borders (bool, optional): If True, sets boundary regions to zero. Default: True.

        Returns:
            NoveltyCurve: Novelty curve with optimized computation for better performance.

        Example:
            >>> ssm = SelfSimilarityMatrixBuilder().build(long_chromagram)
            >>> # Use fast method for large matrices
            >>> novelty = ssm.compute_novelty_curve_fast(kernel_size=20)
            >>> boundaries = novelty.find_peaks(threshold=0.2)

        Note:
            This method is particularly beneficial for large similarity matrices
            where standard convolution approaches become computationally expensive.
        """
        kernel = src.libfmp.c4.compute_kernel_checkerboard_gaussian(
            L=kernel_size, var=variance
        )
        N = self.data().shape[0]
        M = 2 * kernel_size + 1

        # Pad S with zeros manually
        S_padded = np.pad(self.data(), pad_width=kernel_size, mode="constant")

        # Create sliding window view of shape (N, M, M)
        windows = np.lib.stride_tricks.sliding_window_view(S_padded, (M, M))
        # windows shape: (N+2L - M +1, N+2L - M +1, M, M) = (N, N, M, M) here
        # We want only the diagonal patches: windows[i, i, :, :]
        diagonal_windows = windows[np.arange(N), np.arange(N)]

        # Compute novelty by element-wise multiply and sum over kernel dims
        nov = np.einsum("ij,ij->i", diagonal_windows, kernel)

        if exclude_borders:
            right = min(kernel_size, N)
            left = max(0, N - kernel_size)
            nov[:right] = 0
            nov[left:] = 0

        novelty_sr = self.sampling_rate()
        return NoveltyCurve(nov, novelty_sr)


class TimeLagMatrix(SimilarityMatrix):
    """Time-Lag Matrix for analyzing temporal evolution and dynamic similarity patterns.

    Time-Lag Matrix (TLM) captures how similarity patterns evolve over time by
    analyzing the relationship between consecutive time frames and their similarity
    vectors. This advanced technique is particularly useful for detecting gradual
    transitions and temporal dynamics in musical structure.

    Key Features:
        - Temporal evolution analysis of similarity patterns
        - Gradient-based novelty curve computation for smooth transitions
        - Dynamic similarity pattern detection
        - Complementary analysis to standard Self-Similarity Matrix
        - Enhanced sensitivity to gradual structural changes

    Algorithm Foundation:
        TLM analyzes the difference between consecutive columns (or rows) of a
        similarity matrix to capture how similarity patterns change over time.
        This approach is particularly effective for detecting gradual transitions
        that might be missed by block-based SSM analysis.

    Data Format:
        - Shape: Similar to base similarity matrix
        - Values: Temporal similarity evolution measures
        - Focus: Time-based changes rather than static patterns

    Temporal Analysis:
        - Detects gradual transitions between musical sections
        - Captures dynamic changes in harmonic content
        - Identifies tempo or rhythm variations over time
        - Reveals evolutionary patterns in musical development

    Applications:
        - Detection of gradual musical transitions and developments
        - Analysis of temporal dynamics in harmonic progressions
        - Complementary structural analysis with SSM techniques
        - Music evolution and development pattern analysis
        - Enhanced boundary detection for complex transitions

    Example:
        >>> tlm = TimeLagMatrixBuilder().build(chromagram_sequence)
        >>> # Compute novelty from temporal changes
        >>> temporal_novelty = tlm.compute_novelty_curve(padding=True)
        >>> boundaries = temporal_novelty.find_peaks(threshold=0.4)
        >>> tlm.plot()  # Visualize temporal evolution patterns

    Note:
        TLM is particularly effective when combined with SSM analysis for
        comprehensive structural analysis covering both static and dynamic patterns.
    """

    def __init__(self, tlm: np.ndarray, tlm_sr: float):
        """Initialize Time-Lag Matrix with temporal similarity data.

        Args:
            tlm (np.ndarray): Time-lag matrix data showing temporal evolution.
            tlm_sr (float): Feature sampling rate in Hz.
        """
        super().__init__(tlm, tlm_sr, name="Time-Lag Matrix")

    def plot(
        self,
        x_axis_type: str = "time",
        time_annotations: Optional[list] = None,
        original_base_feature: Optional[BaseFeature] = None,
    ):
        """Plot Time-Lag Matrix with temporal evolution visualization.

        Args:
            x_axis_type (str, optional): X-axis format. Default: "time".
            time_annotations (Optional[list], optional): Temporal annotations.
            original_base_feature (Optional[BaseFeature], optional): Reference feature.
        """
        self._plot(
            "Time-Lag Matrix", x_axis_type, time_annotations, original_base_feature
        )

    def compute_novelty_curve(self, padding: bool = True) -> NoveltyCurve:
        """Compute novelty curve from temporal gradient analysis.

        Analyzes the temporal evolution of similarity patterns by computing
        the norm of differences between consecutive time frames, capturing
        gradual transitions and dynamic changes.

        Args:
            padding (bool, optional): If True, maintains original length by padding.
                                    If False, returns (N-1) length array. Default: True.

        Returns:
            NoveltyCurve: Novelty curve indicating temporal change strength.

        Example:
            >>> tlm = TimeLagMatrixBuilder().build(features)
            >>> novelty = tlm.compute_novelty_curve(padding=True)
            >>> gradual_transitions = novelty.find_peaks(threshold=0.2)

        Note:
            This method is particularly effective for detecting gradual transitions
            that develop over multiple time frames rather than abrupt changes.
        """
        N = self.data().shape[0]
        if padding:
            nov = np.zeros(N)
        else:
            nov = np.zeros(N - 1)
        for n in range(N - 1):
            nov[n] = np.linalg.norm(self.data()[:, n + 1] - self.data()[:, n])

        nc: NoveltyCurve = NoveltyCurve(nov, self.sampling_rate())
        return nc


class NoveltyCurve(Feature, PeakFinder):
    """Novelty Curve for structural boundary detection and music segmentation.

    Novelty curves represent the degree of change or novelty at each time point
    in an audio signal, making them essential for automatic music structure analysis,
    boundary detection, and segmentation. They integrate multiple analysis techniques
    and provide robust peak detection for identifying structural transitions.

    Key Features:
        - Normalized novelty representation for consistent analysis
        - Advanced peak detection with configurable thresholds and distances
        - Gaussian smoothing for noise reduction and trend analysis
        - Multiple combination methods for multi-feature analysis
        - Comprehensive visualization with peak marking and annotations
        - Support for both weighted and unweighted curve combination

    Algorithm Foundation:
        Novelty curves are computed from various source analyses including:
        - Spectral change detection (onset detection functions)
        - Self-similarity matrix kernel analysis (checkerboard, Gaussian)
        - Time-lag matrix gradient analysis
        - Harmonic change detection
        - Combined multi-feature analysis

    Data Format:
        - Shape: (time_frames,) - single-dimensional time series
        - Values: Normalized novelty strength in range [0, 1]
        - Peaks: High values indicate structural boundaries or significant changes

    Peak Detection:
        Implements sophisticated peak detection with:
        - Relative height thresholds for adaptive sensitivity
        - Minimum distance constraints to avoid redundant detections
        - Integration with scipy.signal.find_peaks for robust analysis

    Applications:
        - Automatic music structure analysis and segmentation
        - Beat tracking and tempo analysis preprocessing
        - Audio event detection and boundary identification
        - Music information retrieval and automatic annotation
        - Real-time music analysis and live performance systems
        - Cover song identification and music similarity analysis

    Example:
        >>> # Create from SSM analysis
        >>> ssm = SelfSimilarityMatrixBuilder().build(chromagram)
        >>> novelty = ssm.compute_novelty_curve(kernel_size=16)
        >>>
        >>> # Smooth for noise reduction
        >>> smooth_novelty = novelty.smooth(sigma=1.0)
        >>>
        >>> # Detect structural boundaries
        >>> boundaries = smooth_novelty.find_peaks(threshold=0.3, distance_seconds=8)
        >>>
        >>> # Combine with other novelty curves
        >>> combined = novelty.combine_with([other_novelty], method='mean')
        >>>
        >>> # Visualize results
        >>> novelty.plot(peaks=boundaries, x_axis_type='time')

    Note:
        Novelty curves benefit from smoothing to reduce noise while preserving
        structural boundaries. Consider combining multiple novelty sources for
        more robust boundary detection.
    """

    def __init__(self, novelty: np.ndarray, novelty_sr: float):
        """Initialize NoveltyCurve with normalized novelty data.

        Automatically normalizes input data to [0, 1] range for consistent
        analysis and comparison across different novelty computation methods.

        Args:
            novelty (np.ndarray): Raw novelty data array.
            novelty_sr (float): Feature sampling rate in Hz.
        """
        normalized_novelty = self._min_max_normalize(novelty)
        super().__init__(normalized_novelty, novelty_sr, name="Novelty Curve")

    def smooth(self, sigma: float = 1.0) -> "NoveltyCurve":
        """Apply Gaussian smoothing to reduce noise and enhance trends.

        Smoothing helps reduce noise in novelty curves while preserving
        important structural boundaries, making peak detection more reliable.

        Args:
            sigma (float, optional): Gaussian kernel standard deviation. Higher values
                                   provide more smoothing. Use 0 or negative for no smoothing.
                                   Default: 1.0.

        Returns:
            NoveltyCurve: Smoothed novelty curve with same sampling rate.

        Example:
            >>> novelty = ssm.compute_novelty_curve()
            >>> # Light smoothing for noise reduction
            >>> smooth_light = novelty.smooth(sigma=0.5)
            >>> # Heavy smoothing for trend analysis
            >>> smooth_heavy = novelty.smooth(sigma=2.0)

        Raises:
            ValueError: If data is None or invalid.
        """
        if sigma <= 0:
            return self

        data = self.data()
        if data is None:
            raise ValueError("Data cannot be None.")
        smoothed_curve = gaussian_filter1d(data, sigma=sigma)

        return NoveltyCurve(smoothed_curve, self.sampling_rate())

    @override
    def find_peaks(
        self, threshold: float = 0.1, distance_seconds: int = 10
    ) -> np.ndarray:
        """Find peaks in novelty curve for structural boundary detection.

        Detects significant peaks that correspond to structural boundaries,
        transitions, or other significant events in the audio content.

        Args:
            threshold (float, optional): Relative height threshold as fraction of
                                       maximum value (0 to 1). Higher values detect
                                       only prominent boundaries. Default: 0.1.
            distance_seconds (int, optional): Minimum distance between peaks in seconds
                                            to prevent detecting multiple peaks for
                                            the same boundary. Default: 10.

        Returns:
            np.ndarray: Array of peak frame indices corresponding to detected boundaries.

        Example:
            >>> novelty = ssm.compute_novelty_curve()
            >>> # Detect prominent boundaries only
            >>> major_boundaries = novelty.find_peaks(threshold=0.5, distance_seconds=15)
            >>> # Detect all significant changes
            >>> all_boundaries = novelty.find_peaks(threshold=0.2, distance_seconds=5)
            >>> print(f"Found {len(major_boundaries)} major structural boundaries")

        Note:
            Lower thresholds detect more boundaries but may include noise.
            Higher thresholds detect only the most prominent structural changes.
        """
        from scipy.signal import find_peaks as scipy_find_peaks

        # Calculate absolute height threshold based on relative threshold
        height_threshold = np.max(self.data()) * threshold

        # Use scipy's find_peaks to identify peaks
        distance_samples = int(distance_seconds * self.sampling_rate())
        peaks, _ = scipy_find_peaks(
            self.data(), height=height_threshold, distance=distance_samples
        )

        return peaks

    def _min_max_normalize(self, data) -> np.ndarray:
        """Normalize data to [0, 1] range using min-max scaling.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            np.ndarray: Normalized data in range [0, 1].
        """
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    @staticmethod
    def combine(
        ncs: list["NoveltyCurve"],
        weights: list[float] | None = None,
        method: str = "mean",
    ) -> "NoveltyCurve":
        """Combine multiple novelty curves using weighted aggregation methods.

        Enables multi-feature novelty analysis by combining curves from different
        sources (e.g., spectral, harmonic, percussive) with configurable weighting
        and combination strategies.

        Args:
            ncs (list[NoveltyCurve]): List of novelty curves to combine. All curves
                                    must have identical length and sampling rate.
            weights (list[float] | None, optional): Weight for each novelty curve.
                                                   Must match the length of ncs.
                                                   Required parameter.
            method (str, optional): Combination method. Options:
                                  - "mean": Weighted arithmetic mean
                                  - "max": Element-wise maximum of weighted curves
                                  - "weighted": Sum of weighted curves normalized by sum of weights
                                  Default: "mean".

        Returns:
            NoveltyCurve: Combined novelty curve with weighted aggregation.

        Example:
            >>> spectral_novelty = spectral_ssm.compute_novelty_curve()
            >>> harmonic_novelty = harmonic_ssm.compute_novelty_curve()
            >>> percussive_novelty = percussive_ssm.compute_novelty_curve()
            >>>
            >>> # Equal weighting
            >>> combined = NoveltyCurve.combine(
            ...     [spectral_novelty, harmonic_novelty, percussive_novelty],
            ...     weights=[1.0, 1.0, 1.0],
            ...     method="mean"
            ... )
            >>>
            >>> # Emphasize harmonic content
            >>> harmonic_focused = NoveltyCurve.combine(
            ...     [spectral_novelty, harmonic_novelty, percussive_novelty],
            ...     weights=[0.5, 2.0, 0.5],
            ...     method="weighted"
            ... )

        Raises:
            ValueError: If curves have different lengths/sampling rates, or if
                       lists are empty or mismatched in length.
        """
        if not ncs or not weights or len(ncs) != len(weights):
            raise ValueError(
                "The list of novelty curves and weights must be non-empty and of the same length."
            )

        # Ensure all novelty curves have the same length
        length = ncs[0].data().shape[0]
        sr = ncs[0].sampling_rate()
        for nc in ncs:
            if nc.data().shape[0] != length or nc.sampling_rate() != sr:
                raise ValueError(
                    f"All novelty curves must have the same length and sampling rate to combine. {nc.data().shape[0]} != {length} or {nc.sampling_rate()} != {sr}"
                )

        weighted_data = [nc.data() * weight for nc, weight in zip(ncs, weights)]

        # Combine based on the specified method
        if method == "mean":
            combined_data = np.mean(weighted_data, axis=0)
        elif method == "max":
            combined_data = np.max(weighted_data, axis=0)
        elif method == "weighted":
            combined_data = np.sum(weighted_data, axis=0) / np.sum(weights)
        else:
            raise ValueError("Method must be 'mean', 'max', or 'weighted'.")

        return NoveltyCurve(combined_data, ncs[0].sampling_rate())

    def combine_with(
        self,
        novelty_curves: Union[list["NoveltyCurve"], "NoveltyCurve"],
        method: str = "mean",
    ) -> "NoveltyCurve":
        """Combine this novelty curve with others using automatic length alignment.

        Provides a convenient interface for combining the current novelty curve
        with others, automatically handling length mismatches through interpolation.

        Args:
            novelty_curves (Union[list[NoveltyCurve], NoveltyCurve]): Other novelty
                                                                     curves to combine with this one.
            method (str, optional): Combination method. Options:
                                  - "mean": Arithmetic mean of all curves
                                  - "max": Element-wise maximum of all curves
                                  Default: "mean".

        Returns:
            NoveltyCurve: Combined novelty curve with length matching this curve.

        Example:
            >>> primary_novelty = ssm.compute_novelty_curve()
            >>> secondary_novelty = other_ssm.compute_novelty_curve()
            >>>
            >>> # Combine with automatic alignment
            >>> combined = primary_novelty.combine_with(secondary_novelty, method="mean")
            >>>
            >>> # Combine multiple curves
            >>> multi_combined = primary_novelty.combine_with(
            ...     [secondary_novelty, tertiary_novelty],
            ...     method="max"
            ... )

        Note:
            Length mismatches are resolved through linear interpolation, which
            may affect the temporal precision of shorter curves when stretched.

        Raises:
            ValueError: If novelty_curves is empty or method is invalid.
        """
        if not novelty_curves:
            raise ValueError("The list of novelty curves is empty.")

        # If a single NoveltyCurve is provided, convert it to a list
        if isinstance(novelty_curves, NoveltyCurve):
            novelty_curves = [novelty_curves]

        # Stretch all novelty curves to match the length of self
        length = self.data().shape[0]

        # Stretch all novelty curves to match the length -> Modifying nc values
        nc_data_aligned = [self.data()]  # Start with self
        for nc in novelty_curves:
            if nc.data().shape[0] != length:
                # Stretching using linear interpolation
                stretched_data = np.interp(
                    np.linspace(0, nc.data().shape[0] - 1, length),
                    np.arange(nc.data().shape[0]),
                    nc.data(),
                )
                nc_data_aligned.append(stretched_data)
            else:
                # Curve already has the correct length, add it directly
                nc_data_aligned.append(nc.data())

        # Stack the data from all novelty curves
        stacked_data = np.vstack([data for data in nc_data_aligned])

        # Combine based on the specified method
        if method == "mean":
            combined_data = np.mean(stacked_data, axis=0)
        elif method == "max":
            combined_data = np.max(stacked_data, axis=0)
        else:
            raise ValueError("Method must be 'mean' or 'max'.")

        return NoveltyCurve(combined_data, self.sampling_rate())

    def plot(
        self,
        x_axis_type: str = "time",
        novelty_name: str = "Novelty Curve",
        time_annotations: Optional[list] = None,
        peaks: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize=(10, 4),
    ):
        """Plot novelty curve with comprehensive visualization options.

        Creates a detailed visualization of the novelty curve with support for
        peak marking, temporal annotations, and structural analysis overlays.

        Args:
            x_axis_type (str, optional): X-axis format. Options:
                                       - "time": Display in seconds
                                       - "frame": Display in frame indices
                                       Default: "time".
            novelty_name (str, optional): Name for plot title and legend.
                                        Default: "Novelty Curve".
            time_annotations (Optional[list], optional): Temporal annotations for
                                                        structural visualization. Supports:
                                                        - List of [start, end, label] for regions
                                                        - List of floats for transition points
                                                        Default: None.
            peaks (Optional[np.ndarray], optional): Peak frame indices to mark as
                                                  vertical dashed red lines. Default: None.
            save_path (Optional[str], optional): Path to save plot. If None, displays
                                               interactively. Default: None.
            figsize (tuple, optional): Figure size in inches (width, height).
                                     Default: (10, 4).

        Example:
            >>> novelty = ssm.compute_novelty_curve()
            >>> boundaries = novelty.find_peaks(threshold=0.3, distance_seconds=10)
            >>>
            >>> # Basic plot with peaks
            >>> novelty.plot(peaks=boundaries)
            >>>
            >>> # Advanced plot with annotations
            >>> annotations = [[0, 30, 'Verse'], [30, 60, 'Chorus'], [60, 90, 'Bridge']]
            >>> novelty.plot(
            ...     peaks=boundaries,
            ...     time_annotations=annotations,
            ...     novelty_name='Structural Analysis',
            ...     save_path='novelty_analysis.png'
            ... )

        Note:
            Time annotations are always interpreted in seconds regardless of x_axis_type.
            The plot automatically handles conversion for frame-based display.
        """
        import matplotlib.pyplot as plt

        if type(time_annotations) is np.ndarray:
            time_annotations = time_annotations.tolist()

        if x_axis_type not in ["time", "frame"]:
            raise ValueError("x_axis_type must be 'time' or 'frame'")

        n_frames = self.data().shape[0]
        novelty_time = np.arange(n_frames) / self.sampling_rate()

        fig, ax = plt.subplots(figsize=figsize)

        if peaks is None:
            peaks = np.empty((0,), dtype=int)

        # Plot novelty curve depending on x axis type
        if x_axis_type == "time":
            x_values = peaks / self.sampling_rate()
            ax.plot(novelty_time, self.data(), label="Novelty Curve", color="black")
            ax.set_xlabel("Time (s)")
        else:
            x_values = peaks
            ax.plot(
                np.arange(n_frames), self.data(), label="Novelty Curve", color="black"
            )
            ax.set_xlabel("Frames")

        # Draw vertical dashed red lines spanning the full axis height
        for i, xv in enumerate(x_values):
            # label only the first line so legend remains clean
            ax.axvline(
                xv,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label="Predictions" if i == 0 else None,
            )

            ax.set_title(f"{novelty_name}")
            ax.set_ylabel("Novelty")
            max_val = np.max(self.data())
            if max_val <= 0:
                max_val = 1.0
            ax.set_ylim(0, max_val * 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Plot time annotations as colored rectangles on top of novelty curve
        if (
            time_annotations is not None
            and len(time_annotations) > 0
            and type(time_annotations[0]) in [list, tuple]
        ):
            novelty_max = np.max(self.data()) * 1.1
            for i, ann in enumerate(time_annotations):
                if len(ann) >= 2:  # Expecting [start_time, end_time, label] format
                    start_time, end_time = ann[0], ann[1]
                    label = ""

                    # Convert time annotations to appropriate x-coordinates
                    if x_axis_type == "time":
                        start_coord = start_time
                        end_coord = end_time
                    else:
                        # Convert time to frames
                        start_coord = start_time * self.sampling_rate()
                        end_coord = end_time * self.sampling_rate()

                    # Plot colored rectangle over novelty curve
                    ax.axvspan(
                        start_coord,
                        end_coord,
                        alpha=0.2,
                        color=get_color(i),
                        label=label,
                    )

                    # Add text label at the center
                    ax.text(
                        (start_coord + end_coord) / 2,
                        novelty_max * 0.9,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        rotation=0,
                    )
        elif (
            time_annotations is not None
            and len(time_annotations) > 0
            and type(time_annotations[0]) == float
        ):  # only the transitions
            novelty_max = np.max(self.data()) * 1.1
            for i, ann in enumerate(time_annotations):
                # Convert time annotations to appropriate x-coordinates
                if x_axis_type == "time":
                    coord = ann
                else:
                    # Convert time to frames
                    coord = ann * self.sampling_rate()

                # Plot vertical line over novelty curve
                ax.axvline(
                    coord,
                    alpha=0.5,
                    color=get_color(i),
                    linestyle="--",
                    label=f"Transition {i + 1}",
                )

                # Add text label at the top
                ax.text(
                    coord,
                    novelty_max * 0.9,
                    f"Transition {i + 1}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    rotation=90,
                )

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
        if save_path is not None:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()
