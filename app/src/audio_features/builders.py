"""Audio Feature Builders - Comprehensive feature extraction pipeline using the Builder pattern.

This module implements a sophisticated feature extraction system using the Builder design
pattern for constructing various audio features from different input sources. It provides
a unified interface for extracting spectral, temporal, harmonic, and structural features
from audio signals with configurable parameters and processing pipelines.

Key Features:
    - Builder pattern implementation for flexible feature construction
    - Abstract base classes ensuring consistent interfaces
    - Comprehensive feature extractors for multiple audio analysis domains
    - Signal-to-feature, feature-to-feature, and similarity matrix transformations
    - Configurable parameters for domain-specific optimization
    - Memory-efficient processing with librosa integration
    - Advanced techniques like HPSS, MFCC, chromagrams, and self-similarity matrices

Builder Hierarchy:
    - BuilderFromSignal: Extracts features directly from audio signals
    - BuilderFromBaseFeature: Transforms existing features into new representations
    - BuilderFromSimilarityMatrix: Derives features from similarity matrices

Core Feature Builders:
    - SpectrogramBuilder: STFT-based time-frequency representations
    - ChromagramBuilder: Chroma feature extraction for harmonic analysis
    - MFCCBuilder: Mel-Frequency Cepstral Coefficients for timbre analysis
    - TempogramBuilder: Tempo and rhythm feature extraction
    - SilenceCurveBuilder: Silence detection with multiple algorithms
    - HRPSBuilder: Harmonic-Residual-Percussive separation with local energy
    - SSMBuilder: Self-Similarity Matrix computation for structure analysis
    - TLMBuilder: Time-Lag Matrix derivation for repetitive structure analysis

Design Patterns:
    - Builder Pattern: Flexible construction of complex audio features
    - Strategy Pattern: Configurable algorithms within builders
    - Factory Pattern: Consistent feature creation interfaces
    - Template Method: Common processing steps with customizable implementations

Common Use Cases:
    - Music information retrieval (MIR) feature extraction
    - Audio content analysis and classification
    - Music structure analysis and segmentation
    - Audio similarity and recommendation systems
    - Real-time audio processing pipelines
    - Batch feature extraction workflows

Technical Dependencies:
    - librosa: Advanced audio analysis and feature extraction
    - scipy: Signal processing and filtering operations
    - numpy: Numerical operations and array manipulation
    - Custom libfmp modules: Specialized music processing algorithms
    - Custom interfaces: Builder and Feature abstractions

Performance Considerations:
    - Configurable frame and hop lengths for speed/quality trade-offs
    - Memory-efficient processing for long audio files
    - Optional downsampling for computational efficiency
    - Vectorized operations using NumPy and librosa
    - Progress reporting for long-running operations

Author: Hugo Demule
Date: January 2026
"""

from abc import abstractmethod

import librosa
import numpy as np
import scipy
import src.libfmp
import src.libfmp.b
import src.libfmp.c2
import src.libfmp.c3
import src.libfmp.c4
from src.audio.signal import Signal
from src.audio_features.features import (
    HRPS,
    MFCC,
    Chromagram,
    SelfSimilarityMatrix,
    SilenceCurve,
    Spectrogram,
    Tempogram,
    TimeLagMatrix,
)
from src.interfaces.builder import Builder
from src.interfaces.feature import BaseFeature, Feature, SimilarityMatrix


class BuilderFromSignal(Builder):
    """Abstract builder for extracting features directly from audio signals.

    BuilderFromSignal serves as the base class for all feature builders that operate
    directly on raw audio signal data. It defines the interface for signal-to-feature
    transformations, ensuring consistent behavior across different feature extraction
    algorithms.

    This class is part of the Builder pattern hierarchy and represents the first stage
    of the feature extraction pipeline, where raw audio samples are transformed into
    meaningful feature representations for further analysis.

    Design Pattern:
        - Abstract Builder: Defines construction interface for signal-based features
        - Template Method: Common pattern for signal processing workflows
        - Strategy: Different concrete implementations provide various feature types

    Processing Pipeline:
        Signal → Feature Extraction Algorithm → BaseFeature Object

    Common implementations include spectrograms, chromagrams, MFCCs, tempograms,
    and other time-frequency representations that form the foundation of audio
    analysis workflows.

    Example:
        >>> builder = ChromagramBuilder(frame_length=4410, hop_length=2205)
        >>> chromagram = builder.build(audio_signal)
        >>> print(f"Extracted chromagram with shape: {chromagram.data().shape}")

    Note:
        This is an abstract base class and cannot be instantiated directly.
        Use concrete implementations like SpectrogramBuilder, ChromagramBuilder, etc.
    """

    @abstractmethod
    def build(self, signal: Signal) -> BaseFeature:
        """Abstract method for extracting features from an audio signal.

        Performs feature extraction from raw audio signal data, implementing
        the specific algorithm defined by the concrete builder class. This
        method represents the core transformation from time-domain audio
        samples to feature-domain representations.

        Args:
            signal (Signal): Input audio signal containing samples, sample rate,
                           and metadata. The signal should be properly loaded
                           and validated before feature extraction.

        Returns:
            BaseFeature: Extracted feature object containing the computed feature
                        data and associated sampling rate. The specific feature
                        type depends on the concrete implementation.

        Raises:
            NotImplementedError: Always raised since this is an abstract method.

        Note:
            Concrete implementations must handle signal validation, parameter
            configuration, and feature computation according to their specific
            algorithm requirements.
        """
        pass


class BuilderFromBaseFeature(Builder):
    """Abstract builder for transforming existing features into new representations.

    BuilderFromBaseFeature defines the interface for feature-to-feature transformations,
    enabling the construction of higher-level features from existing base features.
    This represents the second stage of hierarchical feature extraction pipelines.

    This pattern allows for complex feature derivations such as self-similarity
    matrices from chromagrams, or structural features from spectral representations,
    enabling multi-level analysis workflows.

    Example Usage:
        BaseFeature (Chromagram) → SSMBuilder → SelfSimilarityMatrix

    Note:
        Abstract class requiring concrete implementation of the build method.
    """

    @abstractmethod
    def build(self, base_feature: BaseFeature) -> Feature:
        """Transform a base feature into a new feature representation.

        Args:
            base_feature (BaseFeature): Input feature to be transformed.

        Returns:
            Feature: New feature derived from the input base feature.
        """
        pass


class BuilderFromSimilarityMatrix(Builder):
    """Abstract builder for deriving features from similarity matrices.

    BuilderFromSimilarityMatrix enables the extraction of structural and temporal
    features from similarity matrix representations, supporting advanced music
    structure analysis workflows.

    This represents the third stage of the feature hierarchy, where similarity
    relationships are transformed into specialized representations for structural
    analysis and pattern detection.

    Example Usage:
        SimilarityMatrix → TLMBuilder → TimeLagMatrix

    Note:
        Abstract class requiring concrete implementation of the build method.
    """

    @abstractmethod
    def build(self, sm: SimilarityMatrix) -> Feature:
        """Extract features from a similarity matrix.

        Args:
            sm (SimilarityMatrix): Input similarity matrix for feature extraction.

        Returns:
            Feature: Feature representation derived from the similarity matrix.
        """
        pass


class SpectrogramBuilder(BuilderFromSignal):
    """Builder for extracting power spectrograms using Short-Time Fourier Transform.

    SpectrogramBuilder computes time-frequency representations of audio signals using
    the STFT algorithm with configurable windowing parameters. The resulting spectrogram
    provides a detailed view of how the frequency content of a signal varies over time,
    forming the foundation for many audio analysis tasks.

    Key Features:
        - STFT-based spectrogram computation using librosa
        - Configurable frame and hop lengths for temporal/frequency resolution trade-offs
        - Power spectrogram output (magnitude squared) for energy-based analysis
        - Automatic sample rate calculation for proper temporal indexing
        - Memory-efficient processing suitable for long audio files

    Algorithm Details:
        1. Apply Short-Time Fourier Transform with specified parameters
        2. Compute magnitude spectrum from complex STFT coefficients
        3. Square magnitudes to obtain power spectrogram
        4. Calculate feature sampling rate from hop length
        5. Return Spectrogram object with data and metadata

    Parameter Guidelines:
        - Larger frame_length: Better frequency resolution, worse time resolution
        - Smaller frame_length: Better time resolution, worse frequency resolution
        - Smaller hop_length: Better time resolution, more computational cost
        - hop_length = frame_length/2 is common for good time-frequency balance

    Attributes:
        frame_length (int): STFT frame length in samples (window size).
        hop_length (int): STFT hop length in samples (step size between frames).

    Example:
        >>> # Standard configuration for music analysis
        >>> builder = SpectrogramBuilder(frame_length=4410, hop_length=2205)
        >>> spectrogram = builder.build(audio_signal)
        >>> print(f"Spectrogram shape: {spectrogram.data().shape}")
        >>> # (frequency_bins, time_frames)

    Use Cases:
        - Time-frequency analysis and visualization
        - Foundation for other feature extractors
        - Spectral content analysis over time
        - Audio classification preprocessing
        - Music information retrieval applications
    """

    def __init__(self, frame_length: int = 4410, hop_length: int = 2205):
        """Initialize SpectrogramBuilder with STFT parameters.

        Args:
            frame_length (int, optional): STFT frame length in samples.
                                        Determines frequency resolution.
                                        Default: 4410 (~100ms at 44.1kHz).
            hop_length (int, optional): STFT hop length in samples.
                                      Determines time resolution.
                                      Default: 2205 (~50ms at 44.1kHz).
        """
        self.frame_length = frame_length
        self.hop_length = hop_length

    def build(self, signal: Signal) -> Spectrogram:
        """Extract power spectrogram from audio signal using STFT.

        Computes the Short-Time Fourier Transform of the input signal and
        converts it to a power spectrogram representation suitable for
        time-frequency analysis.

        Args:
            signal (Signal): Input audio signal with samples and sample rate.

        Returns:
            Spectrogram: Power spectrogram feature with shape (freq_bins, time_frames).
        """
        # Compute STFT
        stft = librosa.stft(
            y=signal.samples, hop_length=self.hop_length, n_fft=self.frame_length
        )

        spec = np.abs(stft) ** 2  # Store magnitude/power spectrogram

        spec_sr = signal.sample_rate / self.hop_length

        return Spectrogram(spec, spec_sr)


class ChromagramBuilder(BuilderFromSignal):
    """Builder for extracting chromagram features for harmonic analysis.

    ChromagramBuilder computes 12-dimensional chroma features that represent the
    relative intensity of each of the 12 pitch classes (C, C#, D, etc.) over time.
    Chromagrams are essential for music analysis tasks involving harmony, chord
    recognition, and key estimation.

    Key Features:
        - 12-bin chroma representation for pitch class analysis
        - STFT-based computation with configurable parameters
        - Automatic octave folding and pitch class mapping
        - Robust to tempo and timbre variations
        - Suitable for harmony-based music analysis

    Algorithm Details:
        1. Compute STFT with specified frame and hop lengths
        2. Apply chroma mapping to fold frequencies into 12 pitch classes
        3. Aggregate energy across octaves for each pitch class
        4. Return normalized chromagram representation

    Applications:
        - Chord recognition and harmony analysis
        - Key estimation and tonal analysis
        - Music structure analysis and segmentation
        - Audio similarity and cover song detection
        - Music alignment and synchronization

    Attributes:
        frame_length (int): STFT frame length in samples.
        hop_length (int): STFT hop length in samples.

    Example:
        >>> builder = ChromagramBuilder(frame_length=4410, hop_length=2205)
        >>> chromagram = builder.build(audio_signal)
        >>> print(f"Chroma shape: {chromagram.data().shape}")  # (12, time_frames)
        >>> # Visualize dominant pitch classes over time
        >>> dominant_pitches = np.argmax(chromagram.data(), axis=0)
    """

    def __init__(self, frame_length: int = 4410, hop_length: int = 2205):
        """Initialize ChromagramBuilder with STFT parameters.

        Args:
            frame_length (int, optional): STFT frame length in samples.
                                        Affects frequency resolution for pitch estimation.
                                        Default: 4410 (~100ms at 44.1kHz).
            hop_length (int, optional): STFT hop length in samples.
                                      Determines temporal resolution of chroma features.
                                      Default: 2205 (~50ms at 44.1kHz).
        """
        self.frame_length = frame_length
        self.hop_length = hop_length

    def build(self, signal: Signal) -> Chromagram:
        """Extract chromagram features from audio signal.

        Computes 12-dimensional chroma features representing the relative
        intensity of each pitch class over time using STFT-based analysis.

        Args:
            signal (Signal): Input audio signal with samples and sample rate.

        Returns:
            Chromagram: Chroma feature with shape (12, time_frames) where each
                       row corresponds to a pitch class (C, C#, D, D#, E, F,
                       F#, G, G#, A, A#, B).
        """
        chroma = librosa.feature.chroma_stft(
            y=signal.samples,
            sr=signal.sample_rate,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        chroma_sr = signal.sample_rate / self.hop_length
        return Chromagram(chroma, chroma_sr)


class MFCCBuilder(BuilderFromSignal):
    """Builder for extracting Mel-Frequency Cepstral Coefficients for timbre analysis.

    MFCCBuilder computes MFCC features that capture the spectral shape characteristics
    of audio signals, making them particularly effective for speech recognition,
    music genre classification, and timbre analysis. MFCCs model human auditory
    perception through mel-scale frequency warping.

    Key Features:
        - Configurable number of MFCC coefficients (typically 12-13)
        - Mel-scale frequency warping for perceptual relevance
        - DCT-based decorrelation for compact representation
        - Robust spectral shape characterization
        - Industry standard for audio classification tasks

    Algorithm Pipeline:
        1. Compute power spectrogram via STFT
        2. Apply mel-scale filterbank to aggregate spectral energy
        3. Take logarithm for dynamic range compression
        4. Apply Discrete Cosine Transform (DCT) for decorrelation
        5. Retain first n_mfcc coefficients as features

    MFCC Coefficient Interpretation:
        - C0: Log energy (often excluded in speech processing)
        - C1-C2: Broad spectral shape and slope
        - C3-C12: Fine spectral details and timbre characteristics
        - Higher coefficients: More detailed spectral information

    Attributes:
        n_mfcc (int): Number of MFCC coefficients to extract.
        frame_length (int): STFT frame length in samples.
        hop_length (int): STFT hop length in samples.

    Example:
        >>> # Standard configuration for music analysis
        >>> builder = MFCCBuilder(n_mfcc=13, frame_length=4410, hop_length=2205)
        >>> mfcc_features = builder.build(audio_signal)
        >>> print(f"MFCC shape: {mfcc_features.data().shape}")  # (13, time_frames)
        >>>
        >>> # First coefficient often represents energy
        >>> energy = mfcc_features.data()[0, :]
        >>> spectral_shape = mfcc_features.data()[1:, :]  # Exclude C0

    Applications:
        - Audio classification and genre recognition
        - Speaker identification and verification
        - Music similarity and recommendation
        - Audio content analysis and indexing
        - Timbre-based music analysis
    """

    def __init__(
        self, n_mfcc: int = 20, frame_length: int = 4410, hop_length: int = 2205
    ):
        """Initialize MFCCBuilder with feature extraction parameters.

        Args:
            n_mfcc (int, optional): Number of MFCC coefficients to extract.
                                  Common values: 12-13 for speech, 20+ for music.
                                  Default: 20.
            frame_length (int, optional): STFT frame length in samples.
                                        Affects frequency resolution for spectral analysis.
                                        Default: 4410 (~100ms at 44.1kHz).
            hop_length (int, optional): STFT hop length in samples.
                                      Determines temporal resolution of MFCC features.
                                      Default: 2205 (~50ms at 44.1kHz).
        """
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length

    def build(self, signal: Signal) -> MFCC:
        """Extract MFCC features from audio signal.

        Computes Mel-Frequency Cepstral Coefficients using the standard algorithm
        with mel-scale filterbank and DCT decorrelation for compact spectral
        shape representation.

        Args:
            signal (Signal): Input audio signal with samples and sample rate.

        Returns:
            MFCC: MFCC feature with shape (n_mfcc, time_frames) where each row
                 represents one cepstral coefficient over time.
        """
        mfcc = librosa.feature.mfcc(
            y=signal.samples,
            sr=signal.sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.frame_length,
        )
        mfcc_sr = signal.sample_rate / self.hop_length
        return MFCC(mfcc, mfcc_sr)


class TempogramBuilder(BuilderFromSignal):
    """Builder for extracting tempogram features for tempo and rhythm analysis.

    TempogramBuilder computes tempograms that represent tempo information over time
    by analyzing the onset strength function and applying Fourier-based tempo
    estimation. Tempograms are essential for rhythm analysis, beat tracking,
    and tempo-based music information retrieval tasks.

    Key Features:
        - Onset strength-based tempo analysis using librosa
        - Fourier tempogram computation for tempo period detection
        - Normalization for consistent amplitude scaling
        - Configurable windowing parameters for temporal resolution
        - Suitable for tempo variation analysis over time

    Algorithm Pipeline:
        1. Compute onset strength function from audio signal
        2. Apply tempogram analysis using Fourier-based period detection
        3. Normalize tempogram values for consistent scaling
        4. Calculate feature sampling rate from hop length
        5. Return Tempogram object with tempo-time representation

    Applications:
        - Beat tracking and tempo estimation
        - Rhythm pattern analysis and classification
        - Music structure analysis based on rhythmic content
        - Tempo variation detection in performances
        - Dance music analysis and BPM estimation

    Attributes:
        frame_length (int): Analysis frame length in samples.
        hop_length (int): Hop length between analysis frames.

    Example:
        >>> builder = TempogramBuilder(frame_length=4410, hop_length=2205)
        >>> tempogram = builder.build(audio_signal)
        >>> print(f"Tempogram shape: {tempogram.data().shape}")
        >>> # (tempo_bins, time_frames) - tempo periods vs. time
    """

    def __init__(self, frame_length: int = 4410, hop_length: int = 2205):
        """Initialize TempogramBuilder with analysis parameters.

        Args:
            frame_length (int, optional): Analysis frame length in samples.
                                        Affects temporal resolution of tempo analysis.
                                        Default: 4410 (~100ms at 44.1kHz).
            hop_length (int, optional): Hop length between analysis frames.
                                      Determines time resolution of tempogram.
                                      Default: 2205 (~50ms at 44.1kHz).
        """
        self.frame_length = frame_length
        self.hop_length = hop_length

    def build(self, signal: Signal) -> Tempogram:
        """Extract tempogram features from audio signal.

        Computes tempo-time representation by analyzing onset strength patterns
        and applying Fourier-based tempogram analysis with normalization.

        Args:
            signal (Signal): Input audio signal with samples and sample rate.

        Returns:
            Tempogram: Tempo feature with shape (tempo_bins, time_frames)
                      representing tempo periods over time.
        """
        onset_env = librosa.onset.onset_strength(
            y=signal.samples, sr=signal.sample_rate, hop_length=self.hop_length
        )
        tempogram = librosa.util.normalize(
            librosa.feature.tempogram(
                onset_envelope=onset_env,
                sr=signal.sample_rate,
                hop_length=self.hop_length,
            )
        )
        tempogram_sr = signal.sample_rate / self.hop_length
        return Tempogram(tempogram, tempogram_sr)

    pass


# Source
class PLPBuilder(BuilderFromSignal):
    """Builder for Predominant Local Pulse (PLP) features for enhanced rhythm analysis.

    PLPBuilder is designed to extract Predominant Local Pulse features that enhance
    the novelty curve of classical tempograms for improved rhythm and beat tracking.
    This advanced technique focuses on the most prominent rhythmic patterns in
    the audio signal.

    Research Background:
        PLP features aim to improve upon traditional tempogram representations by
        emphasizing the predominant rhythmic patterns while suppressing noise and
        less relevant periodicities. This leads to more robust beat tracking and
        rhythm analysis capabilities.

    Development Status:
        This builder is currently under development and serves as a placeholder
        for future PLP implementation. The specific algorithm may be integrated
        with existing tempogram computation or implemented as a separate feature
        extraction pathway.

    Source Reference:
        Based on research available at:
        https://diglib.eg.org/server/api/core/bitstreams/fa7f1cad-a80e-423c-b866-0426908d4dba/content

    Design Considerations:
        The relationship between PLP and Tempogram features is under evaluation.
        This could be implemented as:
        - A separate signal-based builder (current approach)
        - A tempogram-based transformation (alternative approach)
        - An enhanced tempogram computation method

    Note:
        Implementation is not yet complete. Use TempogramBuilder for current
        tempo and rhythm analysis requirements.
    """

    def __init__(self):
        """Initialize PLPBuilder with default parameters.

        Note:
            Specific parameters will be defined once the PLP algorithm
            implementation is finalized.
        """
        super().__init__()

    def build(self, signal: Signal) -> BaseFeature:
        """Extract PLP features from audio signal.

        Args:
            signal (Signal): Input audio signal for PLP extraction.

        Returns:
            BaseFeature: PLP feature representation.

        Raises:
            NotImplementedError: PLP feature extraction is not yet implemented.

        Note:
            This method will be implemented once the PLP algorithm is finalized.
        """
        raise NotImplementedError("PLP feature extraction not implemented yet.")


class SilenceCurveBuilder(BuilderFromSignal):
    """Builder for extracting silence curves using multiple detection algorithms.

    SilenceCurveBuilder implements sophisticated silence detection algorithms that
    compute time-varying measures of audio activity. These features are essential
    for audio segmentation, voice activity detection, and structural analysis of
    audio content with varying dynamic ranges.

    Key Features:
        - Multiple silence detection algorithms (amplitude and spectral)
        - Time-domain RMS energy analysis for amplitude-based detection
        - Frequency-domain spectral energy analysis for spectral detection
        - Configurable windowing parameters for temporal resolution
        - Logarithmic scaling for enhanced dynamic range representation
        - Inverted output for silence curve interpretation (higher = more silent)

    Algorithm Types:
        1. Amplitude-based: Uses RMS energy in time domain
           - Fast computation, direct amplitude measurement
           - Effective for general audio activity detection
           - Recommended for most applications

        2. Spectral-based: Uses spectral energy in frequency domain
           - More sophisticated frequency content analysis
           - Better for distinguishing noise from meaningful content
           - Useful for complex audio mixtures

    Output Interpretation:
        Silence curves use negative energy values where:
        - More negative values indicate higher activity (less silence)
        - Less negative values indicate lower activity (more silence)
        - This inversion allows silence detection thresholding

    Attributes:
        silence_type (str): Type of silence detection algorithm.
        frame_length (int): Analysis frame length in samples.
        hop_length (int): Hop length between analysis frames.

    Example:
        >>> # Amplitude-based silence detection (recommended)
        >>> builder = SilenceCurveBuilder(
        ...     silence_type="amplitude",
        ...     frame_length=4410,
        ...     hop_length=2205
        ... )
        >>> silence_curve = builder.build(audio_signal)
        >>>
        >>> # Find silent regions (less negative values)
        >>> silence_threshold = -0.01
        >>> silent_frames = silence_curve.data() > silence_threshold

    Applications:
        - Audio segmentation and boundary detection
        - Voice activity detection in speech processing
        - Music structure analysis (verse/chorus transitions)
        - Noise reduction preprocessing
        - Audio content analysis and indexing
    """

    def __init__(self, silence_type="amplitude", frame_length=4410, hop_length=2205):
        """Initialize SilenceCurveBuilder with detection algorithm and parameters.

        Args:
            silence_type (str, optional): Type of silence detection method.
                                        Options:
                                        - 'amplitude': RMS energy in time domain (recommended)
                                        - 'spectral': Spectral energy in frequency domain
                                        Default: 'amplitude'.
            frame_length (int, optional): Frame length for analysis windows in samples.
                                        Affects temporal resolution of silence detection.
                                        Default: 4410 (~100ms at 44.1kHz).
            hop_length (int, optional): Hop length between analysis frames in samples.
                                      Determines time resolution of silence curve.
                                      Default: 2205 (~50ms at 44.1kHz).
        """
        super().__init__()
        self.silence_type = silence_type
        self.frame_length = frame_length
        self.hop_length = hop_length

    def compute_amplitude_silence(self, signal):
        """Compute silence curve based on RMS energy in time domain.

        Uses Root Mean Square (RMS) energy computation over windowed frames
        to detect audio activity. This method is computationally efficient
        and effective for general-purpose silence detection.

        Algorithm:
            1. Compute RMS energy per frame using librosa
            2. Invert values (negative RMS) for silence curve interpretation
            3. Higher values indicate more likely silence regions

        Args:
            signal (Signal): Input audio signal for analysis.

        Returns:
            np.ndarray: Amplitude-based silence curve with inverted RMS values.
        """
        # Compute RMS energy per frame
        rms = librosa.feature.rms(
            y=signal.samples, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        # Negative RMS for silence curve (higher values = more likely to be silence)
        return -1 * rms

    def compute_spectral_silence(self, signal):
        """Compute silence curve based on spectral energy in frequency domain.

        Uses Short-Time Fourier Transform (STFT) to analyze frequency content
        and compute total spectral energy per frame. This method provides
        better discrimination between noise and meaningful audio content.

        Algorithm:
            1. Compute STFT with Hann windowing for frequency analysis
            2. Calculate power spectrum (magnitude squared)
            3. Sum energy across all frequency bins per frame
            4. Apply logarithmic scaling for enhanced dynamic range
            5. Invert values for silence curve interpretation

        Args:
            signal (Signal): Input audio signal for spectral analysis.

        Returns:
            np.ndarray: Spectral-based silence curve with log-scaled inverted energy.
        """
        # Compute STFT to get frequency domain representation
        stft = librosa.stft(
            y=signal.samples,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
            window="hann",
        )

        # Compute power spectrum (magnitude squared)
        power_spectrum = np.abs(stft) ** 2

        # Sum across frequency bins to get total spectral energy per frame
        spectral_energy = np.sum(power_spectrum, axis=0)

        # Apply log scaling to compress dynamic range (similar to dB scale)
        spectral_energy_log = np.log1p(spectral_energy)  # log1p for numerical stability

        # Negative energy for silence curve (higher values = more likely to be silence)
        return -1 * spectral_energy_log

    def build(self, signal: Signal) -> SilenceCurve:
        """Extract silence curve features using the configured detection algorithm.

        Applies the specified silence detection method to compute a time-varying
        measure of audio activity suitable for segmentation and analysis tasks.

        Args:
            signal (Signal): Input audio signal for silence detection.

        Returns:
            SilenceCurve: Silence feature with shape (1, time_frames) containing
                         silence probability measures over time.

        Raises:
            ValueError: If silence_type is not supported.

        Example:
            >>> builder = SilenceCurveBuilder(silence_type="amplitude")
            >>> silence_curve = builder.build(audio_signal)
            >>> # Detect silent regions
            >>> threshold = -0.01
            >>> silent_regions = silence_curve.data() > threshold
        """
        if self.silence_type == "amplitude":
            sc = SilenceCurve(
                self.compute_amplitude_silence(signal),
                signal.sample_rate / self.hop_length,
            )
            return sc
        elif self.silence_type == "spectral":
            sc = SilenceCurve(
                self.compute_spectral_silence(signal),
                signal.sample_rate / self.hop_length,
            )
            return sc
        else:
            raise ValueError(
                f"Unsupported silence_type '{self.silence_type}'. Supported types: 'amplitude', 'spectral'"
            )


class HRPSBuilder(BuilderFromSignal):
    """Builder for Harmonic-Residual-Percussive Source Separation with local energy analysis.

    HRPSBuilder implements advanced source separation using median filtering techniques
    to decompose audio signals into harmonic, residual, and percussive components,
    followed by local energy computation for each component. This sophisticated approach
    enables detailed analysis of musical texture and instrument separation.

    Algorithm Overview:
        The HRPS technique uses two-dimensional median filtering on the power spectrogram
        to separate different types of audio content based on their spectro-temporal
        characteristics:

        1. Harmonic Content: Filtered horizontally (across time)
           - Sustained, tonal content like strings and vocals
           - Horizontal filtering preserves frequency consistency

        2. Percussive Content: Filtered vertically (across frequency)
           - Transient, broadband content like drums and attacks
           - Vertical filtering preserves temporal precision

        3. Residual Content: Everything not classified as harmonic or percussive
           - Noise, room tone, and unclassified content
           - Computed as the complement of harmonic and percussive masks

    Key Features:
        - Harmonic/Percussive/Residual source separation using median filtering
        - Configurable filter lengths for different separation characteristics
        - Optional downsampling for computational efficiency
        - Soft masking with configurable beta parameter
        - Local energy computation for each separated component
        - Progress reporting for long-running operations
        - Memory-efficient processing with optional optimization modes

    Mathematical Foundation:
        - Uses 2D median filtering: Y_h = median(Y, [1, L_h]), Y_p = median(Y, [L_p, 1])
        - Masking: M_h = (Y_h >= β * Y_p), M_p = (Y_p > β * Y_h), M_r = 1 - M_h - M_p
        - Reconstruction: X_separated = X * M_component
        - Local energy: E[i] = Σ(x[i*hop:(i+1)*hop]²)

    Performance Optimizations:
        - Optional downsampling reduces computational load
        - Smoothing kernel for downsampled processing
        - Progress reporting for user feedback
        - Memory-efficient ISTFT reconstruction

    Attributes:
        L_h_frames (int): Horizontal median filter length (odd number).
        L_p_bins (int): Vertical median filter length (odd number).
        beta (float): Soft masking exponent for component separation.
        downsampling_factor (int): Temporal downsampling factor for efficiency.
        frame_length (int): STFT frame length in samples.
        hop_length (int): STFT hop length in samples.

    Example:
        >>> # Standard HRPS configuration
        >>> builder = HRPSBuilder(
        ...     L_h_frames=31,      # Harmonic filter length
        ...     L_p_bins=31,        # Percussive filter length
        ...     beta=2.0,           # Soft masking parameter
        ...     downsampling_factor=1  # No downsampling
        ... )
        >>> hrps_features = builder.build(audio_signal)
        >>> print(f"HRPS shape: {hrps_features.data().shape}")  # (3, time_frames)
        >>>
        >>> # Extract individual components
        >>> harmonic_energy = hrps_features.data()[0, :]
        >>> residual_energy = hrps_features.data()[1, :]
        >>> percussive_energy = hrps_features.data()[2, :]

    Applications:
        - Music source separation and analysis
        - Instrument-specific feature extraction
        - Rhythm vs. melody analysis
        - Audio restoration and enhancement
        - Music transcription preprocessing
        - Texture analysis in audio content

    Performance Notes:
        - Processing time scales with audio length and filter sizes
        - Downsampling provides significant speedup with slight quality loss
        - Memory usage peaks during median filtering operations
        - Progress reporting helps track long-running operations
    """

    def __init__(
        self,
        L_h_frames,
        L_p_bins,
        beta: float = 1,
        downsampling_factor: int = 1,
        frame_length: int = 4410,
        hop_length: int = 2205,
    ):
        """Initialize HRPSBuilder with source separation and analysis parameters.

        Configures the Harmonic-Residual-Percussive source separation algorithm
        with median filter lengths, masking parameters, and processing options
        for optimal separation quality and computational efficiency.

        Filter Length Guidelines:
            - L_h_frames (harmonic): Larger values better separate sustained tones
            - L_p_bins (percussive): Larger values better separate transients
            - Both values are automatically adjusted to be odd for proper centering
            - Typical ranges: 15-51 frames/bins depending on audio characteristics

        Beta Parameter Effects:
            - beta = 1: Binary masking (hard separation)
            - beta > 1: Softer masking (more conservative separation)
            - beta < 1: Aggressive masking (stronger separation)
            - Recommended range: 0.5 - 2.0

        Downsampling Trade-offs:
            - Factor = 1: Full resolution, highest quality, slowest
            - Factor > 1: Reduced resolution, faster processing, slight quality loss
            - Recommended for long audio files or real-time applications

        Args:
            L_h_frames (int): Length of median filter for harmonic components (in frames).
                            Will be adjusted to nearest odd number for proper centering.
                            Typical range: 15-51.
            L_p_bins (int): Length of median filter for percussive components (in bins).
                          Will be adjusted to nearest odd number for proper centering.
                          Typical range: 15-51.
            beta (float, optional): Exponent for soft masking control.
                                  beta=1 gives binary masking, higher values give
                                  softer transitions. Default: 1.
            downsampling_factor (int, optional): Temporal downsampling factor for
                                               computational efficiency. Factor of N
                                               processes every Nth frame. Default: 1.
            frame_length (int, optional): STFT frame length in samples.
                                        Affects frequency resolution. Default: 4410.
            hop_length (int, optional): STFT hop length in samples.
                                      Affects time resolution. Default: 2205.

        Example:
            >>> # High-quality separation (slower)
            >>> builder = HRPSBuilder(L_h_frames=31, L_p_bins=31, beta=2.0)
            >>>
            >>> # Fast processing with downsampling
            >>> builder = HRPSBuilder(
            ...     L_h_frames=21, L_p_bins=21,
            ...     beta=1.0, downsampling_factor=4
            ... )
        """
        self.L_h_frames = L_h_frames + ((L_h_frames + 1) % 2)  # Ensure odd length
        self.L_p_bins = L_p_bins + ((L_p_bins + 1) % 2)  # Ensure odd length
        self.beta = beta
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.downsampling_factor = downsampling_factor

    def compute_signals(self, signal: Signal) -> tuple[Signal, Signal, Signal]:
        """Perform complete HRPS decomposition returning separated signal components.

        Executes the full Harmonic-Residual-Percussive source separation pipeline,
        including STFT computation, median filtering, masking, and inverse STFT
        reconstruction to produce three separated audio signals.

        Processing Pipeline:
            1. STFT computation with Hann windowing
            2. Optional downsampling with smoothing for efficiency
            3. Power spectrogram calculation
            4. Median filtering for harmonic and percussive components
            5. Soft masking with configurable beta parameter
            6. Residual mask computation as complement
            7. ISTFT reconstruction for all three components

        Performance Monitoring:
            The method includes progress reporting and timing information
            for monitoring long-running operations on large audio files.

        Args:
            signal (Signal): Input audio signal for source separation.

        Returns:
            tuple[Signal, Signal, Signal]: Tuple of (harmonic, residual, percussive)
                                         separated signals with same length as input.

        Example:
            >>> builder = HRPSBuilder(L_h_frames=31, L_p_bins=31)
            >>> harmonic, residual, percussive = builder.compute_signals(audio_signal)
            >>> # Analyze each component separately
            >>> harmonic.plot()  # Visualize harmonic content

        Note:
            This method performs complete signal reconstruction and is computationally
            intensive. Use build() method for energy-based feature extraction.
        """
        print("Computing STFT...", end="\r")
        X = librosa.stft(
            y=signal.samples,
            hop_length=self.hop_length,
            n_fft=self.frame_length,
            win_length=self.frame_length,
            window="hann",
            center=True,
            pad_mode="constant",
        )

        if self.downsampling_factor > 1:
            self.hop_length *= self.downsampling_factor
            filter_length = 101
            kernel = np.ones(filter_length) / filter_length
            X_smooth = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=X
            )
            # Downsample (take every Nth frame)
            X = X_smooth[:, :: self.downsampling_factor]

        Y = np.abs(X) ** 2  # Power spectrogram

        print("Computed STFT                    \n")

        # Data of a Spectrogram is always complex-valued in linear scale

        print("Applying Median for Harmonic Component (1/2)")
        import time

        start = time.time()
        print("Y.shape:", Y.shape)
        Y_h = scipy.signal.medfilt2d(Y, [1, self.L_h_frames])
        middle = time.time()
        print(f"Median filtering took {middle - start:.2f} seconds")
        print("Applying Median for Percussive Component (2/2)")
        Y_p = scipy.signal.medfilt2d(Y, [self.L_p_bins, 1])
        end = time.time()
        print(f"Median filtering took {end - middle:.2f} seconds")
        print("\n")

        # Masking
        print("Computing Masks...", end="\r")
        M_h = np.int8(Y_h >= self.beta * Y_p)
        M_p = np.int8(Y_p > self.beta * Y_h)
        M_r = 1 - (M_h + M_p)
        X_h = X * M_h
        X_p = X * M_p
        X_r = X * M_r
        print("Computed Masks                    \n")

        # istft
        print("Computing Inverse STFT for x_h (1/3)")
        x_h = librosa.istft(
            X_h,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window="hann",
            center=True,
            length=len(signal.samples),
        )
        print("Computing Inverse STFT for x_r (2/3)")
        x_r = librosa.istft(
            X_r,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window="hann",
            center=True,
            length=len(signal.samples),
        )
        print("Computing Inverse STFT for x_p (3/3)")
        x_p = librosa.istft(
            X_p,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window="hann",
            center=True,
            length=len(signal.samples),
        )
        print("\n")

        return (
            Signal(x_h, signal.sample_rate, ""),
            Signal(x_r, signal.sample_rate, ""),
            Signal(x_p, signal.sample_rate, ""),
        )

    def compute_harmonic_signal(self, signal: Signal) -> Signal:
        print("Computing STFT...", end="\r")
        X = librosa.stft(
            y=signal.samples,
            hop_length=self.hop_length,
            n_fft=self.frame_length,
            win_length=self.frame_length,
            window="hann",
            center=True,
            pad_mode="constant",
        )

        import time

        start = time.time()

        if self.downsampling_factor > 1:
            self.hop_length *= self.downsampling_factor
            filter_length = 101
            kernel = np.ones(filter_length) / filter_length
            X_smooth = np.apply_along_axis(
                lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=X
            )
            # Downsample (take every Nth frame)
            X = X_smooth[:, :: self.downsampling_factor]

        Y = np.abs(X) ** 2  # Power spectrogram

        print("Computed STFT                    \n")

        print("Y.shape:", Y.shape)
        print("Applying Median for Harmonic Component (1/2)")
        Y_h = scipy.signal.medfilt2d(Y, [1, self.L_h_frames])
        print("Applying Median for Percussive Component (2/2)")
        Y_p = scipy.signal.medfilt2d(Y, [self.L_p_bins, 1])

        # Masking
        print("Computing Masks...", end="\r")
        M_h = np.int8(Y_h >= self.beta * Y_p)
        M_p = np.int8(Y_p > self.beta * Y_h)
        M_r = 1 - (M_h + M_p)
        X_r = X * M_r
        print("Computed Masks                    \n")

        print("Computing Inverse STFT for x_h only (1/1)")
        x_r = librosa.istft(
            X_r,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            window="hann",
            center=True,
            length=len(signal.samples),
        )

        end = time.time()
        print(
            f"Total processing with downsampling {self.downsampling_factor} took {end - start:.2f} seconds"
        )

        return Signal(x_r, signal.sample_rate, "")

    def compute_local_energy(self, signal: Signal) -> np.ndarray:
        """Compute local energy for each frame using sliding window technique.

        Calculates the energy (sum of squared samples) for each analysis frame
        of the input signal using a sliding window approach. This provides a
        time-varying measure of signal intensity suitable for rhythm analysis
        and onset detection.

        Algorithm:
            1. Divide signal into overlapping frames based on hop_length
            2. For each frame, extract frame_length samples
            3. Compute energy as sum of squared amplitudes
            4. Return energy time series aligned with STFT frames

        Temporal Alignment:
            The energy frames are computed to align with STFT analysis frames,
            ensuring consistency between spectral and energy-based features.
            Frame boundaries are calculated using the same hop_length parameter.

        Args:
            signal (Signal): Input audio signal for local energy computation.
                           Should be one of the separated components (H, R, or P).

        Returns:
            np.ndarray: Array of local energy values with shape (n_frames,)
                       where n_frames matches the temporal resolution of
                       corresponding STFT analysis.

        Mathematical Formula:
            For frame i: energy[i] = Σ(x[i*hop_length : i*hop_length + frame_length]²)

        Example:
            >>> builder = HRPSBuilder(L_h_frames=31, L_p_bins=31)
            >>> harmonic_signal, _, _ = builder.compute_signals(audio_signal)
            >>> harmonic_energy = builder.compute_local_energy(harmonic_signal)
            >>> print(f"Energy shape: {harmonic_energy.shape}")  # (n_frames,)

        Note:
            This method is used internally by build() to compute energy features
            for each separated component. The sliding window approach ensures
            proper temporal alignment with spectral features.
        """
        # Get signal samples
        x = signal.samples

        # Calculate number of frames
        n_frames = 1 + (len(x) - self.frame_length) // self.hop_length

        # Initialize energy array
        local_energy = np.zeros(n_frames)

        # Compute local energy for each frame using sliding window
        for i in range(n_frames):
            # Calculate frame boundaries
            start = i * self.hop_length
            end = start + self.frame_length

            # Extract frame
            frame = x[start:end]

            # Compute energy as sum of squared samples
            local_energy[i] = np.sum(frame**2)

        return local_energy

    def build(self, signal: Signal) -> HRPS:
        """Extract HRPS features with complete three-component separation.

        Performs full Harmonic-Residual-Percussive source separation followed
        by local energy computation for each component. This provides a
        comprehensive 3-dimensional feature representation suitable for
        advanced music analysis tasks.

        Processing Steps:
            1. Complete HRPS source separation using compute_signals()
            2. Local energy computation for harmonic component
            3. Local energy computation for residual component
            4. Local energy computation for percussive component
            5. Stack energies into 3D feature matrix

        Args:
            signal (Signal): Input audio signal for HRPS feature extraction.

        Returns:
            HRPS: HRPS feature with shape (3, time_frames) where:
                  - Row 0: Harmonic component energy
                  - Row 1: Residual component energy
                  - Row 2: Percussive component energy

        Example:
            >>> builder = HRPSBuilder(L_h_frames=31, L_p_bins=31)
            >>> hrps = builder.build(audio_signal)
            >>> harmonic_energy = hrps.data()[0, :]
            >>> residual_energy = hrps.data()[1, :]
            >>> percussive_energy = hrps.data()[2, :]

        Note:
            This method performs complete separation and is computationally
            intensive. For faster processing with reduced quality, consider
            using build_only_residual() or increasing downsampling_factor.
        """
        sh, sr, sp = self.compute_signals(signal)

        print("Computing Local Energy for Harmonic Signal (1/3)")
        le_h = self.compute_local_energy(sh)

        print("Computing Local Energy for Residual Signal (2/3)")
        le_r = self.compute_local_energy(sr)

        print("Computing Local Energy for Percussive Signal (3/3)")
        le_p = self.compute_local_energy(sp)
        print("\n")

        stacked_energy = np.vstack([le_h, le_r, le_p])

        hrps_sr = signal.sample_rate / self.hop_length
        return HRPS(stacked_energy, hrps_sr)

    def build_only_residual(self, signal: Signal) -> HRPS:
        """Extract HRPS features using only residual component for efficiency.

        Computes only the residual component of HRPS separation for faster
        processing while maintaining the same output format. This optimization
        is useful when full separation is not required or computational
        resources are limited.

        Processing Steps:
            1. Partial HRPS processing focused on residual component
            2. Local energy computation for residual component only
            3. Replicate residual energy for all three components

        Args:
            signal (Signal): Input audio signal for residual-based HRPS extraction.

        Returns:
            HRPS: HRPS feature with shape (3, time_frames) where all three rows
                  contain the same residual component energy for consistency
                  with full HRPS format.

        Example:
            >>> builder = HRPSBuilder(L_h_frames=31, L_p_bins=31)
            >>> hrps_fast = builder.build_only_residual(audio_signal)
            >>> # All components will have identical residual energy

        Note:
            This method provides significant computational savings but loses
            the detailed component separation. Use when processing speed is
            more important than separation quality.
        """
        s_r = self.compute_harmonic_signal(signal)

        print("Computing Local Energy for Residual Signal")
        le_r = self.compute_local_energy(s_r)
        print("\n")

        stacked_energy = np.vstack([le_r, le_r, le_r])

        hrps_sr = signal.sample_rate / self.hop_length
        return HRPS(stacked_energy, hrps_sr)


class SSMBuilder(BuilderFromBaseFeature):
    """Builder for Self-Similarity Matrix computation from base features.

    SSMBuilder constructs Self-Similarity Matrices that reveal repetitive structures
    and patterns within audio signals by computing pairwise similarities between
    feature vectors at different time instances. These matrices are fundamental
    for music structure analysis and pattern detection.

    Key Features:
        - Time-invariant self-similarity computation using libfmp
        - Configurable smoothing filters for noise reduction
        - Support for tempo-relative analysis with multiple tempo ratios
        - Shift-invariant processing for robust pattern detection
        - Flexible filtering directions (forward, backward, both)

    Algorithm Details:
        The SSM computation involves:
        1. Feature vector extraction from base feature at each time frame
        2. Pairwise similarity computation between all frame pairs
        3. Optional smoothing filter application for noise reduction
        4. Tempo-relative and shift-set processing for robustness

    Matrix Interpretation:
        - Diagonal elements: Self-similarity (always maximum)
        - Off-diagonal elements: Cross-similarities between different time points
        - Block structures: Repetitive sections (verses, choruses)
        - Stripe patterns: Sequential similarities (gradual changes)

    Attributes:
        smoothing_filter_length (int): Length of smoothing filter for noise reduction.
        smoothing_filter_direction (int): Direction code for filter application.
        shift_set (np.ndarray): Array of shift values for robustness.
        tempo_relative_set (np.ndarray): Array of tempo ratios for analysis.

    Example:
        >>> # Build SSM from chromagram with smoothing
        >>> builder = SSMBuilder(
        ...     smoothing_filter_length=5,
        ...     smoothing_filter_direction="both",
        ...     tempo_relative_set=np.array([0.5, 1.0, 2.0])
        ... )
        >>> ssm = builder.build(chromagram_feature)
        >>> print(f"SSM shape: {ssm.data().shape}")  # (time_frames, time_frames)

    Applications:
        - Music structure analysis and segmentation
        - Repetition detection in audio content
        - Pattern matching and similarity search
        - Audio thumbnailing and summarization
        - Cover song detection and version identification
    """

    def __init__(
        self,
        smoothing_filter_length: int = 1,
        smoothing_filter_direction: str = "both",
        shift_set: np.ndarray = np.array([0]),
        tempo_relative_set: np.ndarray = np.array([1]),
    ):
        """Initialize SSMBuilder with similarity computation parameters.

        Args:
            smoothing_filter_length (int, optional): Length of smoothing filter
                                                    for noise reduction. Default: 1.
            smoothing_filter_direction (str, optional): Filter direction.
                                                       Options: "both", "backward", "forward".
                                                       Default: "both".
            shift_set (np.ndarray, optional): Array of shift values for robustness.
                                            Default: np.array([0]).
            tempo_relative_set (np.ndarray, optional): Array of tempo ratios.
                                                     Default: np.array([1]).
        """
        self.smoothing_filter_length = smoothing_filter_length
        self.smoothing_filter_direction = (
            2
            if smoothing_filter_direction == "both"
            else (1 if smoothing_filter_direction == "backward" else 0)
        )
        self.shift_set = shift_set
        self.tempo_relative_set = tempo_relative_set

    def build(self, base_feature: BaseFeature) -> SelfSimilarityMatrix:
        """Compute Self-Similarity Matrix from base feature.

        Args:
            base_feature (BaseFeature): Input feature for SSM computation.

        Returns:
            SelfSimilarityMatrix: Computed similarity matrix with same sampling rate.
        """
        S, _ = src.libfmp.c4.compute_sm_ti(
            base_feature.data(),
            base_feature.data(),
            L=self.smoothing_filter_length,
            tempo_rel_set=self.tempo_relative_set,
            shift_set=self.shift_set,
            direction=self.smoothing_filter_direction,
        )

        return SelfSimilarityMatrix(S, base_feature.sampling_rate())


class TLMBuilder(BuilderFromSimilarityMatrix):
    """Builder for Time-Lag Matrix computation from Self-Similarity Matrices.

    TLMBuilder transforms Self-Similarity Matrices into Time-Lag Matrix representations
    that emphasize repetitive structures and periodicities in audio content. This
    transformation is particularly effective for detecting structural patterns
    and recurring musical elements.

    Key Features:
        - Time-lag representation computation using libfmp
        - Circular processing for seamless pattern detection
        - Specialized for Self-Similarity Matrix inputs
        - Enhanced visualization of repetitive structures

    Algorithm Details:
        The Time-Lag Matrix transforms the SSM by reorienting the similarity
        information to emphasize time lags between similar events, making
        repetitive patterns more visually apparent and computationally accessible.

    Matrix Interpretation:
        - Horizontal axis: Time progression through the audio
        - Vertical axis: Time lag between similar events
        - Bright regions: Strong repetitions at specific time lags
        - Horizontal lines: Consistent repetition patterns

    Example:
        >>> # Build TLM from Self-Similarity Matrix
        >>> builder = TLMBuilder()
        >>> tlm = builder.build(self_similarity_matrix)
        >>> print(f"TLM shape: {tlm.data().shape}")

    Applications:
        - Enhanced visualization of repetitive structures
        - Structural pattern analysis in music
        - Automated structure annotation
        - Repetition-based audio segmentation
        - Music form analysis (ABAB, verse-chorus, etc.)
    """

    def __init__(self):
        """Initialize TLMBuilder with default parameters.

        Note:
            TLM computation uses fixed parameters optimized for
            typical music structure analysis tasks.
        """
        pass

    def build(self, sm: SimilarityMatrix) -> TimeLagMatrix:
        """Compute Time-Lag Matrix from similarity matrix.

        Args:
            sm (SimilarityMatrix): Input similarity matrix. Must be a
                                  SelfSimilarityMatrix for proper TLM computation.

        Returns:
            TimeLagMatrix: Time-lag representation with same sampling rate.

        Raises:
            ValueError: If input is not a SelfSimilarityMatrix.
        """
        if not isinstance(sm, SelfSimilarityMatrix):
            raise ValueError(
                "For computing Time-Lag Matrix, the input SimilarityMatrix must be a SelfSimilarityMatrix."
            )

        S = src.libfmp.c4.compute_time_lag_representation(sm.data(), circular=True)

        return TimeLagMatrix(S, sm.sampling_rate())
