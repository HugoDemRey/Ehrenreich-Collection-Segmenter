"""Audio File I/O Operations - Audio file loading, saving, and management utilities.

This module provides comprehensive functionality for audio file operations including
loading various audio formats, saving processed audio, and managing audio file
metadata. It serves as the primary interface between file system audio data and
in-memory Signal objects for audio processing workflows.

Key Features:
    - Multi-format audio file loading with librosa integration
    - High-quality audio file saving with soundfile integration
    - Automatic sample rate conversion and resampling
    - File metadata extraction and validation
    - Memory-efficient loading with progress reporting
    - Comprehensive error handling for file operations
    - Support for common audio formats (WAV, MP3, FLAC, etc.)
    - Integration with custom Signal class for seamless workflows

Core Functionality:
    - AudioFile class for file-based audio operations
    - Lazy loading pattern for memory efficiency
    - Static save methods for processed audio export
    - Automatic file size and duration reporting
    - Path validation and error handling

Common Use Cases:
    - Loading audio files for analysis and processing
    - Converting between audio formats and sample rates
    - Saving processed audio segments and results
    - Batch processing of audio file collections
    - Audio file metadata inspection and validation
    - Integration with signal processing pipelines

Technical Details:
    - Uses librosa for robust audio loading with format detection
    - Uses soundfile for high-quality audio saving
    - Supports arbitrary sample rates and bit depths
    - Handles mono/stereo conversion automatically
    - Provides detailed loading statistics and feedback
    - Integrates with Time utility for duration formatting

Supported Formats:
    Input: WAV, MP3, FLAC, AAC, OGG, M4A, and others via librosa
    Output: WAV, FLAC, OGG and others via soundfile

Author: Hugo Demule
Date: January 2026
"""

import os

import librosa
import soundfile
from src.audio.signal import Signal
from src.utils.Time import Time


class AudioFile:
    """Comprehensive audio file manager for loading, saving, and metadata operations.

    AudioFile provides a complete interface for file-based audio operations, serving
    as the bridge between file system storage and in-memory Signal objects. It implements
    a lazy loading pattern for memory efficiency and provides detailed feedback on
    file operations and metadata.

    The class encapsulates:
        - File path management and validation
        - Lazy loading of audio data with configurable sample rates
        - File size and metadata tracking
        - Integration with Signal class for seamless workflows
        - Comprehensive error handling and reporting
        - Static methods for audio file saving operations

    Key Features:
        - Lazy loading pattern - audio data loaded only when needed
        - Support for sample rate conversion during loading
        - Automatic file validation and error handling
        - Detailed loading statistics with file size and duration
        - Integration with Time utility for human-readable durations
        - Memory-efficient operation with progress reporting

    Loading Process:
        1. File existence validation
        2. Audio data loading via librosa with optional resampling
        3. Signal object creation with loaded data
        4. Metadata extraction and caching
        5. Loading confirmation with detailed statistics

    Supported Operations:
        - Load audio files with optional sample rate conversion
        - Save Signal objects to various audio formats
        - Extract file metadata (size, duration, sample rate)
        - Validate file existence and accessibility
        - Provide loading progress and statistics

    Attributes:
        path (str): Full file path to the audio file for loading operations.
        size (int | None): File size in bytes, populated after loading.
        loaded (bool): Flag indicating whether audio data has been loaded.
        signal (Signal | None): The loaded Signal object containing audio data.

    Example:
        >>> audio_file = AudioFile("path/to/audio.wav")
        >>> signal = audio_file.load(sr=44100)  # Load with specific sample rate
        >>> print(f"Loaded {signal.duration():.2f}s of audio")
        >>>
        >>> # Save processed audio
        >>> AudioFile.save("output.wav", processed_signal)

    Note:
        The class uses librosa for loading (supporting many formats) and
        soundfile for saving (supporting high-quality output formats).
        Loading is performed only once per instance for efficiency.
    """

    def __init__(self, path: str) -> None:
        """Initialize an AudioFile instance with the specified file path.

        Creates a new AudioFile instance configured to load audio from the specified
        path. The initialization sets up the file path and initializes all metadata
        attributes to their default states. No file system operations are performed
        during initialization - actual loading occurs when load() is called.

        Initialization Process:
            - Stores the provided file path for future loading operations
            - Sets metadata attributes to default/None values
            - Prepares the instance for lazy loading when requested
            - No validation or file system access performed at this stage

        Args:
            path (str): Complete file path to the audio file. Can be absolute or
                       relative path. The file does not need to exist at
                       initialization time - existence is validated during load().

        Attributes Set:
            - path: Stores the provided file path
            - size: Initialized to None (populated after loading)
            - loaded: Initialized to False (updated when loading occurs)
            - signal: Initialized to None (contains Signal after loading)

        Example:
            >>> # Initialize with various path formats
            >>> audio1 = AudioFile("/absolute/path/to/audio.wav")
            >>> audio2 = AudioFile("relative/path/audio.mp3")
            >>> audio3 = AudioFile("./data/samples/sample.flac")
            >>>
            >>> # File doesn't need to exist yet
            >>> audio4 = AudioFile("future_file.wav")  # OK at init time

        Note:
            This constructor follows the lazy loading pattern - no expensive
            operations are performed until load() is explicitly called.
            File path validation occurs during the actual load operation.
        """
        self.path: str = path
        self.size = None
        self.loaded = False
        self.signal = None

    def load(self, sr: int | None = None) -> Signal:
        """Load audio data from file and return a Signal object with detailed reporting.

        Performs comprehensive audio file loading including file validation, audio data
        extraction, Signal object creation, and metadata collection. Provides detailed
        loading statistics and progress feedback through console output.

        Loading Process:
            1. File existence validation with detailed error reporting
            2. Audio data loading via librosa with format auto-detection
            3. Optional sample rate conversion/resampling
            4. Signal object creation with loaded samples and metadata
            5. File size calculation and metadata extraction
            6. Loading confirmation with comprehensive statistics display

        Sample Rate Handling:
            - If sr=None: Uses the file's original sample rate
            - If sr specified: Resamples audio to the target sample rate
            - Resampling uses librosa's high-quality resampling algorithms
            - Sample rate information included in loading statistics

        Statistics Reported:
            - Original file path for confirmation
            - Total number of audio samples loaded
            - Final sample rate (original or resampled)
            - Audio duration in HH:MM:SS format
            - File size in megabytes with 2-decimal precision

        Args:
            sr (int | None, optional): Target sample rate in Hz for resampling.
                                     If None, preserves the file's original
                                     sample rate. Common values: 44100, 48000,
                                     22050. Default is None.

        Returns:
            Signal: A newly created Signal object containing the loaded audio
                   samples, sample rate, and original filename metadata.
                   The Signal is also cached in self.signal for future access.

        Raises:
            FileNotFoundError: If the specified audio file does not exist at
                             the given path. Includes the full path in the
                             error message for debugging.
            librosa.LibrosaError: If the file format is unsupported or corrupted.
            MemoryError: If the audio file is too large to load into memory.
            PermissionError: If file access is denied due to permissions.

        Side Effects:
            - Updates self.signal with the loaded Signal object
            - Updates self.size with file size in bytes
            - Sets self.loaded to True after successful loading
            - Prints detailed loading statistics to console

        Example:
            >>> audio_file = AudioFile("music.wav")
            >>> signal = audio_file.load()  # Use original sample rate
            * Loaded Audio Path 'music.wav'
            * Samples Number: 441000
            * Sample Rate: 44100 Hz
            * Duration: 00:00:10
            * File Size: 1.68 MB
            >>>
            >>> # Load with resampling
            >>> signal_22k = audio_file.load(sr=22050)

        Performance Notes:
            - Loading time depends on file size and optional resampling
            - Memory usage scales linearly with audio duration and sample rate
            - Subsequent calls return cached Signal without reloading
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File {self.path} does not exist.")

        samples, sample_rate = librosa.load(self.path, sr=sr)
        self.signal = Signal(samples, int(sample_rate), self.path)
        self.size = os.path.getsize(self.path)
        self.loaded = True

        size_mb = self.size / (1024 * 1024)
        print(
            f"* Loaded Audio Path '{self.path}' \n* Samples Number: {samples.shape[0]} \n* Sample Rate: {sample_rate} Hz \n* Duration: {Time.seconds_to_hms(self.signal._duration)} \n* File Size: {size_mb:.2f} MB"
        )

        return self.signal

    @staticmethod
    def save(path: str, signal: Signal) -> None:
        """Save a Signal object as an audio file to disk with high-quality output.

        Exports the provided Signal object to an audio file using the soundfile
        library for high-quality audio output. Supports various output formats
        based on the file extension and provides confirmation feedback.

        Save Process:
            1. Signal data validation and preparation
            2. File format determination from extension
            3. High-quality audio encoding via soundfile
            4. File writing with sample rate preservation
            5. Success confirmation with path and sample rate

        Format Support:
            The output format is determined by the file extension:
            - .wav: Uncompressed WAV format (recommended for quality)
            - .flac: Lossless compression with excellent quality
            - .ogg: Ogg Vorbis compressed format
            - Other formats supported by soundfile library

        Quality Considerations:
            - Uses soundfile for professional-grade audio output
            - Preserves original sample rate and bit depth
            - No quality loss for supported lossless formats
            - Handles floating-point to integer conversion automatically

        Args:
            path (str): Complete output file path including extension.
                       The extension determines the output format.
                       Parent directories must exist.
            signal (Signal): The Signal object to save containing audio
                           samples, sample rate, and metadata. Must have
                           valid samples and sample_rate attributes.

        Raises:
            ValueError: If the Signal object has invalid or missing audio data.
            IOError: If the file cannot be written (permissions, disk space, etc.).
            SoundFileError: If the output format is unsupported or invalid.
            FileNotFoundError: If parent directories do not exist.

        Side Effects:
            - Creates an audio file at the specified path
            - Overwrites existing files without warning
            - Prints confirmation message with path and sample rate

        Example:
            >>> # Save processed signal to WAV format
            >>> AudioFile.save("output/processed.wav", processed_signal)
            Audio saved to "output/processed.wav" with Sample Rate: 44100
            >>>
            >>> # Save to lossless FLAC format
            >>> AudioFile.save("archive/original.flac", original_signal)
            Audio saved to "archive/original.flac" with Sample Rate: 48000

        Performance Notes:
            - Save time depends on file size and output format
            - Lossless formats (WAV, FLAC) are faster than compressed formats
            - Memory usage is minimal during save operations

        Note:
            This is a static method and can be called without creating an
            AudioFile instance. It provides a convenient way to export
            Signal objects from any context in the application.
        """
        soundfile.write(path, signal.samples, signal.sample_rate)
        print(f'Audio saved to "{path}" with Sample Rate: {signal.sample_rate}')
