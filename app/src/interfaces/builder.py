"""Abstract Builder Interface for Feature Construction and Audio Analysis.

This module provides the foundational Builder interface that defines the contract
for constructing complex audio analysis objects and features. The Builder pattern
enables flexible, configurable construction of audio processing components with
varying parameters and initialization requirements.

Key Components:
    - Builder: Abstract base class for all builder implementations
    - Standardized build() interface for consistent object construction
    - Support for flexible parameter passing and configuration
    - Integration with audio signal processing and feature extraction workflows

Design Pattern:
    Implements the Builder design pattern to separate the construction logic
    from the representation of complex audio analysis objects. This approach
    enables:
    - Flexible configuration of audio processing parameters
    - Step-by-step construction of complex feature extractors
    - Consistent interface across different audio analysis components
    - Easy extensibility for new feature types and analysis methods

Integration:
    This interface is implemented by various builder classes throughout the
    audio analysis framework, including:
    - Feature builders (SpectrogramBuilder, ChromagramBuilder, etc.)
    - Analysis builders (SimilarityMatrixBuilder, etc.)
    - Signal processing builders for complex workflows

Author: Hugo Demule
Date: January 2026
"""

from abc import ABC, abstractmethod
from typing import Any


class Builder(ABC):
    """Abstract base class for implementing the Builder design pattern in audio analysis.

    The Builder interface defines the contract for constructing complex audio analysis
    objects with configurable parameters. This pattern separates the construction logic
    from the object representation, enabling flexible and consistent creation of audio
    processing components.

    Key Features:
        - Abstract build method requiring concrete implementation
        - Flexible parameter passing through *args and **kwargs
        - Consistent interface across all audio analysis builders
        - Support for complex, multi-step object construction
        - Integration with dependency injection and configuration patterns

    Design Philosophy:
        The Builder pattern is particularly valuable in audio analysis where:
        - Feature extractors require complex parameter configurations
        - Multiple initialization strategies may be needed for different contexts
        - Step-by-step construction enables validation and optimization
        - Consistent interfaces improve code maintainability and testing

    Implementation Guidelines:
        Concrete builders should:
        1. Accept configuration parameters in their constructor
        2. Validate parameters during initialization or build time
        3. Return fully constructed, ready-to-use objects from build()
        4. Handle error conditions gracefully with informative messages
        5. Support method chaining for fluent interface design

    Example Usage:
        >>> class SpectrogramBuilder(Builder):
        ...     def __init__(self, n_fft=2048, hop_length=512):
        ...         self.n_fft = n_fft
        ...         self.hop_length = hop_length
        ...
        ...     def build(self, audio_signal):
        ...         return Spectrogram.from_audio(
        ...             audio_signal,
        ...             n_fft=self.n_fft,
        ...             hop_length=self.hop_length
        ...         )
        >>>
        >>> builder = SpectrogramBuilder(n_fft=4096, hop_length=1024)
        >>> spectrogram = builder.build(audio_signal)

    Applications:
        - Feature extraction pipeline construction
        - Configurable audio analysis workflows
        - Plugin architectures for audio processing
        - Testing frameworks with parameterized object creation
        - Dynamic construction based on runtime configuration

    Note:
        This is a pure interface class requiring concrete implementation.
        The build method should return objects ready for immediate use
        in audio analysis workflows.
    """

    @abstractmethod
    def build(self, *args, **kwargs) -> Any:
        """Construct and return the target object with specified configuration.

        This abstract method defines the core construction interface that must
        be implemented by all concrete builders. The method should create,
        configure, and return a fully functional object ready for use.

        Args:
            *args: Variable positional arguments passed to the construction process.
                   Typically includes primary data sources like audio signals,
                   feature matrices, or other required inputs.
            **kwargs: Variable keyword arguments for additional configuration.
                     Allows runtime parameter override and extension of builder
                     functionality without changing the interface.

        Returns:
            Any: The constructed object, typically an audio feature, analysis result,
                 or configured processor. The returned object should be fully
                 initialized and ready for immediate use.

        Raises:
            NotImplementedError: Always raised if called directly on the abstract base.
            ValueError: Should be raised by implementations for invalid parameters.
            TypeError: Should be raised by implementations for incompatible input types.

        Implementation Notes:
            - Validate all input parameters before construction
            - Provide clear error messages for configuration problems
            - Ensure returned objects are in a consistent, usable state
            - Consider caching expensive computations where appropriate
            - Log construction details for debugging when relevant

        Example Implementation:
            >>> def build(self, audio_signal, **overrides):
            ...     # Validate inputs
            ...     if not isinstance(audio_signal, Signal):
            ...         raise TypeError("Expected Signal object")
            ...
            ...     # Apply parameter overrides
            ...     config = {**self.default_config, **overrides}
            ...
            ...     # Construct and return object
            ...     return FeatureExtractor(audio_signal, **config)
        """
        pass
