"""Model for segmentation analysis with novelty curve processing.

Provides segmentation analysis capabilities using novelty curves,
transition detection, and interactive segmentation management.

Author: Hugo Demule
Date: January 2026
"""

import numpy as np
from constants.parameters import NCParameters
from constants.segmenter_configurations import SegmenterConfig
from models.analysis_page.components.interactable_plot_m import InteractablePlotModel
from src.audio.signal import Signal
from src.audio_features.builders import BuilderFromSignal, SSMBuilder
from src.audio_features.features import BaseFeature, NoveltyCurve, SelfSimilarityMatrix


class SegmenterModel(InteractablePlotModel):
    """Model for segmentation analysis with transition detection.

    Handles novelty curve computation, peak detection, and transition
    management for interactive audio segmentation analysis.
    """

    def __init__(self, signal: Signal, config: SegmenterConfig, id: str):
        super().__init__(signal=signal, id=id)
        self.config = config

    def setup_builder(
        self, params: NCParameters, config: SegmenterConfig
    ) -> BuilderFromSignal:
        """Setup the builder based on the configuration and parameters."""
        print(
            f"(MODEL): Setting up builder for config {config} with params: {params.__dict__}"
        )
        if config == SegmenterConfig.CHROMAGRAM:
            from src.audio_features.builders import ChromagramBuilder

            builder = ChromagramBuilder(
                frame_length=params.frame_length,
                hop_length=params.hop_length,
            )
        elif config == SegmenterConfig.MFCC:
            from src.audio_features.builders import MFCCBuilder

            builder = MFCCBuilder(
                frame_length=params.frame_length,
                hop_length=params.hop_length,
            )
        elif config == SegmenterConfig.TEMPOGRAM:
            from src.audio_features.builders import TempogramBuilder

            builder = TempogramBuilder(
                frame_length=params.frame_length,
                hop_length=params.hop_length,
            )
        else:
            raise ValueError(f"Unsupported configuration: {config}")

        self.builder = builder
        return builder

    def compute_base_feature(self, params: NCParameters) -> BaseFeature:
        """Compute the base feature using the builder."""
        print("(MODEL): Building base feature from signal.")

        # Setup the appropriate builder
        self.setup_builder(params, self.config)

        f = self.builder.build(self.signal)
        print(
            f"(MODEL): Normalizing base feature with norm '{params.normalization_mode}'"
        )
        f = f.normalize(norm=params.normalization_mode)
        print(f"(MODEL): Smoothing with filter length {params.smooting_filter_length}")
        f = f.smooth(filter_length=params.smooting_filter_length, window_type="boxcar")
        print(f"(MODEL): Downsampling by factor {params.downsampling_factor}")
        f = f.downsample(factor=params.downsampling_factor)
        print(
            f"(MODEL): Applying log compression with gamma {params.log_compression_factor}"
        )
        f = f.ensure_positive()
        f = f.log_compress(gamma=params.log_compression_factor)

        return f

    def compute_ssm(
        self, base_feature: BaseFeature, params: NCParameters
    ) -> SelfSimilarityMatrix:
        """Compute the self-similarity matrix from the base feature."""
        smoothing_filt_len = params.ssm_smoothing_filter_length
        smoothing_filt_dir = params.ssm_smoothing_filter_direction
        shift_set = np.array([0])
        tempo_relative_set = np.array([1])
        builder: SSMBuilder = SSMBuilder(
            smoothing_filter_length=smoothing_filt_len,
            smoothing_filter_direction=smoothing_filt_dir,
            shift_set=shift_set,
            tempo_relative_set=tempo_relative_set,
        )
        print(
            "(MODEL): Computing self-similarity matrix with params:",
            {
                "smoothing_filt_len": smoothing_filt_len,
                "smoothing_filt_dir": smoothing_filt_dir,
                "shift_set": shift_set,
                "tempo_relative_set": tempo_relative_set,
            },
        )
        ssm: SelfSimilarityMatrix = builder.build(base_feature)

        threshold_value = params.ssm_threshold
        ssm = ssm.threshold(threshold_value, binarize=params.ssm_binarize)

        return ssm

    def compute_novelty_curve(
        self, ssm: SelfSimilarityMatrix, params: NCParameters
    ) -> NoveltyCurve:
        """Compute the novelty curve from the self-similarity matrix."""
        kernel_size = params.nc_kernel_size
        variance = params.nc_variance
        exclude_borders = True
        print(
            "(MODEL): Computing novelty curve with params:",
            {
                "kernel_size": kernel_size,
                "variance": variance,
                "exclude_borders": exclude_borders,
            },
        )
        nc: NoveltyCurve = ssm.compute_novelty_curve(
            kernel_size=kernel_size, variance=variance, exclude_borders=exclude_borders
        )

        sigma = params.nc_smoothing_sigma
        print("(MODEL): Smoothing transitions with sigma:", sigma)
        nc_smoothed = nc.smooth(sigma=sigma)

        return nc_smoothed
