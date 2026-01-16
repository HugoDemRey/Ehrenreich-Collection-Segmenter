"""Model for HRPS-based segmentation analysis.

Provides segmentation analysis using Harmonic-Rhythmic-Percussive Separation
(HRPS) features for advanced audio structure analysis.

Author: Hugo Demule
Date: January 2026
"""

import numpy as np
from constants.parameters import HRPSParameters
from constants.segmenter_configurations import SegmenterConfig
from models.analysis_page.components.interactable_plot_m import InteractablePlotModel
from src.audio.signal import Signal
from src.audio_features.builders import HRPSBuilder
from src.audio_features.features import HRPS
from src.io.session_cache import SessionCache


class SegmenterHrpsModel(InteractablePlotModel):
    """Model for HRPS-based segmentation analysis.

    Uses Harmonic-Rhythmic-Percussive Separation features for
    advanced audio segmentation and transition detection.
    """

    def __init__(self, signal: Signal, id: str):
        super().__init__(signal=signal, id=id)
        self.config = SegmenterConfig.HRPS

    def compute_hrps_feature(self, params: HRPSParameters) -> HRPS | None:
        """Compute the base feature using the builder."""

        try:
            builder = HRPSBuilder(
                frame_length=params.frame_length,
                hop_length=params.hop_length,
                L_h_frames=params.L_h_frames,
                L_p_bins=params.L_p_bins,
                beta=params.beta,
                downsampling_factor=params.downsampling_factor,
            )

            # In the application, we only need the residual component for segmentation (3x faster)
            hrps = builder.build_only_residual(self.signal)

            assert isinstance(hrps, HRPS), "Builder did not return an HRPS feature..."
            return hrps

        except Exception as e:
            print(f"HRPS computation failed: {e}")
            # Return None to signal error to controller
            return None

    def hrps_h_feature_to_time_values(
        self, hrps_feature: HRPS
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert base feature to time-values for plotting."""

        v = hrps_feature.residual_data.T
        v = HRPS._min_max_normalize(v)  # values are now between 0 and 1
        t = np.linspace(0, self.signal.duration_seconds(), len(v))

        SessionCache.save_module_state(
            page_id=self.id,
            module_type=self.config,
            nc_x=t,
            nc_y=v,
        )

        return t, v
