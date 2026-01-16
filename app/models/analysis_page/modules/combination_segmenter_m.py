"""Model for combined segmentation analysis with multiple algorithms.

Provides multi-algorithm segmentation analysis by combining results
from different segmentation approaches and parameter configurations.

Author: Hugo Demule
Date: January 2026
"""

from constants.novelty_curve_code import NoveltyCurveCode
from constants.parameters import NCCombinationParameters
from constants.segmenter_configurations import SegmenterConfig
from models.analysis_page.components.interactable_plot_m import InteractablePlotModel
from src.audio.signal import Signal
from src.audio_features.features import NoveltyCurve


class CombinationSegmenterModel(InteractablePlotModel):
    """Model for multi-algorithm segmentation analysis.

    Combines results from multiple segmentation algorithms to provide
    comparative analysis and enhanced segmentation accuracy.
    """

    def __init__(self, signal: Signal, config: SegmenterConfig, id: str):
        super().__init__(signal=signal, id=id)
        self.config = config
        self.novelty_curve: dict[
            int, "NoveltyCurve"
        ] = {}  # Store novelty curves from source segmenters
        self.novelty_curve_combined: NoveltyCurve | None = None

    def update_novelty_curve(
        self, source_index: int, nc: NoveltyCurve, params: NCCombinationParameters
    ) -> NoveltyCurve | NoveltyCurveCode:
        """Update the novelty curve from a source segmenter and recompute the combined novelty curve."""
        # Update the novelty curve received
        print("(COMB MODEL): Updating novelty curve from source", source_index)
        self.novelty_curve[source_index] = nc

        return self.compute_combined_nc(params)

    def compute_combined_nc(
        self, params: NCCombinationParameters
    ) -> NoveltyCurve | NoveltyCurveCode:
        """Compute the combined novelty curve from source novelty curves."""
        weights = [
            params.chromagram_weight,
            params.mfcc_weight,
            params.tempogram_weight,
        ]
        ncs = [self.novelty_curve.get(i, None) for i in range(len(weights))]

        method = params.combination_method

        # remove the None novelty curves as well as the corresponding weights
        valid_ncs = []
        valid_weights = []
        for nc, weight in zip(ncs, weights):
            if nc is not None:
                valid_ncs.append(nc)
                valid_weights.append(weight)

        if len(valid_ncs) == 0:
            return NoveltyCurveCode.ZERO_CURVE

        try:
            nc = NoveltyCurve.combine(valid_ncs, weights=valid_weights, method=method)
        except ValueError as e:
            print(f"Error combining novelty curves: {e}")
            return NoveltyCurveCode.ERROR_COMBINATION

        self.novelty_curve_combined = nc
        return nc
