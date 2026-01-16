"""Controller for combined segmentation analysis module.

Coordinates multi-algorithm segmentation analysis and manages
interactions between different segmentation approaches.

Author: Hugo Demule
Date: January 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from constants.novelty_curve_code import NoveltyCurveCode
from constants.parameters import NCCombinationParameters
from controllers.analysis_page.components.interactable_plot_c import (
    InteractablePlotController,
)
from controllers.analysis_page.modules.segmenter_c import SegmenterController
from models.analysis_page.modules.combination_segmenter_m import (
    CombinationSegmenterModel,
)
from src.audio_features.features import NoveltyCurve

if TYPE_CHECKING:
    from views.analysis_page.modules.combination_segmenter_with_config import (
        CombinationSegmenterWithConfig,
    )


class CombinationSegmenterController(InteractablePlotController):
    def __init__(
        self,
        view: CombinationSegmenterWithConfig,
        model: CombinationSegmenterModel,
        comb_controllers: list[SegmenterController],
    ):
        super().__init__(view, model)
        self.v = view
        self.m = model
        self.TRANSITION_LINE_COLOR = "#4cc9f0"
        self.comb_controllers = comb_controllers
        self.setup_connections()

    def setup_connections(self):
        for i, controller in enumerate(self.comb_controllers):
            controller.s_novelty_curve_computed.connect(
                lambda nc, idx=i: self.on_source_novelty_curve_computed(nc, idx)
            )

        self.v.s_compute_segmentation_curve.connect(self.on_button_clicked)

    def on_source_novelty_curve_computed(self, nc: NoveltyCurve, source_index: int):
        print(
            f"(COMB CONTROLLER): Received novelty curve from source {source_index}, now updating combination segmenter..."
        )
        self.v.remove_all_transition_lines()
        self.params = cast(NCCombinationParameters, self.v.get_parameters())

        new_nc = self.m.update_novelty_curve(source_index, nc, self.params)

        self.apply_changes(new_nc)

    def compute_transitions_threaded(self, nc: NoveltyCurve):
        self.v.show_loading_overlay(f"4/4 Computing {self.v.name} Transitions...")
        transitions_sec = self.m.find_peaks_seconds(nc, self.params, nc.sampling_rate())
        threshold = self.params.threshold
        for transition in transitions_sec:
            self.v.add_transition_line(transition, color=self.TRANSITION_LINE_COLOR)
        self.v.set_threshold_line(threshold)
        print("(CONTROLLER): Transitions computed and added to view.")
        self.v.hide_loading_overlay()

    def on_button_clicked(self):
        print("(COMB CONTROLLER): Compute combination novelty curve button clicked.")
        self.params = cast(NCCombinationParameters, self.v.get_parameters())
        self.v.remove_all_transition_lines()
        nc = self.m.compute_combined_nc(self.params)

        self.apply_changes(nc)

    def apply_changes(self, nc: NoveltyCurve | NoveltyCurveCode):
        """Apply changes to the view based on the updated novelty curve."""

        if (
            isinstance(nc, NoveltyCurveCode)
            and nc == NoveltyCurveCode.ERROR_COMBINATION
        ):
            self.v.show_loading_overlay(
                f"The combined novelty curve cannot be computed. This may be due to different sampling rates among source segmenters. Try to 1. Keep the same downsampling rate for all source segmenters. 2. Reduce the weight of the segmenters with different sampling rates to 0."
            )
        else:
            t, v = self.m.novelty_curve_to_time_values(nc)
            self.v.create_feature_plot(t, v)
            if not isinstance(nc, NoveltyCurveCode):
                self.compute_transitions_threaded(nc)
