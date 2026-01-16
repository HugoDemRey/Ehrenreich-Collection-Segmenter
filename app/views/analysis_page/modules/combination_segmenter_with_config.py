"""Combined segmentation analysis with multiple algorithm configurations.

Provides interface for running multiple segmentation algorithms with
different parameter configurations for comparative analysis.

Author: Hugo Demule
Date: January 2026
"""

import numpy as np
from constants.parameters import ParameterOption, get_parameters_options
from constants.segmenter_configurations import SegmenterConfig
from controllers.analysis_page.modules.combination_segmenter_c import (
    CombinationSegmenterController,
)
from controllers.analysis_page.modules.segmenter_c import SegmenterController
from models.analysis_page.modules.combination_segmenter_m import (
    CombinationSegmenterModel,
)
from src.audio.signal import Signal
from views.analysis_page.modules.segmenter_with_config import SegmenterWithConfig


class CombinationSegmenterWithConfig(SegmenterWithConfig):
    """Multi-algorithm segmentation interface with parameter controls.

    Extends SegmenterWithConfig to enable comparison of different
    segmentation approaches by running multiple algorithms with
    configurable parameters simultaneously.
    """

    """Combination Segmenter with custom configuration."""

    def __init__(
        self,
        name: str,
        signal: Signal,
        config: SegmenterConfig,
        config_parameters: list[ParameterOption],
        source_controllers: list[SegmenterController],
    ):
        super().__init__(name, signal, config, config_parameters)
        self.source_controllers = source_controllers

        self.create_feature_plot(
            np.linspace(0, self.signal.duration_seconds(), 10),
            np.zeros(10, dtype=float),
        )
        self.on_generate_button_clicked()

    @staticmethod
    def init(
        name: str,
        signal: Signal,
        config: SegmenterConfig,
        source_controllers: list[SegmenterController],
        id: str,
    ) -> "CombinationSegmenterWithConfig":
        """Initialize CombinationSegmenterWithConfig with MVC pattern."""
        config_parameters: list[ParameterOption] = get_parameters_options(config)

        widget = CombinationSegmenterWithConfig(
            name, signal, config, config_parameters, source_controllers
        )
        model = CombinationSegmenterModel(signal=signal, config=config, id=id)
        controller = CombinationSegmenterController(widget, model, source_controllers)

        widget._config = config
        widget._controller = controller
        return widget
