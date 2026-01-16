"""Controller for silence-based segmentation analysis module.

Coordinates silence detection and segmentation analysis based
on audio energy thresholds and pause detection.

Author: Hugo Demule
Date: January 2026
"""

from typing import cast

from constants.parameters import SilenceParameters
from controllers.analysis_page.components.interactable_plot_c import (
    InteractablePlotController,
)
from controllers.multi_threading import Worker
from models.analysis_page.modules.segmenter_silence_m import SegmenterSilenceModel
from PyQt6.QtCore import QThread, pyqtSlot
from src.audio_features.features import SilenceCurve
from views.analysis_page.modules.segmenter import Segmenter


class SegmenterSilenceController(InteractablePlotController):
    """Controller for silence-based segmentation analysis.

    Manages silence curve computation and coordinates between
    silence segmenter view and analysis model for pause-based
    structural audio analysis.
    """

    def __init__(self, view: Segmenter, model: SegmenterSilenceModel):
        super().__init__(view, model)
        self._thread = QThread()
        self._worker = None
        self.v = cast(Segmenter, self.view)
        self.m = cast(SegmenterSilenceModel, self.model)
        self.setup_connections()
        self.TRANSITION_LINE_COLOR = "yellow"

    def setup_connections(self):
        self.v.s_compute_segmentation_curve.connect(self.compute_silence_curve_threaded)

    def compute_silence_curve_threaded(self):
        self.v.show_loading_overlay(f"1/2 Computing {self.v.name}...")
        self.v.remove_all_transition_lines()

        self.params: SilenceParameters = cast(
            SilenceParameters, self.v.get_parameters()
        )

        self.worker = Worker(
            lambda: self.m.compute_silence_curve(self.params),
            self.on_silence_curve_computed,
            self.handle_error,
        )
        self.worker.start()

    @pyqtSlot(object)
    def on_silence_curve_computed(self, sc: SilenceCurve):
        self.v.hide_loading_overlay()

        t, v = self.m.silence_curve_to_time_values(sc)
        self.v.create_feature_plot(t, v, opacity=0.7)

        self.compute_transitions_threaded(sc)

    def compute_transitions_threaded(self, sc: SilenceCurve):
        self.v.show_loading_overlay(f"2/2 Computing transitions...")
        self.worker = Worker(
            lambda: self.m.find_peaks_seconds(sc, self.params, sc.sampling_rate(), sc),
            self.on_transitions_computed,
            self.handle_error,
        )
        self.worker.start()

    @pyqtSlot(object)
    def on_transitions_computed(self, out):
        transitions_sec = out
        threshold = self.params.threshold
        for transition in transitions_sec:
            self.v.add_transition_line(transition, color=self.TRANSITION_LINE_COLOR)
        self.v.set_threshold_line(threshold)
        print("(CONTROLLER): Transitions computed and added to view.")
        self.v.hide_loading_overlay()

    @pyqtSlot(str)
    def handle_error(self, error_msg: str):
        """Handle errors from worker threads"""
        self.v.hide_loading_overlay()
        print(f"Error occurred: {error_msg}")
        # TODO: Show error dialog or notification to user    def handle_error(self, error_msg: str):
        """Handle errors from worker threads"""
        self.v.hide_loading_overlay()
        print(f"Error occurred: {error_msg}")
        # TODO: Show error dialog or notification to user
