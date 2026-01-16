"""Controller for HRPS-based segmentation analysis module.

Coordinates HRPS feature computation and segmentation analysis
using Harmonic-Rhythmic-Percussive Separation techniques.

Author: Hugo Demule
Date: January 2026
"""

from typing import cast

from constants.parameters import HRPSParameters
from controllers.analysis_page.components.interactable_plot_c import (
    InteractablePlotController,
)
from controllers.multi_threading import Worker
from models.analysis_page.modules.segmenter_hrps_m import SegmenterHrpsModel
from PyQt6.QtCore import QThread, pyqtSlot
from src.audio_features.features import HRPS
from views.analysis_page.modules.segmenter import Segmenter


class SegmenterHrpsController(InteractablePlotController):
    """Controller for HRPS-based segmentation analysis.

    Manages HRPS feature computation and coordinates between
    HRPS segmenter view and analysis model for advanced
    harmonic-rhythmic-percussive audio analysis.
    """

    def __init__(self, view: Segmenter, model: SegmenterHrpsModel):
        super().__init__(view, model)
        self._thread = QThread()
        self._worker = None
        self.v = cast(Segmenter, self.view)
        self.m = cast(SegmenterHrpsModel, self.model)
        self.setup_connections()
        self.TRANSITION_LINE_COLOR = "#52b788"

    def setup_connections(self):
        self.v.s_compute_segmentation_curve.connect(self.compute_hrps_feature_threaded)

    def compute_hrps_feature_threaded(self):
        self.v.show_loading_overlay(f"1/2 Computing {self.v.name}...")
        self.v.remove_all_transition_lines()

        self.params = cast(
            HRPSParameters, self.v.get_parameters()
        )  # Assuming the view has a method to get current parameters

        self.worker = Worker(
            lambda: self.m.compute_hrps_feature(self.params),
            self.on_hrps_feature_computed,
            self.handle_error,
        )
        self.worker.start()

    @pyqtSlot(object)
    def on_hrps_feature_computed(self, hrps_feature):
        self.v.hide_loading_overlay()

        # Check if computation failed (model returned None)
        if hrps_feature is None:
            # Show error dialog
            from PyQt6.QtWidgets import QMessageBox

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Error")
            msg_box.setText(
                "The current parameters for HRPS lead to an internal error "
                "(probably due to frame length or hop length).\n\n"
                "Retry with other parameters."
            )
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg_box.exec()

            # Create a zero curve for display
            t, v = self.m.empty_output()
            self.v.create_feature_plot(t, v, opacity=0.7)

            return

        t, v = self.m.hrps_h_feature_to_time_values(hrps_feature)
        self.v.create_feature_plot(t, v, opacity=0.7)

        self.compute_transitions_threaded(hrps_feature)

    def compute_transitions_threaded(self, hrps_feature: HRPS):
        self.v.show_loading_overlay(f"2/2 Computing transitions...")

        self.worker = Worker(
            lambda: self.m.find_peaks_seconds(
                hrps_feature, self.params, hrps_feature.sampling_rate()
            ),
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
        # This handles thread-level errors, HRPS-specific errors are handled in on_hrps_feature_computed
