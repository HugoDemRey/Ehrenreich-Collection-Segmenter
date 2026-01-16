"""Controller for segmentation analysis module.

Coordinates segmentation analysis, novelty curve computation,
and user interactions for the segmenter component.

Author: Hugo Demule
Date: January 2026
"""

from typing import cast

from constants.parameters import NCParameters
from controllers.analysis_page.components.interactable_plot_c import (
    InteractablePlotController,
)
from controllers.multi_threading import Worker
from models.analysis_page.modules.segmenter_m import SegmenterModel
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from src.audio_features.features import BaseFeature, NoveltyCurve, SelfSimilarityMatrix
from views.analysis_page.modules.segmenter import Segmenter


class SegmenterController(InteractablePlotController):
    """Controller for segmentation analysis with novelty curves.

    Manages segmentation computation, peak detection, and coordinates
    between segmenter view and analysis model.

    Signals:
        s_novelty_curve_computed: Emitted when novelty curve is computed
    """

    s_novelty_curve_computed = pyqtSignal(NoveltyCurve)

    def __init__(self, view: Segmenter, model: SegmenterModel):
        super().__init__(view, model)
        self._active_workers = {}  # Dictionary to track active workers
        self.v = cast(Segmenter, self.view)
        self.m = cast(SegmenterModel, self.model)
        self.setup_connections()
        self.TRANSITION_LINE_COLOR = "#4361ee"

    def setup_connections(self):
        self.v.s_compute_segmentation_curve.connect(self.compute_base_feature_threaded)

    def compute_base_feature_threaded(self):
        self.v.show_loading_overlay(f"1/4 Computing {self.v.name}...")
        self.v.remove_all_transition_lines()

        self.params = cast(NCParameters, self.v.get_parameters())

        worker = Worker(
            lambda: self.m.compute_base_feature(self.params),
            self.on_base_feature_computed,
            self.handle_error,
        )
        worker.cleanup_requested.connect(self._cleanup_worker)
        self._active_workers[worker.worker_id] = worker
        print(
            f"(CONTROLLER): Starting base feature computation, active workers: {len(self._active_workers)}"
        )
        worker.start()

    @pyqtSlot(object)
    def on_base_feature_computed(self, base_feature: BaseFeature):
        print("(CONTROLLER): Base feature computed, now computing SSM...")
        self.compute_ssm_threaded(base_feature)

    def compute_ssm_threaded(self, base_feature: BaseFeature):
        self.v.show_loading_overlay(
            f"2/4 Computing {self.v.name} Self-Similarity Matrix..."
        )
        worker = Worker(
            lambda: self.m.compute_ssm(base_feature, self.params),
            self.on_ssm_computed,
            self.handle_error,
        )
        worker.cleanup_requested.connect(self._cleanup_worker)
        self._active_workers[worker.worker_id] = worker
        print(
            f"(CONTROLLER): Starting SSM computation, active workers: {len(self._active_workers)}"
        )
        worker.start()

    @pyqtSlot(object)
    def on_ssm_computed(self, ssm: SelfSimilarityMatrix):
        print("(CONTROLLER): SSM computed, now computing novelty curve...")
        self.compute_novelty_curve_threaded(ssm)

    def compute_novelty_curve_threaded(self, ssm: SelfSimilarityMatrix):
        self.v.show_loading_overlay(f"3/4 Computing {self.v.name} Novelty Curve...")
        worker = Worker(
            lambda: self.m.compute_novelty_curve(ssm, self.params),
            self.on_novelty_curve_computed,
            self.handle_error,
        )
        worker.cleanup_requested.connect(self._cleanup_worker)
        self._active_workers[worker.worker_id] = worker
        print(
            f"(CONTROLLER): Starting novelty curve computation, active workers: {len(self._active_workers)}"
        )
        worker.start()

    @pyqtSlot(object)
    def on_novelty_curve_computed(self, nc: NoveltyCurve):
        print("(CONTROLLER): Novelty curve computed, now computing transitions...")
        self.s_novelty_curve_computed.emit(nc)
        t, v = self.m.novelty_curve_to_time_values(nc)
        self.v.create_feature_plot(t, v)
        self.compute_transitions_threaded(nc)

    def compute_transitions_threaded(self, nc: NoveltyCurve):
        self.v.show_loading_overlay(f"4/4 Computing {self.v.name} Transitions...")
        worker = Worker(
            lambda: self.m.find_peaks_seconds(nc, self.params, nc.sampling_rate()),
            self.on_transitions_computed,
            self.handle_error,
        )
        worker.cleanup_requested.connect(self._cleanup_worker)
        self._active_workers[worker.worker_id] = worker
        print(
            f"(CONTROLLER): Starting transitions computation, active workers: {len(self._active_workers)}"
        )
        worker.start()

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
        # TODO: Show error dialog or notification to user

    @pyqtSlot(str)
    def _cleanup_worker(self, worker_id: str):
        """Remove worker from active workers when it's finished"""
        if worker_id in self._active_workers:
            print(
                f"(CONTROLLER): Cleaning up worker {worker_id}, remaining active workers: {len(self._active_workers) - 1}"
            )
            del self._active_workers[worker_id]
        else:
            print(
                f"(CONTROLLER): Warning - tried to cleanup unknown worker {worker_id}"
            )
