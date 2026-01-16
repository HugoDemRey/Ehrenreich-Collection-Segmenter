"""Manager for synchronizing multiple plot components.

Coordinates interactions between multiple analysis plots,
timeline synchronization, and cross-plot communication.

Author: Hugo Demule
Date: January 2026
"""

from controllers.analysis_page.components.interactable_plot_c import (
    InteractablePlotController,
)
from controllers.analysis_page.modules.naxos_module_c import NaxosModuleController
from controllers.analysis_page.modules.timeline_c import TimelineController
from controllers.audio_c import AudioController
from models.analysis_page.modules.timeline_m import TimelineModel
from PyQt6.QtCore import QObject
from views.analysis_page.modules.segmenter import Segmenter
from views.analysis_page.modules.timeline import Timeline


class PlotSyncManager(QObject):
    """Manager for synchronizing multiple interactive plot components.

    Coordinates cursor positions, zoom levels, and user interactions
    across multiple analysis plots and timeline components.
    """

    def __init__(
        self,
        audio_controller: AudioController,
        out_controller: TimelineController,
        controllers: list[InteractablePlotController],
        naxos_controller: NaxosModuleController,
    ):
        super().__init__()
        self.audio_controller = audio_controller
        self.out_controller = out_controller
        self.controllers = controllers
        self.naxos_controller = naxos_controller
        self.init_signal_connections()

    def init_signal_connections(self):
        self.audio_controller.position_updated.connect(self.on_position_updated)

        for controller in self.controllers:
            controller.change_cursor_pos.connect(self.on_change_cursor_pos)
            if isinstance(controller.view, Segmenter):
                controller.view.s_add_this_transition.connect(
                    lambda pos, ctrl=controller: self.add_transition_at_position(
                        pos, color=ctrl.TRANSITION_LINE_COLOR
                    )
                )
                controller.view.s_add_all_transitions.connect(
                    lambda positions, ctrl=controller: self.add_transitions_at_position(
                        positions, color=ctrl.TRANSITION_LINE_COLOR
                    )
                )

        self.naxos_controller.s_change_cursor_pos.connect(self.on_change_cursor_pos)
        self.naxos_controller.s_add_this_transition.connect(
            lambda pos: self.add_transition_at_position(
                pos, color=self.naxos_controller.TRANSITION_LINE_COLOR
            )
        )
        self.naxos_controller.s_add_all_transitions.connect(
            lambda positions: self.add_transitions_at_position(
                positions, color=self.naxos_controller.TRANSITION_LINE_COLOR
            )
        )

        self.out_controller.change_cursor_pos.connect(self.on_change_cursor_pos)

    def on_change_cursor_pos(self, new_position: float):
        for controller in self.controllers:
            controller.view.set_cursor_position(new_position)

        self.out_controller.view.set_cursor_position(new_position)
        assert isinstance(self.out_controller.view, Timeline), (
            "View must be an instance of TimelineController"
        )
        self.out_controller.view.set_timer(new_position)
        self.audio_controller.model.set_current_position(new_position)

    def on_position_updated(self, updated_position: float):
        for controller in self.controllers:
            controller.view.set_cursor_position(updated_position)

        self.out_controller.view.set_cursor_position(updated_position)
        assert isinstance(self.out_controller.view, Timeline), (
            "View must be an instance of TimelineController"
        )
        self.out_controller.view.set_timer(updated_position)

    def add_transition_at_position(self, position: float, color: str = "white"):
        """Add transition to timeline and update cache."""
        print(f"(SYNC): Adding transition at {position} with color {color}")

        # Add to timeline model cache
        assert isinstance(self.out_controller.model, TimelineModel), (
            "Model must be an instance of TimelineModel"
        )
        self.out_controller.model.add_transition(position, color)

        # Add to timeline view
        self.out_controller.view.add_transition_line(position, color=color)

    def add_transitions_at_position(self, positions: list[float], color: str = "white"):
        """Add multiple transitions to timeline and update cache."""
        print(f"(SYNC): Adding {len(positions)} transitions with color {color}")

        for pos in positions:
            # Add to timeline model cache
            assert isinstance(self.out_controller.model, TimelineModel), (
                "Model must be an instance of TimelineModel"
            )
            self.out_controller.model.add_transition(pos, color)

            # Add to timeline view
            self.out_controller.view.add_transition_line(pos, color=color)
