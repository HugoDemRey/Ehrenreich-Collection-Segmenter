"""Controller for Naxos database integration module.

Coordinates Naxos database interactions, music search and alignment,
and manages integration with local audio analysis.

Author: Hugo Demule
Date: January 2026
"""

from controllers.multi_threading import Worker
from models.analysis_page.modules.naxos_module_m import NaxosModuleModel
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from views.analysis_page.modules.naxos_module import NaxosModule
from src.naxos.exceptions import (
    NaxosScrapingError,
    NaxosDriverError,
    NaxosPageLoadError,
    NaxosContentError,
    NaxosNetworkError,
    NaxosAudioExtractionError,
    NaxosTimeoutError,
)


class NaxosModuleController(QObject):
    """Controller for Naxos database integration and music alignment.

    Manages music search, alignment computation, and transition detection
    for Naxos database integration functionality.

    Signals:
        s_change_cursor_pos (float): Cursor position change notification
        s_add_this_transition (float): Add single transition at position
        s_add_all_transitions (object): Add multiple transitions
    """

    TRANSITION_LINE_COLOR = "orange"

    s_change_cursor_pos = pyqtSignal(float)
    s_add_this_transition = pyqtSignal(float)
    s_add_all_transitions = pyqtSignal(object)

    def __init__(self, view: NaxosModule, model: NaxosModuleModel):
        super().__init__()
        self.view: NaxosModule = view
        self.model: NaxosModuleModel = model
        self._worker = None
        self.percentage = 0

        self.setup_connections()

    def setup_connections(self):
        self.view.s_url_validated.connect(self.scrape_url)
        self.view.s_start_alignment.connect(self.start_alignment)
        self.view.s_change_cursor_pos.connect(
            lambda pos: self.s_change_cursor_pos.emit(pos)
        )
        self.view.s_add_this_alignment.connect(
            lambda pos: self.s_add_this_transition.emit(pos)
        )
        self.view.s_add_all_alignments.connect(
            lambda positions: self.s_add_all_transitions.emit(positions)
        )

        self.model.s_alignment_verbose.connect(self.handle_alignment_verbose)
        self.model.s_scraping_verbose.connect(self.handle_scraping_verbose)

    def handle_alignment_verbose(self, msg: str, percent: int):
        self.view.set_alignment_verbose(msg, percent)

    def handle_scraping_verbose(self, msg: str, percent: int):
        self.view.set_scraping_verbose(msg, percent)

    def scrape_url(self, url: str):
        if self.model.check_url_validity(url):
            self._worker = Worker(
                self.model.run_scraping_pipeline,
                self.on_scrape_completed,
                self.on_error,
                url,
            )
            self._worker.start()
            

    @pyqtSlot(object)
    def on_scrape_completed(self):
        titles = self.model.get_titles()
        durations = self.model.get_durations()
        audio_paths = self.model.get_audio_paths()

        self.view.update_table(titles, durations, audio_paths)

        self.view.show_url_overlay(False)
        self.view.show_content(True)

    def start_alignment(self):
        self._worker = Worker(
            self.model.init_alignment, self.on_alignment_init_completed
        )
        self._worker.start()

    @pyqtSlot(object)
    def on_alignment_init_completed(self):
        self.align_previews()

    def align_previews(self):
        self._worker = Worker(
            self.model.align_all_previews, self.on_all_alignments_completed
        )
        self._worker.start()

    @pyqtSlot(object)
    def on_all_alignments_completed(self, alignments: list[tuple]):
        alignment_starts = [a[0] for a in alignments]
        alignment_ends = [a[1] for a in alignments]
        self.view.show_final_table_view(alignment_starts, alignment_ends)
        print("[CONTROLLER]: All alignments completed.")

    def check_url_validity(self, url: str) -> bool:
        is_url_valid = self.model.check_url_validity(url)

        if not is_url_valid:
            self.on_error("The provided URL is not a valid Naxos catalog page. The URL should start with 'https://www.naxos.com/CatalogueDetail/?id='.")

        return is_url_valid

    def on_error(self, error_msg: str):
        from PyQt6.QtWidgets import QMessageBox

        self.view.reset_module()

        title = "Naxos Scraping Error"
        message = f"An error occurred: {error_msg}"

        msg_box = QMessageBox(
            QMessageBox.Icon.Critical,
            title,
            message,
            QMessageBox.StandardButton.Ok,
        )
        msg_box.exec()
