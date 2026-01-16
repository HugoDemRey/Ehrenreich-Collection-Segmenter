"""Model for Naxos database integration and music alignment.

Provides functionality for searching, fetching, and aligning music
data from the Naxos database with local audio analysis.

Author: Hugo Demule
Date: January 2026
"""

import os
import time

import numpy as np
from constants.naxos import BASE_URL, NO_PREVIEW_LEFT
from constants.paths import NAXOS_CACHE_PATH
from PyQt6.QtCore import QObject, pyqtSignal
from src.audio.audio_file import AudioFile
from src.audio.signal import Signal, SubSignal
from src.audio_features.aligners import ChromagramAligner
from src.audio_features.builders import ChromagramBuilder
from src.audio_features.features import Chromagram
from src.naxos.scraper import NaxosScraper


class NaxosModuleModel(QObject):
    def __init__(self, signal: Signal):
        super().__init__()
        self.signal = signal
        self.preview_chromas = []
        self.aligner = ChromagramAligner()
        self.current_preview_index = 0
        self.expected_starts = []  # Will store expected start positions

        self.offset_time = signal.from_time if isinstance(signal, SubSignal) else 0.0

    scraper: NaxosScraper
    s_alignment_verbose = pyqtSignal(str, int)  # message, percentage
    s_scraping_verbose = pyqtSignal(str, int)  # message, percentage

    def prepare_scraper(self, url: str) -> bool:
        self.s_scraping_verbose.emit(f"Preparing scraper for URL: {url}", 1)
        self.scraper = NaxosScraper(url, NAXOS_CACHE_PATH)
        if self.scraper.is_scraping_complete():
            self.s_scraping_verbose.emit(
                "Retreiving Cached Scraped data for this URL.", 30
            )
            time.sleep(0.3)
            self.s_scraping_verbose.emit(
                "Retreiving Cached Scraped data for this URL.", 60
            )
            time.sleep(0.3)
            self.s_scraping_verbose.emit("Scraper prepared.", 100)
            time.sleep(0.1)
            return True
        self.s_scraping_verbose.emit("Mounting Scraper driver...", 5)
        self.scraper.mount_driver()
        self.s_scraping_verbose.emit("Scraper prepared.", 10)
        return False

    def _require_scraper(self):
        if not hasattr(self, "scraper") or self.scraper is None:
            raise ValueError("Scraper not initialized. Call prepare_scraper() first.")

    def _require_scraper_completed(self):
        self._require_scraper()
        if not self.scraper.is_scraping_complete():
            raise ValueError("Scraper has not completed its tasks yet.")

    def check_url_validity(self, url: str) -> bool:
        self.s_scraping_verbose.emit(f"Checking URL validity for: {url}", 0)
        is_valid = url.startswith(BASE_URL)
        self.s_scraping_verbose.emit(f"Checking URL's validity -> {is_valid}", 1)
        return is_valid

    def scrape_audio_urls(self):
        print("Requiring scraper")
        self._require_scraper()

        print("Scraping audio URLs")
        self.s_scraping_verbose.emit("Scraping audio URLs...", 10)
        self.scraper.scrape_audio_URLs(
            verbose_signal_percentage=self.s_scraping_verbose,
            start_percentage=10,
            end_percentage=60,
        )
        self.s_scraping_verbose.emit("Scraping audio URLs completed.", 60)

    def download_audio_files(self):
        self._require_scraper()
        self.s_scraping_verbose.emit("Downloading audio files...", 60)
        self.scraper.download_audios(
            verbose_signal_percentage=self.s_scraping_verbose,
            start_percentage=60,
            end_percentage=100,
        )
        self.s_scraping_verbose.emit("Downloading audio files completed.", 100)

    def run_scraping_pipeline(self, url: str):
        if self.prepare_scraper(url):
            return
        print("Trying to Run scrape_audio_urls")
        self.scrape_audio_urls()
        print("Trying to Download audio files")
        self.download_audio_files()

    def get_durations(self) -> np.ndarray:
        self._require_scraper()

        durations_path = self.scraper.audio_full_durations_file
        durations = np.load(durations_path)

        return durations

    def get_titles(self) -> np.ndarray:
        self._require_scraper()

        titles_path = self.scraper.audio_titles_file
        titles = np.load(titles_path, allow_pickle=True)
        return titles

    def get_audio_paths(self) -> np.ndarray:
        self._require_scraper()

        audio_paths = self.scraper.audio_files_path
        return np.array(
            [os.path.join(audio_paths, file) for file in os.listdir(audio_paths)]
        )

    def init_alignment(self):
        self._require_scraper_completed()

        self.s_alignment_verbose.emit("Computing main chromagram...", 0)

        chroma = ChromagramBuilder().build(self.signal)
        self.chroma = self._preprocess_chroma(chroma)

        self.s_alignment_verbose.emit("Main chromagram computed.", 10)

        # Calculate expected start positions from durations (like in notebook)
        self.s_alignment_verbose.emit("Calculating expected start positions...", 15)
        self._calculate_expected_starts()
        self.s_alignment_verbose.emit("Expected start positions calculated.", 20)

        preview_paths = self.get_audio_paths()
        for i, preview_path in enumerate(preview_paths):
            # Progress from 20% to 60% during preview chromagram computation
            progress = 20 + int(40 * i / len(preview_paths))
            self.s_alignment_verbose.emit(
                f"Computing Preview Chromagram for {os.path.basename(preview_path)}...",
                progress,
            )
            preview_signal = AudioFile(preview_path).load()
            preview_chroma = ChromagramBuilder().build(preview_signal)
            preview_chroma = self._preprocess_chroma(preview_chroma)
            self.preview_chromas.append(preview_chroma)

    def align_next_preview(
        self, output_type="time", window_size_sec: float = 900
    ) -> tuple:
        if self.current_preview_index >= len(self.preview_chromas):
            return NO_PREVIEW_LEFT, NO_PREVIEW_LEFT

        # Use windowed approach with expected start position
        expected_start = (
            self.expected_starts[self.current_preview_index]
            if self.current_preview_index < len(self.expected_starts)
            else None
        )

        result = self.aligner.align(
            ref=self.chroma,
            query=self.preview_chromas[self.current_preview_index],
            expected_start_sec=expected_start if expected_start else 0.0,
            window_size_sec=window_size_sec,
            offset_second=self.offset_time,
            output_type=output_type,
        )
        self.current_preview_index += 1

        # Progress from 60% to 100% during alignment
        progress = 60 + int(40 * self.current_preview_index / len(self.preview_chromas))
        self.s_alignment_verbose.emit(
            f"Alignment for preview {self.current_preview_index - 1} completed.",
            progress,
        )

        return result

    def align_all_previews(
        self, output_type="time", window_size_sec: float = 900
    ) -> list[tuple]:
        alignments = []
        start, end = self.align_next_preview(
            output_type=output_type, window_size_sec=window_size_sec
        )
        alignments.append((start, end))
        while start != NO_PREVIEW_LEFT:
            start, end = self.align_next_preview(
                output_type=output_type, window_size_sec=window_size_sec
            )
            alignments.append((start, end))

        self.s_alignment_verbose.emit("All alignments completed.", 100)
        return alignments

    def _preprocess_chroma(self, chroma: Chromagram) -> Chromagram:
        ch_p = chroma.normalize("2")
        ch_p = ch_p.smooth(21)
        # ch_p = ch_p.downsample(factor=5)
        ch_p = ch_p.log_compress(500)
        return ch_p

    def _calculate_expected_starts(self):
        """Calculate expected start positions from audio durations, similar to notebook approach."""
        durations = self.get_durations()
        gt_starts = np.cumsum(durations)
        gt_len = gt_starts[-1]
        gt_starts = np.insert(gt_starts, 0, 0)[
            :-1
        ]  # Insert 0 at beginning, remove last

        # Convert to relative positions in the main signal duration
        self.expected_starts = [
            self._get_relative_position(t, gt_len, self.signal.duration_seconds())
            for t in gt_starts
        ]

    def _get_relative_position(
        self, time_sec: float, total_duration: float, reference_duration: float
    ) -> float:
        """Get the relative position of a time in a signal compared to a reference duration."""
        relative_position = (time_sec / total_duration) * reference_duration
        return relative_position
