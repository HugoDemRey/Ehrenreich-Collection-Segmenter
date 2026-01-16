"""Naxos Audio Content Scraper.

This module provides functionality to scrape audio previews and metadata from Naxos
Classical Music Library. It uses Selenium WebDriver to interact with the Naxos website,
extract audio URLs, download preview files, and convert them to WAV format for analysis.

The scraper is designed to handle the dynamic content loading of Naxos pages and
can extract 30-second audio previews along with track metadata including titles
and durations.

Author: Hugo Demule
Date: January 2026
"""

import json
import os
import subprocess
import time

import imageio_ffmpeg
import numpy as np
import requests
from constants.paths import NAXOS_CACHE_PATH
from PyQt6.QtCore import pyqtBoundSignal
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tinytag import TinyTag
from webdriver_manager.chrome import ChromeDriverManager

from src.audio.signal import Signal
from src.naxos.exceptions import (
    NaxosDriverError,
    NaxosPageLoadError,
    NaxosContentError,
    NaxosNetworkError,
    NaxosAudioExtractionError,
    NaxosTimeoutError,
)
from src.naxos.exceptions import (
    NaxosDriverError,
    NaxosPageLoadError,
    NaxosContentError,
    NaxosNetworkError,
    NaxosAudioExtractionError,
    NaxosTimeoutError,
)


class NaxosScraper(object):
    """Web scraper for extracting audio content from Naxos Classical Music Library.

    This class provides comprehensive functionality to scrape audio previews and metadata
    from Naxos catalog pages. It handles dynamic content loading, extracts audio URLs
    using browser network logging, downloads audio files, and converts them to WAV format.

    The scraper uses Selenium WebDriver with Chrome to interact with the Naxos website
    and extract 30-second audio previews along with track information such as titles
    and durations.

    Attributes:
        UNKNOWN_DURATION (int): Constant for unknown duration (-1).
        NA_DURATION (int): Constant for not available duration (-2).
        CONVERSION_FAILED (int): Constant for conversion failure (-3).
        UNKNOWN_DURATION_TAG (str): String tag for unknown duration.
        NA_DURATION_TAG (str): String tag for not available duration.

    Example:
        >>> scraper = NaxosScraper("https://www.naxos.com/CatalogueDetail/?id=CHAN3019-21")
        >>> scraper.mount_driver()
        >>> scraper.scrape_audio_URLs()
        >>> scraper.download_audios()
    """

    UNKNOWN_DURATION = -1
    NA_DURATION = -2
    CONVERSION_FAILED = -3

    UNKNOWN_DURATION_TAG = "Unknown"
    NA_DURATION_TAG = "NA"

    def __init__(self, naxos_url, cache_dir=NAXOS_CACHE_PATH):
        """Initialize the NaxosScraper with a Naxos catalog URL.

        Args:
            naxos_url (str): The full Naxos catalog URL to scrape.
                Example: 'https://www.naxos.com/CatalogueDetail/?id=CHAN3019-21'
            cache_dir (str, optional): Base directory for caching scraped data.
                Defaults to NAXOS_CACHE_PATH.

        Note:
            Creates necessary cache directories automatically. The catalog ID is
            extracted from the URL to create a unique cache subdirectory.
        """
        # Naxos URL example: https://www.naxos.com/CatalogueDetail/?id=CHAN3019-21
        self.naxos_url = naxos_url
        self.id = naxos_url.split("=")[-1]
        self.cache_dir = os.path.join(cache_dir, self.id)
        self.audio_urls_file = os.path.join(self.cache_dir, "audio_URLs.npy")
        self.audio_full_durations_file = os.path.join(
            self.cache_dir, "audio_full_durations.npy"
        )
        self.audio_titles_file = os.path.join(self.cache_dir, "audio_titles.npy")
        self.audio_files_path = os.path.join(self.cache_dir, "audio_files")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.audio_files_path, exist_ok=True)

    def mount_driver(self):
        """Setup Chrome WebDriver with performance logging enabled.

        Initializes a headless Chrome WebDriver instance with performance logging
        capabilities to capture network requests. Includes fallback mechanisms
        for different deployment scenarios including PyInstaller builds.

        Raises:
            RuntimeError: If Chrome driver cannot be initialized after trying
                both ChromeDriverManager and system chromedriver.

        Note:
            The driver is configured with performance logging to capture
            network requests needed for extracting audio URLs from AJAX calls.
        """
        # Setup Chrome with performance logging enabled
        options = Options()
        options.add_argument("--headless=new")  # comment this line to see the browser
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        try:
            # Try to use ChromeDriverManager for normal installations
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            )
        except Exception as e:
            print(f"ChromeDriverManager failed: {e}")
            try:
                # Fallback to system chromedriver or bundled one
                self.driver = webdriver.Chrome(options=options)
            except Exception as e2:
                print(f"Chrome driver initialization failed: {e2}")
                print("Selenium functionality will be disabled.")
                self.driver = None
                raise NaxosDriverError()

    def _extract_content(self, button, pos, idx):
        """Extract content from table cells associated with a play button.

        Navigates up to the parent table row and extracts content from a specific
        table cell position. Used to extract track metadata like duration and title.

        Args:
            button: Selenium WebElement representing the play button.
            pos (int): Position index of the table cell to extract content from.
                      Negative values count from the end (e.g., -1 for last cell).
            idx (int): Button index for logging purposes.

        Returns:
            str: Extracted content text, or UNKNOWN_DURATION_TAG/NA_DURATION_TAG
                 if extraction fails.
        """
        try:
            # Find the parent <tr> element of the play button
            parent_tr = button.find_element(By.XPATH, "./ancestor::tr")
            # Find all <td> elements within this <tr>
            td_elements = parent_tr.find_elements(By.TAG_NAME, "td")
            # Get the last <td> which contains the duration
            if td_elements:
                duration = td_elements[pos].text.strip()
                print(f"Button {idx} duration: {duration}")
            else:
                duration = self.UNKNOWN_DURATION_TAG
                print(f"Button {idx}: No duration found")
        except Exception as e:
            duration = self.NA_DURATION_TAG
            print(f"Button {idx}: Failed to extract duration - {e}")

        return duration

    def _duration_txt_to_seconds(self, duration_txt):
        """Convert duration text in MM:SS format to total seconds.

        Args:
            duration_txt (str): Duration string in "MM:SS" format, or special tags
                               like UNKNOWN_DURATION_TAG or NA_DURATION_TAG.

        Returns:
            int: Total duration in seconds, or one of the error constants:
                 UNKNOWN_DURATION, NA_DURATION, or CONVERSION_FAILED.

        Example:
            >>> scraper._duration_txt_to_seconds("3:45")
            225
        """
        if duration_txt == self.UNKNOWN_DURATION_TAG:
            return self.UNKNOWN_DURATION
        elif duration_txt == self.NA_DURATION_TAG:
            return self.NA_DURATION

        try:
            minutes, seconds = map(int, duration_txt.split(":"))
            total_seconds = minutes * 60 + seconds
        except:
            total_seconds = self.CONVERSION_FAILED
        return total_seconds

    def scrape_audio_URLs(
        self,
        verbose_signal_percentage: pyqtBoundSignal | None = None,
        start_percentage: int = 0,
        end_percentage: int = 100,
    ):
        """Scrape audio URLs, durations, and titles from the Naxos catalog page.

        Navigates to the Naxos catalog page, handles dynamic content loading,
        clicks play buttons to trigger AJAX requests, and captures audio URLs
        from network traffic using Chrome DevTools Protocol.

        Args:
            verbose_signal_percentage (pyqtBoundSignal, optional): Qt signal for
                emitting progress updates. Defaults to None.
            start_percentage (int): Starting percentage for progress reporting.
                Defaults to 0.
            end_percentage (int): Ending percentage for progress reporting.
                Defaults to 100.

        Raises:
            Exception: If page loading fails or driver is not initialized.

        Note:
            Results are cached as numpy arrays in the cache directory:
            - audio_URLs.npy: Array of audio preview URLs
            - audio_full_durations.npy: Array of track durations in seconds
            - audio_titles.npy: Array of track titles

            If cache files exist, the method returns early without re-scraping.
        """
        if not self.driver:
            print("Chrome driver not initialized. Cannot scrape audio URLs.")
            raise NaxosDriverError()

        # Helper to emit signal if provided
        def emit_signal(message, percentage):
            if verbose_signal_percentage is not None:
                # print("Should Emit:", message, percentage)
                verbose_signal_percentage.emit(message, percentage)

        # Step 1: Check cache (5%)
        step1_start = start_percentage
        step1_end = start_percentage + int(0.05 * (end_percentage - start_percentage))
        emit_signal("Checking cache for audio URLs...", step1_start)
        if os.path.exists(self.audio_urls_file):
            print(f"Audio URLs already scraped and cached at {self.audio_urls_file}.")
            emit_signal("Audio URLs found in cache.", step1_end)
            return
        emit_signal("No cached audio URLs found.", step1_end)

        # Step 2: Load page and expand content (30%)
        step2_start = step1_end
        step2_end = step2_start + int(0.30 * (end_percentage - start_percentage))
        emit_signal("Loading Naxos Page...", step2_start)
        try:
            self.driver.get(self.naxos_url)

            # Wait up to 30 seconds for the page to load and content to be ready
            wait = WebDriverWait(self.driver, 30)

            # Wait for the page content to load by checking for either:
            # 1. The "More" button (indicating content is loaded but needs expansion)
            # 2. Play buttons (indicating content is already fully loaded)
            try:
                print("Waiting for page content to load...")
                # First, try to wait for the More button to appear (content partially loaded)
                more_button = wait.until(
                    EC.presence_of_element_located((By.ID, "myMore"))
                )
                print("Page content loaded - 'More' button found.")

                # Now wait for it to be clickable and click it
                more_button = wait.until(EC.element_to_be_clickable((By.ID, "myMore")))
                print("'More' button is clickable, clicking to reveal all tracks...")
                self.driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'});", more_button
                )
                more_button.click()
                print("Clicked 'More' button to reveal additional tracks.")

                # Wait for the expanded content to load (wait for more play buttons to appear)
                wait.until(
                    EC.presence_of_element_located(
                        (
                            By.CSS_SELECTOR,
                            "td[id^='divplaystop_'] > a[onclick*='fnPlayStop30']",
                        )
                    )
                )
                print("Expanded content loaded successfully.")
            except Exception as e:
                print("No 'More' button found or content already fully loaded.")
                # If no More button, just make sure play buttons are present
                try:
                    wait.until(
                        EC.presence_of_element_located(
                            (
                                By.CSS_SELECTOR,
                                "td[id^='divplaystop_'] > a[onclick*='fnPlayStop30']",
                            )
                        )
                    )
                    print("Play buttons are available - content is ready.")
                except Exception as content_error:
                    raise NaxosContentError()

            # Step 3: Find play buttons (5%)
            step3_start = step2_end
            step3_end = step3_start + int(0.05 * (end_percentage - start_percentage))
            emit_signal("Finding Tracks on Page...", step3_start)
            time.sleep(5)  # Extra wait to ensure all JS has executed
            play_buttons = self.driver.find_elements(
                By.CSS_SELECTOR, "td[id^='divplaystop_'] > a[onclick*='fnPlayStop30']"
            )
            print(f"Found {len(play_buttons)} play buttons.")
            
            if len(play_buttons) == 0:
                raise NaxosContentError()

            # Step 4: Scrape each play button (50%)
            step4_start = step3_end
            step4_end = step4_start + int(0.50 * (end_percentage - start_percentage))
            emit_signal("Retreiving Audio Tracks...", step4_start)
            audio_URLs = []
            durations = []
            titles = []
            num_buttons = len(play_buttons)
            if num_buttons > 0:
                step4_increment = (step4_end - step4_start) / num_buttons
            else:
                step4_increment = 0

            for idx, button in enumerate(play_buttons, start=1):
                emit_signal(
                    f"Retreiving Audio Track {idx}/{num_buttons}...",
                    int(step4_start + (idx - 1) * step4_increment),
                )
                onclick_js = button.get_attribute("onclick")
                if not onclick_js:
                    print(f"Button {idx}: No onclick attribute found.")
                    emit_signal(
                        f"No onclick attribute found for button {idx}.",
                        int(step4_start + idx * step4_increment),
                    )
                    continue

                import re

                match = re.search(
                    r"fnPlayStop30\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", onclick_js
                )
                if not match:
                    print(f"Button {idx}: Could not parse onclick parameters.")
                    emit_signal(
                        f"Could not parse onclick parameters for button {idx}.",
                        int(step4_start + idx * step4_increment),
                    )
                    continue
                btn_id, btn_token = match.group(1), match.group(2)

                # For the first button, wait a bit longer before clicking to ensure everything is loaded
                # if idx == 1:
                #     print("Extra wait for first button to ensure page is ready...")
                #     time.sleep(2)

                duration_txt = self._extract_content(button, -1, idx)
                title_txt = self._extract_content(button, -2, idx)
                print(f"\nClicking button {idx} with id: {btn_id}")
                print(f"Token: {btn_token[:60]}...")

                self.driver.get_log("performance")
                try:
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block:'center'});", button
                    )
                    button.click()
                except Exception as e:
                    print(f"Button {idx}: Click Excepted, skipping.")
                    emit_signal(
                        f"Click Excepted for button {idx}, skipping.",
                        int(step4_start + idx * step4_increment),
                    )
                    continue

                found_audio_url = False
                timeout = 10.0
                poll_interval = 0.1
                waited = 0.0
                audio_url = None
                while waited < timeout:
                    logs = self.driver.get_log("performance")
                    for entry in logs:
                        message = entry["message"]
                        if (
                            "Network.responseReceived" in message
                            and "AjxGetAudioUrl" in message
                        ):
                            message_json = json.loads(message)["message"]
                            request_id = message_json["params"]["requestId"]
                            try:
                                response_body = self.driver.execute_cdp_cmd(
                                    "Network.getResponseBody", {"requestId": request_id}
                                )
                                audio_url = response_body.get("body", "")
                                if audio_url:
                                    print(
                                        f"Audio URL for button {btn_id}:\n{audio_url}\n"
                                    )
                                    found_audio_url = True
                                    break
                            except Exception as e:
                                print(f"Failed to get response body: {e}")
                    if found_audio_url:
                        break
                    time.sleep(poll_interval)
                    waited += poll_interval

                if found_audio_url and audio_url:
                    audio_URLs.append(audio_url)
                    durations.append(self._duration_txt_to_seconds(duration_txt))
                    titles.append(title_txt)
                else:
                    print(
                        f"\033[91m /!\\ No audio URL found for button {btn_id}.\033[0m"
                    )
                    emit_signal(
                        f"No audio URL found for button {btn_id}.",
                        int(step4_start + idx * step4_increment),
                    )

            # Step 5: Finalize and save (10%)
            step5_start = step4_end
            step5_end = end_percentage
            emit_signal("Finalizing and saving data...", step5_start)
            self.driver.quit()
            print("All extracted audio URLs:")
            for url in audio_URLs:
                print(f"* {url}")
            np.save(self.audio_urls_file, np.array(audio_URLs))
            np.save(self.audio_full_durations_file, np.array(durations))
            np.save(self.audio_titles_file, np.array(titles))
            emit_signal("Finalizing and saving data...", step5_end)

        except (NaxosDriverError, NaxosPageLoadError, NaxosContentError, 
                NaxosAudioExtractionError, NaxosTimeoutError) as e:
            # Re-raise custom exceptions as-is
            if hasattr(self, "driver") and self.driver:
                self.driver.quit()
            raise e
        except Exception as e:
            # Convert generic exceptions to user-friendly ones
            if hasattr(self, "driver") and self.driver:
                self.driver.quit()
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise NaxosTimeoutError()
            elif "network" in str(e).lower() or "connection" in str(e).lower():
                raise NaxosPageLoadError(self.naxos_url)
            else:
                raise NaxosPageLoadError(self.naxos_url, f"Unexpected error occurred while loading page: {str(e)[:100]}")

    def download_audios(
        self,
        verbose_signal_percentage: pyqtBoundSignal | None = None,
        start_percentage: int = 0,
        end_percentage: int = 100,
    ):
        """Download audio files from scraped URLs and convert them to WAV format.

        Downloads MP4 audio previews from the URLs obtained by scrape_audio_URLs(),
        then converts them to WAV format using FFmpeg. Original MP4 files are
        deleted after conversion to save disk space.

        Args:
            verbose_signal_percentage (pyqtBoundSignal, optional): Qt signal for
                emitting progress updates. Defaults to None.
            start_percentage (int): Starting percentage for progress reporting.
                Defaults to 0.
            end_percentage (int): Ending percentage for progress reporting.
                Defaults to 100.

        Raises:
            FileNotFoundError: If audio URLs have not been scraped yet.

        Note:
            Requires scrape_audio_URLs() to be called first. Files are saved
            with zero-padded numerical prefixes for proper sorting.
            If audio files are already cached, the method returns early.
        """
        if not os.path.exists(self.audio_urls_file):
            raise FileNotFoundError(
                f"Audio URLs file not found at {self.audio_urls_file}. Please run scrape_audio_URLs() first."
            )

        if self._is_cached(self.audio_files_path):
            print(
                f"Audio files already cached in {self.audio_files_path}. Skipping download."
            )
            return

        # Helper to emit signal if provided
        def emit_signal(message, percentage):
            if verbose_signal_percentage is not None:
                # print("Should Emit:", message, percentage)
                verbose_signal_percentage.emit(message, percentage)

        audio_URLs = np.load(self.audio_urls_file, allow_pickle=True)

        # Pad the index with zeros for proper alphabetical sorting (e.g., 001, 002, ... 010, 011, ...)
        num_digits = len(str(len(audio_URLs)))

        step_1_start = start_percentage
        step_1_end = ((end_percentage - start_percentage) / 2) + step_1_start
        ratio_per_file = (step_1_end - step_1_start) / len(audio_URLs)

        for idx, url in enumerate(audio_URLs):
            emit_signal(
                f"Downloading audio file {idx + 1}/{len(audio_URLs)}...",
                int(step_1_start + idx * ratio_per_file),
            )
            if isinstance(url, str) and url.startswith("http"):
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()  # Raise exception for bad status codes
                    file_path = os.path.join(
                        self.audio_files_path,
                        f"30s_preview_{str(idx + 1).zfill(num_digits)}.mp4",
                    )
                    with open(file_path, "wb") as file:
                        file.write(response.content)
                except requests.RequestException as e:
                    print(f"Failed to download audio file {idx + 1}: {e}")
                    if "timeout" in str(e).lower():
                        raise NaxosTimeoutError("audio download")
                    else:
                        raise NaxosNetworkError()
            else:
                print(f"Skipping invalid URL at index {idx}: {url}")

        mp4_files_in_preview = [
            f for f in os.listdir(self.audio_files_path) if f.endswith(".mp4")
        ]

        step_2_start = step_1_end
        step_2_end = end_percentage
        ratio_per_file = (step_2_end - step_2_start) / len(mp4_files_in_preview)

        for filename in mp4_files_in_preview:
            emit_signal(
                f"Converting {filename} to WAV...",
                int(
                    step_2_start + mp4_files_in_preview.index(filename) * ratio_per_file
                ),
            )
            audio_file_path = os.path.join(self.audio_files_path, filename)
            sr, ch = self._extract_sr_and_channels_from_mp4(audio_file_path)
            samples = self._extract_samples_from_mp4(
                audio_file_path, sample_rate=sr, channels=ch
            )
            print(f"Samples shape: {samples.shape}, Sample rate: {sr}")

            signal = Signal(
                samples=samples, sample_rate=sr, origine_filename=audio_file_path
            )
            signal.save_wav(audio_file_path.replace(".mp4", ".wav"))
            # delete the mp4 file to save space
            os.remove(audio_file_path)

    def _extract_samples_from_mp4(self, file_path, sample_rate=44100, channels=2):
        """Extract raw audio samples from MP4 file using FFmpeg.

        Uses FFmpeg to decode MP4 audio to raw PCM samples, then converts
        to numpy array. Multi-channel audio is automatically converted to mono
        by averaging channels.

        Args:
            file_path (str): Path to the MP4 audio file.
            sample_rate (int): Target sample rate for extraction. Defaults to 44100.
            channels (int): Number of audio channels. Defaults to 2 (stereo).

        Returns:
            numpy.ndarray: Mono audio samples as int16 array.

        Raises:
            subprocess.CalledProcessError: If FFmpeg conversion fails.
        """
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_path,
            "-i",
            file_path,
            "-f",
            "s16le",  # output raw 16-bit PCM audio
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),  # set audio sampling rate
            "-ac",
            str(channels),  # number of audio channels
            "-",
        ]

        # Run ffmpeg and capture output bytes
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        raw_audio = process.stdout

        # Convert raw bytes to numpy array of int16 samples
        audio_data = np.frombuffer(raw_audio, dtype=np.int16)

        # Reshape for multi-channel audio
        if channels > 1:
            audio_data = audio_data.reshape(-1, channels)
            # Average channels to mono (np.mean produces float64, convert back to int16)
            mono_audio = audio_data.mean(axis=1).astype(np.int16)
        else:
            mono_audio = audio_data

        return mono_audio

    def _extract_sr_and_channels_from_mp4(self, file_path):
        """Extract sample rate and channel count from MP4 file metadata.

        Uses TinyTag library to read audio metadata from the MP4 file
        to determine the appropriate parameters for audio extraction.

        Args:
            file_path (str): Path to the MP4 audio file.

        Returns:
            tuple: (sample_rate, channels) as integers.

        Raises:
            ValueError: If sample rate or channel count cannot be determined.
        """
        tag = TinyTag.get(file_path)
        sr, channels = tag.samplerate, tag.channels
        if sr is None or channels is None:
            raise ValueError(
                f"Could not extract sample rate or channels from {file_path}"
            )
        return sr, channels

    def _clear_cache(self, path):
        """Clear all files in the specified cache directory.

        Removes all regular files from the given directory path,
        leaving subdirectories intact.

        Args:
            path (str): Directory path to clear.

        Note:
            Only removes files, not subdirectories. Does nothing if
            the path doesn't exist.
        """
        if os.path.exists(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def _is_cached(self, path):
        """Check if a directory exists and contains files.

        Args:
            path (str): Directory path to check.

        Returns:
            bool: True if directory exists and contains at least one file,
                  False otherwise.
        """
        return os.path.exists(path) and len(os.listdir(path)) > 0

    def is_scraping_complete(self):
        """Check if both scraping and downloading phases are complete.

        Verifies that all necessary files and data have been successfully
        scraped and downloaded, including audio files and metadata.

        Returns:
            bool: True if scraping is complete (both metadata files exist
                  and audio files directory contains files), False otherwise.

        Note:
            This method checks for:
            - Presence of audio files in the cache directory
            - Existence of metadata files (URLs, durations, titles)
        """
        contains_audios = self._is_cached(self.audio_files_path)

        contains_meta_files = (
            os.path.exists(self.audio_urls_file)
            and os.path.exists(self.audio_full_durations_file)
            and os.path.exists(self.audio_titles_file)
        )

        return contains_audios and contains_meta_files
