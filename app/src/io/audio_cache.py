"""Audio Cache Manager - Advanced temporary audio file management system.

This module provides a sophisticated singleton-based cache management system for
handling temporary audio files in multi-threaded audio analysis applications.
It addresses common issues with file locking, resource contention, and cleanup
in Windows environments where multiple processes may access the same audio files.

Key Features:
    - Singleton pattern ensuring single instance across application
    - Thread-safe operations with reentrant locking mechanisms
    - Reference counting to prevent premature file deletion
    - Deduplication based on source file hashing
    - Aggressive Windows-specific deletion strategies
    - Automatic cleanup on application shutdown
    - Support for multiple AudioModel instances sharing files
    - Centralized cache statistics and monitoring

The cache manager creates temporary WAV files from various source formats,
manages their lifecycle through reference counting, and ensures proper cleanup
even in case of application crashes or improper shutdown.

Author: Hugo Demule
Date: January 2026
"""

import atexit
import gc
import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import librosa
import soundfile as sf
from constants.paths import CACHE_PATH, NAXOS_CACHE_PATH

logger = logging.getLogger(__name__)


class AudioCacheManager:
    """Singleton cache manager for temporary audio files with advanced lifecycle management.

    This class implements a thread-safe singleton pattern for managing temporary audio
    files used across multiple AudioModel instances. It provides reference counting,
    deduplication, and aggressive cleanup strategies specifically designed for Windows
    environments where file locking can be problematic.

    Architecture:
        - Singleton pattern ensures single instance across the application
        - Thread-safe operations using reentrant locks
        - Reference counting prevents premature file deletion
        - Hash-based deduplication reduces storage requirements
        - Multiple deletion strategies for Windows compatibility

    Features:
        - Automatic temporary file creation from various audio formats
        - Reference counting with automatic cleanup when unused
        - Thread-safe operations for multi-threaded applications
        - Deduplication based on source file hash to save space
        - Aggressive deletion methods for Windows file locking issues
        - Centralized cleanup for application shutdown scenarios
        - Statistics and monitoring capabilities

    Attributes:
        _instance (AudioCacheManager): Singleton instance.
        _lock (threading.Lock): Class-level lock for singleton creation.
        _temp_files (Dict): Mapping of temp paths to file metadata.
        _source_to_temp (Dict): Mapping of source paths to temp paths.
        _file_lock (threading.RLock): Reentrant lock for file operations.
        _temp_dir (Path): Directory for temporary audio files.

    Example:
        >>> cache = AudioCacheManager()  # Gets singleton instance
        >>> temp_path = cache.create_temp_file("/path/to/audio.mp3")
        >>> # Use temp_path for audio processing
        >>> cache.release_temp_file(temp_path)  # Cleanup when done

    Note:
        This class addresses specific Windows file locking issues that can occur
        when multiple processes or threads access the same audio files simultaneously.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Implement thread-safe singleton pattern.

        Ensures only one instance of AudioCacheManager exists throughout
        the application lifecycle, using double-checked locking for
        thread safety.

        Returns:
            AudioCacheManager: The singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the cache manager (called only once due to singleton pattern).

        Sets up the temporary directory structure, initializes tracking dictionaries,
        configures thread-safe locks, and registers cleanup handlers. Also performs
        initial cleanup of any temporary files from previous application sessions.

        Note:
            Due to the singleton pattern, this method only executes once regardless
            of how many times AudioCacheManager() is called.
        """
        if self._initialized:
            return

        self._initialized = True
        self._temp_files: Dict[str, dict] = (
            {}
        )  # temp_path -> {ref_count, source_path, hash}
        self._source_to_temp: Dict[str, str] = {}  # source_path -> temp_path mapping
        self._file_lock = threading.RLock()  # Reentrant lock for nested calls

        # Create temp directory
        self._temp_dir = Path(CACHE_PATH) / "audio_temp"
        self._temp_dir.mkdir(exist_ok=True)

        # Register cleanup on application exit
        atexit.register(self.cleanup_all)

        # Clean up old files from previous sessions
        self._cleanup_old_files()

        logger.info("AudioCacheManager initialized")

    def create_temp_file(self, source_audio_path: str) -> Optional[str]:
        """Create or reuse a temporary audio file for the specified source.

        This method implements intelligent caching by checking if a temporary file
        already exists for the given source audio. If found, it increments the
        reference count and returns the existing path. Otherwise, it creates a
        new temporary WAV file with proper reference tracking.

        The method performs:
            - Source path validation and mapping lookup
            - Reference count management for existing files
            - New temporary file creation with audio format conversion
            - Thread-safe operations with proper locking
            - Error handling with graceful fallback

        Args:
            source_audio_path (str): Path to the original audio file to cache.
                                   Can be any format supported by librosa.

        Returns:
            Optional[str]: Path to the temporary WAV file if successful,
                          None if creation failed due to I/O errors or
                          invalid source path.

        Note:
            The returned temporary file will be in WAV format regardless of
            the source format, ensuring consistent audio handling across
            the application.
        """
        with self._file_lock:
            try:
                # Check if we already have a temp file for this source
                if source_audio_path in self._source_to_temp:
                    temp_path = self._source_to_temp[source_audio_path]
                    if temp_path in self._temp_files and os.path.exists(temp_path):
                        # Increment reference count
                        self._temp_files[temp_path]["ref_count"] += 1
                        logger.info(
                            f"Reusing temp file: {temp_path} (refs: {self._temp_files[temp_path]['ref_count']})"
                        )
                        return temp_path
                    else:
                        # Clean up stale mapping
                        del self._source_to_temp[source_audio_path]
                        if temp_path in self._temp_files:
                            del self._temp_files[temp_path]

                # Create new temp file
                return self._create_new_temp_file(source_audio_path)

            except Exception as e:
                logger.error(f"Failed to create temp file for {source_audio_path}: {e}")
                return None

    def _create_new_temp_file(self, source_audio_path: str) -> str:
        """Create a new temporary audio file from the source.

        Generates a deterministic filename based on the source path hash,
        loads the audio using librosa, converts it to WAV format, and
        registers it in the cache tracking system.

        Args:
            source_audio_path (str): Path to the source audio file.

        Returns:
            str: Path to the newly created temporary WAV file.

        Raises:
            Various exceptions may be raised during audio loading or file I/O.

        Note:
            The temporary filename is generated using an MD5 hash of the source
            path, ensuring consistent naming and deduplication.
        """
        # Generate deterministic filename based on source
        file_hash = hashlib.md5(source_audio_path.encode()).hexdigest()[:8]
        temp_filename = f"audio_{file_hash}.wav"
        temp_path = str(self._temp_dir / temp_filename)

        logger.info(f"Creating temp file for: {source_audio_path}")

        # Load and process audio
        audio_data, sample_rate = librosa.load(source_audio_path, sr=None)
        sf.write(temp_path, audio_data, sample_rate)

        # Register the temp file
        self._temp_files[temp_path] = {
            "ref_count": 1,
            "source_path": source_audio_path,
            "hash": file_hash,
            "created_time": time.time(),
        }
        self._source_to_temp[source_audio_path] = temp_path

        logger.info(f"Created temp file: {temp_path}")
        return temp_path

    def release_temp_file(self, temp_path: str) -> bool:
        """Release a reference to a temporary file with automatic cleanup.

        Decrements the reference count for the specified temporary file.
        If the reference count reaches zero, the file is automatically
        deleted using aggressive deletion strategies suitable for Windows.

        This method implements proper reference counting to ensure temporary
        files are only deleted when no longer needed by any part of the
        application, preventing premature deletion and access errors.

        Args:
            temp_path (str): Path to the temporary file to release.

        Returns:
            bool: True if the file was successfully released (and deleted if
                  ref count reached zero), False if the file was unknown or
                  deletion failed.

        Note:
            It's safe to call this method even if the file has already been
            deleted or was never registered - it will log a warning but not fail.
        """
        with self._file_lock:
            if temp_path not in self._temp_files:
                logger.warning(f"Attempted to release unknown temp file: {temp_path}")
                return False

            # Decrement reference count
            self._temp_files[temp_path]["ref_count"] -= 1
            ref_count = self._temp_files[temp_path]["ref_count"]

            logger.info(
                f"Released temp file: {temp_path} (refs remaining: {ref_count})"
            )

            # If no more references, delete the file
            if ref_count <= 0:
                return self._delete_temp_file(temp_path)

            return True

    def _delete_temp_file(self, temp_path: str) -> bool:
        """Delete a temporary file using aggressive deletion strategies.

        Removes the file from tracking dictionaries and attempts to delete
        the actual file using multiple deletion methods specifically designed
        for Windows file locking issues.

        Args:
            temp_path (str): Path to the temporary file to delete.

        Returns:
            bool: True if the file was successfully deleted, False otherwise.

        Note:
            This method uses aggressive deletion strategies including file
            renaming, command-line tools, and PowerShell commands to handle
            Windows file locking scenarios.
        """
        try:
            file_info = self._temp_files[temp_path]
            source_path = file_info["source_path"]

            # Remove from tracking
            del self._temp_files[temp_path]
            if source_path in self._source_to_temp:
                del self._source_to_temp[source_path]

            # Force garbage collection to release any references
            gc.collect()

            # Attempt aggressive deletion
            success = self._force_delete_file(Path(temp_path))

            if success:
                logger.info(f"Successfully deleted temp file: {temp_path}")
            else:
                logger.warning(f"Failed to delete temp file: {temp_path}")

            return success

        except Exception as e:
            logger.error(f"Error deleting temp file {temp_path}: {e}")
            return False

    def _force_delete_file(self, file_path: Path) -> bool:
        """Use multiple Windows-specific methods to force delete a file.

        Implements a cascading series of deletion strategies specifically
        designed to handle Windows file locking issues that can occur when
        files are being accessed by multiple processes or audio libraries.

        Deletion strategies tried in order:
        1. Standard os.remove() - fastest method when it works
        2. Rename then delete - Windows-specific trick to bypass locks
        3. Windows DEL command with force flags - command-line approach
        4. PowerShell Remove-Item with force - PowerShell approach

        Args:
            file_path (Path): Path object pointing to the file to delete.

        Returns:
            bool: True if any deletion method succeeded, False if all failed.

        Note:
            This method is specifically designed for Windows environments
            where file locking can prevent normal deletion operations.
            On other platforms, it gracefully falls back to standard methods.
        """
        if not file_path.exists():
            return True

        logger.info(f"Attempting to force delete: {file_path}")

        # Method 1: Try normal deletion first
        try:
            os.remove(file_path)
            logger.debug(f"✓ Normal deletion succeeded: {file_path}")
            return True
        except (PermissionError, OSError) as e:
            print(f"Normal deletion failed: {e}")

        # Method 2: Try renaming first, then delete (Windows trick)
        try:
            temp_name = str(file_path) + ".deleting"
            os.rename(str(file_path), temp_name)
            os.remove(temp_name)
            logger.debug(f"✓ Rename+delete succeeded: {file_path}")
            return True
        except (PermissionError, OSError) as e:
            print(f"Rename+delete failed: {e}")

        # Method 3: Use Windows DEL command with force flags
        if sys.platform == "win32":
            try:
                result = subprocess.run(
                    ["cmd", "/c", "del", "/f", "/q", str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and not file_path.exists():
                    logger.debug(f"✓ Windows DEL command succeeded: {file_path}")
                    return True
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                print(f"Windows DEL command failed: {e}")

        # Method 4: Use PowerShell Remove-Item with force
        if sys.platform == "win32":
            try:
                powershell_cmd = f"Remove-Item -Path '{file_path}' -Force -ErrorAction SilentlyContinue"
                result = subprocess.run(
                    ["powershell", "-Command", powershell_cmd],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if not file_path.exists():
                    logger.debug(f"✓ PowerShell force delete succeeded: {file_path}")
                    return True
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

        logger.warning(f"✗ All deletion methods failed for: {file_path}")
        return False

    def cleanup_all(self) -> None:
        """Force cleanup of all temporary files during application shutdown.

        Performs comprehensive cleanup of all tracked temporary files using
        aggressive deletion methods. This method is automatically called during
        application shutdown via atexit registration.

        The cleanup process includes:
            - Force garbage collection to release file handles
            - Iteration through all tracked temporary files
            - Aggressive deletion using multiple strategies
            - Tracking dictionary cleanup regardless of deletion success
            - Statistical logging of cleanup results

        Note:
            This method is designed to be robust and will continue cleaning up
            even if individual file deletions fail. It logs detailed statistics
            about the cleanup process for debugging purposes.
        """
        with self._file_lock:
            logger.info("AudioCacheManager: Starting cleanup of all temp files")

            # Force garbage collection first
            gc.collect()

            temp_files_copy = list(self._temp_files.keys())
            deleted_count = 0

            for temp_path in temp_files_copy:
                try:
                    if self._force_delete_file(Path(temp_path)):
                        deleted_count += 1
                    # Remove from tracking regardless of deletion success
                    if temp_path in self._temp_files:
                        source_path = self._temp_files[temp_path]["source_path"]
                        del self._temp_files[temp_path]
                        if source_path in self._source_to_temp:
                            del self._source_to_temp[source_path]
                except Exception as e:
                    logger.error(f"Error during cleanup of {temp_path}: {e}")

            logger.info(
                f"AudioCacheManager: Cleanup complete. Deleted {deleted_count}/{len(temp_files_copy)} files"
            )

            # Clear all tracking
            self._temp_files.clear()
            self._source_to_temp.clear()

    def cleanup_unused(self) -> int:
        """Clean up temporary files that are no longer referenced.

        Identifies and removes temporary files with zero or negative reference
        counts, indicating they are no longer needed by any part of the application.
        This is useful for periodic cleanup during application runtime.

        Returns:
            int: Number of files successfully cleaned up.

        Note:
            This method only removes files with ref_count <= 0, preserving
            files that are still in use. It's safe to call during normal
            application operation.
        """
        with self._file_lock:
            unused_files = [
                path
                for path, info in self._temp_files.items()
                if info["ref_count"] <= 0
            ]

            cleaned_count = 0
            for temp_path in unused_files:
                if self._delete_temp_file(temp_path):
                    cleaned_count += 1

            logger.info(f"Cleaned up {cleaned_count} unused temp files")
            return cleaned_count

    def _cleanup_old_files(self) -> None:
        """Clean up old temporary files from previous application sessions.

        Removes any temporary files that may have been left behind by previous
        application instances that didn't shut down cleanly. This ensures a
        clean starting state for the current session.

        This method is called during initialization and attempts to delete
        all files found in the temporary directory, regardless of their age
        or properties.

        Note:
            This is an aggressive cleanup that removes ALL files in the temp
            directory, assuming they are orphaned from previous sessions.
        """
        try:
            if not self._temp_dir.exists():
                return

            current_time = time.time()
            old_files = []

            for temp_file in self._temp_dir.iterdir():
                try:
                    is_gitkeep = temp_file.name == ".gitkeep"
                    if is_gitkeep:
                        continue
                    old_files.append(temp_file)
                except OSError:
                    # File might be in use or deleted, skip
                    continue

            deleted_count = 0
            for old_file in old_files:
                print(" * Deleting old temp file:", old_file)
                if self._force_delete_file(old_file):
                    deleted_count += 1

            if deleted_count > 0:
                logger.info(
                    f"Cleaned up {deleted_count} old temp files from previous sessions"
                )

        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")

    def get_stats(self) -> dict:
        """Get comprehensive statistics about the audio cache state.

        Provides detailed information about the current cache state including
        file counts, reference counts, directory information, and per-file
        reference tracking for debugging and monitoring purposes.

        Returns:
            dict: Dictionary containing cache statistics:
                - total_files (int): Total number of cached files
                - total_references (int): Sum of all reference counts
                - temp_directory (str): Path to temporary directory
                - files_by_ref_count (Dict[str, int]): Mapping of file paths
                  to their current reference counts

        Example:
            >>> cache = AudioCacheManager()
            >>> stats = cache.get_stats()
            >>> print(f"Cache has {stats['total_files']} files")
            >>> print(f"Total references: {stats['total_references']}")
        """
        with self._file_lock:
            total_refs = sum(info["ref_count"] for info in self._temp_files.values())
            return {
                "total_files": len(self._temp_files),
                "total_references": total_refs,
                "temp_directory": str(self._temp_dir),
                "files_by_ref_count": {
                    path: info["ref_count"] for path, info in self._temp_files.items()
                },
            }

    def force_release_all_references(self, source_path: str) -> bool:
        """Force release all references to files from a specific source.

        Immediately removes all references and deletes the temporary file
        associated with the specified source path. This is useful for recovery
        scenarios where AudioModel instances have crashed or failed to clean
        up properly.

        This method bypasses the normal reference counting mechanism and
        forces immediate cleanup, making it suitable for error recovery
        situations.

        Args:
            source_path (str): Path to the original audio file whose temporary
                             file should be forcibly cleaned up.

        Returns:
            bool: True if successful or if no temporary file existed for the
                  source, False if deletion failed.

        Warning:
            This method should only be used in error recovery scenarios as
            it can cause issues if other parts of the application are still
            using the temporary file.
        """
        with self._file_lock:
            if source_path not in self._source_to_temp:
                return True

            temp_path = self._source_to_temp[source_path]
            if temp_path in self._temp_files:
                logger.warning(f"Force releasing all references for {source_path}")
                return self._delete_temp_file(temp_path)

            return True

    def initial_cleanup(self) -> None:
        """Perform initial cleanup of old temporary files on startup.

        This is a convenience method that delegates to _cleanup_old_files()
        and can be called explicitly if needed during application initialization.

        Note:
            This method is automatically called during AudioCacheManager
            initialization, so manual calls are typically unnecessary.
        """
        self._cleanup_old_files()

    def clear_naxos_cache(self) -> None:
        """Clear the Naxos-specific cache directory completely.

        Removes all files and subdirectories from the Naxos cache directory
        while preserving the main cache directory structure. This is useful
        for clearing downloaded audio previews and metadata from Naxos.

        The method performs:
            - Recursive file deletion in the Naxos cache directory
            - Subdirectory removal (bottom-up to handle nested structures)
            - Error handling for locked or in-use files
            - Preservation of the main cache directory structure

        Note:
            This method specifically targets the NAXOS_CACHE_PATH and does
            not affect the general audio temporary file cache managed by
            this class.
        """
        if os.path.exists(NAXOS_CACHE_PATH):
            for root, dirs, files in os.walk(NAXOS_CACHE_PATH):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted cached file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting file {file_path}: {e}")

            # Remove all subdirectories but keep the main NAXOS_CACHE folder
            for root, dirs, files in os.walk(NAXOS_CACHE_PATH, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.rmdir(dir_path)
                        print(f"Deleted directory: {dir_path}")
                    except Exception as e:
                        print(f"Error deleting directory {dir_path}: {e}")
        else:
            print("Cache directory does not exist.")


# Global singleton instance - use this throughout the application
# This ensures consistent cache management across all audio processing components
AudioCache = AudioCacheManager()
