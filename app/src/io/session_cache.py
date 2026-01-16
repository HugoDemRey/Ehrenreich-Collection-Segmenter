"""Session Cache Manager - Advanced state persistence for audio analysis applications.

This module provides comprehensive session management and caching functionality for
audio analysis applications. It handles the persistence and restoration of complex
application states including timeline data, analysis module results, and cross-module
relationships.

Key Features:
    - Single active session management with automatic cleanup
    - Persistent storage of analysis results and parameters
    - Timeline state management with user-added transitions
    - Module state caching with lazy loading capabilities
    - Session serialization to custom .ehra (Ehrenreich Analysis) files
    - Thread-safe operations with proper locking mechanisms
    - Signal-based notifications for session lifecycle events

The cache manager supports:
    - Timeline state (transitions with colors and metadata)
    - Module states (novelty curves, parameters, transitions)
    - Cross-module relationships and dependencies
    - Audio signal fingerprinting for consistency validation
    - Automatic cleanup on application exit

Author: Hugo Demule
Date: January 2026
"""

import atexit
import hashlib
import json
import logging
import os
import pickle
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from constants.paths import CACHE_PATH
from constants.segmenter_configurations import SegmenterConfig
from views.loading_dialog import LoadingWindow

from src.audio.signal import Signal

logger = logging.getLogger(__name__)


class CacheEntryType(Enum):
    """Enumeration of cache entry types for categorizing stored data.

    This enum defines the different types of data that can be cached by the
    session manager, enabling proper categorization and handling of various
    data structures.

    Attributes:
        TIMELINE: Timeline-related data including transitions and user annotations.
        MODULE: Analysis module data including parameters and computed results.
        SESSION: Session metadata and overall application state.
    """

    TIMELINE = "timeline"
    MODULE = "module"
    SESSION = "session"


@dataclass
class TimelineState:
    """Represents timeline state with user-added transitions and data access methods.

    This class encapsulates the timeline state of an audio analysis session,
    providing access to user-defined transitions, their properties, and associated
    metadata. It implements lazy loading through cache manager integration.

    The timeline state includes:
        - User-added segment transitions with timestamps
        - Transition colors and visual properties
        - Metadata associated with timeline segments
        - References to the parent session for data persistence

    Attributes:
        session_id (str): Unique identifier of the parent session.
        _cache_manager (SessionCacheManager, optional): Reference to the cache
            manager for data access operations.

    Note:
        This class uses lazy loading - actual timeline data is retrieved from
        the cache manager on demand rather than being stored directly.
    """

    session_id: str  # Reference to parent session
    _cache_manager: Optional["SessionCacheManager"] = None  # Reference to cache manager

    def _set_cache_context(self, session_id: str, cache_manager: "SessionCacheManager"):
        """Set cache context for data access operations.

        Internal method used to establish the connection between this timeline
        state and its cache manager after deserialization or initialization.

        Args:
            session_id (str): Unique session identifier.
            cache_manager (SessionCacheManager): Cache manager instance for data access.
        """
        self.session_id = session_id
        self._cache_manager = cache_manager

    def _get_timeline_data(self) -> Optional[Dict]:
        """Retrieve complete timeline data from the cache manager.

        Returns:
            Optional[Dict]: Timeline data dictionary containing transitions and
                           metadata, or None if unavailable.
        """
        if not self._cache_manager or not self.session_id:
            return None
        return self._cache_manager._load_timeline_data(self.session_id)

    def get_transitions_sec(self) -> List[Dict[str, Any]]:
        """Get all timeline transitions with their properties.

        Returns:
            List[Dict[str, Any]]: List of transition dictionaries containing
                                 timestamps, colors, and other properties.
                                 Returns empty list if no transitions exist.
        """
        data = self._get_timeline_data()
        if data and "transitions_sec" in data:
            return data["transitions_sec"]
        return []

    def get_transition_count(self) -> int:
        """Get the total number of transitions in the timeline.

        Returns:
            int: Number of transitions currently stored in the timeline.
        """
        return len(self.get_transitions_sec())

    def get_data(self) -> Optional[Dict]:
        """Get all available timeline data.

        Returns:
            Optional[Dict]: Complete timeline data dictionary, or None if unavailable.
        """
        return self._get_timeline_data()

    def to_dict(self) -> Dict:
        """Serialize timeline state to dictionary for JSON storage.

        Returns:
            Dict: Dictionary representation containing serializable fields only.
        """
        return {"session_id": self.session_id}

    @classmethod
    def from_dict(cls, data: Dict) -> "TimelineState":
        """Deserialize timeline state from dictionary.

        Args:
            data (Dict): Dictionary containing serialized timeline state data.

        Returns:
            TimelineState: Reconstructed timeline state instance.
        """
        return cls(
            session_id=data.get("session_id", ""),
        )


@dataclass
class ModuleState:
    """Represents the complete state of a single analysis module.

    This class encapsulates all data associated with an analysis module including
    configuration parameters, computed results, and metadata. It provides lazy
    loading access to heavy data objects through cache manager integration.

    Module state includes:
        - Analysis parameters and configuration settings
        - Computed novelty curves and feature data
        - Detected transitions and boundaries
        - Computation timestamps and provenance information

    Attributes:
        module_type (SegmenterConfig): Type/configuration of the analysis module.
        computation_timestamp (datetime): When the module computation was performed.
        page_id (str): Identifier of the UI page/context where module was created.
        _module_id (str, optional): Unique module identifier for cache access.
        _cache_manager (SessionCacheManager, optional): Reference to cache manager.

    Note:
        Heavy data (numpy arrays, large lists) are stored separately in the cache
        and loaded on demand to optimize memory usage and serialization performance.
    """

    module_type: SegmenterConfig
    computation_timestamp: datetime
    page_id: str
    _module_id: Optional[str] = None
    _cache_manager: Optional["SessionCacheManager"] = None

    def _set_cache_context(self, module_id: str, cache_manager: "SessionCacheManager"):
        """Set cache context for data access operations.

        Internal method used to establish the connection between this module
        state and its cache manager after deserialization or initialization.

        Args:
            module_id (str): Unique module identifier.
            cache_manager (SessionCacheManager): Cache manager instance for data access.
        """
        self._module_id = module_id
        self._cache_manager = cache_manager

    def _get_module_data(self) -> Optional[Dict]:
        """Retrieve complete module data from the cache manager.

        Returns:
            Optional[Dict]: Module data dictionary containing all computed results,
                           or None if unavailable.
        """
        if not self._cache_manager or not self._module_id:
            return None
        return self._cache_manager._load_module_data(self._module_id)

    def get_nc_x(self) -> Optional[np.ndarray]:
        """Get novelty curve X-axis values (time coordinates).

        Returns:
            Optional[np.ndarray]: Time values for the novelty curve, typically
                                 in seconds. Returns None if not available.
        """
        data = self._get_module_data()
        if data and "nc_x" in data:
            return data["nc_x"]
        return None

    def get_nc_y(self) -> Optional[np.ndarray]:
        """Get novelty curve Y-axis values (novelty/feature values).

        Returns:
            Optional[np.ndarray]: Novelty curve values representing detected
                                 changes or features. Returns None if not available.
        """
        data = self._get_module_data()
        if data and "nc_y" in data:
            return data["nc_y"]
        return None

    def get_transitions_sec(self) -> Optional[List[float]]:
        """Get detected transitions/boundaries in seconds.

        Returns:
            Optional[List[float]]: List of transition timestamps in seconds,
                                  or None if not available.
        """
        data = self._get_module_data()
        if data and "transitions_sec" in data:
            return data["transitions_sec"]
        return None

    def get_parameters(self) -> Optional[Dict[str, Any]]:
        """Get analysis parameters used for this module.

        Returns:
            Optional[Dict[str, Any]]: Dictionary of parameter names and values
                                     used for the analysis, or None if not available.
        """
        data = self._get_module_data()
        if data and "parameters" in data:
            return data["parameters"]
        return None

    def get_data(self) -> Optional[Dict]:
        """Get all available data from this module.

        Returns:
            Optional[Dict]: Complete module data dictionary containing all
                           computed results and metadata, or None if unavailable.
        """
        return self._get_module_data()

    def to_dict(self) -> Dict:
        """Serialize module state to dictionary for JSON storage.

        Returns:
            Dict: Dictionary representation containing serializable metadata only.
                 Heavy data objects are stored separately in the cache.
        """
        # Manually construct dict to avoid serializing internal fields
        return {
            "module_type": self.module_type.value,
            "computation_timestamp": self.computation_timestamp.isoformat(),
            "page_id": self.page_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModuleState":
        """Deserialize module state from dictionary.

        Args:
            data (Dict): Dictionary containing serialized module state metadata.

        Returns:
            ModuleState: Reconstructed module state instance.
        """
        return cls(
            module_type=SegmenterConfig(data["module_type"]),
            computation_timestamp=datetime.fromisoformat(data["computation_timestamp"]),
            page_id=data["page_id"],
        )


@dataclass
class SessionState:
    """Represents the complete application session state.

    This class encapsulates all data associated with an active analysis session,
    including timeline state, module states, audio file information, and metadata.
    It serves as the top-level container for session persistence and restoration.

    Session state includes:
        - Session identification and timestamps
        - Audio file path and signal fingerprint
        - Timeline state with user transitions
        - Collection of all analysis module states
        - Cross-module relationships and dependencies

    Attributes:
        session_id (str): Unique session identifier.
        created_timestamp (datetime): When the session was first created.
        last_modified (datetime): Last modification timestamp.
        timeline_state (TimelineState): Timeline data and user transitions.
        module_states (Dict[str, ModuleState]): Mapping of module IDs to states.
        audio_file_path (str): Path to the associated audio file.
        signal_hash (str): Hash of audio signal properties for consistency validation.

    Note:
        The signal hash is used to ensure that loaded sessions match the current
        audio file, preventing inconsistencies when restoring analysis results.
    """

    session_id: str
    created_timestamp: datetime
    last_modified: datetime
    timeline_state: TimelineState
    module_states: Dict[str, ModuleState]  # module_id -> ModuleState
    audio_file_path: str
    signal_hash: str

    def to_dict(self) -> Dict:
        """Serialize session state to dictionary for JSON storage.

        Returns:
            Dict: Complete dictionary representation of the session state,
                 including all nested timeline and module states.
        """
        data = {
            "session_id": self.session_id,
            "created_timestamp": self.created_timestamp.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "timeline_state": self.timeline_state.to_dict(),
            "module_states": {k: v.to_dict() for k, v in self.module_states.items()},
            "audio_file_path": self.audio_file_path,
            "signal_hash": self.signal_hash,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "SessionState":
        """Deserialize session state from dictionary.

        Args:
            data (Dict): Dictionary containing serialized session state.

        Returns:
            SessionState: Reconstructed session state instance with all
                         nested states properly restored.
        """
        return cls(
            session_id=data["session_id"],
            created_timestamp=datetime.fromisoformat(data["created_timestamp"]),
            last_modified=datetime.fromisoformat(data["last_modified"]),
            timeline_state=TimelineState.from_dict(data["timeline_state"]),
            module_states={
                k: ModuleState.from_dict(v) for k, v in data["module_states"].items()
            },
            audio_file_path=data["audio_file_path"],
            signal_hash=data["signal_hash"],
        )


from PyQt6.QtCore import QObject, pyqtSignal


class SessionCacheManager(QObject):
    """Advanced session cache manager with comprehensive state persistence.

    This class provides a sophisticated caching system for audio analysis applications,
    managing the complete lifecycle of analysis sessions from creation to persistence.
    It implements a single-session architecture with automatic cleanup and supports
    serialization to custom .ehra (Ehrenreich Analysis) files.

    Key Features:
        - Single active session management with thread-safe operations
        - Lazy loading of heavy data objects (numpy arrays, large lists)
        - Custom .ehra file format for session persistence
        - Audio signal fingerprinting for consistency validation
        - Automatic cache cleanup on application exit
        - Qt signal emissions for session lifecycle events
        - Module state management with parameter copying
        - Timeline state persistence with user transitions

    Architecture:
        - Uses temporary cache directory for active session data
        - Separates metadata (JSON) from heavy data (pickle)
        - Thread-safe with RLock for concurrent access
        - Implements proper cleanup patterns

    Signals:
        s_session_loaded: Emitted when a session is successfully loaded

    Example:
        >>> cache_manager = SessionCacheManager()
        >>> session_id = cache_manager.create_session("/path/to/audio.wav", signal)
        >>> cache_manager.save_session_to_file("/path/to/project.ehra")
        >>> success, msg = cache_manager.load_session_from_file("/path/to/project.ehra")

    Note:
        Only one session can be active at a time. Creating a new session
        automatically clears any existing session data.
    """

    s_session_loaded = pyqtSignal(str)  # Emitted when a session is loaded

    def __init__(self):
        """Initialize the session cache manager.

        Sets up the cache directory structure, initializes thread-safe locks,
        clears any existing cache data, and registers cleanup handlers.

        The initialization process:
        - Creates cache directory structure if needed
        - Clears any existing active session data
        - Sets up thread-safe locks for concurrent access
        - Registers automatic cleanup on application exit
        - Initializes internal state variables

        Note:
            Any existing cache data is cleared during initialization to ensure
            a clean starting state for the new session.
        """
        super().__init__()
        self._initialized = True
        self._cache_lock = threading.RLock()

        # Single session directories (cleared on startup/close)
        Path(CACHE_PATH).mkdir(exist_ok=True)
        self._cache_dir = Path(CACHE_PATH) / "active_session"
        self._temp_session_file = self._cache_dir / "session.json"
        self._temp_modules_dir = self._cache_dir / "modules"
        self._temp_timeline_file = (
            self._cache_dir / "timeline.pkl"
        )  # New timeline PKL file

        # Clear any existing cache and recreate
        self._clear_active_cache()
        self._cache_dir.mkdir(exist_ok=True)
        self._temp_modules_dir.mkdir(exist_ok=True)

        # Current session state
        self._current_session: Optional[SessionState] = None
        self._module_cache: Dict[str, ModuleState] = {}
        self._copied_parameters: Dict[str, Any] = {}  # Store copied parameters

        # Register cleanup on app close
        atexit.register(self.cleanup_on_exit)

        logger.info("SessionCacheManager initialized (single session mode)")

    def _clear_active_cache(self):
        """Clear the active session cache directory completely.

        Removes the entire active session cache directory and all its contents,
        ensuring a clean state for new sessions.
        """
        if self._cache_dir.exists():
            shutil.rmtree(self._cache_dir)
        logger.info("Cleared active session cache")

    def cleanup_on_exit(self):
        """Perform cleanup operations when the application exits.

        This method is automatically called via atexit registration and ensures
        that temporary cache data is properly cleaned up when the application closes.
        """
        self._clear_active_cache()
        logger.info("Session cache cleaned up on exit")

    # === SESSION MANAGEMENT ===

    def create_session(self, audio_file_path: str, signal: Signal) -> str:
        """Create a new active session, replacing any existing session.

        Initializes a new analysis session with the provided audio file and signal.
        Any existing session data is automatically cleared to maintain the
        single-session architecture.

        Args:
            audio_file_path (str): Path to the audio file being analyzed.
            signal (Signal): Audio signal object containing the audio data.

        Returns:
            str: Unique session identifier for the created session.

        Note:
            The session ID is generated based on the audio file path and signal
            properties, ensuring consistency across application restarts.
        """
        with self._cache_lock:
            # Clear any existing session
            self._clear_active_cache()
            self._cache_dir.mkdir(exist_ok=True)
            self._temp_modules_dir.mkdir(exist_ok=True)

            signal_hash = self._generate_signal_hash(signal)
            session_id = self._generate_session_id(audio_file_path, signal_hash)

            timeline_state = TimelineState(session_id=session_id)
            timeline_state._set_cache_context(session_id, self)

            self._current_session = SessionState(
                session_id=session_id,
                created_timestamp=datetime.now(),
                last_modified=datetime.now(),
                timeline_state=timeline_state,
                module_states={},
                audio_file_path=audio_file_path,
                signal_hash=signal_hash,
            )

            self._module_cache.clear()
            self._save_temp_session()

            logger.info(f"Created new session: {session_id}")
            return session_id

    def _save_temp_session(self):
        """Save current session metadata to temporary cache file.

        Serializes the current session state to a JSON file in the temporary
        cache directory for quick access and recovery.
        """
        if self._current_session:
            with open(self._temp_session_file, "w") as f:
                json.dump(self._current_session.to_dict(), f, indent=2)

    def get_current_session(self) -> Optional[SessionState]:
        """Get the current active session state.

        Returns:
            Optional[SessionState]: The currently active session, or None
                                   if no session is active.
        """
        return self._current_session

    # === SAVE/LOAD TO FILES ===

    def save_session_to_file(self, file_path: str) -> bool:
        """Save current session to a .ehra (Ehrenreich Analysis) file.

        Creates a complete session bundle containing all metadata, module data,
        and timeline information, then serializes it to a custom .ehra file
        using pickle format for efficient storage.

        Args:
            file_path (str): Path where the session file should be saved.
                           The .ehra extension is added automatically if not present.

        Returns:
            bool: True if the session was saved successfully, False otherwise.

        Note:
            The .ehra file format includes:
            - Session metadata (timestamps, audio file info)
            - All module states with computed data
            - Timeline data with user transitions
            - Cross-module relationships and dependencies
        """
        with self._cache_lock:
            if not self._current_session:
                logger.error("No active session to save")
                return False

            try:
                save_path = Path(file_path)
                if not save_path.suffix:
                    save_path = save_path.with_suffix(".ehra")

                # Create session bundle
                session_data = {
                    "session_metadata": self._current_session.to_dict(),
                    "modules": {},
                    "timeline_data": self._load_timeline_data(
                        self._current_session.session_id
                    ),
                }

                # Include all module data
                for (
                    module_id,
                    module_state,
                ) in self._current_session.module_states.items():
                    full_module_data = self._load_module_data(module_id)
                    if full_module_data:
                        session_data["modules"][module_id] = full_module_data

                # Save as pickle file with .ehra extension
                with open(save_path, "wb") as f:
                    pickle.dump(session_data, f)

                logger.info(f"Session saved to: {save_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to save session: {e}")
                return False

    def load_session_from_file(self, file_path: str) -> Tuple[bool, str]:
        """Load session from a .ehra file, replacing the current session.

        Restores a complete analysis session from a .ehra file, including all
        module states, timeline data, and metadata. Performs validation to ensure
        the loaded session is compatible with the current audio context.

        The loading process includes:
        - File existence and format validation
        - Audio file path and signal hash verification
        - Session metadata restoration
        - Module data reconstruction with cache integration
        - Timeline data restoration
        - UI notification via signal emission

        Args:
            file_path (str): Path to the .ehra file to load.

        Returns:
            Tuple[bool, str]: Success status and descriptive message.
                             The message explains success or the reason for failure.

        Raises:
            Exception: Various exceptions may be raised during file I/O or
                      data deserialization, which are caught and logged.

        Note:
            The method validates that the loaded session's audio file and signal
            hash match the current session to prevent inconsistencies. If they
            don't match, loading fails with a descriptive error message.
        """
        LoadingWindow.show(
            title="Loading Project",
            message="Please wait while the project is being loaded...",
        )

        with self._cache_lock:
            try:
                load_path = Path(file_path)
                if not load_path.exists():
                    logger.error(f"Session file not found: {load_path}")
                    return False, f"File not found: {load_path}"

                # Load session data
                with open(load_path, "rb") as f:
                    session_data = pickle.load(f)

                current_session = self.get_current_session()
                if not current_session:
                    return (
                        False,
                        "No current audio to compare. Try to load the project from the main page.",
                    )

                current_audio_file = os.path.abspath(current_session.audio_file_path)
                new_audio_file = os.path.abspath(
                    session_data["session_metadata"]["audio_file_path"]
                )
                if new_audio_file != current_audio_file:
                    return (
                        False,
                        f"Current Audio File does not match the loaded project. {current_audio_file} != {new_audio_file}. Try to load the project from the main page.",
                    )

                current_signal_hash = current_session.signal_hash
                new_signal_hash = session_data["session_metadata"]["signal_hash"]
                if new_signal_hash != current_signal_hash:
                    return (
                        False,
                        f"Current Audio Signal does not match the loaded project. {current_signal_hash} != {new_signal_hash}. Try to load the project from the main page.",
                    )

                # Clear current session
                self._clear_active_cache()
                self._cache_dir.mkdir(exist_ok=True)
                self._temp_modules_dir.mkdir(exist_ok=True)

                # Restore session metadata
                self._current_session = SessionState.from_dict(
                    session_data["session_metadata"]
                )

                # Restore modules
                self._module_cache.clear()
                for module_id, module_data in session_data.get("modules", {}).items():
                    # Save module to temp cache
                    module_path = self._temp_modules_dir / f"{module_id}.pkl"
                    with open(module_path, "wb") as f:
                        pickle.dump(module_data, f)

                # Restore timeline data
                if "timeline_data" in session_data and session_data["timeline_data"]:
                    timeline_path = self._temp_timeline_file
                    with open(timeline_path, "wb") as f:
                        pickle.dump(session_data["timeline_data"], f)
                    logger.info("Restored timeline data from session file")

                # Save temp session
                self._save_temp_session()

                logger.info(f"Session loaded from: {load_path}")

                # Emit signal to notify session load
                self.s_session_loaded.emit(self._current_session.session_id)

                return True, f"Project loaded successfully at {load_path}."

            except Exception as e:
                logger.error(f"Failed to load session: {e}")
                return False, str(e)
            finally:
                LoadingWindow.hide()

    def _load_module_data(self, module_id: str) -> Optional[Dict]:
        """Load complete module data including heavy computational results.

        Retrieves all data associated with a specific module from the cache,
        including numpy arrays, parameter dictionaries, and computed results.

        Args:
            module_id (str): Unique identifier of the module to load.

        Returns:
            Optional[Dict]: Complete module data dictionary containing:
                - parameters: Analysis parameters used
                - nc_x, nc_y: Novelty curve data arrays
                - transitions_sec: Detected boundary timestamps
                - Additional computed results and metadata
                Returns None if module data cannot be loaded.
        """
        try:
            module_path = self._temp_modules_dir / f"{module_id}.pkl"
            if module_path.exists():
                with open(module_path, "rb") as f:
                    data = pickle.load(f)

                return data

        except Exception as e:
            logger.error(f"Failed to load module data {module_id}: {e}")
        return None

    def _load_timeline_data(self, session_id: str) -> Optional[Dict]:
        """Load timeline data including user transitions and annotations.

        Retrieves timeline-specific data for the specified session, including
        user-added transitions, colors, and timeline metadata.

        Args:
            session_id (str): Session identifier to match against stored data.

        Returns:
            Optional[Dict]: Timeline data dictionary containing:
                - session_id: Associated session identifier
                - transitions_sec: List of transition dictionaries
                - last_modified: Timestamp of last modification
                Returns None if timeline data cannot be loaded or session mismatch.
        """
        try:
            if self._temp_timeline_file.exists():
                with open(self._temp_timeline_file, "rb") as f:
                    data = pickle.load(f)
                    if data.get("session_id") == session_id:
                        return data
        except Exception as e:
            logger.error(f"Failed to load timeline data: {e}")
        return None

    # === MODULE MANAGEMENT (Updated for single session) ===

    def save_module_state(
        self,
        page_id: str,
        module_type: SegmenterConfig,
        params: Optional[Dict[str, Any]] = None,
        nc_x: Optional[np.ndarray] = None,
        nc_y: Optional[np.ndarray] = None,
        transitions_sec: Optional[List[float]] = None,
    ) -> str:
        """Save or update module state data to the active session.

        Creates or updates a module's complete state including metadata and
        computed results. The method preserves existing data when updating,
        only replacing fields that are explicitly provided.

        Args:
            page_id (str): Identifier of the UI page/context where module was created.
            module_type (SegmenterConfig): Type/configuration of the analysis module.
            params (Dict[str, Any], optional): Analysis parameters. If None,
                existing parameters are preserved.
            nc_x (np.ndarray, optional): Novelty curve X-axis (time) values.
                If None, existing values are preserved.
            nc_y (np.ndarray, optional): Novelty curve Y-axis (novelty) values.
                If None, existing values are preserved.
            transitions_sec (List[float], optional): Detected transition timestamps.
                If None, existing transitions are preserved.

        Returns:
            str: Unique module identifier for the saved/updated module.

        Raises:
            RuntimeError: If no active session exists.

        Note:
            The method implements incremental updates - only non-None parameters
            are updated, allowing partial updates without losing existing data.
        """
        with self._cache_lock:
            if not self._current_session:
                raise RuntimeError("No active session")

            module_id = self._generate_module_id(page_id, module_type)

            # Load existing module data to preserve it
            existing_data = self._load_module_data(module_id) or {}

            # Create module state (metadata only - for JSON)
            module_state = ModuleState(
                module_type=module_type,
                computation_timestamp=datetime.now(),
                page_id=page_id,
            )

            # Merge new data with existing data (only update non-None values)
            module_data = dict(existing_data)  # Start with existing data

            if params is not None:
                module_data["parameters"] = params
            if nc_x is not None:
                module_data["nc_x"] = nc_x
            if nc_y is not None:
                module_data["nc_y"] = nc_y
            if transitions_sec is not None:
                module_data["transitions_sec"] = transitions_sec

            # Save to temp cache
            module_path = self._temp_modules_dir / f"{module_id}.pkl"
            with open(module_path, "wb") as f:
                pickle.dump(module_data, f)

            # Update session metadata
            self._current_session.module_states[module_id] = module_state
            self._current_session.last_modified = datetime.now()
            module_state._set_cache_context(module_id, self)
            self._module_cache[module_id] = module_state

            # Update temp session file
            self._save_temp_session()

            logger.info(f"Saved/updated module: {module_id}")
            return module_id

    def load_module_state(self, module_id: str) -> Optional[ModuleState]:
        """Load module state metadata from the current session.

        Retrieves the metadata and provides data access methods for a specific
        analysis module. The returned ModuleState object provides lazy access
        to the module's computed data through cache integration.

        Args:
            module_id (str): Unique identifier of the module to load.

        Returns:
            Optional[ModuleState]: Module state object with data access methods,
                                  or None if the module doesn't exist or no
                                  active session.

        Note:
            The returned ModuleState object provides methods like get_nc_x(),
            get_nc_y(), get_transitions_sec() for accessing the actual computed
            data through the cache manager.
        """
        if not self._current_session:
            return None

        if module_id in self._module_cache:
            return self._module_cache[module_id]

        if module_id in self._current_session.module_states:
            state = self._current_session.module_states[module_id]
            # Set cache context for data access methods
            state._set_cache_context(module_id, self)
            self._module_cache[module_id] = state
            return state

        return None

    # === TIMELINE MANAGEMENT ===

    def save_timeline_state(self, transitions_sec: List[Dict[str, Any]]) -> bool:
        """Save timeline transitions and metadata to persistent storage.

        Stores the complete timeline state including user-added transitions,
        their properties (colors, positions), and associated metadata.

        Args:
            transitions_sec (List[Dict[str, Any]]): List of transition dictionaries
                containing timestamps, colors, and other properties.

        Returns:
            bool: True if timeline data was saved successfully, False otherwise.

        Note:
            Timeline data is stored in pickle format for efficient serialization
            of complex nested structures and numpy arrays.
        """
        with self._cache_lock:
            if not self._current_session:
                return False

            # Save transitions to PKL file
            timeline_data = {
                "session_id": self._current_session.session_id,
                "transitions_sec": transitions_sec,
                "last_modified": datetime.now().isoformat(),
            }

            try:
                with open(self._temp_timeline_file, "wb") as f:
                    pickle.dump(timeline_data, f)

                self._current_session.last_modified = datetime.now()
                self._save_temp_session()
                return True
            except Exception as e:
                logger.error(f"Failed to save timeline data: {e}")
                return False

    def load_timeline_state(self) -> Optional[TimelineState]:
        """Load timeline state with proper cache context integration.

        Retrieves the timeline state for the current session and ensures
        it has proper cache manager integration for data access methods.

        Returns:
            Optional[TimelineState]: Timeline state object with data access
                                    methods, or None if no active session.

        Note:
            The returned TimelineState object provides methods like
            get_transitions_sec() and get_transition_count() for accessing
            the actual timeline data through the cache manager.
        """
        if self._current_session:
            timeline_state = self._current_session.timeline_state
            # Ensure cache context is set
            timeline_state._set_cache_context(self._current_session.session_id, self)
            return timeline_state
        return None

    # === UTILITY METHODS ===

    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about the current session.

        Provides a summary of the current session including metadata,
        statistics, and content information useful for UI display and
        session management.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing:
                - session_id: Unique session identifier
                - audio_file: Path to the associated audio file
                - created: Session creation timestamp (ISO format)
                - last_modified: Last modification timestamp (ISO format)
                - module_count: Number of analysis modules in session
                - timeline_transitions: Number of user-added transitions
                Returns None if no active session.
        """
        if self._current_session:
            timeline_state = self.load_timeline_state()
            transition_count = (
                timeline_state.get_transition_count() if timeline_state else 0
            )

            return {
                "session_id": self._current_session.session_id,
                "audio_file": self._current_session.audio_file_path,
                "created": self._current_session.created_timestamp.isoformat(),
                "last_modified": self._current_session.last_modified.isoformat(),
                "module_count": len(self._current_session.module_states),
                "timeline_transitions": transition_count,
            }
        return None

    def save_parameters_copy(self, parameters: Dict[str, Any]) -> bool:
        """Save a copy of parameters for cross-module sharing.

        Stores analysis parameters that can be copied between different
        analysis modules, enabling parameter reuse and consistency.

        Args:
            parameters (Dict[str, Any]): Analysis parameters to store for copying.

        Returns:
            bool: Always returns True (operation cannot fail).

        Note:
            These parameters are session-scoped and persist until the session
            is closed or new parameters are saved.
        """
        with self._cache_lock:
            self._copied_parameters = parameters.copy()
            return True

    def get_parameters_copy(self) -> Optional[Dict[str, Any]]:
        """Get previously copied parameters for module reuse.

        Retrieves parameters that were stored using save_parameters_copy(),
        enabling parameter sharing between analysis modules.

        Returns:
            Optional[Dict[str, Any]]: Copy of stored parameters, or empty
                                     dictionary if no parameters were stored.
        """
        return self._copied_parameters if self._copied_parameters else {}

    def _generate_signal_hash(self, signal: Signal) -> str:
        """Generate hash from audio signal properties for consistency validation.

        Creates a unique hash based on signal characteristics that can be used
        to verify that loaded sessions match the current audio context.

        Args:
            signal (Signal): Audio signal object to hash.

        Returns:
            str: MD5 hash string based on signal duration, sample rate, and length.

        Note:
            The hash includes duration, sample rate, and sample count to ensure
            that sessions are only restored for matching audio files.
        """
        signal_data = (
            f"{signal.duration_seconds()}_{signal.sample_rate}_{len(signal.samples)}"
        )
        return hashlib.md5(signal_data.encode()).hexdigest()

    def _generate_module_id(self, page_id: str, module_type: SegmenterConfig) -> str:
        """Generate unique module identifier based on context and type.

        Creates a deterministic unique identifier for analysis modules based
        on their UI context and analysis type, ensuring consistent IDs across
        application sessions.

        Args:
            page_id (str): Identifier of the UI page/context.
            module_type (SegmenterConfig): Type/configuration of analysis module.

        Returns:
            str: MD5 hash string serving as unique module identifier.
        """
        combined = f"{page_id}_{module_type.value}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _generate_session_id(self, audio_file_path: str, signal_hash: str) -> str:
        """Generate unique session identifier based on audio file and signal.

        Creates a deterministic session identifier that uniquely identifies
        sessions for specific audio files and their signal characteristics.

        Args:
            audio_file_path (str): Path to the audio file.
            signal_hash (str): Hash of the audio signal properties.

        Returns:
            str: MD5 hash string serving as unique session identifier.

        Note:
            The session ID ensures that sessions are uniquely tied to specific
            audio files and their signal characteristics.
        """
        combined = f"{audio_file_path}_{signal_hash}"
        return hashlib.md5(combined.encode()).hexdigest()


# Global instance
SessionCache = SessionCacheManager()
