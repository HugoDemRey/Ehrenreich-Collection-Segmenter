"""Multi-threading utilities for background task execution.

Provides worker classes and thread management for executing
long-running tasks without blocking the UI thread.

Author: Hugo Demule
Date: January 2026
"""

import uuid
from typing import Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class Worker(QObject):
    """Self-contained worker for background task execution.

    Manages its own thread lifecycle and provides callback mechanisms
    for task completion and error handling.
    """

    def __init__(
        self,
        func: Callable,
        callback_func: Callable,
        error_func: Callable | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.callback_func = callback_func
        self.error_func = error_func
        self.worker_id = str(uuid.uuid4())[:8]
        self._thread = QThread()
        self._thread.setObjectName(f"WorkerThread-{self.worker_id}")
        self.moveToThread(self._thread)

        # Connect signals for automatic cleanup
        self._thread.started.connect(self.run)
        self._thread.finished.connect(self._cleanup_thread)
        self.finished.connect(self._finish_work)
        self.error.connect(self._handle_error)

        # Connect to provided output signal
        self.finished.connect(self.callback_func)
        if self.error_func:
            self.error.connect(self.error_func)

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cleanup_requested = pyqtSignal(str)  # Signal to notify controller of cleanup

    def start(self):
        """Start the worker in its own thread"""
        print(f"(WORKER-{self.worker_id}): Starting worker thread")
        self._thread.start()

    @pyqtSlot()
    def run(self):
        try:
            print(f"(WORKER-{self.worker_id}): Executing task")
            result = self.func(*self.args, **self.kwargs)
            print(f"(WORKER-{self.worker_id}): Task completed successfully")
            self.finished.emit(result)
        except Exception as e:
            print(f"(WORKER-{self.worker_id}): Task failed with error: {str(e)}")
            self.error.emit(str(e))

    @pyqtSlot()
    def _finish_work(self):
        """Handle work completion"""
        print(f"(WORKER-{self.worker_id}): Finishing work, requesting thread quit")
        if self._thread.isRunning():
            self._thread.quit()
            # Don't emit cleanup signal here - wait for thread to actually finish

    @pyqtSlot()
    def _handle_error(self):
        """Handle work error"""
        print(f"(WORKER-{self.worker_id}): Handling error, requesting thread quit")
        if self._thread.isRunning():
            self._thread.quit()
            # Don't emit cleanup signal here - wait for thread to actually finish

    @pyqtSlot()
    def _cleanup_thread(self):
        """Clean up thread resources"""
        print(f"(WORKER-{self.worker_id}): Thread finished, scheduling deletion")
        # Emit cleanup signal only when thread has actually finished
        self.cleanup_requested.emit(self.worker_id)
        self._thread.deleteLater()
        self.deleteLater()
