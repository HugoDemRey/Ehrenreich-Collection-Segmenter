"""Custom exceptions for Naxos scraper operations.

Provides user-friendly exception classes for different failure scenarios
in Naxos scraping operations.

Author: Hugo Demule  
Date: January 2026
"""


class NaxosScrapingError(Exception):
    """Base exception for Naxos scraping operations."""
    pass


class NaxosDriverError(NaxosScrapingError):
    """Exception raised when Chrome driver cannot be initialized."""
    def __init__(self, message="Unable to initialize chrome driver. Make you are connected to internet and retry."):
        super().__init__(message)


class NaxosPageLoadError(NaxosScrapingError):
    """Exception raised when Naxos page fails to load."""
    def __init__(self, url, message=None):
        if message is None:
            message = f"Failed to load Naxos page. Please check your internet connection and verify the URL is correct."
        super().__init__(message)
        self.url = url


class NaxosContentError(NaxosScrapingError):
    """Exception raised when expected content is not found on the page."""
    def __init__(self, message="No audio tracks found on this page. This may not be a valid Naxos catalog page with audio previews."):
        super().__init__(message)


class NaxosNetworkError(NaxosScrapingError):
    """Exception raised when network requests fail."""
    def __init__(self, message="Network error occurred while downloading audio files. Please check your internet connection."):
        super().__init__(message)


class NaxosAudioExtractionError(NaxosScrapingError):
    """Exception raised when audio URLs cannot be extracted."""
    def __init__(self, message="Failed to extract audio previews from the page. The page structure may have changed."):
        super().__init__(message)


class NaxosTimeoutError(NaxosScrapingError):
    """Exception raised when operations timeout."""
    def __init__(self, operation="page loading", timeout_seconds=30):
        message = f"Operation timed out after {timeout_seconds} seconds during {operation}. Please try again or check your internet connection."
        super().__init__(message)