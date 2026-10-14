"""Exception hierarchy for Zero-Context memory engine."""

from __future__ import annotations


class ZeroContextError(Exception):
    """Base exception for all Zero-Context errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MemoryNotFoundError(ZeroContextError):
    """Raised when an operation references a memory item that does not exist."""


class StorageError(ZeroContextError):
    """Raised when loading or persisting memory data to disk fails."""
