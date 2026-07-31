# VENDORED COPY — adapted from backend/app/core/exceptions.py
# Part of the neuralviz CLI package. Sync manually if upstream changes.

"""
core/exceptions.py — domain-specific exceptions with structured error reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class NNAException(Exception):
    """Base exception for all NeuralNetworkAnalyzer domain errors."""


class InvalidFileTypeError(NNAException):
    """Raised when an uploaded file's extension is not supported."""


class FileTooLargeError(NNAException):
    """Raised when an uploaded file exceeds the configured size limit."""


class JobNotFoundError(NNAException):
    """Raised when a requested job_id does not exist in storage."""


class FrameworkNotSupportedError(NNAException):
    """Raised when the detected framework has no parser implemented yet."""


class FrameworkNotDetectedError(NNAException):
    """Raised when no known framework import could be found in the code."""


class ModelParsingError(NNAException):
    """Raised when a single parsing strategy failed. Caught by the tier chain."""


@dataclass
class ParsingFailure:
    """Records a single tier's failure with context and a user-facing suggestion."""
    tier: str
    error: str
    suggestion: str = ""
    is_fatal: bool = False


@dataclass
class ParserWarning:
    """A non-fatal advisory message from parsing with a severity level."""
    level: Literal["info", "warning", "degraded"] = "info"
    message: str = ""
    tier: str = ""


class ParseChainError(NNAException):
    """
    Raised when every parsing tier for the detected framework has failed.
    Carries the full ordered list of per-tier failures for structured display.
    """

    def __init__(self, failures: list[ParsingFailure]) -> None:
        self.failures = failures
        super().__init__(self._format())

    def _format(self) -> str:
        lines = ["All parsing strategies failed:"]
        for f in self.failures:
            lines.append(f"  [{f.tier}] {f.error}")
            if f.suggestion:
                lines.append(f"    → {f.suggestion}")
        return "\n".join(lines)
