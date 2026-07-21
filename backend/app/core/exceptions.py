"""
core/exceptions.py

Why this file exists:
    Defines domain-specific exceptions so that services can raise meaningful
    errors instead of generic Exceptions, and the API layer can map each
    one to the correct HTTP status code in one place.

What it does:
    Declares a small hierarchy of exceptions used by the upload, detection,
    and parsing services.

How it connects:
    Raised inside services/*.py and engines/*.py. Caught by
    api/routes/*.py and translated into HTTPException responses.
"""


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
    """Raised when every available parsing strategy failed."""
