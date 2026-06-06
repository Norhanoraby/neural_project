"""
utils/exception_handling.py
-----------------------------
Custom exception classes for the pneumonia detection project.
Raising typed exceptions makes it easier to catch specific failures
in the Lambda handler, pipelines, and CLI entry point.
"""


class PneumoniaVitBaseError(Exception):
    """Base class for all project-specific exceptions."""
    pass


# ── Data exceptions ──────────────────────────────────────────────────────────

class DataPipelineError(PneumoniaVitBaseError):
    """Raised when the data loading or preprocessing pipeline fails."""
    pass


class MissingCSVError(DataPipelineError):
    """Raised when an expected CSV file (chexpert / metadata / split) is not found."""
    pass


class ImageNotFoundError(DataPipelineError):
    """Raised when an image referenced in the DataFrame cannot be found on disk."""
    pass


class InvalidImageError(DataPipelineError):
    """Raised when an image file exists but cannot be opened or processed."""
    pass


# ── Model exceptions ─────────────────────────────────────────────────────────

class ModelNotLoadedError(PneumoniaVitBaseError):
    """Raised when inference is attempted before the model has been loaded."""
    pass


class TrainingPipelineError(PneumoniaVitBaseError):
    """Raised when the training pipeline encounters an unrecoverable error."""
    pass


class CheckpointNotFoundError(PneumoniaVitBaseError):
    """Raised when a .pth checkpoint file is expected but does not exist."""
    pass


# ── Handler / API exceptions ─────────────────────────────────────────────────

class InvalidRequestError(PneumoniaVitBaseError):
    """Raised when the Lambda handler receives a malformed or incomplete request."""
    pass


class UnsupportedImageFormatError(InvalidRequestError):
    """Raised when the uploaded file is not a supported image format (JPEG/PNG)."""
    pass
