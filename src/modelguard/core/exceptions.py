"""Custom exceptions for ModelGuard."""


class ModelGuardError(Exception):
    """Base exception for all ModelGuard errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(ModelGuardError):
    """Raised when there's a configuration error."""
    pass


class BaselineError(ModelGuardError):
    """Raised when there's an error with baseline operations."""
    pass


class DriftDetectionError(ModelGuardError):
    """Raised when drift detection fails."""
    pass


class StorageError(ModelGuardError):
    """Raised when there's a storage/database error."""
    pass


class ValidationError(ModelGuardError):
    """Raised when validation fails."""
    pass


class ModelNotFoundError(ModelGuardError):
    """Raised when a model is not found."""
    pass


class BaselineNotFoundError(ModelGuardError):
    """Raised when a baseline is not found."""
    pass


class AlertNotFoundError(ModelGuardError):
    """Raised when an alert is not found."""
    pass


class InsufficientDataError(ModelGuardError):
    """Raised when there's not enough data for an operation."""
    pass


class RetrainingError(ModelGuardError):
    """Raised when retraining fails."""
    pass


class DeploymentError(ModelGuardError):
    """Raised when model deployment fails."""
    pass
