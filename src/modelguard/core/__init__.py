"""Core module containing interfaces, types, configuration, and exceptions."""

from modelguard.core.types import (
    DriftResult,
    DriftType,
    SeverityLevel,
    SeverityScore,
    ActionType,
    ActionRecommendation,
    FeatureStatistics,
    PredictionStatistics,
    PerformanceMetrics,
    Baseline,
    DriftReport,
    Alert,
    AlertStatus,
)
from modelguard.core.config import Config, load_config
from modelguard.core.exceptions import (
    ModelGuardError,
    ConfigurationError,
    BaselineError,
    DriftDetectionError,
    StorageError,
    ValidationError,
)

__all__ = [
    # Types
    "DriftResult",
    "DriftType",
    "SeverityLevel",
    "SeverityScore",
    "ActionType",
    "ActionRecommendation",
    "FeatureStatistics",
    "PredictionStatistics",
    "PerformanceMetrics",
    "Baseline",
    "DriftReport",
    "Alert",
    "AlertStatus",
    # Config
    "Config",
    "load_config",
    # Exceptions
    "ModelGuardError",
    "ConfigurationError",
    "BaselineError",
    "DriftDetectionError",
    "StorageError",
    "ValidationError",
]
