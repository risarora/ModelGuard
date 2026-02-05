"""
ModelGuard: Data Drift, Model Decay & Auto-Retraining System

A comprehensive MLOps system that monitors deployed ML models for drift,
assesses severity, generates actionable recommendations, and orchestrates
retraining when needed - with human oversight.
"""

__version__ = "0.1.0"

from modelguard.core.types import (
    DriftResult,
    DriftType,
    SeverityLevel,
    SeverityScore,
    ActionType,
    ActionRecommendation,
)

__all__ = [
    "__version__",
    "DriftResult",
    "DriftType",
    "SeverityLevel",
    "SeverityScore",
    "ActionType",
    "ActionRecommendation",
]
