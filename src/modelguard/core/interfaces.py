"""Protocol definitions for ModelGuard components."""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from modelguard.core.types import (
    ActionRecommendation,
    Alert,
    Baseline,
    DriftReport,
    DriftResult,
    SeverityScore,
)


@runtime_checkable
class DriftDetector(Protocol):
    """Protocol for drift detection methods."""

    @property
    def name(self) -> str:
        """Unique identifier for this detector."""
        ...

    @property
    def supported_dtypes(self) -> List[str]:
        """Data types this detector supports: 'numerical', 'categorical'."""
        ...

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        **kwargs: Any,
    ) -> DriftResult:
        """
        Detect drift between reference and current distributions.

        Args:
            reference: Baseline/reference data distribution
            current: Current production data distribution
            feature_name: Name of the feature being tested
            **kwargs: Method-specific parameters

        Returns:
            DriftResult with detection outcome
        """
        ...


@runtime_checkable
class SeverityScorer(Protocol):
    """Protocol for severity scoring strategies."""

    def score(
        self,
        drift_results: List[DriftResult],
        feature_metadata: Dict[str, Any],
        prediction_impact: Optional[float] = None,
    ) -> SeverityScore:
        """
        Compute severity from drift results.

        Args:
            drift_results: List of drift detection results
            feature_metadata: Metadata about features (importance, etc.)
            prediction_impact: Optional measure of prediction quality degradation

        Returns:
            SeverityScore with overall assessment
        """
        ...


@runtime_checkable
class ActionRecommender(Protocol):
    """Protocol for action recommendation engines."""

    def recommend(
        self,
        severity: SeverityScore,
        historical_context: Dict[str, Any],
        model_metadata: Dict[str, Any],
    ) -> ActionRecommendation:
        """
        Generate action recommendation from severity assessment.

        Args:
            severity: Severity score from drift detection
            historical_context: Historical drift/action information
            model_metadata: Information about the model

        Returns:
            ActionRecommendation with suggested action
        """
        ...


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for ML framework adapters."""

    @property
    def model_type(self) -> str:
        """Type of model: 'classification' or 'regression'."""
        ...

    def get_feature_names(self) -> List[str]:
        """Get feature names from model."""
        ...

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        ...

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities if available."""
        ...

    def serialize(self) -> bytes:
        """Serialize model to bytes."""
        ...

    @classmethod
    def deserialize(cls, data: bytes) -> "ModelAdapter":
        """Deserialize model from bytes."""
        ...


@runtime_checkable
class NotificationChannel(Protocol):
    """Protocol for notification channels."""

    def send(
        self,
        alert: Alert,
        recipients: List[str],
    ) -> bool:
        """
        Send alert notification.

        Args:
            alert: Alert to send
            recipients: List of recipient identifiers

        Returns:
            True if notification was sent successfully
        """
        ...


@runtime_checkable
class DataCollector(Protocol):
    """Protocol for production data collectors."""

    def collect(
        self,
        model_id: str,
        start_time: Any,
        end_time: Any,
    ) -> pd.DataFrame:
        """
        Collect data for the specified time window.

        Args:
            model_id: ID of the model to collect data for
            start_time: Start of collection window
            end_time: End of collection window

        Returns:
            DataFrame with collected data
        """
        ...


@runtime_checkable
class BaselineRepository(Protocol):
    """Protocol for baseline storage operations."""

    def save(self, baseline: Baseline) -> str:
        """Save a baseline and return its ID."""
        ...

    def get(self, baseline_id: str) -> Optional[Baseline]:
        """Get a baseline by ID."""
        ...

    def get_active(self, model_id: str) -> Optional[Baseline]:
        """Get the active baseline for a model."""
        ...

    def list_for_model(self, model_id: str) -> List[Baseline]:
        """List all baselines for a model."""
        ...

    def delete(self, baseline_id: str) -> bool:
        """Delete a baseline."""
        ...


@runtime_checkable
class AlertRepository(Protocol):
    """Protocol for alert storage operations."""

    def save(self, alert: Alert) -> str:
        """Save an alert and return its ID."""
        ...

    def get(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        ...

    def list_pending(self, model_id: Optional[str] = None) -> List[Alert]:
        """List pending alerts, optionally filtered by model."""
        ...

    def update(self, alert: Alert) -> bool:
        """Update an alert."""
        ...


@runtime_checkable
class DriftReportRepository(Protocol):
    """Protocol for drift report storage operations."""

    def save(self, report: DriftReport) -> str:
        """Save a drift report and return its ID."""
        ...

    def get(self, report_id: str) -> Optional[DriftReport]:
        """Get a drift report by ID."""
        ...

    def list_for_model(
        self,
        model_id: str,
        limit: int = 100,
    ) -> List[DriftReport]:
        """List drift reports for a model."""
        ...


@runtime_checkable
class ModelRegistry(Protocol):
    """Protocol for model registry operations."""

    def register(
        self,
        model: ModelAdapter,
        name: str,
        version: str,
        metrics: Dict[str, float],
    ) -> str:
        """Register a model and return its ID."""
        ...

    def get(self, model_id: str) -> Optional[ModelAdapter]:
        """Get a model by ID."""
        ...

    def get_latest(self, name: str) -> Optional[ModelAdapter]:
        """Get the latest version of a model."""
        ...

    def list_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a model."""
        ...
