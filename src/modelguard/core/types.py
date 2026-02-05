"""Core type definitions for ModelGuard."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    DATA = "data"
    PREDICTION = "prediction"
    CONCEPT = "concept"


class SeverityLevel(str, Enum):
    """Severity levels for drift detection."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """Recommended actions for drift response."""
    IGNORE = "ignore"
    MONITOR = "monitor"
    RETRAIN = "retrain"
    ROLLBACK = "rollback"


class AlertStatus(str, Enum):
    """Status of an alert in the review queue."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class Urgency(str, Enum):
    """Urgency levels for actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMEDIATE = "immediate"


@dataclass
class DriftResult:
    """Result from a single drift detection method."""
    method_name: str
    feature_name: str
    drift_detected: bool
    statistic: float
    p_value: Optional[float]
    threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_name": self.method_name,
            "feature_name": self.feature_name,
            "drift_detected": self.drift_detected,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


@dataclass
class SeverityScore:
    """Structured severity assessment."""
    overall_score: float  # 0.0 - 1.0
    level: SeverityLevel
    affected_features: List[str]
    feature_scores: Dict[str, float]
    impacts_predictions: bool
    confidence: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "level": self.level.value,
            "affected_features": self.affected_features,
            "feature_scores": self.feature_scores,
            "impacts_predictions": self.impacts_predictions,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class ActionRecommendation:
    """Recommended action with justification."""
    action: ActionType
    confidence: float
    reasoning: List[str]
    urgency: Urgency
    estimated_impact: str
    prerequisite_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "urgency": self.urgency.value,
            "estimated_impact": self.estimated_impact,
            "prerequisite_actions": self.prerequisite_actions,
        }


@dataclass
class FeatureStatistics:
    """Statistics for a single feature."""
    name: str
    dtype: str  # 'numerical' or 'categorical'
    count: int
    null_count: int
    null_ratio: float

    # Numerical stats (None for categorical)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    percentiles: Optional[Dict[int, float]] = None
    histogram_bins: Optional[List[float]] = None
    histogram_counts: Optional[List[int]] = None

    # Categorical stats (None for numerical)
    unique_count: Optional[int] = None
    value_counts: Optional[Dict[str, int]] = None
    mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "null_count": self.null_count,
            "null_ratio": self.null_ratio,
        }
        if self.dtype == "numerical":
            result.update({
                "mean": self.mean,
                "std": self.std,
                "min": self.min_val,
                "max": self.max_val,
                "percentiles": self.percentiles,
                "histogram_bins": self.histogram_bins,
                "histogram_counts": self.histogram_counts,
            })
        else:
            result.update({
                "unique_count": self.unique_count,
                "value_counts": self.value_counts,
                "mode": self.mode,
            })
        return result


@dataclass
class PredictionStatistics:
    """Statistics for model predictions."""
    prediction_type: str  # 'classification' or 'regression'

    # Classification stats
    class_distribution: Optional[Dict[str, float]] = None
    probability_mean: Optional[Dict[str, float]] = None
    probability_std: Optional[Dict[str, float]] = None

    # Regression stats
    mean: Optional[float] = None
    std: Optional[float] = None
    percentiles: Optional[Dict[int, float]] = None
    histogram_bins: Optional[List[float]] = None
    histogram_counts: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"prediction_type": self.prediction_type}
        if self.prediction_type == "classification":
            result.update({
                "class_distribution": self.class_distribution,
                "probability_mean": self.probability_mean,
                "probability_std": self.probability_std,
            })
        else:
            result.update({
                "mean": self.mean,
                "std": self.std,
                "percentiles": self.percentiles,
                "histogram_bins": self.histogram_bins,
                "histogram_counts": self.histogram_counts,
            })
        return result


@dataclass
class PerformanceMetrics:
    """Model performance metrics (when labels available)."""
    metric_type: str  # 'classification' or 'regression'

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[Dict[str, float]] = None
    recall: Optional[Dict[str, float]] = None
    f1_score: Optional[Dict[str, float]] = None
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None

    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {"metric_type": self.metric_type}
        if self.metric_type == "classification":
            result.update({
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "auc_roc": self.auc_roc,
                "confusion_matrix": self.confusion_matrix,
            })
        else:
            result.update({
                "mse": self.mse,
                "rmse": self.rmse,
                "mae": self.mae,
                "r2": self.r2,
            })
        return result


@dataclass
class Baseline:
    """Complete baseline snapshot for a model."""
    id: str
    model_id: str
    created_at: datetime
    feature_statistics: Dict[str, FeatureStatistics]
    prediction_statistics: PredictionStatistics
    performance_metrics: Optional[PerformanceMetrics] = None
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        model_id: str,
        feature_statistics: Dict[str, FeatureStatistics],
        prediction_statistics: PredictionStatistics,
        performance_metrics: Optional[PerformanceMetrics] = None,
        sample_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Baseline":
        """Create a new baseline with generated ID and timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            model_id=model_id,
            created_at=datetime.utcnow(),
            feature_statistics=feature_statistics,
            prediction_statistics=prediction_statistics,
            performance_metrics=performance_metrics,
            sample_size=sample_size,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat(),
            "feature_statistics": {
                k: v.to_dict() for k, v in self.feature_statistics.items()
            },
            "prediction_statistics": self.prediction_statistics.to_dict(),
            "performance_metrics": (
                self.performance_metrics.to_dict()
                if self.performance_metrics else None
            ),
            "sample_size": self.sample_size,
            "metadata": self.metadata,
        }


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    id: str
    baseline_id: str
    model_id: str
    timestamp: datetime

    # Per-feature results
    feature_results: Dict[str, List[DriftResult]]

    # Aggregated flags
    data_drift_detected: bool
    prediction_drift_detected: bool
    concept_drift_detected: bool

    # Summary statistics
    features_with_drift: List[str]
    drift_percentage: float

    # Sample info
    current_sample_size: int
    reference_sample_size: int

    # Severity assessment
    severity: Optional[SeverityScore] = None

    # Recommended action
    recommendation: Optional[ActionRecommendation] = None

    @classmethod
    def create(
        cls,
        baseline_id: str,
        model_id: str,
        feature_results: Dict[str, List[DriftResult]],
        current_sample_size: int,
        reference_sample_size: int,
    ) -> "DriftReport":
        """Create a new drift report with computed aggregates."""
        # Compute aggregates
        features_with_drift = [
            feature for feature, results in feature_results.items()
            if any(r.drift_detected for r in results)
        ]

        total_features = len(feature_results)
        drift_percentage = (
            len(features_with_drift) / total_features * 100
            if total_features > 0 else 0.0
        )

        return cls(
            id=str(uuid.uuid4()),
            baseline_id=baseline_id,
            model_id=model_id,
            timestamp=datetime.utcnow(),
            feature_results=feature_results,
            data_drift_detected=len(features_with_drift) > 0,
            prediction_drift_detected=False,  # Set separately
            concept_drift_detected=False,  # Set separately
            features_with_drift=features_with_drift,
            drift_percentage=drift_percentage,
            current_sample_size=current_sample_size,
            reference_sample_size=reference_sample_size,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "baseline_id": self.baseline_id,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "feature_results": {
                k: [r.to_dict() for r in v]
                for k, v in self.feature_results.items()
            },
            "data_drift_detected": self.data_drift_detected,
            "prediction_drift_detected": self.prediction_drift_detected,
            "concept_drift_detected": self.concept_drift_detected,
            "features_with_drift": self.features_with_drift,
            "drift_percentage": self.drift_percentage,
            "current_sample_size": self.current_sample_size,
            "reference_sample_size": self.reference_sample_size,
            "severity": self.severity.to_dict() if self.severity else None,
            "recommendation": (
                self.recommendation.to_dict() if self.recommendation else None
            ),
        }


@dataclass
class Alert:
    """Alert for human review."""
    id: str
    model_id: str
    created_at: datetime

    # Classification
    alert_type: str  # drift, performance, anomaly
    severity: SeverityLevel
    urgency: Urgency

    # Context
    drift_report_id: str

    # Status tracking
    status: AlertStatus
    assigned_to: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Human decision
    decision: Optional[ActionType] = None
    decision_notes: Optional[str] = None
    decided_by: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        model_id: str,
        alert_type: str,
        severity: SeverityLevel,
        urgency: Urgency,
        drift_report_id: str,
    ) -> "Alert":
        """Create a new alert."""
        return cls(
            id=str(uuid.uuid4()),
            model_id=model_id,
            created_at=datetime.utcnow(),
            alert_type=alert_type,
            severity=severity,
            urgency=urgency,
            drift_report_id=drift_report_id,
            status=AlertStatus.PENDING,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "created_at": self.created_at.isoformat(),
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "urgency": self.urgency.value,
            "drift_report_id": self.drift_report_id,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "resolved_at": (
                self.resolved_at.isoformat() if self.resolved_at else None
            ),
            "decision": self.decision.value if self.decision else None,
            "decision_notes": self.decision_notes,
            "decided_by": self.decided_by,
            "metadata": self.metadata,
        }


@dataclass
class HumanFeedback:
    """Human feedback on alerts and recommendations."""
    id: str
    alert_id: str
    original_recommendation: ActionType
    human_decision: ActionType
    agreed_with_recommendation: bool
    feedback_type: str  # correct, incorrect, partially_correct
    reason: Optional[str]
    created_at: datetime

    # Outcome tracking
    outcome_observed: bool = False
    outcome_was_correct: Optional[bool] = None
    outcome_notes: Optional[str] = None

    @classmethod
    def create(
        cls,
        alert_id: str,
        original_recommendation: ActionType,
        human_decision: ActionType,
        feedback_type: str,
        reason: Optional[str] = None,
    ) -> "HumanFeedback":
        """Create new feedback record."""
        return cls(
            id=str(uuid.uuid4()),
            alert_id=alert_id,
            original_recommendation=original_recommendation,
            human_decision=human_decision,
            agreed_with_recommendation=original_recommendation == human_decision,
            feedback_type=feedback_type,
            reason=reason,
            created_at=datetime.utcnow(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "original_recommendation": self.original_recommendation.value,
            "human_decision": self.human_decision.value,
            "agreed_with_recommendation": self.agreed_with_recommendation,
            "feedback_type": self.feedback_type,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "outcome_observed": self.outcome_observed,
            "outcome_was_correct": self.outcome_was_correct,
            "outcome_notes": self.outcome_notes,
        }


@dataclass
class RetrainingJob:
    """Tracks a retraining job through its lifecycle."""
    id: str
    model_id: str
    triggered_by: str  # alert_id, manual, scheduled
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None

    # Artifacts
    new_model_id: Optional[str] = None
    training_metrics: Optional[Dict[str, float]] = None
    validation_metrics: Optional[Dict[str, float]] = None

    # Comparison
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    improvement: Optional[Dict[str, float]] = None

    # Outcome
    deployed: bool = False
    error_message: Optional[str] = None

    @classmethod
    def create(
        cls,
        model_id: str,
        triggered_by: str,
    ) -> "RetrainingJob":
        """Create a new retraining job."""
        return cls(
            id=str(uuid.uuid4()),
            model_id=model_id,
            triggered_by=triggered_by,
            status="pending",
            started_at=datetime.utcnow(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "triggered_by": self.triggered_by,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "new_model_id": self.new_model_id,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "baseline_metrics": self.baseline_metrics,
            "improvement": self.improvement,
            "deployed": self.deployed,
            "error_message": self.error_message,
        }
