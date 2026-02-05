"""SQLAlchemy ORM models for ModelGuard."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class ModelRecord(Base):
    """Tracked ML models."""

    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    framework = Column(String(50))  # sklearn, pytorch, etc.
    model_type = Column(String(50))  # classification, regression

    # Status
    is_active = Column(Boolean, default=True)
    deployed_at = Column(DateTime)

    # Storage reference
    artifact_path = Column(String(500))

    # Metadata
    feature_names = Column(JSON)
    target_name = Column(String(100))
    hyperparameters = Column(JSON)
    extra_data = Column(JSON)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    baselines = relationship("BaselineRecord", back_populates="model")
    alerts = relationship("AlertRecord", back_populates="model")
    retraining_jobs = relationship(
        "RetrainingJobRecord",
        back_populates="model",
        foreign_keys="RetrainingJobRecord.model_id",
    )
    drift_metrics = relationship("DriftMetricRecord", back_populates="model")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "framework": self.framework,
            "model_type": self.model_type,
            "is_active": self.is_active,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "artifact_path": self.artifact_path,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "hyperparameters": self.hyperparameters,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BaselineRecord(Base):
    """Model baseline snapshots."""

    __tablename__ = "baselines"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False, index=True)

    # Versioning
    version = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, default=True)

    # Statistics (stored as JSON)
    feature_statistics = Column(JSON, nullable=False)
    prediction_statistics = Column(JSON, nullable=False)
    performance_metrics = Column(JSON)

    # Sample info
    sample_size = Column(Integer)
    data_start_date = Column(DateTime)
    data_end_date = Column(DateTime)

    # Extra data
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    notes = Column(Text)
    extra_data = Column(JSON)

    # Relationships
    model = relationship("ModelRecord", back_populates="baselines")
    drift_metrics = relationship("DriftMetricRecord", back_populates="baseline")

    __table_args__ = (
        Index("ix_baselines_model_active", "model_id", "is_active"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "version": self.version,
            "is_active": self.is_active,
            "feature_statistics": self.feature_statistics,
            "prediction_statistics": self.prediction_statistics,
            "performance_metrics": self.performance_metrics,
            "sample_size": self.sample_size,
            "data_start_date": self.data_start_date.isoformat() if self.data_start_date else None,
            "data_end_date": self.data_end_date.isoformat() if self.data_end_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "notes": self.notes,
            "extra_data": self.extra_data,
        }


class DriftMetricRecord(Base):
    """Time-series drift metrics."""

    __tablename__ = "drift_metrics"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False)
    baseline_id = Column(String(36), ForeignKey("baselines.id"), nullable=False)

    # Time window
    window_start = Column(DateTime, nullable=False)
    window_end = Column(DateTime, nullable=False)

    # Drift type
    drift_type = Column(String(50))  # data, prediction, concept

    # Per-feature metrics
    feature_name = Column(String(100))
    method_name = Column(String(50))  # ks_test, psi, etc.

    # Results
    statistic = Column(Float)
    p_value = Column(Float)
    threshold = Column(Float)
    drift_detected = Column(Boolean)

    # Additional data
    extra_data = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    model = relationship("ModelRecord", back_populates="drift_metrics")
    baseline = relationship("BaselineRecord", back_populates="drift_metrics")

    __table_args__ = (
        Index("ix_drift_metrics_model_time", "model_id", "window_start"),
        Index("ix_drift_metrics_feature", "model_id", "feature_name"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "baseline_id": self.baseline_id,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "drift_type": self.drift_type,
            "feature_name": self.feature_name,
            "method_name": self.method_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DriftReportRecord(Base):
    """Drift detection reports."""

    __tablename__ = "drift_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False, index=True)
    baseline_id = Column(String(36), ForeignKey("baselines.id"), nullable=False)

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Results (stored as JSON)
    feature_results = Column(JSON)

    # Aggregated flags
    data_drift_detected = Column(Boolean, default=False)
    prediction_drift_detected = Column(Boolean, default=False)
    concept_drift_detected = Column(Boolean, default=False)

    # Summary
    features_with_drift = Column(JSON)
    drift_percentage = Column(Float)

    # Sample info
    current_sample_size = Column(Integer)
    reference_sample_size = Column(Integer)

    # Severity and recommendation (stored as JSON)
    severity = Column(JSON)
    recommendation = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_drift_reports_model_time", "model_id", "timestamp"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "baseline_id": self.baseline_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "feature_results": self.feature_results,
            "data_drift_detected": self.data_drift_detected,
            "prediction_drift_detected": self.prediction_drift_detected,
            "concept_drift_detected": self.concept_drift_detected,
            "features_with_drift": self.features_with_drift,
            "drift_percentage": self.drift_percentage,
            "current_sample_size": self.current_sample_size,
            "reference_sample_size": self.reference_sample_size,
            "severity": self.severity,
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SeverityRecord(Base):
    """Severity assessments over time."""

    __tablename__ = "severity_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False, index=True)
    drift_report_id = Column(String(36), ForeignKey("drift_reports.id"))

    timestamp = Column(DateTime, nullable=False)

    # Scores
    overall_score = Column(Float, nullable=False)
    level = Column(String(20), nullable=False)  # NONE, LOW, MEDIUM, HIGH, CRITICAL

    # Details
    affected_features = Column(JSON)
    feature_scores = Column(JSON)
    impacts_predictions = Column(Boolean)
    confidence = Column(Float)
    explanation = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "drift_report_id": self.drift_report_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "overall_score": self.overall_score,
            "level": self.level,
            "affected_features": self.affected_features,
            "feature_scores": self.feature_scores,
            "impacts_predictions": self.impacts_predictions,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AlertRecord(Base):
    """Alerts for human review."""

    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False, index=True)
    drift_report_id = Column(String(36), ForeignKey("drift_reports.id"))

    # Classification
    alert_type = Column(String(50), nullable=False)  # drift, performance, anomaly
    severity = Column(String(20), nullable=False)
    urgency = Column(String(20), nullable=False)

    # Context (stored as JSON)
    drift_summary = Column(JSON)
    severity_details = Column(JSON)
    recommendation = Column(JSON)

    # Status
    status = Column(String(20), default="pending", index=True)
    assigned_to = Column(String(100))

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)

    # Human decision
    decision = Column(String(50))
    decision_notes = Column(Text)
    decided_by = Column(String(100))

    # Metadata
    extra_data = Column(JSON)

    # Relationships
    model = relationship("ModelRecord", back_populates="alerts")
    feedback = relationship("HumanFeedbackRecord", back_populates="alert", uselist=False)

    __table_args__ = (
        Index("ix_alerts_status_severity", "status", "severity"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "drift_report_id": self.drift_report_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "urgency": self.urgency,
            "drift_summary": self.drift_summary,
            "severity_details": self.severity_details,
            "recommendation": self.recommendation,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "decision": self.decision,
            "decision_notes": self.decision_notes,
            "decided_by": self.decided_by,
            "extra_data": self.extra_data,
        }


class HumanFeedbackRecord(Base):
    """Human feedback on alerts and recommendations."""

    __tablename__ = "human_feedback"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    alert_id = Column(String(36), ForeignKey("alerts.id"), nullable=False, unique=True)

    # Original recommendation
    original_recommendation = Column(String(50))

    # Human decision
    human_decision = Column(String(50))
    agreed_with_recommendation = Column(Boolean)

    # Feedback details
    feedback_type = Column(String(50))  # correct, incorrect, partially_correct
    reason = Column(Text)

    # Outcome tracking
    outcome_observed = Column(Boolean, default=False)
    outcome_was_correct = Column(Boolean)
    outcome_notes = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    alert = relationship("AlertRecord", back_populates="feedback")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "original_recommendation": self.original_recommendation,
            "human_decision": self.human_decision,
            "agreed_with_recommendation": self.agreed_with_recommendation,
            "feedback_type": self.feedback_type,
            "reason": self.reason,
            "outcome_observed": self.outcome_observed,
            "outcome_was_correct": self.outcome_was_correct,
            "outcome_notes": self.outcome_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class RetrainingJobRecord(Base):
    """Retraining job tracking."""

    __tablename__ = "retraining_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False, index=True)

    # Trigger
    triggered_by = Column(String(100))  # alert_id, manual, scheduled
    approved_by = Column(String(100))

    # Status
    status = Column(String(50), nullable=False, index=True)

    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # Artifacts
    new_model_id = Column(String(36), ForeignKey("models.id"))

    # Metrics
    training_metrics = Column(JSON)
    validation_metrics = Column(JSON)
    baseline_metrics = Column(JSON)
    improvement = Column(JSON)

    # Outcome
    deployed = Column(Boolean, default=False)
    error_message = Column(Text)

    # Relationships
    model = relationship(
        "ModelRecord",
        back_populates="retraining_jobs",
        foreign_keys=[model_id],
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "triggered_by": self.triggered_by,
            "approved_by": self.approved_by,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "new_model_id": self.new_model_id,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "baseline_metrics": self.baseline_metrics,
            "improvement": self.improvement,
            "deployed": self.deployed,
            "error_message": self.error_message,
        }


class AuditLogRecord(Base):
    """Audit trail for all actions."""

    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)

    # What
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50))
    entity_id = Column(String(36))

    # Who
    user_id = Column(String(100))

    # Details
    old_value = Column(JSON)
    new_value = Column(JSON)
    extra_data = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "action": self.action,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "user_id": self.user_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
