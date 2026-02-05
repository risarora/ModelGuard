"""Repository for alert operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from modelguard.storage.models import AlertRecord, HumanFeedbackRecord


class AlertRepository:
    """Repository for alert CRUD operations."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        model_id: str,
        alert_type: str,
        severity: str,
        urgency: str,
        drift_report_id: Optional[str] = None,
        drift_summary: Optional[Dict[str, Any]] = None,
        severity_details: Optional[Dict[str, Any]] = None,
        recommendation: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AlertRecord:
        """Create a new alert."""
        record = AlertRecord(
            model_id=model_id,
            drift_report_id=drift_report_id,
            alert_type=alert_type,
            severity=severity,
            urgency=urgency,
            drift_summary=drift_summary,
            severity_details=severity_details,
            recommendation=recommendation,
            status="pending",
            metadata=metadata,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, alert_id: str) -> Optional[AlertRecord]:
        """Get an alert by ID."""
        return self.session.query(AlertRecord).filter(
            AlertRecord.id == alert_id
        ).first()

    def list_pending(
        self,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AlertRecord]:
        """List pending alerts."""
        query = self.session.query(AlertRecord).filter(
            AlertRecord.status == "pending"
        )

        if model_id:
            query = query.filter(AlertRecord.model_id == model_id)

        return query.order_by(AlertRecord.created_at.desc()).limit(limit).all()

    def list_by_status(
        self,
        status: str,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AlertRecord]:
        """List alerts by status."""
        query = self.session.query(AlertRecord).filter(
            AlertRecord.status == status
        )

        if model_id:
            query = query.filter(AlertRecord.model_id == model_id)

        return query.order_by(AlertRecord.created_at.desc()).limit(limit).all()

    def list_for_model(
        self,
        model_id: str,
        limit: int = 100,
    ) -> List[AlertRecord]:
        """List all alerts for a model."""
        return self.session.query(AlertRecord).filter(
            AlertRecord.model_id == model_id
        ).order_by(AlertRecord.created_at.desc()).limit(limit).all()

    def acknowledge(
        self,
        alert_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[AlertRecord]:
        """Acknowledge an alert."""
        record = self.get(alert_id)
        if record is None:
            return None

        record.status = "acknowledged"
        record.acknowledged_at = datetime.utcnow()
        if user_id:
            record.assigned_to = user_id
        self.session.flush()
        return record

    def resolve(
        self,
        alert_id: str,
        decision: str,
        decided_by: Optional[str] = None,
        decision_notes: Optional[str] = None,
    ) -> Optional[AlertRecord]:
        """Resolve an alert with a decision."""
        record = self.get(alert_id)
        if record is None:
            return None

        record.status = "resolved"
        record.resolved_at = datetime.utcnow()
        record.decision = decision
        record.decided_by = decided_by
        record.decision_notes = decision_notes
        self.session.flush()
        return record

    def dismiss(
        self,
        alert_id: str,
        decided_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Optional[AlertRecord]:
        """Dismiss an alert."""
        record = self.get(alert_id)
        if record is None:
            return None

        record.status = "dismissed"
        record.resolved_at = datetime.utcnow()
        record.decision = "dismiss"
        record.decided_by = decided_by
        record.decision_notes = reason
        self.session.flush()
        return record

    def assign(
        self,
        alert_id: str,
        user_id: str,
    ) -> Optional[AlertRecord]:
        """Assign an alert to a user."""
        record = self.get(alert_id)
        if record is None:
            return None

        record.assigned_to = user_id
        record.status = "in_review"
        self.session.flush()
        return record

    def delete(self, alert_id: str) -> bool:
        """Delete an alert."""
        record = self.get(alert_id)
        if record is None:
            return False

        self.session.delete(record)
        self.session.flush()
        return True


class FeedbackRepository:
    """Repository for human feedback CRUD operations."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        alert_id: str,
        original_recommendation: str,
        human_decision: str,
        feedback_type: str,
        reason: Optional[str] = None,
    ) -> HumanFeedbackRecord:
        """Create a new feedback record."""
        agreed = original_recommendation == human_decision

        record = HumanFeedbackRecord(
            alert_id=alert_id,
            original_recommendation=original_recommendation,
            human_decision=human_decision,
            agreed_with_recommendation=agreed,
            feedback_type=feedback_type,
            reason=reason,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, feedback_id: str) -> Optional[HumanFeedbackRecord]:
        """Get feedback by ID."""
        return self.session.query(HumanFeedbackRecord).filter(
            HumanFeedbackRecord.id == feedback_id
        ).first()

    def get_for_alert(self, alert_id: str) -> Optional[HumanFeedbackRecord]:
        """Get feedback for an alert."""
        return self.session.query(HumanFeedbackRecord).filter(
            HumanFeedbackRecord.alert_id == alert_id
        ).first()

    def list_recent(self, limit: int = 100) -> List[HumanFeedbackRecord]:
        """List recent feedback records."""
        return self.session.query(HumanFeedbackRecord).order_by(
            HumanFeedbackRecord.created_at.desc()
        ).limit(limit).all()

    def list_disagreements(self, limit: int = 100) -> List[HumanFeedbackRecord]:
        """List feedback where human disagreed with recommendation."""
        return self.session.query(HumanFeedbackRecord).filter(
            HumanFeedbackRecord.agreed_with_recommendation == False
        ).order_by(HumanFeedbackRecord.created_at.desc()).limit(limit).all()

    def update_outcome(
        self,
        feedback_id: str,
        outcome_was_correct: bool,
        outcome_notes: Optional[str] = None,
    ) -> Optional[HumanFeedbackRecord]:
        """Update the outcome observation for feedback."""
        record = self.get(feedback_id)
        if record is None:
            return None

        record.outcome_observed = True
        record.outcome_was_correct = outcome_was_correct
        record.outcome_notes = outcome_notes
        record.updated_at = datetime.utcnow()
        self.session.flush()
        return record

    def get_agreement_stats(self) -> Dict[str, Any]:
        """Get statistics on recommendation agreement."""
        total = self.session.query(HumanFeedbackRecord).count()
        agreed = self.session.query(HumanFeedbackRecord).filter(
            HumanFeedbackRecord.agreed_with_recommendation == True
        ).count()

        return {
            "total_feedback": total,
            "agreed_count": agreed,
            "disagreed_count": total - agreed,
            "agreement_rate": agreed / total if total > 0 else 0.0,
        }
