"""Alert management for human-in-the-loop review."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from modelguard.core.config import Config, get_config
from modelguard.core.types import (
    ActionRecommendation,
    ActionType,
    Alert,
    AlertStatus,
    DriftReport,
    SeverityLevel,
    SeverityScore,
    Urgency,
)


class AlertManager:
    """
    Manages alert lifecycle for human review.

    Alerts are created when drift is detected and requires human attention.
    This manager handles:
    - Alert creation with full context
    - Alert routing and assignment
    - Status tracking
    - Resolution and feedback collection
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize alert manager.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()

    def create_alert(
        self,
        model_id: str,
        drift_report: DriftReport,
        severity: SeverityScore,
        recommendation: ActionRecommendation,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """
        Create a new alert from drift analysis results.

        Args:
            model_id: ID of the affected model
            drift_report: Drift detection report
            severity: Severity assessment
            recommendation: Action recommendation
            metadata: Additional metadata

        Returns:
            Created Alert object
        """
        # Determine urgency based on severity and recommendation
        urgency = self._determine_urgency(severity, recommendation)

        alert = Alert.create(
            model_id=model_id,
            alert_type="drift",
            severity=severity.level,
            urgency=urgency,
            drift_report_id=drift_report.id,
        )

        # Add metadata
        alert.metadata = {
            "drift_percentage": drift_report.drift_percentage,
            "affected_features": drift_report.features_with_drift,
            "recommendation": recommendation.action.value,
            "recommendation_confidence": recommendation.confidence,
            "severity_score": severity.overall_score,
            **(metadata or {}),
        }

        return alert

    def should_create_alert(
        self,
        severity: SeverityScore,
        recommendation: ActionRecommendation,
    ) -> bool:
        """
        Determine if an alert should be created.

        Not all drift detections require human attention.

        Args:
            severity: Severity assessment
            recommendation: Action recommendation

        Returns:
            True if an alert should be created
        """
        # Always create alerts for high/critical severity
        if severity.level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            return True

        # Create alerts for medium severity if action is needed
        if severity.level == SeverityLevel.MEDIUM:
            if recommendation.action in [ActionType.RETRAIN, ActionType.ROLLBACK]:
                return True

        # Create alerts for any severity if auto-action is disabled
        # and action is not IGNORE
        if not self.config.actions.auto_action_enabled:
            if recommendation.action != ActionType.IGNORE:
                return True

        # Don't create alerts for low/none severity with IGNORE recommendation
        return False

    def _determine_urgency(
        self,
        severity: SeverityScore,
        recommendation: ActionRecommendation,
    ) -> Urgency:
        """Determine alert urgency from severity and recommendation."""
        # Use recommendation urgency as primary
        if recommendation.urgency == Urgency.IMMEDIATE:
            return Urgency.IMMEDIATE

        # Escalate based on severity
        if severity.level == SeverityLevel.CRITICAL:
            return Urgency.IMMEDIATE
        elif severity.level == SeverityLevel.HIGH:
            return max(recommendation.urgency, Urgency.HIGH, key=lambda u: ["low", "medium", "high", "immediate"].index(u.value))
        elif severity.level == SeverityLevel.MEDIUM:
            return max(recommendation.urgency, Urgency.MEDIUM, key=lambda u: ["low", "medium", "high", "immediate"].index(u.value))
        else:
            return recommendation.urgency

    def acknowledge_alert(
        self,
        alert: Alert,
        user_id: Optional[str] = None,
    ) -> Alert:
        """
        Mark an alert as acknowledged.

        Args:
            alert: Alert to acknowledge
            user_id: ID of user acknowledging

        Returns:
            Updated Alert
        """
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        if user_id:
            alert.assigned_to = user_id
        return alert

    def start_review(
        self,
        alert: Alert,
        user_id: str,
    ) -> Alert:
        """
        Mark an alert as being reviewed.

        Args:
            alert: Alert to review
            user_id: ID of reviewing user

        Returns:
            Updated Alert
        """
        alert.status = AlertStatus.IN_REVIEW
        alert.assigned_to = user_id
        return alert

    def resolve_alert(
        self,
        alert: Alert,
        decision: ActionType,
        decided_by: str,
        notes: Optional[str] = None,
    ) -> Alert:
        """
        Resolve an alert with a decision.

        Args:
            alert: Alert to resolve
            decision: Action decision made
            decided_by: User who made the decision
            notes: Optional decision notes

        Returns:
            Updated Alert
        """
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.decision = decision
        alert.decided_by = decided_by
        alert.decision_notes = notes
        return alert

    def dismiss_alert(
        self,
        alert: Alert,
        decided_by: str,
        reason: Optional[str] = None,
    ) -> Alert:
        """
        Dismiss an alert as not actionable.

        Args:
            alert: Alert to dismiss
            decided_by: User dismissing
            reason: Reason for dismissal

        Returns:
            Updated Alert
        """
        alert.status = AlertStatus.DISMISSED
        alert.resolved_at = datetime.utcnow()
        alert.decided_by = decided_by
        alert.decision_notes = reason or "Alert dismissed"
        return alert

    def get_alert_summary(self, alert: Alert) -> Dict[str, Any]:
        """
        Get a summary of an alert for display.

        Args:
            alert: Alert to summarize

        Returns:
            Dictionary with summary information
        """
        return {
            "id": alert.id,
            "model_id": alert.model_id,
            "severity": alert.severity.value,
            "urgency": alert.urgency.value,
            "status": alert.status.value,
            "created_at": alert.created_at.isoformat(),
            "affected_features": alert.metadata.get("affected_features", []),
            "drift_percentage": alert.metadata.get("drift_percentage", 0),
            "recommendation": alert.metadata.get("recommendation", "unknown"),
            "assigned_to": alert.assigned_to,
            "age_hours": self._calculate_age_hours(alert),
        }

    def _calculate_age_hours(self, alert: Alert) -> float:
        """Calculate alert age in hours."""
        if alert.resolved_at:
            delta = alert.resolved_at - alert.created_at
        else:
            delta = datetime.utcnow() - alert.created_at
        return delta.total_seconds() / 3600

    def is_escalation_needed(self, alert: Alert) -> bool:
        """
        Check if an alert needs escalation.

        Args:
            alert: Alert to check

        Returns:
            True if escalation is needed
        """
        if alert.status in [AlertStatus.RESOLVED, AlertStatus.DISMISSED]:
            return False

        age_hours = self._calculate_age_hours(alert)
        timeout = self.config.human_loop.escalation_timeout_hours

        return age_hours > timeout

    def prioritize_alerts(
        self,
        alerts: List[Alert],
    ) -> List[Alert]:
        """
        Sort alerts by priority for review.

        Priority order:
        1. Urgency (immediate > high > medium > low)
        2. Severity (critical > high > medium > low > none)
        3. Age (older first)

        Args:
            alerts: List of alerts to prioritize

        Returns:
            Sorted list of alerts
        """
        urgency_order = {"immediate": 0, "high": 1, "medium": 2, "low": 3}
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}

        def priority_key(alert: Alert):
            return (
                urgency_order.get(alert.urgency.value, 4),
                severity_order.get(alert.severity.value, 5),
                alert.created_at,  # Older alerts first
            )

        return sorted(alerts, key=priority_key)
