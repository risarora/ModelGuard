"""Action recommendation engine for drift response."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from modelguard.core.config import Config, get_config
from modelguard.core.types import (
    ActionRecommendation,
    ActionType,
    SeverityLevel,
    SeverityScore,
    Urgency,
)


@dataclass
class ActionRule:
    """A rule for action recommendation."""

    name: str
    priority: int
    action: ActionType
    confidence: float
    urgency: Urgency
    prerequisites: List[str]
    condition: Callable[[SeverityScore, Dict[str, Any], Dict[str, Any]], bool]
    reasoning_template: str

    def matches(
        self,
        severity: SeverityScore,
        historical_context: Dict[str, Any],
        model_metadata: Dict[str, Any],
    ) -> bool:
        """Check if this rule matches the current situation."""
        return self.condition(severity, historical_context, model_metadata)

    def generate_reasoning(
        self,
        severity: SeverityScore,
        historical_context: Dict[str, Any],
        model_metadata: Dict[str, Any],
    ) -> List[str]:
        """Generate reasoning for this recommendation."""
        reasoning = [self.reasoning_template]

        # Add context-specific reasoning
        if severity.level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            reasoning.append(
                f"Severity level is {severity.level.value}, "
                f"with {len(severity.affected_features)} features affected."
            )

        if severity.impacts_predictions:
            reasoning.append("Drift is impacting prediction quality.")

        consecutive_days = historical_context.get("consecutive_drift_days", 0)
        if consecutive_days > 0:
            reasoning.append(f"Drift has persisted for {consecutive_days} consecutive days.")

        return reasoning


class ActionRecommender:
    """
    Rule-based action recommendation engine.

    Analyzes severity scores and context to recommend appropriate actions:
    - IGNORE: Temporary noise, no action needed
    - MONITOR: Early warning, continue watching
    - RETRAIN: Model needs updating with fresh data
    - ROLLBACK: Critical issue, revert to previous model version

    The engine uses a priority-based rule system that can be extended
    with ML-based recommendations in the future.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize action recommender.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.rules = self._build_rules()

    def recommend(
        self,
        severity: SeverityScore,
        historical_context: Optional[Dict[str, Any]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionRecommendation:
        """
        Generate action recommendation based on severity and context.

        Args:
            severity: Severity score from drift analysis
            historical_context: Historical information about drift patterns
            model_metadata: Information about the model

        Returns:
            ActionRecommendation with suggested action and reasoning
        """
        historical_context = historical_context or {}
        model_metadata = model_metadata or {}

        # Find matching rules
        matching_rules = [
            rule
            for rule in self.rules
            if rule.matches(severity, historical_context, model_metadata)
        ]

        # Select highest priority matching rule
        if matching_rules:
            best_rule = max(matching_rules, key=lambda r: r.priority)
            return self._create_recommendation(
                rule=best_rule,
                severity=severity,
                historical_context=historical_context,
                model_metadata=model_metadata,
            )

        # Default recommendation if no rules match
        return self._default_recommendation(severity)

    def _build_rules(self) -> List[ActionRule]:
        """Build the rule set for action recommendation."""
        return [
            # Rule 1: Critical severity -> Immediate rollback
            ActionRule(
                name="critical_severity_rollback",
                priority=100,
                action=ActionType.ROLLBACK,
                confidence=0.95,
                urgency=Urgency.IMMEDIATE,
                prerequisites=["notify_on_call", "create_incident"],
                condition=lambda s, h, m: s.level == SeverityLevel.CRITICAL,
                reasoning_template="Critical drift detected. Immediate rollback recommended to prevent service degradation.",
            ),
            # Rule 2: High severity with prediction impact -> Retrain urgently
            ActionRule(
                name="high_severity_with_impact",
                priority=90,
                action=ActionType.RETRAIN,
                confidence=0.90,
                urgency=Urgency.HIGH,
                prerequisites=["validate_data_quality", "backup_current_model"],
                condition=lambda s, h, m: (
                    s.level == SeverityLevel.HIGH and s.impacts_predictions
                ),
                reasoning_template="High severity drift affecting predictions. Urgent retraining recommended.",
            ),
            # Rule 3: High severity -> Retrain
            ActionRule(
                name="high_severity_retrain",
                priority=80,
                action=ActionType.RETRAIN,
                confidence=0.85,
                urgency=Urgency.MEDIUM,
                prerequisites=["validate_data_quality"],
                condition=lambda s, h, m: s.level == SeverityLevel.HIGH,
                reasoning_template="High severity drift detected. Retraining recommended.",
            ),
            # Rule 4: Persistent medium drift -> Retrain
            ActionRule(
                name="persistent_medium_drift",
                priority=70,
                action=ActionType.RETRAIN,
                confidence=0.80,
                urgency=Urgency.MEDIUM,
                prerequisites=["validate_data_quality"],
                condition=lambda s, h, m: (
                    s.level == SeverityLevel.MEDIUM
                    and h.get("consecutive_drift_days", 0) >= 7
                ),
                reasoning_template="Medium drift persisting for over a week. Retraining recommended.",
            ),
            # Rule 5: Medium severity -> Monitor closely
            ActionRule(
                name="medium_severity_monitor",
                priority=60,
                action=ActionType.MONITOR,
                confidence=0.75,
                urgency=Urgency.MEDIUM,
                prerequisites=[],
                condition=lambda s, h, m: s.level == SeverityLevel.MEDIUM,
                reasoning_template="Moderate drift detected. Close monitoring recommended.",
            ),
            # Rule 6: Persistent low drift -> Monitor
            ActionRule(
                name="persistent_low_drift",
                priority=50,
                action=ActionType.MONITOR,
                confidence=0.70,
                urgency=Urgency.LOW,
                prerequisites=[],
                condition=lambda s, h, m: (
                    s.level == SeverityLevel.LOW
                    and h.get("consecutive_drift_days", 0) >= 3
                ),
                reasoning_template="Low-level drift persisting. Continued monitoring recommended.",
            ),
            # Rule 7: Low severity, recent -> Ignore
            ActionRule(
                name="low_severity_recent",
                priority=40,
                action=ActionType.IGNORE,
                confidence=0.65,
                urgency=Urgency.LOW,
                prerequisites=[],
                condition=lambda s, h, m: (
                    s.level == SeverityLevel.LOW
                    and h.get("consecutive_drift_days", 0) < 3
                    and s.confidence < 0.7
                ),
                reasoning_template="Minor drift detected, likely temporary noise. No action needed.",
            ),
            # Rule 8: Low confidence, any severity -> Monitor
            ActionRule(
                name="low_confidence_monitor",
                priority=35,
                action=ActionType.MONITOR,
                confidence=0.60,
                urgency=Urgency.LOW,
                prerequisites=[],
                condition=lambda s, h, m: s.confidence < 0.5,
                reasoning_template="Detection confidence is low. Monitoring recommended to confirm findings.",
            ),
            # Rule 9: Critical model, medium+ drift -> Retrain
            ActionRule(
                name="critical_model_retrain",
                priority=85,
                action=ActionType.RETRAIN,
                confidence=0.85,
                urgency=Urgency.HIGH,
                prerequisites=["validate_data_quality", "backup_current_model"],
                condition=lambda s, h, m: (
                    m.get("criticality", "normal") == "critical"
                    and s.level in [SeverityLevel.MEDIUM, SeverityLevel.HIGH]
                ),
                reasoning_template="Drift detected in critical model. Proactive retraining recommended.",
            ),
            # Rule 10: No significant drift -> Ignore
            ActionRule(
                name="no_drift_ignore",
                priority=20,
                action=ActionType.IGNORE,
                confidence=0.90,
                urgency=Urgency.LOW,
                prerequisites=[],
                condition=lambda s, h, m: s.level == SeverityLevel.NONE,
                reasoning_template="No significant drift detected. No action needed.",
            ),
        ]

    def _create_recommendation(
        self,
        rule: ActionRule,
        severity: SeverityScore,
        historical_context: Dict[str, Any],
        model_metadata: Dict[str, Any],
    ) -> ActionRecommendation:
        """Create a recommendation from a matched rule."""
        reasoning = rule.generate_reasoning(
            severity, historical_context, model_metadata
        )

        # Estimate impact
        estimated_impact = self._estimate_impact(
            rule.action, severity, model_metadata
        )

        return ActionRecommendation(
            action=rule.action,
            confidence=rule.confidence,
            reasoning=reasoning,
            urgency=rule.urgency,
            estimated_impact=estimated_impact,
            prerequisite_actions=rule.prerequisites,
        )

    def _default_recommendation(
        self,
        severity: SeverityScore,
    ) -> ActionRecommendation:
        """Generate default recommendation when no rules match."""
        if severity.level in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            action = ActionType.RETRAIN
            urgency = Urgency.HIGH
            reasoning = ["Significant drift detected but no specific rule matched. Retraining recommended as precaution."]
        elif severity.level == SeverityLevel.MEDIUM:
            action = ActionType.MONITOR
            urgency = Urgency.MEDIUM
            reasoning = ["Moderate drift detected. Monitoring recommended."]
        else:
            action = ActionType.IGNORE
            urgency = Urgency.LOW
            reasoning = ["Drift levels within acceptable range. No action needed."]

        return ActionRecommendation(
            action=action,
            confidence=0.60,
            reasoning=reasoning,
            urgency=urgency,
            estimated_impact="Unknown - using default recommendation",
            prerequisite_actions=[],
        )

    def _estimate_impact(
        self,
        action: ActionType,
        severity: SeverityScore,
        model_metadata: Dict[str, Any],
    ) -> str:
        """Estimate the impact of taking or not taking the recommended action."""
        if action == ActionType.ROLLBACK:
            return (
                "Rollback will immediately restore previous model version. "
                "Current predictions may be unreliable until rollback completes."
            )
        elif action == ActionType.RETRAIN:
            affected_pct = len(severity.affected_features) / max(
                len(severity.feature_scores), 1
            ) * 100
            return (
                f"Retraining will update model to handle drifted features "
                f"({affected_pct:.0f}% affected). Expected to restore prediction quality."
            )
        elif action == ActionType.MONITOR:
            return (
                "Continued monitoring will track drift progression. "
                "Early detection of worsening conditions."
            )
        else:  # IGNORE
            return (
                "No immediate action required. Current drift levels are acceptable."
            )

    def get_allowed_actions(self) -> List[str]:
        """Get list of allowed actions from configuration."""
        return self.config.actions.allowed_actions

    def is_auto_action_enabled(self) -> bool:
        """Check if automatic actions are enabled."""
        return self.config.actions.auto_action_enabled

    def can_auto_execute(
        self,
        recommendation: ActionRecommendation,
        severity: SeverityScore,
    ) -> bool:
        """
        Check if a recommendation can be auto-executed without human approval.

        Only low-severity actions can be auto-executed (if enabled).
        """
        if not self.is_auto_action_enabled():
            return False

        max_severity = self.config.actions.auto_action_max_severity
        severity_order = ["none", "low", "medium", "high", "critical"]

        current_idx = severity_order.index(severity.level.value)
        max_idx = severity_order.index(max_severity)

        return current_idx <= max_idx
