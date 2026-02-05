"""Feedback collection and learning from human decisions."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from modelguard.core.types import ActionType, Alert, HumanFeedback


class FeedbackManager:
    """
    Manages human feedback collection and analysis.

    Collects feedback on:
    - Whether recommendations were followed
    - Why decisions differed from recommendations
    - Outcomes of decisions

    This data can be used to improve the recommendation engine.
    """

    def __init__(self):
        """Initialize feedback manager."""
        pass

    def record_feedback(
        self,
        alert: Alert,
        original_recommendation: ActionType,
        human_decision: ActionType,
        feedback_type: str,
        reason: Optional[str] = None,
    ) -> HumanFeedback:
        """
        Record feedback on an alert resolution.

        Args:
            alert: The resolved alert
            original_recommendation: What the system recommended
            human_decision: What the human decided
            feedback_type: Type of feedback (correct, incorrect, partially_correct)
            reason: Optional explanation of disagreement

        Returns:
            HumanFeedback record
        """
        return HumanFeedback.create(
            alert_id=alert.id,
            original_recommendation=original_recommendation,
            human_decision=human_decision,
            feedback_type=feedback_type,
            reason=reason,
        )

    def record_outcome(
        self,
        feedback: HumanFeedback,
        was_correct: bool,
        notes: Optional[str] = None,
    ) -> HumanFeedback:
        """
        Record the outcome of a decision.

        Called after some time has passed to evaluate if the
        decision (human or system) was correct.

        Args:
            feedback: Existing feedback record
            was_correct: Whether the decision turned out to be correct
            notes: Optional notes on the outcome

        Returns:
            Updated HumanFeedback
        """
        feedback.outcome_observed = True
        feedback.outcome_was_correct = was_correct
        feedback.outcome_notes = notes
        return feedback

    def analyze_feedback(
        self,
        feedback_records: List[HumanFeedback],
    ) -> Dict[str, Any]:
        """
        Analyze feedback patterns to improve recommendations.

        Args:
            feedback_records: List of feedback records

        Returns:
            Analysis results with patterns and insights
        """
        if not feedback_records:
            return {
                "total_feedback": 0,
                "agreement_rate": 0.0,
                "insights": ["No feedback data available"],
            }

        total = len(feedback_records)
        agreed = sum(1 for f in feedback_records if f.agreed_with_recommendation)

        # Analyze by action type
        action_stats: Dict[str, Dict[str, int]] = {}
        for f in feedback_records:
            rec = f.original_recommendation.value
            if rec not in action_stats:
                action_stats[rec] = {"total": 0, "agreed": 0, "disagreed": 0}
            action_stats[rec]["total"] += 1
            if f.agreed_with_recommendation:
                action_stats[rec]["agreed"] += 1
            else:
                action_stats[rec]["disagreed"] += 1

        # Analyze outcomes (where available)
        with_outcome = [f for f in feedback_records if f.outcome_observed]
        correct_decisions = sum(1 for f in with_outcome if f.outcome_was_correct)

        # Generate insights
        insights = self._generate_insights(
            total, agreed, action_stats, with_outcome, correct_decisions
        )

        return {
            "total_feedback": total,
            "agreement_rate": agreed / total if total > 0 else 0.0,
            "action_stats": action_stats,
            "outcomes_tracked": len(with_outcome),
            "outcome_accuracy": correct_decisions / len(with_outcome) if with_outcome else None,
            "insights": insights,
        }

    def _generate_insights(
        self,
        total: int,
        agreed: int,
        action_stats: Dict[str, Dict[str, int]],
        with_outcome: List[HumanFeedback],
        correct_decisions: int,
    ) -> List[str]:
        """Generate insights from feedback analysis."""
        insights = []

        # Overall agreement insight
        agreement_rate = agreed / total if total > 0 else 0
        if agreement_rate < 0.5:
            insights.append(
                f"Low agreement rate ({agreement_rate:.0%}) - "
                "recommendation rules may need adjustment"
            )
        elif agreement_rate > 0.8:
            insights.append(
                f"High agreement rate ({agreement_rate:.0%}) - "
                "recommendations are generally trusted"
            )

        # Per-action insights
        for action, stats in action_stats.items():
            action_agreement = stats["agreed"] / stats["total"] if stats["total"] > 0 else 0
            if action_agreement < 0.4 and stats["total"] >= 5:
                insights.append(
                    f"'{action}' recommendations often overridden ({action_agreement:.0%} agreement) - "
                    "consider adjusting threshold"
                )

        # Outcome insights
        if with_outcome:
            accuracy = correct_decisions / len(with_outcome)
            if accuracy < 0.6:
                insights.append(
                    f"Decision accuracy ({accuracy:.0%}) suggests model may need updating"
                )
            elif accuracy > 0.9:
                insights.append(
                    f"High decision accuracy ({accuracy:.0%}) indicates effective drift management"
                )

        if not insights:
            insights.append("Insufficient data for detailed insights")

        return insights

    def get_disagreement_reasons(
        self,
        feedback_records: List[HumanFeedback],
    ) -> Dict[str, List[str]]:
        """
        Collect reasons for disagreements by action type.

        Args:
            feedback_records: List of feedback records

        Returns:
            Dictionary mapping action types to lists of reasons
        """
        disagreements = [f for f in feedback_records if not f.agreed_with_recommendation]

        reasons: Dict[str, List[str]] = {}
        for f in disagreements:
            action = f.original_recommendation.value
            if action not in reasons:
                reasons[action] = []
            if f.reason:
                reasons[action].append(f.reason)

        return reasons

    def suggest_rule_adjustments(
        self,
        feedback_records: List[HumanFeedback],
    ) -> List[Dict[str, Any]]:
        """
        Suggest adjustments to recommendation rules based on feedback.

        Args:
            feedback_records: List of feedback records

        Returns:
            List of suggested adjustments
        """
        suggestions = []

        # Analyze patterns
        analysis = self.analyze_feedback(feedback_records)
        action_stats = analysis.get("action_stats", {})

        for action, stats in action_stats.items():
            if stats["total"] < 5:
                continue

            agreement_rate = stats["agreed"] / stats["total"]

            if agreement_rate < 0.4:
                # Often overridden - maybe threshold too aggressive
                suggestions.append({
                    "action": action,
                    "type": "threshold_adjustment",
                    "direction": "increase",
                    "reason": f"Humans override {action} {(1-agreement_rate):.0%} of the time",
                    "confidence": min(stats["total"] / 20, 1.0),
                })
            elif agreement_rate > 0.95 and stats["total"] >= 10:
                # Always accepted - maybe could be auto-executed
                suggestions.append({
                    "action": action,
                    "type": "auto_execution",
                    "reason": f"Humans always accept {action} recommendations",
                    "confidence": min(stats["total"] / 50, 1.0),
                })

        return suggestions
