"""Severity scoring for drift detection results."""

from typing import Any, Dict, List, Optional

import numpy as np

from modelguard.core.config import Config, SeverityConfig, get_config
from modelguard.core.types import DriftReport, DriftResult, SeverityLevel, SeverityScore


class SeverityScorer:
    """
    Computes severity scores from drift detection results.

    Instead of binary "drift detected" outputs, this scorer produces
    nuanced severity assessments that consider:
    - Number of features affected
    - Magnitude of drift statistics
    - Feature importance weights
    - Impact on predictions
    - Confidence in the assessment

    Severity Levels:
    - NONE: No significant drift detected
    - LOW: Minor drift, monitoring recommended
    - MEDIUM: Moderate drift, investigation needed
    - HIGH: Significant drift, retraining recommended
    - CRITICAL: Severe drift, immediate action required
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize severity scorer.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.severity_config = self.config.severity

    def score(
        self,
        drift_results: List[DriftResult],
        feature_metadata: Optional[Dict[str, Any]] = None,
        prediction_impact: Optional[float] = None,
    ) -> SeverityScore:
        """
        Compute severity from drift results.

        Args:
            drift_results: List of drift detection results
            feature_metadata: Optional metadata about features (importance weights)
            prediction_impact: Optional measure of prediction quality degradation (0-1)

        Returns:
            SeverityScore with overall assessment
        """
        if not drift_results:
            return self._create_empty_score()

        feature_metadata = feature_metadata or {}

        # Group results by feature
        feature_groups = self._group_by_feature(drift_results)

        # Calculate per-feature severity
        feature_scores: Dict[str, float] = {}
        affected_features: List[str] = []

        for feature, results in feature_groups.items():
            feature_severity = self._calculate_feature_severity(results)

            # Apply feature weight
            weight = self._get_feature_weight(feature, feature_metadata)
            feature_scores[feature] = feature_severity * weight

            # Track affected features
            if any(r.drift_detected for r in results):
                affected_features.append(feature)

        # Aggregate to overall score
        overall_score = self._aggregate_scores(feature_scores)

        # Adjust for prediction impact
        if prediction_impact is not None and prediction_impact > 0:
            impact_multiplier = 1 + (
                prediction_impact * self.severity_config.impact_multiplier
            )
            overall_score = min(1.0, overall_score * impact_multiplier)

        # Calculate confidence
        confidence = self._calculate_confidence(drift_results)

        # Determine severity level
        level = self._classify_level(overall_score)

        # Generate explanation
        explanation = self._generate_explanation(
            overall_score=overall_score,
            level=level,
            affected_features=affected_features,
            feature_scores=feature_scores,
            prediction_impact=prediction_impact,
        )

        return SeverityScore(
            overall_score=overall_score,
            level=level,
            affected_features=affected_features,
            feature_scores=feature_scores,
            impacts_predictions=prediction_impact is not None and prediction_impact > 0.1,
            confidence=confidence,
            explanation=explanation,
        )

    def score_report(
        self,
        report: DriftReport,
        feature_metadata: Optional[Dict[str, Any]] = None,
        prediction_impact: Optional[float] = None,
    ) -> SeverityScore:
        """
        Compute severity from a drift report.

        Args:
            report: DriftReport with detection results
            feature_metadata: Optional feature metadata
            prediction_impact: Optional prediction impact measure

        Returns:
            SeverityScore
        """
        # Flatten all results
        all_results = []
        for feature_results in report.feature_results.values():
            all_results.extend(feature_results)

        return self.score(
            drift_results=all_results,
            feature_metadata=feature_metadata,
            prediction_impact=prediction_impact,
        )

    def _create_empty_score(self) -> SeverityScore:
        """Create an empty severity score for no drift."""
        return SeverityScore(
            overall_score=0.0,
            level=SeverityLevel.NONE,
            affected_features=[],
            feature_scores={},
            impacts_predictions=False,
            confidence=1.0,
            explanation="No drift results to evaluate.",
        )

    def _group_by_feature(
        self,
        results: List[DriftResult],
    ) -> Dict[str, List[DriftResult]]:
        """Group drift results by feature name."""
        groups: Dict[str, List[DriftResult]] = {}
        for result in results:
            if result.feature_name not in groups:
                groups[result.feature_name] = []
            groups[result.feature_name].append(result)
        return groups

    def _calculate_feature_severity(
        self,
        results: List[DriftResult],
    ) -> float:
        """
        Calculate severity for a single feature from multiple method results.

        Combines:
        - Whether drift was detected
        - The magnitude of the drift statistic
        - The p-value (lower = more significant)
        """
        if not results:
            return 0.0

        severities = []

        for result in results:
            severity = 0.0

            if result.drift_detected:
                # Base severity from detection
                severity = 0.3

                # Add severity based on statistic magnitude
                # Normalize statistic relative to threshold
                if result.threshold > 0:
                    relative_magnitude = result.statistic / result.threshold
                    # Cap at 3x threshold for max contribution
                    magnitude_contribution = min(relative_magnitude / 3, 1.0) * 0.4
                    severity += magnitude_contribution

                # Add severity based on p-value (if available)
                if result.p_value is not None:
                    # Lower p-value = higher severity
                    # p=0.05 -> 0.1 contribution, p=0.001 -> 0.3 contribution
                    if result.p_value < 0.001:
                        pvalue_contribution = 0.3
                    elif result.p_value < 0.01:
                        pvalue_contribution = 0.2
                    elif result.p_value < 0.05:
                        pvalue_contribution = 0.1
                    else:
                        pvalue_contribution = 0.0
                    severity += pvalue_contribution

            severities.append(severity)

        # Take maximum across methods (most severe detection wins)
        return max(severities) if severities else 0.0

    def _get_feature_weight(
        self,
        feature: str,
        metadata: Dict[str, Any],
    ) -> float:
        """Get importance weight for a feature."""
        # Check config first
        if feature in self.severity_config.feature_weights:
            return self.severity_config.feature_weights[feature]

        # Check metadata
        feature_weights = metadata.get("feature_weights", {})
        if feature in feature_weights:
            return feature_weights[feature]

        # Check importance from metadata
        feature_importance = metadata.get("feature_importance", {})
        if feature in feature_importance:
            return feature_importance[feature]

        return self.severity_config.default_weight

    def _aggregate_scores(
        self,
        feature_scores: Dict[str, float],
    ) -> float:
        """Aggregate feature-level scores to overall score."""
        if not feature_scores:
            return 0.0

        scores = list(feature_scores.values())
        strategy = self.severity_config.aggregation

        if strategy == "max":
            return max(scores)
        elif strategy == "mean":
            return float(np.mean(scores))
        elif strategy == "weighted_mean":
            # Weighted by feature weights (already applied)
            return float(np.mean(scores))
        else:
            return float(np.mean(scores))

    def _calculate_confidence(
        self,
        results: List[DriftResult],
    ) -> float:
        """
        Calculate confidence in the severity assessment.

        Higher confidence when:
        - Multiple methods agree
        - Results have p-values
        - Larger sample sizes
        """
        if not results:
            return 0.0

        confidence_factors = []

        # Factor 1: Method agreement
        drift_detected_count = sum(1 for r in results if r.drift_detected)
        total_count = len(results)

        if total_count > 1:
            agreement = max(
                drift_detected_count / total_count,
                (total_count - drift_detected_count) / total_count,
            )
            confidence_factors.append(agreement)

        # Factor 2: P-value availability and significance
        results_with_pvalue = [r for r in results if r.p_value is not None]
        if results_with_pvalue:
            # Higher confidence when p-values are extreme (very low or very high)
            avg_pvalue = np.mean([r.p_value for r in results_with_pvalue])
            pvalue_confidence = 1 - 2 * abs(avg_pvalue - 0.5)  # Max at 0 or 1
            confidence_factors.append(max(0.5, pvalue_confidence))

        # Factor 3: Sample size consideration
        sample_sizes = [
            r.metadata.get("current_size", 0) for r in results
            if "current_size" in r.metadata
        ]
        if sample_sizes:
            avg_size = np.mean(sample_sizes)
            # Confidence increases with sample size, max at ~1000 samples
            size_confidence = min(1.0, avg_size / 1000)
            confidence_factors.append(max(0.5, size_confidence))

        if confidence_factors:
            return float(np.mean(confidence_factors))
        return 0.7  # Default moderate confidence

    def _classify_level(self, score: float) -> SeverityLevel:
        """Classify numeric score into severity level."""
        thresholds = self.severity_config.thresholds

        if score >= thresholds.get("critical", 0.8):
            return SeverityLevel.CRITICAL
        elif score >= thresholds.get("high", 0.5):
            return SeverityLevel.HIGH
        elif score >= thresholds.get("medium", 0.3):
            return SeverityLevel.MEDIUM
        elif score >= thresholds.get("low", 0.1):
            return SeverityLevel.LOW
        else:
            return SeverityLevel.NONE

    def _generate_explanation(
        self,
        overall_score: float,
        level: SeverityLevel,
        affected_features: List[str],
        feature_scores: Dict[str, float],
        prediction_impact: Optional[float],
    ) -> str:
        """Generate human-readable explanation of severity assessment."""
        parts = []

        # Overall summary
        if level == SeverityLevel.NONE:
            parts.append("No significant drift detected.")
        elif level == SeverityLevel.LOW:
            parts.append("Minor drift detected. Monitoring recommended.")
        elif level == SeverityLevel.MEDIUM:
            parts.append("Moderate drift detected. Investigation recommended.")
        elif level == SeverityLevel.HIGH:
            parts.append("Significant drift detected. Retraining may be needed.")
        elif level == SeverityLevel.CRITICAL:
            parts.append("Critical drift detected. Immediate action required.")

        # Affected features
        if affected_features:
            n_affected = len(affected_features)
            n_total = len(feature_scores)
            parts.append(
                f"{n_affected} of {n_total} features show drift "
                f"({n_affected/n_total*100:.0f}%)."
            )

            # Top affected features
            top_features = sorted(
                [(f, s) for f, s in feature_scores.items() if s > 0.1],
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            if top_features:
                feature_list = ", ".join(
                    f"{f} ({s:.2f})" for f, s in top_features
                )
                parts.append(f"Most affected: {feature_list}.")

        # Prediction impact
        if prediction_impact is not None and prediction_impact > 0.1:
            parts.append(
                f"Prediction quality degradation: {prediction_impact*100:.1f}%."
            )

        return " ".join(parts)
