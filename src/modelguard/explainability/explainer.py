"""Drift explainability analysis."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modelguard.core.types import DriftReport, DriftResult, SeverityScore


@dataclass
class FeatureDriftExplanation:
    """Explanation for drift in a single feature."""

    feature_name: str
    drift_detected: bool
    severity_score: float
    methods_detected: List[str]  # Which methods detected drift
    methods_clear: List[str]  # Which methods found no drift
    statistics_summary: Dict[str, float]  # Key statistics from detectors
    distribution_change: str  # Text description of change
    potential_causes: List[str]  # Potential explanations
    recommended_investigation: List[str]  # Next steps to investigate


@dataclass
class DriftExplanation:
    """Comprehensive explanation of drift analysis."""

    summary: str  # High-level summary
    feature_explanations: List[FeatureDriftExplanation]
    top_drifted_features: List[str]  # Features ranked by drift severity
    potential_root_causes: List[str]  # System-wide potential causes
    drift_pattern: str  # Pattern type: sudden, gradual, seasonal, etc.
    recommendations: List[str]  # Investigation recommendations
    confidence_assessment: str  # Assessment of detection confidence


class DriftExplainer:
    """
    Generates human-readable explanations for drift detection results.

    This module answers:
    - Which features drifted most?
    - When did drift start?
    - What might have caused the drift?
    - How confident are we in the detection?
    """

    def explain(
        self,
        report: DriftReport,
        severity: Optional[SeverityScore] = None,
        historical_reports: Optional[List[DriftReport]] = None,
    ) -> DriftExplanation:
        """
        Generate comprehensive explanation for a drift report.

        Args:
            report: DriftReport to explain
            severity: Optional severity score for additional context
            historical_reports: Optional historical reports for trend analysis

        Returns:
            DriftExplanation with detailed analysis
        """
        # Explain each feature
        feature_explanations = []
        for feature_name, results in report.feature_results.items():
            explanation = self._explain_feature(feature_name, results)
            feature_explanations.append(explanation)

        # Rank features by drift severity
        top_drifted = self._rank_features(feature_explanations)

        # Analyze drift pattern
        drift_pattern = self._analyze_pattern(historical_reports)

        # Generate potential root causes
        root_causes = self._identify_root_causes(
            feature_explanations, report, historical_reports
        )

        # Generate summary
        summary = self._generate_summary(
            report, severity, feature_explanations, drift_pattern
        )

        # Assess confidence
        confidence_assessment = self._assess_confidence(
            report, feature_explanations
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            feature_explanations, drift_pattern, root_causes
        )

        return DriftExplanation(
            summary=summary,
            feature_explanations=feature_explanations,
            top_drifted_features=top_drifted,
            potential_root_causes=root_causes,
            drift_pattern=drift_pattern,
            recommendations=recommendations,
            confidence_assessment=confidence_assessment,
        )

    def _explain_feature(
        self,
        feature_name: str,
        results: List[DriftResult],
    ) -> FeatureDriftExplanation:
        """Generate explanation for a single feature."""
        methods_detected = [r.method_name for r in results if r.drift_detected]
        methods_clear = [r.method_name for r in results if not r.drift_detected]

        # Collect statistics
        statistics = {}
        for r in results:
            statistics[f"{r.method_name}_statistic"] = r.statistic
            if r.p_value is not None:
                statistics[f"{r.method_name}_pvalue"] = r.p_value

        # Determine drift severity
        severity_score = max(
            (r.statistic / r.threshold if r.threshold > 0 else 0)
            for r in results
        ) if results else 0.0

        # Generate distribution change description
        distribution_change = self._describe_distribution_change(results)

        # Generate potential causes
        potential_causes = self._identify_feature_causes(feature_name, results)

        # Generate investigation recommendations
        investigation = self._recommend_investigation(feature_name, results)

        return FeatureDriftExplanation(
            feature_name=feature_name,
            drift_detected=any(r.drift_detected for r in results),
            severity_score=severity_score,
            methods_detected=methods_detected,
            methods_clear=methods_clear,
            statistics_summary=statistics,
            distribution_change=distribution_change,
            potential_causes=potential_causes,
            recommended_investigation=investigation,
        )

    def _describe_distribution_change(
        self,
        results: List[DriftResult],
    ) -> str:
        """Describe how the distribution changed."""
        descriptions = []

        for r in results:
            meta = r.metadata

            # Check for mean shift
            if "reference_mean" in meta and "current_mean" in meta:
                ref_mean = meta["reference_mean"]
                cur_mean = meta["current_mean"]
                shift = cur_mean - ref_mean
                if abs(shift) > 0.01 * abs(ref_mean) if ref_mean != 0 else abs(shift) > 0.01:
                    direction = "increased" if shift > 0 else "decreased"
                    descriptions.append(
                        f"Mean has {direction} from {ref_mean:.2f} to {cur_mean:.2f}"
                    )

            # Check for variance change
            if "reference_std" in meta and "current_std" in meta:
                ref_std = meta["reference_std"]
                cur_std = meta["current_std"]
                if abs(cur_std - ref_std) > 0.1 * ref_std if ref_std > 0 else abs(cur_std - ref_std) > 0.01:
                    change = "more variable" if cur_std > ref_std else "less variable"
                    descriptions.append(f"Distribution has become {change}")

            # Check severity from PSI
            if "severity" in meta:
                descriptions.append(f"PSI indicates {meta['severity']} drift")

        return "; ".join(descriptions) if descriptions else "No significant distribution change details available"

    def _identify_feature_causes(
        self,
        feature_name: str,
        results: List[DriftResult],
    ) -> List[str]:
        """Identify potential causes for feature drift."""
        causes = []

        for r in results:
            meta = r.metadata

            # Mean shift might indicate
            if "mean_shift" in meta and abs(meta["mean_shift"]) > 0:
                causes.append("Population segment shift (different user demographics)")
                causes.append("Upstream data processing change")

            # Variance change might indicate
            if "reference_std" in meta and "current_std" in meta:
                ref_std = meta["reference_std"]
                cur_std = meta["current_std"]
                if cur_std > ref_std * 1.5:
                    causes.append("Data quality issues (outliers or noise)")
                    causes.append("Measurement system changes")
                elif cur_std < ref_std * 0.5:
                    causes.append("Data pipeline filtering changes")
                    causes.append("Missing value handling changes")

        # Generic causes
        if not causes:
            causes = [
                "Natural population drift over time",
                "Seasonal or temporal patterns",
                "External event impact",
                "Data collection changes",
            ]

        return list(set(causes))[:4]  # Limit to 4 unique causes

    def _recommend_investigation(
        self,
        feature_name: str,
        results: List[DriftResult],
    ) -> List[str]:
        """Recommend investigation steps for a feature."""
        recommendations = [
            f"Plot distribution comparison for {feature_name}",
            f"Check data pipeline logs for {feature_name}",
        ]

        has_high_severity = any(
            r.statistic / r.threshold > 2 if r.threshold > 0 else False
            for r in results
        )

        if has_high_severity:
            recommendations.extend([
                f"Review recent data source changes affecting {feature_name}",
                f"Check for outliers or data quality issues in {feature_name}",
            ])

        return recommendations

    def _rank_features(
        self,
        explanations: List[FeatureDriftExplanation],
    ) -> List[str]:
        """Rank features by drift severity."""
        drifted = [e for e in explanations if e.drift_detected]
        sorted_features = sorted(
            drifted,
            key=lambda e: e.severity_score,
            reverse=True,
        )
        return [e.feature_name for e in sorted_features]

    def _analyze_pattern(
        self,
        historical_reports: Optional[List[DriftReport]],
    ) -> str:
        """Analyze drift pattern from historical data."""
        if not historical_reports or len(historical_reports) < 2:
            return "unknown (insufficient historical data)"

        # Sort by timestamp
        sorted_reports = sorted(historical_reports, key=lambda r: r.timestamp)

        # Check drift percentages over time
        drift_percentages = [r.drift_percentage for r in sorted_reports]

        if len(drift_percentages) < 3:
            return "unknown (insufficient data points)"

        # Calculate trend
        recent = drift_percentages[-3:]
        older = drift_percentages[:-3] if len(drift_percentages) > 3 else [0]

        avg_recent = np.mean(recent)
        avg_older = np.mean(older)

        if avg_recent > avg_older * 2:
            return "sudden (rapid increase in drift)"
        elif avg_recent > avg_older * 1.2:
            return "gradual (steady increase in drift)"
        elif avg_recent < avg_older * 0.5:
            return "recovering (drift decreasing)"
        else:
            return "stable (consistent drift levels)"

    def _identify_root_causes(
        self,
        explanations: List[FeatureDriftExplanation],
        report: DriftReport,
        historical_reports: Optional[List[DriftReport]],
    ) -> List[str]:
        """Identify potential system-wide root causes."""
        causes = []

        # Check affected feature count
        affected_count = len([e for e in explanations if e.drift_detected])
        total_count = len(explanations)

        if affected_count > total_count * 0.7:
            causes.append(
                "Widespread drift suggests global data pipeline or population change"
            )
        elif affected_count > total_count * 0.3:
            causes.append(
                "Moderate number of features affected - possible correlated changes"
            )

        # Check sample size changes
        if report.current_sample_size < report.reference_sample_size * 0.5:
            causes.append("Significant sample size decrease - possible data collection issue")
        elif report.current_sample_size > report.reference_sample_size * 2:
            causes.append("Large sample size increase - possible new user segment")

        # Add generic causes
        causes.extend([
            "Temporal drift (model aging)",
            "Concept drift (real-world patterns changing)",
            "Data quality degradation",
        ])

        return causes[:5]  # Limit to 5 causes

    def _generate_summary(
        self,
        report: DriftReport,
        severity: Optional[SeverityScore],
        explanations: List[FeatureDriftExplanation],
        pattern: str,
    ) -> str:
        """Generate high-level summary."""
        affected = len([e for e in explanations if e.drift_detected])
        total = len(explanations)

        parts = []

        if affected == 0:
            parts.append("No significant drift detected across monitored features.")
        else:
            parts.append(
                f"Drift detected in {affected} of {total} features "
                f"({report.drift_percentage:.1f}%)."
            )

        if severity:
            parts.append(f"Overall severity: {severity.level.value}.")

        if pattern != "unknown":
            parts.append(f"Drift pattern appears {pattern}.")

        top_features = [e for e in explanations if e.drift_detected][:3]
        if top_features:
            top_names = [e.feature_name for e in top_features]
            parts.append(f"Most affected features: {', '.join(top_names)}.")

        return " ".join(parts)

    def _assess_confidence(
        self,
        report: DriftReport,
        explanations: List[FeatureDriftExplanation],
    ) -> str:
        """Assess confidence in the drift detection."""
        assessments = []

        # Check method agreement
        for e in explanations:
            if e.drift_detected:
                agreement = len(e.methods_detected) / (
                    len(e.methods_detected) + len(e.methods_clear)
                )
                if agreement < 0.5:
                    assessments.append("low")
                elif agreement < 0.8:
                    assessments.append("medium")
                else:
                    assessments.append("high")

        if not assessments:
            return "High confidence - no drift detected by any method"

        avg_confidence = {
            "low": assessments.count("low"),
            "medium": assessments.count("medium"),
            "high": assessments.count("high"),
        }

        dominant = max(avg_confidence, key=avg_confidence.get)

        if dominant == "high":
            return "High confidence - multiple methods agree on drift detection"
        elif dominant == "medium":
            return "Medium confidence - some disagreement between detection methods"
        else:
            return "Low confidence - significant disagreement between methods, results should be verified"

    def _generate_recommendations(
        self,
        explanations: List[FeatureDriftExplanation],
        pattern: str,
        root_causes: List[str],
    ) -> List[str]:
        """Generate investigation and action recommendations."""
        recommendations = []

        # Pattern-based recommendations
        if "sudden" in pattern:
            recommendations.append(
                "Investigate recent changes to data pipelines or upstream systems"
            )
        elif "gradual" in pattern:
            recommendations.append(
                "Consider scheduled retraining to adapt to evolving patterns"
            )

        # Feature-based recommendations
        affected = [e for e in explanations if e.drift_detected]
        if affected:
            top = affected[0]
            recommendations.extend(top.recommended_investigation[:2])

        # General recommendations
        recommendations.extend([
            "Review feature distributions using visualization tools",
            "Check for correlation between drifted features",
            "Monitor prediction quality metrics alongside drift metrics",
        ])

        return recommendations[:5]  # Limit to 5 recommendations
