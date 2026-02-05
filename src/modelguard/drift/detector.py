"""Main drift detector orchestrator."""

from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from modelguard.core.config import Config, DriftMethodConfig, get_config
from modelguard.core.exceptions import DriftDetectionError
from modelguard.core.types import Baseline, DriftReport, DriftResult, FeatureStatistics
from modelguard.drift.base import BaseDriftDetector
from modelguard.drift.statistical.ks_test import KolmogorovSmirnovTest
from modelguard.drift.statistical.chi_square import ChiSquareTest
from modelguard.drift.distance.psi import PopulationStabilityIndex
from modelguard.drift.distance.kl_divergence import KLDivergence
from modelguard.drift.distance.wasserstein import WassersteinDistance
from modelguard.drift.distance.jensen_shannon import JensenShannonDivergence


# Registry of available detectors
DETECTOR_REGISTRY: Dict[str, Type[BaseDriftDetector]] = {
    "ks_test": KolmogorovSmirnovTest,
    "chi_square": ChiSquareTest,
    "psi": PopulationStabilityIndex,
    "kl_divergence": KLDivergence,
    "wasserstein": WassersteinDistance,
    "jensen_shannon": JensenShannonDivergence,
}


class DriftDetector:
    """
    Orchestrates multiple drift detection methods.

    This class coordinates different drift detection algorithms and
    combines their results using configurable consensus strategies.

    Usage:
        detector = DriftDetector()
        report = detector.detect(baseline, current_data)
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize drift detector.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self._numerical_detectors: List[BaseDriftDetector] = []
        self._categorical_detectors: List[BaseDriftDetector] = []
        self._initialize_detectors()

    def _initialize_detectors(self) -> None:
        """Initialize detectors from configuration."""
        # Initialize numerical detectors
        for method_config in self.config.drift.numerical_methods:
            if method_config.enabled:
                detector = self._create_detector(method_config)
                if detector and "numerical" in detector.supported_dtypes:
                    self._numerical_detectors.append(detector)

        # Initialize categorical detectors
        for method_config in self.config.drift.categorical_methods:
            if method_config.enabled:
                detector = self._create_detector(method_config)
                if detector and "categorical" in detector.supported_dtypes:
                    self._categorical_detectors.append(detector)

    def _create_detector(
        self,
        method_config: DriftMethodConfig,
    ) -> Optional[BaseDriftDetector]:
        """Create a detector instance from configuration."""
        detector_class = DETECTOR_REGISTRY.get(method_config.name)
        if detector_class is None:
            return None

        # Build kwargs from config
        kwargs = {"threshold": method_config.threshold}
        kwargs.update(method_config.params)

        return detector_class(**kwargs)

    def detect(
        self,
        baseline: Baseline,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Detect data drift between baseline and current data.

        Args:
            baseline: Reference baseline
            current_data: Current production data
            features: Specific features to check (all if None)

        Returns:
            DriftReport with comprehensive drift analysis
        """
        # Determine features to check
        if features is None:
            features = list(baseline.feature_statistics.keys())

        # Ensure features exist in current data
        available_features = [f for f in features if f in current_data.columns]

        if not available_features:
            raise DriftDetectionError(
                "No matching features found between baseline and current data",
                details={
                    "baseline_features": list(baseline.feature_statistics.keys()),
                    "current_features": list(current_data.columns),
                },
            )

        # Run detection for each feature
        feature_results: Dict[str, List[DriftResult]] = {}

        for feature in available_features:
            feature_stats = baseline.feature_statistics.get(feature)
            if feature_stats is None:
                continue

            current_values = current_data[feature].values
            results = self._detect_feature_drift(
                feature_stats=feature_stats,
                current_values=current_values,
            )
            feature_results[feature] = results

        # Create drift report
        report = DriftReport.create(
            baseline_id=baseline.id,
            model_id=baseline.model_id,
            feature_results=feature_results,
            current_sample_size=len(current_data),
            reference_sample_size=baseline.sample_size,
        )

        return report

    def _detect_feature_drift(
        self,
        feature_stats: FeatureStatistics,
        current_values: np.ndarray,
    ) -> List[DriftResult]:
        """Run drift detection for a single feature."""
        results = []
        dtype = feature_stats.dtype

        # Get reference data from statistics
        reference_data = self._reconstruct_reference_data(feature_stats)

        if reference_data is None or len(reference_data) == 0:
            return results

        # Select appropriate detectors
        if dtype == "numerical":
            detectors = self._numerical_detectors
        else:
            detectors = self._categorical_detectors

        # Run each detector
        for detector in detectors:
            try:
                result = detector.detect(
                    reference=reference_data,
                    current=current_values,
                    feature_name=feature_stats.name,
                    dtype=dtype,
                )
                results.append(result)
            except Exception as e:
                # Log error but continue with other detectors
                results.append(
                    DriftResult(
                        method_name=detector.name,
                        feature_name=feature_stats.name,
                        drift_detected=False,
                        statistic=0.0,
                        p_value=None,
                        threshold=0.0,
                        metadata={"error": str(e)},
                    )
                )

        return results

    def _reconstruct_reference_data(
        self,
        feature_stats: FeatureStatistics,
    ) -> Optional[np.ndarray]:
        """
        Reconstruct reference data from statistics.

        For numerical features, we can use histogram to generate samples.
        For categorical features, we use value counts.
        """
        if feature_stats.dtype == "numerical":
            return self._reconstruct_numerical(feature_stats)
        else:
            return self._reconstruct_categorical(feature_stats)

    def _reconstruct_numerical(
        self,
        stats: FeatureStatistics,
    ) -> Optional[np.ndarray]:
        """Reconstruct numerical data from histogram."""
        if stats.histogram_bins is None or stats.histogram_counts is None:
            # Fallback: generate from mean and std
            if stats.mean is not None and stats.std is not None:
                # Generate samples from normal distribution
                n_samples = stats.count - stats.null_count
                return np.random.normal(stats.mean, max(stats.std, 0.01), n_samples)
            return None

        # Generate samples from histogram
        bins = np.array(stats.histogram_bins)
        counts = np.array(stats.histogram_counts)

        if len(bins) < 2 or len(counts) == 0:
            return None

        # Calculate bin centers
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Generate samples
        samples = []
        for center, count in zip(bin_centers, counts):
            if count > 0:
                # Add some noise around bin center
                bin_width = bins[1] - bins[0]
                noise = np.random.uniform(-bin_width / 2, bin_width / 2, count)
                samples.extend(center + noise)

        return np.array(samples) if samples else None

    def _reconstruct_categorical(
        self,
        stats: FeatureStatistics,
    ) -> Optional[np.ndarray]:
        """Reconstruct categorical data from value counts."""
        if stats.value_counts is None:
            return None

        samples = []
        for value, count in stats.value_counts.items():
            samples.extend([value] * count)

        return np.array(samples) if samples else None

    def detect_prediction_drift(
        self,
        baseline: Baseline,
        current_predictions: np.ndarray,
        current_probabilities: Optional[np.ndarray] = None,
    ) -> DriftReport:
        """
        Detect drift in model predictions.

        Args:
            baseline: Reference baseline
            current_predictions: Current model predictions
            current_probabilities: Current prediction probabilities (optional)

        Returns:
            DriftReport focused on prediction drift
        """
        pred_stats = baseline.prediction_statistics
        feature_results: Dict[str, List[DriftResult]] = {}

        # Reconstruct reference predictions from statistics
        if pred_stats.prediction_type == "classification":
            reference_preds = self._reconstruct_classification_predictions(pred_stats)
        else:
            reference_preds = self._reconstruct_regression_predictions(pred_stats)

        if reference_preds is None:
            raise DriftDetectionError("Could not reconstruct reference predictions")

        # Detect drift on predictions
        results = []
        for detector in self._numerical_detectors:
            try:
                if pred_stats.prediction_type == "regression":
                    result = detector.detect(
                        reference=reference_preds,
                        current=current_predictions,
                        feature_name="predictions",
                        dtype="numerical",
                    )
                    results.append(result)
            except Exception:
                pass

        # For classification, also check class distribution
        if pred_stats.prediction_type == "classification":
            for detector in self._categorical_detectors:
                try:
                    result = detector.detect(
                        reference=reference_preds.astype(str),
                        current=current_predictions.astype(str),
                        feature_name="predictions",
                        dtype="categorical",
                    )
                    results.append(result)
                except Exception:
                    pass

        feature_results["predictions"] = results

        # Create report
        report = DriftReport.create(
            baseline_id=baseline.id,
            model_id=baseline.model_id,
            feature_results=feature_results,
            current_sample_size=len(current_predictions),
            reference_sample_size=baseline.sample_size,
        )
        report.prediction_drift_detected = any(
            r.drift_detected for r in results
        )

        return report

    def _reconstruct_classification_predictions(self, pred_stats) -> Optional[np.ndarray]:
        """Reconstruct classification predictions from statistics."""
        if pred_stats.class_distribution is None:
            return None

        samples = []
        total_samples = 1000  # Generate fixed number of samples

        for cls, proportion in pred_stats.class_distribution.items():
            count = int(proportion * total_samples)
            samples.extend([cls] * count)

        return np.array(samples) if samples else None

    def _reconstruct_regression_predictions(self, pred_stats) -> Optional[np.ndarray]:
        """Reconstruct regression predictions from statistics."""
        if pred_stats.mean is None or pred_stats.std is None:
            return None

        n_samples = 1000
        return np.random.normal(pred_stats.mean, max(pred_stats.std, 0.01), n_samples)

    def check_consensus(
        self,
        results: List[DriftResult],
    ) -> bool:
        """
        Check if drift is detected based on consensus strategy.

        Args:
            results: List of drift results from different methods

        Returns:
            True if drift is detected according to consensus strategy
        """
        if not results:
            return False

        drift_count = sum(1 for r in results if r.drift_detected)
        total_count = len(results)

        strategy = self.config.drift.consensus.strategy
        min_agreement = self.config.drift.consensus.min_agreement

        if strategy == "any":
            return drift_count >= min_agreement
        elif strategy == "majority":
            return drift_count > total_count / 2
        elif strategy == "all":
            return drift_count == total_count
        else:
            return drift_count >= min_agreement

    def get_available_methods(self) -> Dict[str, List[str]]:
        """Get list of available drift detection methods."""
        return {
            "numerical": [d.name for d in self._numerical_detectors],
            "categorical": [d.name for d in self._categorical_detectors],
        }
