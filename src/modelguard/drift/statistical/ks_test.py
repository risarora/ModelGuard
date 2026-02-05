"""Kolmogorov-Smirnov test for drift detection."""

from typing import Any, List

import numpy as np
from scipy import stats

from modelguard.core.types import DriftResult
from modelguard.drift.base import BaseDriftDetector


class KolmogorovSmirnovTest(BaseDriftDetector):
    """
    Two-sample Kolmogorov-Smirnov test for drift detection.

    The KS test compares two samples to determine if they come from the
    same distribution. It measures the maximum distance between the
    empirical cumulative distribution functions (ECDFs) of the two samples.

    Best for: Continuous numerical features
    Detects: Any difference in distribution shape (location, scale, shape)

    Interpretation:
    - statistic: Maximum distance between ECDFs (0 to 1)
    - p_value: Probability of seeing this difference if distributions are the same
    - drift_detected: True if p_value < threshold

    References:
        https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test
    """

    def __init__(self, threshold: float = 0.05):
        """
        Initialize KS test detector.

        Args:
            threshold: P-value threshold for drift detection.
                      Lower values require stronger evidence of drift.
                      Default 0.05 (5% significance level).
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "ks_test"

    @property
    def supported_dtypes(self) -> List[str]:
        return ["numerical"]

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        **kwargs: Any,
    ) -> DriftResult:
        """
        Perform KS test between reference and current data.

        Args:
            reference: Reference/baseline data
            current: Current production data
            feature_name: Name of the feature
            **kwargs: Additional parameters (ignored)

        Returns:
            DriftResult with:
            - statistic: KS statistic (max distance between CDFs)
            - p_value: Two-sided p-value
            - drift_detected: True if p_value < threshold
        """
        self._validate_inputs(reference, current)

        # Remove NaN values
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return DriftResult(
                method_name=self.name,
                feature_name=feature_name,
                drift_detected=False,
                statistic=0.0,
                p_value=1.0,
                threshold=self.threshold,
                metadata={
                    "error": "Insufficient data after removing NaN values",
                    "reference_size": len(reference),
                    "current_size": len(current),
                },
            )

        # Perform KS test
        statistic, p_value = stats.ks_2samp(reference, current)

        return DriftResult(
            method_name=self.name,
            feature_name=feature_name,
            drift_detected=p_value < self.threshold,
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=self.threshold,
            metadata={
                "reference_size": len(reference),
                "current_size": len(current),
                "reference_mean": float(np.mean(reference)),
                "current_mean": float(np.mean(current)),
                "reference_std": float(np.std(reference)),
                "current_std": float(np.std(current)),
            },
        )
