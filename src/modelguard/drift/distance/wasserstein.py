"""Wasserstein Distance (Earth Mover's Distance) for drift detection."""

from typing import Any, List

import numpy as np
from scipy import stats

from modelguard.core.types import DriftResult
from modelguard.drift.base import BaseDriftDetector


class WassersteinDistance(BaseDriftDetector):
    """
    Wasserstein Distance (Earth Mover's Distance) for drift detection.

    The Wasserstein distance measures the minimum "cost" of transforming
    one distribution into another, where cost is the amount of distribution
    weight times the distance it needs to be moved.

    Intuition: Imagine distributions as piles of dirt. Wasserstein distance
    is the minimum work needed to reshape one pile to match the other.

    Formula (1D):
        W(P, Q) = integral(|F_P(x) - F_Q(x)|) dx

    Where F_P and F_Q are the cumulative distribution functions.

    Best for: Numerical features, especially when you care about
              the "shape" difference between distributions
    Detects: Any distribution shift, robust to outliers

    Advantages over KL divergence:
    - Always finite (doesn't require support overlap)
    - Symmetric
    - Meaningful even when distributions have different supports

    References:
        https://en.wikipedia.org/wiki/Wasserstein_metric
    """

    def __init__(self, threshold: float = 0.1, normalize: bool = True):
        """
        Initialize Wasserstein distance detector.

        Args:
            threshold: Distance threshold for drift detection
            normalize: If True, normalize distance by data range
        """
        self.threshold = threshold
        self.normalize = normalize

    @property
    def name(self) -> str:
        return "wasserstein"

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
        Calculate Wasserstein distance between reference and current distributions.

        Args:
            reference: Reference/baseline data
            current: Current production data
            feature_name: Name of the feature
            **kwargs: Additional parameters

        Returns:
            DriftResult with Wasserstein distance
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
                p_value=None,
                threshold=self.threshold,
                metadata={"error": "Insufficient data after removing NaN values"},
            )

        # Calculate Wasserstein distance (Earth Mover's Distance)
        distance = stats.wasserstein_distance(reference, current)

        # Optionally normalize by data range
        if self.normalize:
            data_range = max(
                reference.max() - reference.min(),
                current.max() - current.min(),
            )
            if data_range > 0:
                normalized_distance = distance / data_range
            else:
                normalized_distance = 0.0
        else:
            normalized_distance = distance

        drift_detected = normalized_distance >= self.threshold

        return DriftResult(
            method_name=self.name,
            feature_name=feature_name,
            drift_detected=drift_detected,
            statistic=float(normalized_distance),
            p_value=None,
            threshold=self.threshold,
            metadata={
                "raw_distance": float(distance),
                "normalized": self.normalize,
                "reference_mean": float(np.mean(reference)),
                "current_mean": float(np.mean(current)),
                "mean_shift": float(np.mean(current) - np.mean(reference)),
                "reference_std": float(np.std(reference)),
                "current_std": float(np.std(current)),
                "reference_size": len(reference),
                "current_size": len(current),
            },
        )
