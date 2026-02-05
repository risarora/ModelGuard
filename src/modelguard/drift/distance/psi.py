"""Population Stability Index for drift detection."""

from typing import Any, List, Optional

import numpy as np

from modelguard.core.types import DriftResult
from modelguard.drift.base import BaseDriftDetector


class PopulationStabilityIndex(BaseDriftDetector):
    """
    Population Stability Index (PSI) for drift detection.

    PSI measures the shift in distribution between two datasets by
    comparing the proportion of observations in each bin/category.

    Interpretation:
    - PSI < 0.1: No significant change (NONE)
    - 0.1 <= PSI < 0.2: Slight change, monitoring recommended (LOW)
    - 0.2 <= PSI < 0.25: Moderate change (MEDIUM)
    - PSI >= 0.25: Significant change, action required (HIGH)

    Formula:
        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

    Best for: Both numerical and categorical features
    Detects: Distribution shifts

    References:
        https://www.listendata.com/2015/05/population-stability-index.html
    """

    def __init__(
        self,
        threshold: float = 0.1,
        n_bins: int = 10,
        threshold_high: float = 0.2,
    ):
        """
        Initialize PSI detector.

        Args:
            threshold: PSI threshold for drift detection (default 0.1)
            n_bins: Number of bins for numerical features
            threshold_high: Higher threshold for significant drift
        """
        self.threshold = threshold
        self.n_bins = n_bins
        self.threshold_high = threshold_high

    @property
    def name(self) -> str:
        return "psi"

    @property
    def supported_dtypes(self) -> List[str]:
        return ["numerical", "categorical"]

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        dtype: str = "numerical",
        bins: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> DriftResult:
        """
        Calculate PSI between reference and current distributions.

        Args:
            reference: Reference/baseline data
            current: Current production data
            feature_name: Name of the feature
            dtype: Data type ('numerical' or 'categorical')
            bins: Pre-computed bin edges (for numerical)
            **kwargs: Additional parameters

        Returns:
            DriftResult with PSI value
        """
        self._validate_inputs(reference, current)

        if dtype == "categorical":
            psi = self._calculate_categorical_psi(reference, current)
        else:
            psi = self._calculate_numerical_psi(reference, current, bins)

        drift_detected = psi >= self.threshold
        severity = self._classify_severity(psi)

        return DriftResult(
            method_name=self.name,
            feature_name=feature_name,
            drift_detected=drift_detected,
            statistic=float(psi),
            p_value=None,  # PSI doesn't produce p-values
            threshold=self.threshold,
            metadata={
                "severity": severity,
                "threshold_low": self.threshold,
                "threshold_high": self.threshold_high,
                "dtype": dtype,
                "reference_size": len(reference),
                "current_size": len(current),
            },
        )

    def _calculate_numerical_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate PSI for numerical features."""
        # Remove NaN values
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Create bins from reference data if not provided
        if bins is None:
            # Use percentile-based bins for better handling of outliers
            bins = np.percentile(
                reference,
                np.linspace(0, 100, self.n_bins + 1)
            )
            # Ensure unique bin edges
            bins = np.unique(bins)
            if len(bins) < 3:
                # Fallback to equal-width bins
                bins = np.linspace(
                    min(reference.min(), current.min()),
                    max(reference.max(), current.max()),
                    self.n_bins + 1,
                )

        # Calculate histogram counts
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        return self._calculate_psi_from_counts(ref_counts, cur_counts)

    def _calculate_categorical_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Calculate PSI for categorical features."""
        # Convert to string for consistent handling
        reference = np.array([str(x) for x in reference])
        current = np.array([str(x) for x in current])

        # Get all unique categories
        all_categories = list(set(reference) | set(current))

        # Count frequencies
        ref_counts = np.array([np.sum(reference == cat) for cat in all_categories])
        cur_counts = np.array([np.sum(current == cat) for cat in all_categories])

        return self._calculate_psi_from_counts(ref_counts, cur_counts)

    def _calculate_psi_from_counts(
        self,
        ref_counts: np.ndarray,
        cur_counts: np.ndarray,
    ) -> float:
        """Calculate PSI from count arrays."""
        # Convert to percentages
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        # Avoid division by zero and log(0)
        # Replace zeros with small value
        epsilon = 1e-10
        ref_pct = np.clip(ref_pct, epsilon, 1 - epsilon)
        cur_pct = np.clip(cur_pct, epsilon, 1 - epsilon)

        # PSI formula: sum((cur - ref) * ln(cur/ref))
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def _classify_severity(self, psi: float) -> str:
        """Classify PSI value into severity level."""
        if psi < self.threshold:
            return "NONE"
        elif psi < self.threshold_high:
            return "LOW"
        elif psi < 0.25:
            return "MEDIUM"
        else:
            return "HIGH"
