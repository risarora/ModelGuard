"""Chi-square test for categorical drift detection."""

from typing import Any, List

import numpy as np
from scipy import stats

from modelguard.core.types import DriftResult
from modelguard.drift.base import BaseDriftDetector


class ChiSquareTest(BaseDriftDetector):
    """
    Chi-square test for categorical drift detection.

    The Chi-square test compares the observed frequency distribution
    in the current data against the expected distribution from the
    reference data.

    Best for: Categorical features with discrete values
    Detects: Changes in category proportions

    Interpretation:
    - statistic: Chi-square statistic (sum of squared deviations)
    - p_value: Probability of seeing this deviation if distributions are the same
    - drift_detected: True if p_value < threshold

    References:
        https://en.wikipedia.org/wiki/Chi-squared_test
    """

    def __init__(self, threshold: float = 0.05, min_frequency: int = 5):
        """
        Initialize Chi-square test detector.

        Args:
            threshold: P-value threshold for drift detection.
            min_frequency: Minimum expected frequency per category.
                          Categories with lower expected frequency are merged.
        """
        self.threshold = threshold
        self.min_frequency = min_frequency

    @property
    def name(self) -> str:
        return "chi_square"

    @property
    def supported_dtypes(self) -> List[str]:
        return ["categorical"]

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        **kwargs: Any,
    ) -> DriftResult:
        """
        Perform Chi-square test between reference and current data.

        Args:
            reference: Reference/baseline categorical data
            current: Current production categorical data
            feature_name: Name of the feature
            **kwargs: Additional parameters

        Returns:
            DriftResult with Chi-square test results
        """
        self._validate_inputs(reference, current)

        # Convert to string for consistent handling
        reference = np.array([str(x) for x in reference if pd.notna(x)])
        current = np.array([str(x) for x in current if pd.notna(x)])

        if len(reference) == 0 or len(current) == 0:
            return DriftResult(
                method_name=self.name,
                feature_name=feature_name,
                drift_detected=False,
                statistic=0.0,
                p_value=1.0,
                threshold=self.threshold,
                metadata={"error": "Insufficient data after removing NaN values"},
            )

        # Get all unique categories
        all_categories = list(set(reference) | set(current))

        # Count frequencies
        ref_counts = {cat: 0 for cat in all_categories}
        cur_counts = {cat: 0 for cat in all_categories}

        for val in reference:
            ref_counts[val] += 1
        for val in current:
            cur_counts[val] += 1

        # Convert to arrays
        ref_freq = np.array([ref_counts[cat] for cat in all_categories])
        cur_freq = np.array([cur_counts[cat] for cat in all_categories])

        # Normalize reference to expected frequencies for current sample size
        ref_total = len(reference)
        cur_total = len(current)
        expected = (ref_freq / ref_total) * cur_total

        # Handle categories with low expected frequency
        # Merge categories below minimum frequency
        if self.min_frequency > 0:
            cur_freq, expected = self._merge_low_frequency(
                cur_freq, expected, self.min_frequency
            )

        # Check if we have enough categories
        if len(cur_freq) < 2:
            return DriftResult(
                method_name=self.name,
                feature_name=feature_name,
                drift_detected=False,
                statistic=0.0,
                p_value=1.0,
                threshold=self.threshold,
                metadata={
                    "error": "Not enough categories for Chi-square test",
                    "num_categories": len(all_categories),
                },
            )

        # Perform Chi-square test
        # Add small epsilon to avoid division by zero
        expected = np.maximum(expected, 1e-10)
        statistic, p_value = stats.chisquare(cur_freq, expected)

        return DriftResult(
            method_name=self.name,
            feature_name=feature_name,
            drift_detected=p_value < self.threshold,
            statistic=float(statistic),
            p_value=float(p_value),
            threshold=self.threshold,
            metadata={
                "num_categories": len(all_categories),
                "degrees_of_freedom": len(cur_freq) - 1,
                "reference_size": ref_total,
                "current_size": cur_total,
            },
        )

    def _merge_low_frequency(
        self,
        observed: np.ndarray,
        expected: np.ndarray,
        min_freq: int,
    ) -> tuple:
        """Merge categories with low expected frequency."""
        # Identify categories to merge
        low_freq_mask = expected < min_freq

        if not np.any(low_freq_mask):
            return observed, expected

        # Merge low-frequency categories into "other"
        high_freq_observed = observed[~low_freq_mask]
        high_freq_expected = expected[~low_freq_mask]

        other_observed = observed[low_freq_mask].sum()
        other_expected = expected[low_freq_mask].sum()

        if other_expected >= min_freq:
            # Include merged "other" category
            return (
                np.append(high_freq_observed, other_observed),
                np.append(high_freq_expected, other_expected),
            )
        else:
            # Just use high-frequency categories
            return high_freq_observed, high_freq_expected


# Import pandas for notna check
import pandas as pd
