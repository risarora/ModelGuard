"""Jensen-Shannon Divergence for drift detection."""

from typing import Any, List

import numpy as np
from scipy.spatial.distance import jensenshannon

from modelguard.core.types import DriftResult
from modelguard.drift.base import BaseDriftDetector


class JensenShannonDivergence(BaseDriftDetector):
    """
    Jensen-Shannon Divergence for drift detection.

    JS divergence is a symmetric and smoothed version of KL divergence.
    It measures the similarity between two probability distributions.

    Formula:
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        where M = 0.5 * (P + Q)

    Properties:
    - Symmetric: JS(P||Q) = JS(Q||P)
    - Bounded: 0 <= JS <= 1 (when using base-2 logarithm)
    - Always finite (unlike KL divergence)
    - The square root of JS divergence is a proper metric

    Best for: Both numerical and categorical features
    Detects: Distribution divergence

    Interpretation:
    - JS = 0: Distributions are identical
    - JS = 1: Distributions have no overlap
    - JS < 0.1: Very similar distributions
    - JS > 0.2: Significant difference

    References:
        https://en.wikipedia.org/wiki/Jensen-Shannon_divergence
    """

    def __init__(
        self,
        threshold: float = 0.1,
        n_bins: int = 50,
    ):
        """
        Initialize JS divergence detector.

        Args:
            threshold: JS divergence threshold for drift detection
            n_bins: Number of bins for histogram estimation
        """
        self.threshold = threshold
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "jensen_shannon"

    @property
    def supported_dtypes(self) -> List[str]:
        return ["numerical", "categorical"]

    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        dtype: str = "numerical",
        **kwargs: Any,
    ) -> DriftResult:
        """
        Calculate JS divergence between reference and current distributions.

        Args:
            reference: Reference/baseline data
            current: Current production data
            feature_name: Name of the feature
            dtype: Data type ('numerical' or 'categorical')
            **kwargs: Additional parameters

        Returns:
            DriftResult with JS divergence value
        """
        self._validate_inputs(reference, current)

        if dtype == "categorical":
            js_divergence = self._calculate_categorical_js(reference, current)
        else:
            js_divergence = self._calculate_numerical_js(reference, current)

        if js_divergence is None:
            return DriftResult(
                method_name=self.name,
                feature_name=feature_name,
                drift_detected=False,
                statistic=0.0,
                p_value=None,
                threshold=self.threshold,
                metadata={"error": "Could not compute JS divergence"},
            )

        drift_detected = js_divergence >= self.threshold

        return DriftResult(
            method_name=self.name,
            feature_name=feature_name,
            drift_detected=drift_detected,
            statistic=float(js_divergence),
            p_value=None,
            threshold=self.threshold,
            metadata={
                "dtype": dtype,
                "n_bins": self.n_bins if dtype == "numerical" else None,
                "reference_size": len(reference),
                "current_size": len(current),
                # JS distance (square root of JS divergence) is a proper metric
                "js_distance": float(np.sqrt(js_divergence)),
            },
        )

    def _calculate_numerical_js(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Calculate JS divergence for numerical features."""
        # Remove NaN values
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return None

        # Create common bins
        all_data = np.concatenate([reference, current])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)

        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Normalize to probability distributions
        ref_prob = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
        cur_prob = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist

        # Add small epsilon to avoid issues with zero probabilities
        epsilon = 1e-10
        ref_prob = ref_prob + epsilon
        cur_prob = cur_prob + epsilon

        # Renormalize after adding epsilon
        ref_prob = ref_prob / ref_prob.sum()
        cur_prob = cur_prob / cur_prob.sum()

        # Calculate JS divergence using scipy
        # Note: scipy returns JS distance (sqrt of divergence), so we square it
        js_distance = jensenshannon(ref_prob, cur_prob)
        js_divergence = js_distance ** 2

        return js_divergence

    def _calculate_categorical_js(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Calculate JS divergence for categorical features."""
        # Convert to string
        reference = np.array([str(x) for x in reference])
        current = np.array([str(x) for x in current])

        # Get all unique categories
        all_categories = list(set(reference) | set(current))

        # Calculate probability distributions
        ref_prob = np.array([np.sum(reference == cat) for cat in all_categories])
        cur_prob = np.array([np.sum(current == cat) for cat in all_categories])

        # Normalize
        ref_prob = ref_prob / ref_prob.sum() if ref_prob.sum() > 0 else ref_prob
        cur_prob = cur_prob / cur_prob.sum() if cur_prob.sum() > 0 else cur_prob

        # Add small epsilon
        epsilon = 1e-10
        ref_prob = ref_prob + epsilon
        cur_prob = cur_prob + epsilon

        # Renormalize
        ref_prob = ref_prob / ref_prob.sum()
        cur_prob = cur_prob / cur_prob.sum()

        # Calculate JS divergence
        js_distance = jensenshannon(ref_prob, cur_prob)
        js_divergence = js_distance ** 2

        return js_divergence
