"""Kullback-Leibler Divergence for drift detection."""

from typing import Any, List, Optional

import numpy as np
from scipy import stats

from modelguard.core.types import DriftResult
from modelguard.drift.base import BaseDriftDetector


class KLDivergence(BaseDriftDetector):
    """
    Kullback-Leibler Divergence for drift detection.

    KL divergence measures how one probability distribution diverges
    from a second, expected probability distribution. It is asymmetric:
    KL(P||Q) != KL(Q||P).

    We compute KL(current || reference), which measures how much
    information is lost when reference is used to approximate current.

    Formula:
        KL(P||Q) = sum(P(x) * log(P(x) / Q(x)))

    Note: KL divergence is undefined when Q(x) = 0 and P(x) > 0.
    We handle this by adding small epsilon to probabilities.

    Best for: Numerical features (continuous distributions)
    Detects: Distribution divergence (especially tail behavior)

    References:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
    """

    def __init__(
        self,
        threshold: float = 0.1,
        n_bins: int = 50,
        symmetric: bool = True,
    ):
        """
        Initialize KL divergence detector.

        Args:
            threshold: KL divergence threshold for drift detection
            n_bins: Number of bins for histogram estimation
            symmetric: If True, compute symmetric KL (average of both directions)
        """
        self.threshold = threshold
        self.n_bins = n_bins
        self.symmetric = symmetric

    @property
    def name(self) -> str:
        return "kl_divergence"

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
        Calculate KL divergence between reference and current distributions.

        Args:
            reference: Reference/baseline data
            current: Current production data
            feature_name: Name of the feature
            **kwargs: Additional parameters

        Returns:
            DriftResult with KL divergence value
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

        # Create common bins
        all_data = np.concatenate([reference, current])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)

        # Calculate histograms (probability distributions)
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)

        # Normalize to ensure they sum to 1
        ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
        cur_hist = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist

        # Add small epsilon to avoid log(0) and division by zero
        epsilon = 1e-10
        ref_hist = np.clip(ref_hist, epsilon, None)
        cur_hist = np.clip(cur_hist, epsilon, None)

        # Calculate KL divergence
        kl_cur_ref = stats.entropy(cur_hist, ref_hist)  # KL(current || reference)

        if self.symmetric:
            kl_ref_cur = stats.entropy(ref_hist, cur_hist)  # KL(reference || current)
            kl_divergence = (kl_cur_ref + kl_ref_cur) / 2
        else:
            kl_divergence = kl_cur_ref

        drift_detected = kl_divergence >= self.threshold

        return DriftResult(
            method_name=self.name,
            feature_name=feature_name,
            drift_detected=drift_detected,
            statistic=float(kl_divergence),
            p_value=None,  # KL divergence doesn't produce p-values
            threshold=self.threshold,
            metadata={
                "symmetric": self.symmetric,
                "kl_cur_ref": float(kl_cur_ref),
                "kl_ref_cur": float(kl_ref_cur) if self.symmetric else None,
                "n_bins": self.n_bins,
                "reference_size": len(reference),
                "current_size": len(current),
            },
        )
