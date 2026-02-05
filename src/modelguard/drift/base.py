"""Base class for drift detectors."""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from modelguard.core.types import DriftResult


class BaseDriftDetector(ABC):
    """Abstract base class for drift detection methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this detector."""
        pass

    @property
    @abstractmethod
    def supported_dtypes(self) -> List[str]:
        """Data types this detector supports: 'numerical', 'categorical'."""
        pass

    @abstractmethod
    def detect(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str = "unknown",
        **kwargs: Any,
    ) -> DriftResult:
        """
        Detect drift between reference and current distributions.

        Args:
            reference: Baseline/reference data distribution
            current: Current production data distribution
            feature_name: Name of the feature being tested
            **kwargs: Method-specific parameters

        Returns:
            DriftResult with detection outcome
        """
        pass

    def _validate_inputs(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> None:
        """Validate input arrays."""
        if len(reference) == 0:
            raise ValueError("Reference data cannot be empty")
        if len(current) == 0:
            raise ValueError("Current data cannot be empty")
