"""Tests for drift detection module."""

import numpy as np
import pytest

from modelguard.drift.statistical.ks_test import KolmogorovSmirnovTest
from modelguard.drift.statistical.chi_square import ChiSquareTest
from modelguard.drift.distance.psi import PopulationStabilityIndex
from modelguard.drift.distance.kl_divergence import KLDivergence
from modelguard.drift.distance.wasserstein import WassersteinDistance
from modelguard.drift.distance.jensen_shannon import JensenShannonDivergence


class TestKolmogorovSmirnovTest:
    """Tests for KS test drift detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions show no drift."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        detector = KolmogorovSmirnovTest(threshold=0.05)
        result = detector.detect(reference, current, feature_name="test")

        # Same distribution should not detect drift
        assert result.p_value > 0.05
        assert result.drift_detected == False

    def test_drift_different_mean(self):
        """Test that different means are detected as drift."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)  # Shifted mean

        detector = KolmogorovSmirnovTest(threshold=0.05)
        result = detector.detect(reference, current, feature_name="test")

        # Different distribution should detect drift
        assert result.drift_detected == True
        assert result.p_value < 0.05

    def test_drift_different_variance(self):
        """Test that different variances are detected as drift."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 3, 1000)  # Larger variance

        detector = KolmogorovSmirnovTest(threshold=0.05)
        result = detector.detect(reference, current, feature_name="test")

        assert result.drift_detected == True


class TestPopulationStabilityIndex:
    """Tests for PSI drift detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have PSI near zero."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        detector = PopulationStabilityIndex(threshold=0.1)
        result = detector.detect(reference, current, feature_name="test")

        assert result.statistic < 0.1
        assert result.drift_detected == False

    def test_drift_shifted_distribution(self):
        """Test that shifted distributions have high PSI."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(3, 1, 1000)

        detector = PopulationStabilityIndex(threshold=0.1)
        result = detector.detect(reference, current, feature_name="test")

        assert result.statistic > 0.1
        assert result.drift_detected == True

    def test_categorical_psi(self):
        """Test PSI on categorical data."""
        np.random.seed(42)
        reference = np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])
        current = np.random.choice(["A", "B", "C"], 1000, p=[0.3, 0.4, 0.3])

        detector = PopulationStabilityIndex(threshold=0.1)
        result = detector.detect(reference, current, feature_name="test", dtype="categorical")

        # Distribution changed, should detect some drift
        assert result.statistic > 0


class TestWassersteinDistance:
    """Tests for Wasserstein distance drift detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have zero distance."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        detector = WassersteinDistance(threshold=0.1)
        result = detector.detect(reference, current, feature_name="test")

        # Small distance for same distribution
        assert result.statistic < 0.1

    def test_drift_shifted_distribution(self):
        """Test that shifted distributions have high distance."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(5, 1, 1000)

        detector = WassersteinDistance(threshold=0.1)
        result = detector.detect(reference, current, feature_name="test")

        assert result.drift_detected == True


class TestJensenShannonDivergence:
    """Tests for Jensen-Shannon divergence drift detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical distributions have JS near zero."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        detector = JensenShannonDivergence(threshold=0.1)
        result = detector.detect(reference, current, feature_name="test")

        assert result.statistic < 0.1

    def test_drift_different_distribution(self):
        """Test that different distributions have high JS."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(5, 2, 1000)

        detector = JensenShannonDivergence(threshold=0.1)
        result = detector.detect(reference, current, feature_name="test")

        assert result.statistic > 0.1
        assert result.drift_detected == True


class TestChiSquareTest:
    """Tests for Chi-square test drift detector."""

    def test_no_drift_same_distribution(self):
        """Test that identical categorical distributions show no drift."""
        np.random.seed(42)
        reference = np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])
        current = np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])

        detector = ChiSquareTest(threshold=0.05)
        result = detector.detect(reference, current, feature_name="test")

        assert result.p_value > 0.05

    def test_drift_different_distribution(self):
        """Test that different categorical distributions show drift."""
        np.random.seed(42)
        reference = np.random.choice(["A", "B", "C"], 1000, p=[0.8, 0.1, 0.1])
        current = np.random.choice(["A", "B", "C"], 1000, p=[0.2, 0.4, 0.4])

        detector = ChiSquareTest(threshold=0.05)
        result = detector.detect(reference, current, feature_name="test")

        assert result.drift_detected == True
        assert result.p_value < 0.05
