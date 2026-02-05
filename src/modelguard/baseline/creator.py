"""Baseline creator for ML models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from modelguard.core.config import Config, get_config
from modelguard.core.exceptions import BaselineError, ValidationError
from modelguard.core.types import Baseline, FeatureStatistics
from modelguard.baseline.statistics import (
    compute_feature_statistics,
    compute_performance_metrics,
    compute_prediction_statistics,
)


class BaselineCreator:
    """Creates baseline snapshots from training data and model predictions."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize baseline creator.

        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()

    def create(
        self,
        model_id: str,
        training_data: pd.DataFrame,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        prediction_type: str = "classification",
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Baseline:
        """
        Create a baseline from training data and model predictions.

        Args:
            model_id: ID of the model this baseline is for
            training_data: Feature DataFrame used for training
            predictions: Model predictions on training data
            labels: Actual labels (if available) for computing performance metrics
            probabilities: Prediction probabilities (for classification)
            prediction_type: 'classification' or 'regression'
            feature_names: Feature names (inferred from DataFrame if not provided)
            metadata: Additional metadata to store

        Returns:
            Baseline object with all computed statistics

        Raises:
            BaselineError: If baseline creation fails
            ValidationError: If input data is invalid
        """
        # Validate inputs
        self._validate_inputs(training_data, predictions, labels)

        # Get feature names
        if feature_names is None:
            feature_names = list(training_data.columns)

        # Compute feature statistics
        feature_statistics = compute_feature_statistics(
            data=training_data,
            percentiles=self.config.baseline.numerical.percentiles,
            histogram_bins=self.config.baseline.numerical.histogram_bins,
            max_categories=self.config.baseline.categorical.max_categories,
            rare_threshold=self.config.baseline.categorical.rare_threshold,
        )

        # Compute prediction statistics
        prediction_statistics = compute_prediction_statistics(
            predictions=predictions,
            probabilities=probabilities,
            prediction_type=prediction_type,
            histogram_bins=self.config.baseline.numerical.histogram_bins,
            percentiles=self.config.baseline.numerical.percentiles,
        )

        # Compute performance metrics if labels provided
        performance_metrics = None
        if labels is not None:
            performance_metrics = compute_performance_metrics(
                predictions=predictions,
                labels=labels,
                probabilities=probabilities,
                metric_type=prediction_type,
            )

        # Build metadata
        full_metadata = {
            "feature_names": feature_names,
            "prediction_type": prediction_type,
            "created_at": datetime.utcnow().isoformat(),
        }
        if metadata:
            full_metadata.update(metadata)

        # Create baseline
        baseline = Baseline.create(
            model_id=model_id,
            feature_statistics=feature_statistics,
            prediction_statistics=prediction_statistics,
            performance_metrics=performance_metrics,
            sample_size=len(training_data),
            metadata=full_metadata,
        )

        return baseline

    def create_from_reference_data(
        self,
        model_id: str,
        reference_data: pd.DataFrame,
        model_adapter: Any,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Baseline:
        """
        Create a baseline by running the model on reference data.

        Args:
            model_id: ID of the model
            reference_data: Reference dataset (features only)
            model_adapter: Model adapter with predict/predict_proba methods
            labels: Actual labels (if available)
            metadata: Additional metadata

        Returns:
            Baseline object
        """
        # Get predictions
        predictions = model_adapter.predict(reference_data)

        # Get probabilities if available
        probabilities = None
        if hasattr(model_adapter, "predict_proba"):
            try:
                probabilities = model_adapter.predict_proba(reference_data)
            except Exception:
                pass  # Not all models support predict_proba

        # Determine prediction type
        prediction_type = getattr(model_adapter, "model_type", "classification")

        return self.create(
            model_id=model_id,
            training_data=reference_data,
            predictions=predictions,
            labels=labels,
            probabilities=probabilities,
            prediction_type=prediction_type,
            metadata=metadata,
        )

    def _validate_inputs(
        self,
        training_data: pd.DataFrame,
        predictions: np.ndarray,
        labels: Optional[np.ndarray],
    ) -> None:
        """Validate input data for baseline creation."""
        if training_data.empty:
            raise ValidationError("Training data cannot be empty")

        if len(predictions) != len(training_data):
            raise ValidationError(
                f"Predictions length ({len(predictions)}) does not match "
                f"training data length ({len(training_data)})"
            )

        if labels is not None and len(labels) != len(training_data):
            raise ValidationError(
                f"Labels length ({len(labels)}) does not match "
                f"training data length ({len(training_data)})"
            )

    def validate_baseline(self, baseline: Baseline) -> bool:
        """
        Validate a baseline for completeness.

        Args:
            baseline: Baseline to validate

        Returns:
            True if valid

        Raises:
            ValidationError: If baseline is invalid
        """
        if not baseline.feature_statistics:
            raise ValidationError("Baseline has no feature statistics")

        if baseline.prediction_statistics is None:
            raise ValidationError("Baseline has no prediction statistics")

        if baseline.sample_size == 0:
            raise ValidationError("Baseline has zero sample size")

        return True

    def compare_baselines(
        self,
        baseline1: Baseline,
        baseline2: Baseline,
    ) -> Dict[str, Any]:
        """
        Compare two baselines to understand differences.

        Args:
            baseline1: First baseline (typically older)
            baseline2: Second baseline (typically newer)

        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "sample_size_change": baseline2.sample_size - baseline1.sample_size,
            "feature_changes": {},
            "prediction_changes": {},
            "performance_changes": {},
        }

        # Compare feature statistics
        all_features = set(baseline1.feature_statistics.keys()) | set(
            baseline2.feature_statistics.keys()
        )

        for feature in all_features:
            stats1 = baseline1.feature_statistics.get(feature)
            stats2 = baseline2.feature_statistics.get(feature)

            if stats1 is None:
                comparison["feature_changes"][feature] = {"status": "added"}
            elif stats2 is None:
                comparison["feature_changes"][feature] = {"status": "removed"}
            else:
                changes = self._compare_feature_stats(stats1, stats2)
                if changes:
                    comparison["feature_changes"][feature] = changes

        # Compare prediction statistics
        pred1 = baseline1.prediction_statistics
        pred2 = baseline2.prediction_statistics

        if pred1.prediction_type == pred2.prediction_type:
            if pred1.prediction_type == "classification":
                comparison["prediction_changes"] = self._compare_classification_predictions(
                    pred1, pred2
                )
            else:
                comparison["prediction_changes"] = self._compare_regression_predictions(
                    pred1, pred2
                )

        # Compare performance metrics
        if baseline1.performance_metrics and baseline2.performance_metrics:
            perf1 = baseline1.performance_metrics
            perf2 = baseline2.performance_metrics

            if perf1.metric_type == "classification":
                if perf1.accuracy is not None and perf2.accuracy is not None:
                    comparison["performance_changes"]["accuracy_change"] = (
                        perf2.accuracy - perf1.accuracy
                    )
                if perf1.auc_roc is not None and perf2.auc_roc is not None:
                    comparison["performance_changes"]["auc_change"] = (
                        perf2.auc_roc - perf1.auc_roc
                    )
            else:
                if perf1.rmse is not None and perf2.rmse is not None:
                    comparison["performance_changes"]["rmse_change"] = (
                        perf2.rmse - perf1.rmse
                    )
                if perf1.r2 is not None and perf2.r2 is not None:
                    comparison["performance_changes"]["r2_change"] = (
                        perf2.r2 - perf1.r2
                    )

        return comparison

    def _compare_feature_stats(
        self,
        stats1: FeatureStatistics,
        stats2: FeatureStatistics,
    ) -> Dict[str, Any]:
        """Compare statistics for a single feature."""
        changes = {}

        if stats1.dtype != stats2.dtype:
            changes["dtype_changed"] = {"from": stats1.dtype, "to": stats2.dtype}
            return changes

        if stats1.dtype == "numerical":
            if stats1.mean is not None and stats2.mean is not None:
                mean_change = stats2.mean - stats1.mean
                if abs(mean_change) > 0.01 * abs(stats1.mean or 1):
                    changes["mean_change"] = mean_change

            if stats1.std is not None and stats2.std is not None:
                std_change = stats2.std - stats1.std
                if abs(std_change) > 0.01 * abs(stats1.std or 1):
                    changes["std_change"] = std_change
        else:
            # Categorical
            if stats1.unique_count != stats2.unique_count:
                changes["unique_count_change"] = (
                    stats2.unique_count - stats1.unique_count
                )

        if abs(stats2.null_ratio - stats1.null_ratio) > 0.01:
            changes["null_ratio_change"] = stats2.null_ratio - stats1.null_ratio

        return changes

    def _compare_classification_predictions(
        self,
        pred1,
        pred2,
    ) -> Dict[str, Any]:
        """Compare classification prediction statistics."""
        changes = {}

        if pred1.class_distribution and pred2.class_distribution:
            for cls in set(pred1.class_distribution.keys()) | set(
                pred2.class_distribution.keys()
            ):
                dist1 = pred1.class_distribution.get(cls, 0)
                dist2 = pred2.class_distribution.get(cls, 0)
                if abs(dist2 - dist1) > 0.01:
                    changes[f"class_{cls}_distribution_change"] = dist2 - dist1

        return changes

    def _compare_regression_predictions(
        self,
        pred1,
        pred2,
    ) -> Dict[str, Any]:
        """Compare regression prediction statistics."""
        changes = {}

        if pred1.mean is not None and pred2.mean is not None:
            changes["mean_change"] = pred2.mean - pred1.mean

        if pred1.std is not None and pred2.std is not None:
            changes["std_change"] = pred2.std - pred1.std

        return changes
