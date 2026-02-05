"""Statistical computations for baseline creation."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from modelguard.core.types import (
    FeatureStatistics,
    PerformanceMetrics,
    PredictionStatistics,
)


def compute_feature_statistics(
    data: pd.DataFrame,
    percentiles: Optional[List[int]] = None,
    histogram_bins: int = 50,
    max_categories: int = 100,
    rare_threshold: float = 0.01,
) -> Dict[str, FeatureStatistics]:
    """
    Compute statistics for all features in a DataFrame.

    Args:
        data: Input DataFrame
        percentiles: Percentiles to compute for numerical features
        histogram_bins: Number of bins for histograms
        max_categories: Max categories to track for categorical features
        rare_threshold: Threshold for collapsing rare categories

    Returns:
        Dictionary mapping feature names to their statistics
    """
    if percentiles is None:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    stats = {}

    for col in data.columns:
        series = data[col]
        dtype = _infer_dtype(series)

        if dtype == "numerical":
            stats[col] = _compute_numerical_stats(
                series,
                percentiles=percentiles,
                histogram_bins=histogram_bins,
            )
        else:
            stats[col] = _compute_categorical_stats(
                series,
                max_categories=max_categories,
                rare_threshold=rare_threshold,
            )

    return stats


def _infer_dtype(series: pd.Series) -> str:
    """Infer whether a series is numerical or categorical."""
    if pd.api.types.is_numeric_dtype(series):
        # Check if it's actually categorical (few unique values)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.05 and series.nunique() < 20:
            return "categorical"
        return "numerical"
    return "categorical"


def _compute_numerical_stats(
    series: pd.Series,
    percentiles: List[int],
    histogram_bins: int,
) -> FeatureStatistics:
    """Compute statistics for a numerical feature."""
    clean_series = series.dropna()

    # Basic stats - convert to native Python types for JSON serialization
    count = int(len(series))
    null_count = int(series.isna().sum())
    null_ratio = float(null_count / count) if count > 0 else 0.0

    if len(clean_series) == 0:
        return FeatureStatistics(
            name=series.name,
            dtype="numerical",
            count=count,
            null_count=null_count,
            null_ratio=null_ratio,
        )

    # Numerical stats
    mean = float(clean_series.mean())
    std = float(clean_series.std())
    min_val = float(clean_series.min())
    max_val = float(clean_series.max())

    # Percentiles
    pct_values = np.percentile(clean_series, percentiles)
    percentile_dict = {p: float(v) for p, v in zip(percentiles, pct_values)}

    # Histogram
    hist_counts, hist_bins = np.histogram(clean_series, bins=histogram_bins)
    histogram_bins_list = [float(b) for b in hist_bins]
    histogram_counts_list = [int(c) for c in hist_counts]

    return FeatureStatistics(
        name=series.name,
        dtype="numerical",
        count=count,
        null_count=null_count,
        null_ratio=null_ratio,
        mean=mean,
        std=std,
        min_val=min_val,
        max_val=max_val,
        percentiles=percentile_dict,
        histogram_bins=histogram_bins_list,
        histogram_counts=histogram_counts_list,
    )


def _compute_categorical_stats(
    series: pd.Series,
    max_categories: int,
    rare_threshold: float,
) -> FeatureStatistics:
    """Compute statistics for a categorical feature."""
    clean_series = series.dropna().astype(str)

    # Basic stats - convert to native Python types for JSON serialization
    count = int(len(series))
    null_count = int(series.isna().sum())
    null_ratio = float(null_count / count) if count > 0 else 0.0

    if len(clean_series) == 0:
        return FeatureStatistics(
            name=series.name,
            dtype="categorical",
            count=count,
            null_count=null_count,
            null_ratio=null_ratio,
        )

    # Value counts
    value_counts = clean_series.value_counts()
    unique_count = len(value_counts)

    # Collapse rare categories if needed
    if len(value_counts) > max_categories:
        threshold = int(len(clean_series) * rare_threshold)
        common = value_counts[value_counts >= threshold]
        rare_count = value_counts[value_counts < threshold].sum()
        value_counts = common.to_dict()
        if rare_count > 0:
            value_counts["__OTHER__"] = rare_count
    else:
        value_counts = value_counts.to_dict()

    # Mode
    mode = clean_series.mode()[0] if len(clean_series.mode()) > 0 else None

    return FeatureStatistics(
        name=series.name,
        dtype="categorical",
        count=count,
        null_count=null_count,
        null_ratio=null_ratio,
        unique_count=unique_count,
        value_counts={str(k): int(v) for k, v in value_counts.items()},
        mode=str(mode) if mode is not None else None,
    )


def compute_prediction_statistics(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    prediction_type: str = "classification",
    histogram_bins: int = 50,
    percentiles: Optional[List[int]] = None,
) -> PredictionStatistics:
    """
    Compute statistics for model predictions.

    Args:
        predictions: Model predictions (class labels or values)
        probabilities: Prediction probabilities (for classification)
        prediction_type: 'classification' or 'regression'
        histogram_bins: Number of bins for regression histograms
        percentiles: Percentiles for regression statistics

    Returns:
        PredictionStatistics object
    """
    if percentiles is None:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    if prediction_type == "classification":
        return _compute_classification_prediction_stats(predictions, probabilities)
    else:
        return _compute_regression_prediction_stats(
            predictions, histogram_bins, percentiles
        )


def _compute_classification_prediction_stats(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
) -> PredictionStatistics:
    """Compute prediction statistics for classification."""
    # Class distribution
    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)
    class_distribution = {str(c): float(cnt / total) for c, cnt in zip(unique, counts)}

    # Probability statistics
    probability_mean = None
    probability_std = None

    if probabilities is not None:
        if len(probabilities.shape) == 1:
            # Binary classification, single probability
            probability_mean = {"1": float(np.mean(probabilities))}
            probability_std = {"1": float(np.std(probabilities))}
        else:
            # Multi-class
            probability_mean = {
                str(i): float(np.mean(probabilities[:, i]))
                for i in range(probabilities.shape[1])
            }
            probability_std = {
                str(i): float(np.std(probabilities[:, i]))
                for i in range(probabilities.shape[1])
            }

    return PredictionStatistics(
        prediction_type="classification",
        class_distribution=class_distribution,
        probability_mean=probability_mean,
        probability_std=probability_std,
    )


def _compute_regression_prediction_stats(
    predictions: np.ndarray,
    histogram_bins: int,
    percentiles: List[int],
) -> PredictionStatistics:
    """Compute prediction statistics for regression."""
    mean = float(np.mean(predictions))
    std = float(np.std(predictions))

    pct_values = np.percentile(predictions, percentiles)
    percentile_dict = {p: float(v) for p, v in zip(percentiles, pct_values)}

    hist_counts, hist_bins = np.histogram(predictions, bins=histogram_bins)
    histogram_bins_list = [float(b) for b in hist_bins]
    histogram_counts_list = [int(c) for c in hist_counts]

    return PredictionStatistics(
        prediction_type="regression",
        mean=mean,
        std=std,
        percentiles=percentile_dict,
        histogram_bins=histogram_bins_list,
        histogram_counts=histogram_counts_list,
    )


def compute_performance_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    metric_type: str = "classification",
) -> PerformanceMetrics:
    """
    Compute performance metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        probabilities: Prediction probabilities (for AUC)
        metric_type: 'classification' or 'regression'

    Returns:
        PerformanceMetrics object
    """
    if metric_type == "classification":
        return _compute_classification_metrics(predictions, labels, probabilities)
    else:
        return _compute_regression_metrics(predictions, labels)


def _compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray],
) -> PerformanceMetrics:
    """Compute classification metrics."""
    # Determine if binary or multiclass
    unique_labels = np.unique(labels)
    is_binary = len(unique_labels) == 2

    accuracy = float(accuracy_score(labels, predictions))

    # Per-class metrics
    if is_binary:
        precision = {
            str(unique_labels[1]): float(
                precision_score(labels, predictions, zero_division=0)
            )
        }
        recall = {
            str(unique_labels[1]): float(
                recall_score(labels, predictions, zero_division=0)
            )
        }
        f1 = {
            str(unique_labels[1]): float(
                f1_score(labels, predictions, zero_division=0)
            )
        }
    else:
        precision_values = precision_score(
            labels, predictions, average=None, zero_division=0
        )
        recall_values = recall_score(
            labels, predictions, average=None, zero_division=0
        )
        f1_values = f1_score(labels, predictions, average=None, zero_division=0)

        precision = {str(c): float(v) for c, v in zip(unique_labels, precision_values)}
        recall = {str(c): float(v) for c, v in zip(unique_labels, recall_values)}
        f1 = {str(c): float(v) for c, v in zip(unique_labels, f1_values)}

    # AUC-ROC
    auc_roc = None
    if probabilities is not None:
        try:
            if is_binary:
                if len(probabilities.shape) == 2:
                    probs = probabilities[:, 1]
                else:
                    probs = probabilities
                auc_roc = float(roc_auc_score(labels, probs))
            else:
                auc_roc = float(
                    roc_auc_score(labels, probabilities, multi_class="ovr")
                )
        except ValueError:
            pass  # AUC not computable

    # Confusion matrix - convert to native Python ints for JSON serialization
    cm_array = confusion_matrix(labels, predictions)
    cm = [[int(val) for val in row] for row in cm_array]

    return PerformanceMetrics(
        metric_type="classification",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        auc_roc=auc_roc,
        confusion_matrix=cm,
    )


def _compute_regression_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> PerformanceMetrics:
    """Compute regression metrics."""
    mse = float(mean_squared_error(labels, predictions))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(labels, predictions))
    r2 = float(r2_score(labels, predictions))

    return PerformanceMetrics(
        metric_type="regression",
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
    )
