"""
ModelGuard Quickstart Example

This script demonstrates the full ModelGuard workflow:
1. Register a model
2. Create a baseline from training data
3. Simulate production data with drift
4. Detect drift and get recommendations
5. Review alerts

Run this script after installing ModelGuard:
    pip install -e .
    python examples/quickstart.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize ModelGuard
from modelguard.core.config import load_config
from modelguard.storage.database import init_database, get_database
from modelguard.storage.repositories.model_repo import ModelRepository
from modelguard.storage.repositories.baseline_repo import BaselineRepository
from modelguard.baseline.creator import BaselineCreator
from modelguard.drift.detector import DriftDetector
from modelguard.severity.scorer import SeverityScorer
from modelguard.actions.recommender import ActionRecommender
from modelguard.explainability.explainer import DriftExplainer
from modelguard.human_loop.alert_manager import AlertManager
from modelguard.core.types import Baseline, FeatureStatistics, PredictionStatistics
from modelguard.storage.repositories.alert_repo import AlertRepository


def generate_training_data(n_samples=1000, random_state=42):
    """Generate synthetic training data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=random_state,
    )

    # Create DataFrame with named features
    feature_names = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)

    return df, y


def generate_drifted_data(n_samples=500, drift_magnitude=0.5, random_state=123):
    """Generate production data with drift."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=random_state,
    )

    feature_names = [f"feature_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)

    # Introduce drift in some features
    # Shift mean of first 3 features
    df["feature_0"] = df["feature_0"] + drift_magnitude * 2
    df["feature_1"] = df["feature_1"] - drift_magnitude * 1.5
    df["feature_2"] = df["feature_2"] * (1 + drift_magnitude)

    # Increase variance of feature_3
    df["feature_3"] = df["feature_3"] * (1 + drift_magnitude * 2)

    return df, y


def main():
    print("=" * 60)
    print("ModelGuard Quickstart Example")
    print("=" * 60)

    # Step 1: Initialize database
    print("\n[1] Initializing ModelGuard...")
    config = load_config()
    init_database(config)
    db = get_database()
    print("    Database initialized.")

    # Step 2: Generate training data and train a model
    print("\n[2] Generating training data...")
    X_train, y_train = generate_training_data(n_samples=1000)
    print(f"    Generated {len(X_train)} training samples with {len(X_train.columns)} features.")

    print("\n[3] Training a RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    probabilities = model.predict_proba(X_train)
    accuracy = (predictions == y_train).mean()
    print(f"    Training accuracy: {accuracy:.2%}")

    # Step 3: Register model in ModelGuard
    print("\n[4] Registering model in ModelGuard...")
    with db.session() as session:
        model_repo = ModelRepository(session)
        model_record = model_repo.create(
            name="fraud_detection_model",
            version="1.0.0",
            framework="sklearn",
            model_type="classification",
            feature_names=list(X_train.columns),
        )
        model_id = model_record.id
    print(f"    Model registered with ID: {model_id[:8]}...")

    # Step 4: Create baseline
    print("\n[5] Creating baseline from training data...")
    baseline_creator = BaselineCreator(config)
    baseline = baseline_creator.create(
        model_id=model_id,
        training_data=X_train,
        predictions=predictions,
        labels=y_train,
        probabilities=probabilities,
        prediction_type="classification",
    )

    # Save baseline
    with db.session() as session:
        baseline_repo = BaselineRepository(session)
        baseline_record = baseline_repo.create(
            model_id=model_id,
            feature_statistics={k: v.to_dict() for k, v in baseline.feature_statistics.items()},
            prediction_statistics=baseline.prediction_statistics.to_dict(),
            performance_metrics=baseline.performance_metrics.to_dict() if baseline.performance_metrics else None,
            sample_size=baseline.sample_size,
        )
        baseline_id = baseline_record.id
    print(f"    Baseline created with ID: {baseline_id[:8]}...")
    print(f"    Features tracked: {len(baseline.feature_statistics)}")
    print(f"    Sample size: {baseline.sample_size}")

    # Step 5: Simulate production data with drift
    print("\n[6] Generating production data with simulated drift...")
    X_prod, y_prod = generate_drifted_data(n_samples=500, drift_magnitude=0.8)
    print(f"    Generated {len(X_prod)} production samples.")
    print("    Drift injected in features: feature_0, feature_1, feature_2, feature_3")

    # Step 6: Run drift detection
    print("\n[7] Running drift detection...")

    # Reconstruct baseline object from stored data
    with db.session() as session:
        baseline_repo = BaselineRepository(session)
        baseline_record = baseline_repo.get(baseline_id)
        # Access data within session to avoid DetachedInstanceError
        stored_feature_stats = dict(baseline_record.feature_statistics)
        stored_sample_size = baseline_record.sample_size
        stored_model_id = baseline_record.model_id
        stored_created_at = baseline_record.created_at
        stored_id = baseline_record.id

    feature_stats = {}
    for name, stats in stored_feature_stats.items():
        feature_stats[name] = FeatureStatistics(
            name=name,
            dtype=stats.get("dtype", "numerical"),
            count=stats.get("count", 0),
            null_count=stats.get("null_count", 0),
            null_ratio=stats.get("null_ratio", 0),
            mean=stats.get("mean"),
            std=stats.get("std"),
            min_val=stats.get("min"),
            max_val=stats.get("max"),
            percentiles=stats.get("percentiles"),
            histogram_bins=stats.get("histogram_bins"),
            histogram_counts=stats.get("histogram_counts"),
        )

    pred_stats = PredictionStatistics(prediction_type="classification")

    baseline_obj = Baseline(
        id=stored_id,
        model_id=stored_model_id,
        created_at=stored_created_at,
        feature_statistics=feature_stats,
        prediction_statistics=pred_stats,
        sample_size=stored_sample_size or 0,
    )

    detector = DriftDetector(config)
    drift_report = detector.detect(baseline_obj, X_prod)

    print(f"    Report ID: {drift_report.id[:8]}...")
    print(f"    Data drift detected: {drift_report.data_drift_detected}")
    print(f"    Features with drift: {len(drift_report.features_with_drift)}")
    print(f"    Drift percentage: {drift_report.drift_percentage:.1f}%")

    if drift_report.features_with_drift:
        print(f"    Drifted features: {', '.join(drift_report.features_with_drift)}")

    # Step 7: Calculate severity
    print("\n[8] Calculating severity...")
    scorer = SeverityScorer(config)
    severity = scorer.score_report(drift_report)
    drift_report.severity = severity

    print(f"    Severity level: {severity.level.value.upper()}")
    print(f"    Severity score: {severity.overall_score:.2f}")
    print(f"    Confidence: {severity.confidence:.2f}")
    print(f"    Explanation: {severity.explanation}")

    # Step 8: Get action recommendation
    print("\n[9] Getting action recommendation...")
    recommender = ActionRecommender(config)
    recommendation = recommender.recommend(severity)
    drift_report.recommendation = recommendation

    print(f"    Recommended action: {recommendation.action.value.upper()}")
    print(f"    Urgency: {recommendation.urgency.value}")
    print(f"    Confidence: {recommendation.confidence:.2f}")
    print("    Reasoning:")
    for reason in recommendation.reasoning:
        print(f"      - {reason}")

    # Step 9: Generate explanation
    print("\n[10] Generating drift explanation...")
    explainer = DriftExplainer()
    explanation = explainer.explain(drift_report, severity)

    print(f"    Summary: {explanation.summary}")
    print(f"    Drift pattern: {explanation.drift_pattern}")
    print("    Top root causes:")
    for cause in explanation.potential_root_causes[:3]:
        print(f"      - {cause}")

    # Step 10: Create alert if needed
    print("\n[11] Alert management...")
    alert_manager = AlertManager(config)

    if alert_manager.should_create_alert(severity, recommendation):
        alert = alert_manager.create_alert(
            model_id=model_id,
            drift_report=drift_report,
            severity=severity,
            recommendation=recommendation,
        )

        # Persist alert to database
        with db.session() as session:
            alert_repo = AlertRepository(session)
            alert_record = alert_repo.create(
                model_id=alert.model_id,
                alert_type=alert.alert_type,
                severity=alert.severity.value,
                urgency=alert.urgency.value,
                drift_report_id=alert.drift_report_id,
                drift_summary={
                    "drift_percentage": drift_report.drift_percentage,
                    "features_with_drift": drift_report.features_with_drift,
                },
                recommendation={
                    "action": recommendation.action.value,
                    "confidence": recommendation.confidence,
                    "reasoning": recommendation.reasoning,
                },
            )
            alert_id = alert_record.id

        print(f"    Alert created: {alert_id[:8]}...")
        print(f"    Severity: {alert.severity.value}")
        print(f"    Urgency: {alert.urgency.value}")
        print(f"    Status: {alert.status.value}")
        print("\n    This alert would be routed to the ML team for review.")
    else:
        print("    No alert needed - drift level is acceptable.")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Model: fraud_detection_model v1.0.0")
    print(f"  Baseline samples: {baseline.sample_size}")
    print(f"  Production samples: {len(X_prod)}")
    print(f"  Drift detected: {drift_report.data_drift_detected}")
    print(f"  Severity: {severity.level.value.upper()} ({severity.overall_score:.2f})")
    print(f"  Recommended action: {recommendation.action.value.upper()}")
    print()
    print("ModelGuard provides continuous monitoring to prevent silent model failure.")
    print("=" * 60)


if __name__ == "__main__":
    main()
