"""
Fraud Detection Model - Complete Drift Monitoring Demo

This demo shows the full ModelGuard workflow:
1. Train a fraud detection model
2. Create baseline from training data
3. Simulate production drift over time
4. Detect drift with multiple methods
5. Score severity and get recommendations
6. Create alert for human review
7. Demonstrate the decision workflow

Run this script:
    python examples/fraud_detection_demo.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ModelGuard imports
from modelguard.core.config import load_config
from modelguard.storage.database import init_database, get_database
from modelguard.storage.repositories.model_repo import ModelRepository
from modelguard.storage.repositories.baseline_repo import BaselineRepository
from modelguard.storage.repositories.alert_repo import AlertRepository
from modelguard.baseline.creator import BaselineCreator
from modelguard.drift.detector import DriftDetector
from modelguard.severity.scorer import SeverityScorer
from modelguard.actions.recommender import ActionRecommender
from modelguard.explainability.explainer import DriftExplainer
from modelguard.human_loop.alert_manager import AlertManager
from modelguard.core.types import Baseline, FeatureStatistics, PredictionStatistics


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    """Print a section header."""
    print(f"\n{'-' * 50}")
    print(f"  {text}")
    print("-" * 50)


def generate_fraud_data(
    n_samples: int = 1000,
    fraud_rate: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """
    Generate synthetic fraud detection data.

    Features:
    - transaction_amount: Transaction value
    - merchant_category: Type of merchant (encoded)
    - time_since_last: Hours since last transaction
    - distance_from_home: Miles from home location
    - velocity_24h: Number of transactions in last 24h
    - avg_amount_7d: Average transaction amount in last 7 days
    - is_online: Whether transaction was online
    - device_age_days: Age of device used
    - failed_attempts: Failed transaction attempts recently
    - account_age_months: How long account has been active
    """
    np.random.seed(random_state)

    # Generate base features
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit_data = {
        'transaction_amount': np.random.lognormal(4, 1, n_legit),
        'merchant_category': np.random.choice([0, 1, 2, 3, 4], n_legit, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'time_since_last': np.random.exponential(12, n_legit),
        'distance_from_home': np.random.exponential(5, n_legit),
        'velocity_24h': np.random.poisson(2, n_legit),
        'avg_amount_7d': np.random.lognormal(4, 0.5, n_legit),
        'is_online': np.random.choice([0, 1], n_legit, p=[0.6, 0.4]),
        'device_age_days': np.random.exponential(180, n_legit),
        'failed_attempts': np.random.poisson(0.1, n_legit),
        'account_age_months': np.random.exponential(24, n_legit),
    }

    # Fraudulent transactions (different patterns)
    fraud_data = {
        'transaction_amount': np.random.lognormal(5, 1.5, n_fraud),  # Higher amounts
        'merchant_category': np.random.choice([0, 1, 2, 3, 4], n_fraud, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'time_since_last': np.random.exponential(2, n_fraud),  # Rapid transactions
        'distance_from_home': np.random.exponential(50, n_fraud),  # Far from home
        'velocity_24h': np.random.poisson(8, n_fraud),  # Many transactions
        'avg_amount_7d': np.random.lognormal(4, 0.5, n_fraud),
        'is_online': np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),  # More online
        'device_age_days': np.random.exponential(30, n_fraud),  # New devices
        'failed_attempts': np.random.poisson(2, n_fraud),  # More failures
        'account_age_months': np.random.exponential(6, n_fraud),  # Newer accounts
    }

    # Combine
    df_legit = pd.DataFrame(legit_data)
    df_fraud = pd.DataFrame(fraud_data)

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    labels = np.array([0] * n_legit + [1] * n_fraud)

    # Shuffle
    idx = np.random.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)
    labels = labels[idx]

    return df, labels


def inject_drift(
    df: pd.DataFrame,
    drift_type: str = "gradual",
    magnitude: float = 0.5,
) -> pd.DataFrame:
    """
    Inject drift into production data.

    Drift types:
    - gradual: Slow mean shift over time
    - sudden: Abrupt distribution change
    - seasonal: Periodic pattern changes
    """
    df = df.copy()

    if drift_type == "gradual":
        # Shift means gradually
        df['transaction_amount'] = df['transaction_amount'] * (1 + magnitude)
        df['distance_from_home'] = df['distance_from_home'] * (1 + magnitude * 0.5)
        df['velocity_24h'] = df['velocity_24h'] + int(magnitude * 3)
        df['time_since_last'] = df['time_since_last'] * (1 - magnitude * 0.3)

    elif drift_type == "sudden":
        # Abrupt change in distributions
        df['transaction_amount'] = df['transaction_amount'] * (1 + magnitude * 2)
        df['is_online'] = np.where(df['is_online'] == 0, 1, df['is_online'])  # All online now
        df['device_age_days'] = df['device_age_days'] * 0.3  # Much newer devices

    elif drift_type == "seasonal":
        # Simulate holiday season (higher amounts, more velocity)
        df['transaction_amount'] = df['transaction_amount'] * (1 + magnitude * 1.5)
        df['velocity_24h'] = df['velocity_24h'] * 2
        df['merchant_category'] = np.random.choice([0, 1, 2, 3, 4], len(df),
                                                    p=[0.1, 0.15, 0.35, 0.25, 0.15])

    return df


def main():
    print_header("FRAUD DETECTION MODEL - DRIFT MONITORING DEMO")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # =========================================================================
    # PHASE 1: Setup and Training
    # =========================================================================

    print_section("Phase 1: Model Training")

    # Initialize ModelGuard
    config = load_config()
    init_database(config)
    db = get_database()
    print("  [+] ModelGuard initialized")

    # Generate training data
    X_train, y_train = generate_fraud_data(n_samples=2000, fraud_rate=0.1, random_state=42)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"  [+] Training data: {len(X_train_split)} samples")
    print(f"  [+] Validation data: {len(X_val)} samples")
    print(f"  [+] Fraud rate: {y_train.mean():.1%}")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_split, y_train_split)

    train_preds = model.predict(X_train_split)
    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)

    train_acc = accuracy_score(y_train_split, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)

    print(f"  [+] Training accuracy: {train_acc:.2%}")
    print(f"  [+] Validation accuracy: {val_acc:.2%}")
    print(f"  [+] Validation F1: {val_f1:.2f}")

    # =========================================================================
    # PHASE 2: Register Model and Create Baseline
    # =========================================================================

    print_section("Phase 2: Register Model & Create Baseline")

    # Register model
    with db.session() as session:
        model_repo = ModelRepository(session)
        baseline_repo = BaselineRepository(session)
        alert_repo = AlertRepository(session)

        # Clean up any existing demo models (must delete related records first)
        existing = model_repo.get_by_name_version("fraud_detector", "1.0.0")
        if existing:
            # Delete alerts first
            alerts = alert_repo.list_for_model(existing.id)
            for alert in alerts:
                alert_repo.delete(alert.id)
            # Delete baselines
            baselines = baseline_repo.list_for_model(existing.id)
            for bl in baselines:
                baseline_repo.delete(bl.id)
            # Now delete the model
            model_repo.delete(existing.id)

        model_record = model_repo.create(
            name="fraud_detector",
            version="1.0.0",
            framework="sklearn",
            model_type="classification",
            feature_names=list(X_train.columns),
        )
        model_id = model_record.id

    print(f"  [+] Model registered: fraud_detector v1.0.0")
    print(f"  [+] Model ID: {model_id[:8]}...")

    # Create baseline
    baseline_creator = BaselineCreator(config)
    all_train_preds = model.predict(X_train)
    all_train_proba = model.predict_proba(X_train)

    baseline = baseline_creator.create(
        model_id=model_id,
        training_data=X_train,
        predictions=all_train_preds,
        labels=y_train,
        probabilities=all_train_proba,
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

    print(f"  [+] Baseline created: {baseline_id[:8]}...")
    print(f"  [+] Features tracked: {len(baseline.feature_statistics)}")
    print(f"  [+] Sample size: {baseline.sample_size}")

    # =========================================================================
    # PHASE 3: Simulate Production with Drift
    # =========================================================================

    print_section("Phase 3: Simulate Production Drift")

    # Generate production data with drift
    X_prod_base, y_prod = generate_fraud_data(n_samples=500, fraud_rate=0.15, random_state=99)

    # Inject gradual drift (simulating real-world changes)
    X_prod = inject_drift(X_prod_base, drift_type="gradual", magnitude=0.8)

    print(f"  [+] Production samples: {len(X_prod)}")
    print(f"  [+] Drift injected: gradual (magnitude=0.8)")
    print("\n  Drift effects:")
    print(f"    - transaction_amount: mean +80%")
    print(f"    - distance_from_home: mean +40%")
    print(f"    - velocity_24h: +2-3 additional transactions")
    print(f"    - time_since_last: mean -24%")

    # Check production model performance
    prod_preds = model.predict(X_prod)
    prod_acc = accuracy_score(y_prod, prod_preds)
    prod_f1 = f1_score(y_prod, prod_preds)

    print(f"\n  Production performance:")
    print(f"    - Baseline accuracy: {val_acc:.2%}")
    print(f"    - Production accuracy: {prod_acc:.2%} ({(prod_acc - val_acc)*100:+.1f}%)")
    print(f"    - Production F1: {prod_f1:.2f}")

    # =========================================================================
    # PHASE 4: Drift Detection
    # =========================================================================

    print_section("Phase 4: Drift Detection")

    # Reconstruct baseline object
    with db.session() as session:
        baseline_repo = BaselineRepository(session)
        baseline_record = baseline_repo.get(baseline_id)
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

    baseline_obj = Baseline(
        id=stored_id,
        model_id=stored_model_id,
        created_at=stored_created_at,
        feature_statistics=feature_stats,
        prediction_statistics=PredictionStatistics(prediction_type="classification"),
        sample_size=stored_sample_size or 0,
    )

    # Run drift detection
    detector = DriftDetector(config)
    drift_report = detector.detect(baseline_obj, X_prod)

    print(f"  Report ID: {drift_report.id[:8]}...")
    print(f"  Data drift detected: {drift_report.data_drift_detected}")
    print(f"  Features with drift: {len(drift_report.features_with_drift)} / {len(X_prod.columns)}")
    print(f"  Drift percentage: {drift_report.drift_percentage:.1f}%")

    print("\n  Per-feature results:")
    print(f"  {'Feature':<25} {'Method':<12} {'Statistic':<10} {'Drift?':<8}")
    print(f"  {'-'*55}")

    for feature_name, results in drift_report.feature_results.items():
        for result in results[:1]:  # Show first method per feature
            drift_marker = "YES" if result.drift_detected else "no"
            print(f"  {feature_name:<25} {result.method_name:<12} {result.statistic:<10.3f} {drift_marker:<8}")

    # =========================================================================
    # PHASE 5: Severity Scoring
    # =========================================================================

    print_section("Phase 5: Severity Scoring")

    scorer = SeverityScorer(config)
    severity = scorer.score_report(drift_report)
    drift_report.severity = severity

    print(f"  Overall score: {severity.overall_score:.2f}")
    print(f"  Severity level: {severity.level.value.upper()}")
    print(f"  Confidence: {severity.confidence:.2f}")
    print(f"  Impacts predictions: {severity.impacts_predictions}")
    print(f"\n  Explanation:")
    print(f"    {severity.explanation}")

    print("\n  Feature severity scores:")
    sorted_features = sorted(severity.feature_scores.items(), key=lambda x: x[1], reverse=True)
    for feature, score in sorted_features[:5]:
        print(f"    {feature}: {score:.2f}")

    # =========================================================================
    # PHASE 6: Action Recommendation
    # =========================================================================

    print_section("Phase 6: Action Recommendation")

    recommender = ActionRecommender(config)
    recommendation = recommender.recommend(severity)
    drift_report.recommendation = recommendation

    print(f"  Recommended action: {recommendation.action.value.upper()}")
    print(f"  Urgency: {recommendation.urgency.value}")
    print(f"  Confidence: {recommendation.confidence:.2f}")
    print(f"  Estimated impact: {recommendation.estimated_impact}")

    print("\n  Reasoning:")
    for reason in recommendation.reasoning:
        print(f"    - {reason}")

    if recommendation.prerequisite_actions:
        print("\n  Prerequisite actions:")
        for action in recommendation.prerequisite_actions:
            print(f"    - {action}")

    # =========================================================================
    # PHASE 7: Generate Explanation
    # =========================================================================

    print_section("Phase 7: Drift Explanation")

    explainer = DriftExplainer()
    explanation = explainer.explain(drift_report, severity)

    print(f"  Summary:")
    print(f"    {explanation.summary}")

    print(f"\n  Drift pattern: {explanation.drift_pattern}")

    print("\n  Potential root causes:")
    for i, cause in enumerate(explanation.potential_root_causes[:3], 1):
        print(f"    {i}. {cause}")

    print("\n  Recommendations:")
    for i, rec in enumerate(explanation.recommendations[:3], 1):
        print(f"    {i}. {rec}")

    # =========================================================================
    # PHASE 8: Alert Creation
    # =========================================================================

    print_section("Phase 8: Alert Management")

    alert_manager = AlertManager(config)

    if alert_manager.should_create_alert(severity, recommendation):
        alert = alert_manager.create_alert(
            model_id=model_id,
            drift_report=drift_report,
            severity=severity,
            recommendation=recommendation,
        )

        # Persist alert
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
                    "severity_score": severity.overall_score,
                },
                recommendation={
                    "action": recommendation.action.value,
                    "confidence": recommendation.confidence,
                    "reasoning": recommendation.reasoning,
                },
            )
            alert_id = alert_record.id

        print(f"  ALERT CREATED")
        print(f"  -----------------------------------------")
        print(f"  ID: {alert_id}")
        print(f"  Severity: {alert.severity.value.upper()}")
        print(f"  Urgency: {alert.urgency.value}")
        print(f"  Status: {alert.status.value}")
        print(f"\n  This alert requires human review before action.")
        print(f"\n  To resolve via CLI:")
        print(f"    modelguard alert resolve {alert_id[:8]}... retrain --user admin")
    else:
        print("  No alert needed - drift level is acceptable.")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_header("DEMO SUMMARY")

    print(f"""
  Model: fraud_detector v1.0.0
  ---------------------------------------------------------

  BASELINE (Training)
    Samples: {baseline.sample_size}
    Accuracy: {val_acc:.2%}
    F1 Score: {val_f1:.2f}

  PRODUCTION (With Drift)
    Samples: {len(X_prod)}
    Accuracy: {prod_acc:.2%} ({(prod_acc - val_acc)*100:+.1f}%)
    Drift detected: {drift_report.drift_percentage:.1f}% of features

  ASSESSMENT
    Severity: {severity.level.value.upper()} ({severity.overall_score:.2f})
    Recommended: {recommendation.action.value.upper()}
    Urgency: {recommendation.urgency.value}

  OUTCOME
    Alert created: Yes
    Status: Pending human review

  ---------------------------------------------------------

  The model has detected significant drift in production data.
  Key features affected: {', '.join(drift_report.features_with_drift[:3])}

  Recommended next steps:
  1. Review the alert and drifting features
  2. Investigate data pipeline for changes
  3. Consider retraining with recent data
  4. Monitor post-retrain performance
    """)


if __name__ == "__main__":
    main()
