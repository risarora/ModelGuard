"""
Fraud Detection Model - Complete Drift Monitoring Demo

Uses the Kaggle Credit Card Fraud Detection Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset: 284,807 transactions, 492 frauds (0.172%)
Features: V1-V28 (PCA), Time, Amount
Source: ULB Machine Learning Group

This demo shows the full ModelGuard workflow:
1. Load real credit card transaction data
2. Train a fraud detection model
3. Create baseline from training data
4. Simulate production drift over time
5. Detect drift with multiple methods
6. Score severity and get recommendations
7. Create alert for human review

Run this script:
    python examples/fraud_detection_demo.py

To use the real dataset:
    1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    2. Place creditcard.csv in the examples/ directory
    3. Run the demo
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

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


# Dataset paths to check
DATASET_PATHS = [
    "examples/creditcard.csv",
    "creditcard.csv",
    "data/creditcard.csv",
    Path.home() / "Downloads" / "creditcard.csv",
]


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


def load_kaggle_creditcard_data(sample_size: int = 10000) -> tuple:
    """
    Load the Kaggle Credit Card Fraud Detection dataset.

    Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    - 284,807 transactions made by European cardholders in September 2013
    - 492 frauds out of 284,807 transactions (0.172%)
    - Features V1-V28 are PCA transformed (confidential original features)
    - Time: seconds elapsed between each transaction and first transaction
    - Amount: transaction amount
    - Class: 1 for fraud, 0 otherwise

    Returns:
        Tuple of (X, y, dataset_info) or None if dataset not found
    """
    for path in DATASET_PATHS:
        path = Path(path)
        if path.exists():
            print(f"  [+] Found Kaggle dataset: {path}")

            # Load the dataset
            df = pd.read_csv(path)

            # Sample if needed (full dataset is 284k rows)
            if len(df) > sample_size:
                # Stratified sample to maintain fraud ratio
                fraud = df[df['Class'] == 1]
                legit = df[df['Class'] == 0]

                # Keep all frauds, sample from legitimate
                n_fraud = len(fraud)
                n_legit = sample_size - n_fraud

                if n_legit > 0 and n_legit < len(legit):
                    legit_sample = legit.sample(n=n_legit, random_state=42)
                    df = pd.concat([fraud, legit_sample]).sample(frac=1, random_state=42)

                print(f"  [+] Sampled {len(df):,} transactions from {284807:,}")

            # Prepare features and target
            feature_cols = [c for c in df.columns if c not in ['Class']]
            X = df[feature_cols]
            y = df['Class']

            dataset_info = {
                "name": "Kaggle Credit Card Fraud Detection",
                "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
                "total_transactions": 284807,
                "total_frauds": 492,
                "fraud_rate": 0.00172,
                "features": feature_cols,
                "pca_features": [f"V{i}" for i in range(1, 29)],
                "original_features": ["Time", "Amount"],
            }

            return X, y, dataset_info

    return None


def generate_synthetic_fraud_data(
    n_samples: int = 2000,
    fraud_rate: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """
    Generate synthetic fraud detection data (fallback when Kaggle data unavailable).
    """
    np.random.seed(random_state)

    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Legitimate transactions - mimic PCA-like features
    legit_features = np.random.randn(n_legit, 28)  # V1-V28
    legit_time = np.random.uniform(0, 172800, n_legit)  # 2 days in seconds
    legit_amount = np.random.lognormal(4, 1, n_legit)

    # Fraudulent transactions - different patterns
    fraud_features = np.random.randn(n_fraud, 28) * 1.5 + 0.5  # Shifted
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    fraud_amount = np.random.lognormal(5, 1.5, n_fraud)  # Higher amounts

    # Combine
    features = np.vstack([legit_features, fraud_features])
    time_col = np.concatenate([legit_time, fraud_time])
    amount_col = np.concatenate([legit_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Create DataFrame with same structure as Kaggle data
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    data = np.column_stack([features, time_col, amount_col])
    X = pd.DataFrame(data, columns=feature_names)
    y = pd.Series(labels, name='Class')

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)

    dataset_info = {
        "name": "Synthetic Credit Card Data",
        "source": "Generated (Kaggle format)",
        "note": "Download real data from kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "features": feature_names,
    }

    return X, y, dataset_info


def inject_drift(df: pd.DataFrame, drift_type: str = "gradual", magnitude: float = 0.5) -> pd.DataFrame:
    """
    Inject realistic drift into credit card transaction data.

    Simulates real-world scenarios:
    - gradual: Economic changes affecting transaction patterns
    - sudden: Fraud ring attack or system change
    - seasonal: Holiday shopping patterns
    """
    df = df.copy()

    if drift_type == "gradual":
        # Simulate economic shift - higher transaction amounts, different patterns
        df['Amount'] = df['Amount'] * (1 + magnitude)
        df['V1'] = df['V1'] + magnitude * 0.5  # Shift PCA components
        df['V2'] = df['V2'] - magnitude * 0.3
        df['V14'] = df['V14'] * (1 + magnitude * 0.4)  # V14 often correlates with fraud
        df['Time'] = df['Time'] * 1.2  # Time pattern shift

    elif drift_type == "sudden":
        # Simulate fraud ring attack - massive distribution shifts across many features
        df['Amount'] = df['Amount'] * (1 + magnitude * 3)  # 7x higher amounts
        df['Time'] = df['Time'] * 2.5  # Different time patterns
        df['V1'] = df['V1'] + magnitude * 3
        df['V2'] = df['V2'] - magnitude * 2.5
        df['V3'] = df['V3'] - magnitude * 2.5
        df['V4'] = df['V4'] + magnitude * 2
        df['V5'] = df['V5'] * (1 + magnitude * 1.5)
        df['V6'] = df['V6'] + magnitude * 2
        df['V7'] = df['V7'] - magnitude * 1.8
        df['V8'] = df['V8'] + magnitude * 1.5
        df['V9'] = df['V9'] - magnitude * 1.2
        df['V10'] = df['V10'] * (1 + magnitude)
        df['V11'] = df['V11'] + magnitude * 2
        df['V12'] = df['V12'] - magnitude * 1.5
        df['V14'] = df['V14'] * (1 + magnitude * 1.5)
        df['V16'] = df['V16'] + magnitude * 1.2
        df['V17'] = df['V17'] * 0.2  # Dramatic shift
        df['V19'] = df['V19'] - magnitude

    elif drift_type == "seasonal":
        # Simulate holiday shopping - higher amounts, more transactions
        df['Amount'] = df['Amount'] * (1 + magnitude * 1.5)
        df['V1'] = df['V1'] + np.random.normal(0, magnitude, len(df))
        df['V2'] = df['V2'] + np.random.normal(0, magnitude, len(df))
        df['Time'] = df['Time'] * 0.8  # Faster transaction velocity

    return df


def main():
    print_header("CREDIT CARD FRAUD DETECTION - DRIFT MONITORING DEMO")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # =========================================================================
    # PHASE 1: Load Data and Train Model
    # =========================================================================

    print_section("Phase 1: Data Loading & Model Training")

    # Initialize ModelGuard
    config = load_config()
    init_database(config)
    db = get_database()
    print("  [+] ModelGuard initialized")

    # Try to load Kaggle dataset, fall back to synthetic
    result = load_kaggle_creditcard_data(sample_size=10000)

    if result is not None:
        X_full, y_full, dataset_info = result
        using_real_data = True
    else:
        print("  [!] Kaggle dataset not found, using synthetic data")
        print("  [!] For real data, download from:")
        print("      https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        X_full, y_full, dataset_info = generate_synthetic_fraud_data(n_samples=5000)
        using_real_data = False

    print(f"\n  Dataset: {dataset_info['name']}")
    if 'source' in dataset_info:
        print(f"  Source: {dataset_info['source']}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42, stratify=y_full
    )
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\n  Training samples: {len(X_train_split):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples (for production sim): {len(X_test):,}")
    print(f"  Fraud rate: {y_full.mean():.2%} (balanced resample; original Kaggle is 0.17%)")
    print(f"  Features: {len(X_full.columns)} (V1-V28 PCA + Time + Amount)")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_split, y_train_split)

    train_preds = model.predict(X_train_split)
    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)

    train_acc = accuracy_score(y_train_split, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)

    print(f"\n  Model Performance:")
    print(f"    Validation F1 (fraud class): {val_f1:.2f}")
    print(f"    Validation accuracy: {val_acc:.2%} (less meaningful for imbalanced data)")

    # =========================================================================
    # PHASE 2: Register Model and Create Baseline
    # =========================================================================

    print_section("Phase 2: Register Model & Create Baseline")

    # Register model
    with db.session() as session:
        model_repo = ModelRepository(session)
        baseline_repo = BaselineRepository(session)
        alert_repo = AlertRepository(session)

        # Clean up any existing demo models
        existing = model_repo.get_by_name_version("creditcard_fraud_detector", "1.0.0")
        if existing:
            alerts = alert_repo.list_for_model(existing.id)
            for alert in alerts:
                alert_repo.delete(alert.id)
            baselines = baseline_repo.list_for_model(existing.id)
            for bl in baselines:
                baseline_repo.delete(bl.id)
            model_repo.delete(existing.id)

        model_record = model_repo.create(
            name="creditcard_fraud_detector",
            version="1.0.0",
            framework="sklearn",
            model_type="classification",
            feature_names=list(X_train.columns),
            metadata={"dataset": dataset_info["name"], "source": dataset_info.get("source", "synthetic")},
        )
        model_id = model_record.id

    print(f"  [+] Model registered: creditcard_fraud_detector v1.0.0")
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
    print(f"  [+] Baseline samples: {baseline.sample_size:,}")

    # =========================================================================
    # PHASE 3: Simulate Production Drift
    # =========================================================================

    print_section("Phase 3: Simulate Production Drift")

    # Use test set as "production" data and inject drift
    drift_type = "sudden"
    drift_magnitude = 2.0  # High magnitude to trigger HIGH severity

    X_prod = inject_drift(X_test, drift_type=drift_type, magnitude=drift_magnitude)
    y_prod = y_test

    print(f"  [+] Production samples: {len(X_prod):,}")
    print(f"  [+] Drift injected: {drift_type} (magnitude={drift_magnitude})")

    print(f"\n  Simulated drift scenario: FRAUD RING ATTACK")
    print(f"    - Transaction amounts surged {drift_magnitude*300:.0f}%")
    print(f"    - 18+ PCA features shifted (coordinated attack pattern)")
    print(f"    - Time patterns disrupted (off-hours transactions)")
    print(f"    - This simulates a real-world fraud ring exploiting the model")

    # Check model performance on drifted data
    prod_preds = model.predict(X_prod)
    prod_acc = accuracy_score(y_prod, prod_preds)
    prod_f1 = f1_score(y_prod, prod_preds)

    print(f"\n  Production performance (F1 = fraud detection quality):")
    print(f"    Baseline F1: {val_f1:.2f}")
    print(f"    Production F1: {prod_f1:.2f} ({(prod_f1 - val_f1)*100:+.1f}%)")
    print(f"    (Accuracy: {prod_acc:.2%} - not ideal for imbalanced data)")

    # =========================================================================
    # PHASE 4: Drift Detection
    # =========================================================================

    print_section("Phase 4: Drift Detection")

    detector = DriftDetector(config)
    drift_report = detector.detect(baseline, X_prod)

    print(f"  Report ID: {drift_report.id[:8]}...")
    print(f"  Data drift detected: {drift_report.data_drift_detected}")
    print(f"  Features with drift: {len(drift_report.features_with_drift)} / {len(drift_report.feature_results)}")
    print(f"  Drift percentage: {drift_report.drift_percentage:.1f}%")

    print(f"\n  Per-feature results (top 10):")
    print(f"  {'Feature':<25} {'Method':<12} {'Statistic':<10} {'Drift?':<8}")
    print(f"  {'-'*55}")

    # Show top features by drift
    feature_stats = []
    for feature_name, results in drift_report.feature_results.items():
        if results:
            max_stat = max(r.statistic for r in results)
            has_drift = any(r.drift_detected for r in results)
            feature_stats.append((feature_name, results[0].method_name, max_stat, has_drift))

    feature_stats.sort(key=lambda x: x[2], reverse=True)
    for feat, method, stat, drift in feature_stats[:10]:
        drift_str = "YES" if drift else "no"
        print(f"  {feat:<25} {method:<12} {stat:<10.3f} {drift_str:<8}")

    # =========================================================================
    # PHASE 5: Severity Scoring
    # =========================================================================

    print_section("Phase 5: Severity Scoring")

    scorer = SeverityScorer(config)
    severity = scorer.score_report(drift_report)

    print(f"  Overall score: {severity.overall_score:.2f}")
    print(f"  Severity level: {severity.level.value.upper()}")
    print(f"  Confidence: {severity.confidence:.2f}")
    print(f"  Impacts predictions: {severity.impacts_predictions}")

    print(f"\n  Explanation:")
    print(f"    {severity.explanation}")

    # Show feature severity scores
    print(f"\n  Top feature severity scores:")
    sorted_features = sorted(
        severity.feature_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for feat, score in sorted_features:
        print(f"    {feat}: {score:.2f}")

    # =========================================================================
    # PHASE 6: Action Recommendation
    # =========================================================================

    print_section("Phase 6: Action Recommendation")

    recommender = ActionRecommender(config)
    recommendation = recommender.recommend(severity)

    print(f"  Recommended action: {recommendation.action.value.upper()}")
    print(f"  Urgency: {recommendation.urgency.value}")
    print(f"  Confidence: {recommendation.confidence:.2f}")
    print(f"  Estimated impact: {recommendation.estimated_impact}")

    print(f"\n  Reasoning:")
    for reason in recommendation.reasoning[:3]:
        print(f"    - {reason}")

    if recommendation.prerequisite_actions:
        print(f"\n  Prerequisite actions:")
        for action in recommendation.prerequisite_actions:
            print(f"    - {action}")

    # =========================================================================
    # PHASE 7: Drift Explanation
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
    alert_created = False

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

        alert_created = True
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
        print("  No alert needed - drift within acceptable thresholds")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print_header("DEMO SUMMARY")

    dataset_label = "Kaggle Credit Card Fraud" if using_real_data else "Synthetic (Kaggle format)"

    f1_drop = val_f1 - prod_f1
    f1_drop_pct = (f1_drop / val_f1 * 100) if val_f1 > 0 else 0

    print(f"""
  Dataset: {dataset_label}
  Note: Balanced resample (10% fraud rate) for demo. Original Kaggle is 0.17%.
  Model: creditcard_fraud_detector v1.0.0
  ---------------------------------------------------------

  BASELINE (Training Data)
    Samples: {baseline.sample_size:,}
    F1 (fraud class): {val_f1:.2f}

  PRODUCTION (With Simulated Drift)
    Samples: {len(X_prod):,}
    F1 (fraud class): {prod_f1:.2f} ({-f1_drop_pct:.0f}%)
    Features drifted: {drift_report.drift_percentage:.0f}%

  ASSESSMENT
    Severity: {severity.level.value.upper()} ({severity.overall_score:.2f})
    Recommended: {recommendation.action.value.upper()}
    Urgency: {recommendation.urgency.value}

  OUTCOME
    Alert created: {"Yes" if alert_created else "No"}
    Status: {"Pending human review" if alert_created else "Within acceptable thresholds"}

  ---------------------------------------------------------

  Key findings:
  - Drift detected in {len(drift_report.features_with_drift)} features
  - Most affected: {', '.join(drift_report.features_with_drift[:3])}
  - F1 score dropped {f1_drop_pct:.0f}% (fraud detection degraded)

  Recommended next steps:
  1. Review the alert and drifting features
  2. Investigate data pipeline for changes
  3. Consider retraining with recent data
  4. Monitor post-retrain performance
""")

    if not using_real_data:
        print(f"""
  ---------------------------------------------------------
  NOTE: Running with synthetic data. For real-world demo:

  1. Download the Kaggle Credit Card Fraud dataset:
     https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

  2. Place creditcard.csv in examples/ directory

  3. Re-run the demo
  ---------------------------------------------------------
""")


if __name__ == "__main__":
    main()
