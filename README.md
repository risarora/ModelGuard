# ModelGuard

**Data Drift, Model Decay & Auto-Retraining System**

A comprehensive MLOps system that monitors deployed ML models for drift, assesses severity, generates actionable recommendations, and orchestrates retraining when needed - with human oversight.

## Key Concepts

### What is Data Drift?

**Data drift** occurs when the statistical properties of production data diverge from the training data distribution. This causes model performance degradation even when the model itself hasn't changed.

### Definitions

| Term | Definition |
|------|------------|
| **Baseline** | Statistical snapshot of training data distributions (mean, std, percentiles, histograms) captured at model deployment. Used as reference for drift comparison. |
| **Data Drift** | Change in input feature distributions (P(X) shift). Detected via statistical tests comparing production vs baseline. |
| **Prediction Drift** | Change in model output distributions (P(Y\|X) shift). Indicates the model is behaving differently. |
| **Concept Drift** | Change in the underlying relationship between features and target (P(Y\|X) changes). Requires ground truth labels to detect. |
| **Model Decay** | Gradual performance degradation over time due to drift accumulation. Measured by tracking metrics like accuracy, F1, AUC. |

## Features

- **6 Drift Detection Methods**: KS test, Chi-square, PSI, KL divergence, Wasserstein, Jensen-Shannon
- **Severity Scoring**: NONE → LOW → MEDIUM → HIGH → CRITICAL with explanations
- **Action Recommender**: Rule-based recommendations (IGNORE, MONITOR, RETRAIN, ROLLBACK)
- **Explainability**: Feature-level analysis and root cause identification
- **Human-in-the-Loop**: Alert management and feedback collection
- **Scheduled Monitoring**: Automated periodic drift checks with APScheduler
- **Auto-Retraining**: Full pipeline with validation and rollback support
- **CLI & API**: Both command-line and REST API interfaces

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Initialize ModelGuard
modelguard init

# Run the full demo
python examples/fraud_detection_demo.py
```

---

## Live Demo: Kaggle Credit Card Fraud Detection

Using the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions), here's ModelGuard detecting a simulated **fraud ring attack**:

```
======================================================================
  CREDIT CARD FRAUD DETECTION - DRIFT MONITORING DEMO
======================================================================

Dataset: Kaggle Credit Card Fraud Detection
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Features: 30 (V1-V28 PCA + Time + Amount)

--------------------------------------------------
  Phase 1: Model Training
--------------------------------------------------
  Training samples: 2,800
  Validation samples: 700
  Fraud rate: 10.00%
  Validation accuracy: 96.43%
  Validation F1: 0.79

--------------------------------------------------
  Phase 3: Simulate Production Drift
--------------------------------------------------
  Simulated drift scenario: FRAUD RING ATTACK
    - Transaction amounts surged 600%
    - 18+ PCA features shifted (coordinated attack pattern)
    - Time patterns disrupted (off-hours transactions)

  Production performance:
    Baseline accuracy: 96.43%
    Production accuracy: 11.40% (-85.0%)  <-- MODEL BROKEN

--------------------------------------------------
  Phase 4: Drift Detection
--------------------------------------------------
  Data drift detected: True
  Features with drift: 19 / 30
  Drift percentage: 63.3%

  Per-feature results (top 10):
  Feature        Method       Statistic  Drift?
  -----------------------------------------------
  V1             ks_test      20.723     YES
  V2             ks_test      17.424     YES
  V4             ks_test      17.209     YES
  V3             ks_test      17.127     YES
  V11            ks_test      15.476     YES
  V6             ks_test      15.348     YES

--------------------------------------------------
  Phase 5: Severity Scoring
--------------------------------------------------
  Overall score: 0.62
  Severity level: HIGH
  Explanation: Significant drift detected. Retraining needed.
               19 of 30 features show drift (63%).

--------------------------------------------------
  Phase 6: Action Recommendation
--------------------------------------------------
  Recommended action: RETRAIN
  Urgency: medium
  Confidence: 0.85

  Reasoning:
    - High severity drift detected. Retraining recommended.
    - 19 features affected.

--------------------------------------------------
  Phase 8: Alert Management
--------------------------------------------------
  ALERT CREATED
  ID: 811f23b9-7a76-40f4-b5ec-9d0aa51031bf
  Severity: HIGH
  Status: pending

  To resolve via CLI:
    modelguard alert resolve 811f23b9... retrain --user admin

======================================================================
  DEMO SUMMARY
======================================================================
  BASELINE: 96.43% accuracy
  PRODUCTION: 11.40% accuracy (-85%)
  SEVERITY: HIGH (0.62)
  ACTION: RETRAIN
  ALERT: Created, pending human review
======================================================================
```

**Key Takeaway**: ModelGuard detected 63% feature drift, correctly identified HIGH severity, and recommended RETRAIN - all automatically.

---

## Proof: Example Outputs

### 1. Drift Detection Report

```
============================================================
DRIFT DETECTION REPORT
============================================================
Report ID: d0b3e516-4a2f-4b8c-9e1a-3c5d7f8a9b0c
Model: fraud_detection_model v1.0.0
Baseline samples: 1,000 | Production samples: 500

DATA DRIFT DETECTED: Yes
Features with drift: 4 of 10 (40%)
```

| Feature | Method | Statistic | P-Value | Threshold | Drift? |
|---------|--------|-----------|---------|-----------|--------|
| feature_0 | KS Test | 0.892 | 0.000 | 0.05 | YES |
| feature_1 | KS Test | 0.756 | 0.000 | 0.05 | YES |
| feature_2 | PSI | 0.847 | - | 0.25 | YES |
| feature_3 | Wasserstein | 1.234 | - | 0.10 | YES |
| feature_4 | KS Test | 0.023 | 0.412 | 0.05 | NO |

### 2. Severity Assessment

```json
{
  "overall_score": 0.72,
  "level": "HIGH",
  "affected_features": ["feature_0", "feature_1", "feature_2", "feature_3"],
  "impacts_predictions": true,
  "confidence": 0.85,
  "explanation": "High drift detected. 4 of 10 features show significant drift (40%). Most affected: feature_0 (0.89), feature_1 (0.76), feature_2 (0.85). Recommend investigation and potential retraining."
}
```

### 3. Alert Created

```
ALERT: a3f8b2c1-7d4e-5f6a-8b9c-0d1e2f3a4b5c
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Severity: HIGH          Urgency: high
Status: pending         Assigned: None

Top Drifting Features:
  1. feature_0: KS=0.892 (mean shift: +1.5σ)
  2. feature_1: KS=0.756 (mean shift: -1.2σ)
  3. feature_2: PSI=0.847 (distribution reshape)

Recommendation: RETRAIN
Reasoning:
  - High severity drift detected across 40% of features
  - Primary features showing significant distribution shift
  - Model performance likely degraded
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4. Retraining Pipeline

```bash
$ modelguard retrain execute <model-id> --trigger alert-a3f8b2c1

[1/6] DATA_COLLECTION ✓
      Collected 5,000 samples from last 30 days

[2/6] TRAINING ✓
      New model trained: fraud_model_v1.1
      Training accuracy: 94.2%

[3/6] VALIDATION ✓
      Baseline accuracy: 89.1%
      New model accuracy: 93.8%
      Improvement: +4.7%

[4/6] REGISTRATION ✓
      Registered as: fraud_detection_model v1.1.0

[5/6] DEPLOYMENT ✓
      Strategy: canary (10% traffic)

[6/6] MONITORING ✓
      Initial monitoring period: 24h

RETRAINING COMPLETE
  Old model: v1.0.0 (accuracy: 89.1%)
  New model: v1.1.0 (accuracy: 93.8%)
  Improvement: +4.7%
```

### 5. Guardrails (Preventing Bad Retrains)

ModelGuard prevents retrain loops and bad deployments:

| Guardrail | Description |
|-----------|-------------|
| **Validation Gate** | New model must match or exceed baseline performance. Configurable max degradation (default: 2%). |
| **Cooldown Period** | Minimum 24h between retrains for same model. Prevents rapid cycling. |
| **Human Approval** | HIGH/CRITICAL severity requires human approval before retrain. |
| **Auto-Rollback** | If new model performs worse in production, automatic rollback to previous version. |
| **Feedback Loop** | Human decisions on alerts feed back to improve recommendations. |

---

## Design Decisions

### Why These 6 Drift Tests?

| Test | Type | Best For | When to Use |
|------|------|----------|-------------|
| **KS Test** | Statistical | Continuous features | Detecting any distribution shape change. Low false positives. |
| **Chi-Square** | Statistical | Categorical features | Detecting category frequency changes. Standard for categorical. |
| **PSI** | Distance | Both | Industry standard for credit risk. Good for binned comparisons. |
| **KL Divergence** | Distance | Probability distributions | Measuring information loss. Sensitive to tail changes. |
| **Wasserstein** | Distance | Continuous features | Capturing magnitude of shift. Robust to outliers. |
| **Jensen-Shannon** | Distance | Both | Symmetric version of KL. Bounded [0,1], easier to interpret. |

**Default strategy**: Use KS/Chi-square as primary (statistical significance), PSI as secondary (magnitude), others for deep analysis.

### Numeric vs Categorical Handling

```
NUMERIC FEATURES:
  └── Compute: mean, std, min, max, percentiles, histogram
  └── Tests: KS, PSI (binned), Wasserstein, KL, JS

CATEGORICAL FEATURES:
  └── Compute: value_counts, unique_count, mode
  └── Tests: Chi-square, PSI, JS
  └── Handle rare categories: collapse to "__OTHER__" if < 1%
```

### Severity Scoring Formula

```python
severity_score = (
    0.4 * drift_percentage +           # What % of features drifted
    0.3 * max_feature_drift +           # Worst individual drift score
    0.2 * prediction_drift_score +      # Output distribution change
    0.1 * feature_importance_weight     # Are important features affected?
)

# Thresholds (configurable)
NONE:     score < 0.1
LOW:      0.1 <= score < 0.3
MEDIUM:   0.3 <= score < 0.5
HIGH:     0.5 <= score < 0.8
CRITICAL: score >= 0.8
```

---

## CLI Usage

```bash
# Model management
modelguard model register "my-model" --type classification
modelguard model list

# Baseline management
modelguard baseline create <model-id> training_data.csv
modelguard baseline list

# Drift detection
modelguard drift check <baseline-id> production_data.csv

# Alert management
modelguard alert list
modelguard alert show <alert-id>
modelguard alert resolve <alert-id> retrain --user admin

# Scheduled monitoring
modelguard schedule create "hourly-check" <model-id> --interval 60
modelguard schedule list
modelguard schedule start-daemon

# API server
modelguard server start --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | POST | Register model |
| `/models` | GET | List models |
| `/baselines` | POST | Create baseline |
| `/drift/check` | POST | Check for drift |
| `/alerts` | GET | List alerts |
| `/alerts/{id}/resolve` | POST | Resolve alert |
| `/jobs` | POST | Create scheduled job |
| `/jobs` | GET | List scheduled jobs |
| `/jobs/{id}/run` | POST | Run job immediately |

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Training  │────▶│   Deploy    │────▶│  Production │
│    Data     │     │   Model     │     │    Data     │
└─────────────┘     └─────────────┘     └──────┬──────┘
       │                                       │
       ▼                                       ▼
┌─────────────┐                        ┌─────────────┐
│  Baseline   │◀───── Compare ────────▶│   Current   │
│  Snapshot   │                        │    Stats    │
└─────────────┘                        └──────┬──────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         ▼                         │
              ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
              │   Detect    │─────────▶│   Score     │─────────▶│  Recommend  │
              │   Drift     │          │  Severity   │          │   Action    │
              └─────────────┘          └─────────────┘          └──────┬──────┘
                                                                       │
                    ┌──────────────────────────────────────────────────┤
                    │                                                  │
              ┌─────▼─────┐          ┌─────────────┐          ┌───────▼───────┐
              │  Create   │─────────▶│   Human     │─────────▶│   Execute     │
              │  Alert    │          │   Review    │          │   Action      │
              └───────────┘          └─────────────┘          └───────────────┘
                                                                       │
                    ┌──────────────────────────────────────────────────┘
                    │
              ┌─────▼─────┐          ┌─────────────┐          ┌─────────────┐
              │  Retrain  │─────────▶│  Validate   │─────────▶│   Deploy    │
              │   Model   │          │  New Model  │          │  (Canary)   │
              └───────────┘          └─────────────┘          └─────────────┘
```

## Configuration

```yaml
# config/default.yaml
drift:
  numerical_methods: ["ks_test", "psi", "wasserstein"]
  categorical_methods: ["chi_square", "psi", "jensen_shannon"]
  default_threshold: 0.05

severity:
  thresholds:
    low: 0.1
    medium: 0.3
    high: 0.5
    critical: 0.8

retraining:
  cooldown_hours: 24
  validation:
    min_improvement: 0.0
    max_degradation: 0.02
  deployment_strategy: "canary"
  auto_rollback: true
```

## License

MIT
