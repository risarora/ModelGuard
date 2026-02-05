# ModelGuard

**ML Drift Detection & Auto-Retraining System**

Monitors deployed ML models for data drift, scores severity, recommends actions, and orchestrates retraining with human oversight.

```bash
pip install -e . && modelguard init && python examples/fraud_detection_demo.py
```

---

## 10-Second Proof

Using the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), we simulate a fraud ring attack and watch ModelGuard catch it:

```
Drift Detection Results:
  Features drifted:  19 / 30 (63%)
  Severity:          HIGH (0.62)
  F1 Score drop:     0.79 → 0.18 (-77%)
  Recommendation:    RETRAIN
  Alert:             Created, pending human review
```

> **Note on demo data**: Original Kaggle dataset has 0.17% fraud rate. Demo uses balanced resampling (10% fraud) for faster iteration. Drift is synthetically injected to simulate a fraud ring attack. All drift statistics are real calculations.

---

## Features

- **6 Drift Detection Methods** - KS test, Chi-square, PSI, KL divergence, Wasserstein, Jensen-Shannon
- **Severity Scoring** - NONE → LOW → MEDIUM → HIGH → CRITICAL with explanations
- **Action Recommender** - Rule-based: IGNORE, MONITOR, RETRAIN, ROLLBACK
- **Human-in-the-Loop** - Alerts require approval before action
- **Scheduled Monitoring** - Cron/interval-based drift checks
- **CLI & REST API** - Full control via both interfaces

---

## What's Real vs Simulated

| Component | Status |
|-----------|--------|
| Drift detection (KS, PSI, etc.) | **Real** - actual statistical tests |
| Severity scoring | **Real** - configurable formula |
| Alert management | **Real** - persisted to SQLite |
| Scheduled jobs | **Real** - APScheduler with persistence |
| Retraining pipeline | **Simulated** - orchestration stubs, bring your own trainer |
| Canary deployment | **Simulated** - versioned registry, no real traffic splitting |
| Auto-rollback | **Simulated** - version pointer swap |

Deployment integration is designed to be extensible. Plug in your own training scripts and serving infrastructure.

---

## Quick Start

```bash
# Install
pip install -e .

# Initialize (creates config + SQLite DB)
modelguard init

# Run the fraud detection demo
python examples/fraud_detection_demo.py

# Or use CLI
modelguard model register "my-model" --type classification
modelguard baseline create <model-id> training_data.csv
modelguard drift check <baseline-id> production_data.csv
modelguard alert list
```

---

## Demo Output (Full)

```
======================================================================
  CREDIT CARD FRAUD DETECTION - DRIFT MONITORING DEMO
======================================================================
Dataset: Kaggle Credit Card Fraud (balanced resample, 10% fraud rate)
Features: 30 (V1-V28 PCA + Time + Amount)

Phase 1: Model Training
  Training samples: 2,800
  Validation F1 (fraud class): 0.79

Phase 3: Simulated Drift (Fraud Ring Attack)
  - Transaction amounts surged 600%
  - 18+ PCA features shifted
  - Time patterns disrupted

  Performance after drift:
    F1 (fraud class): 0.79 → 0.18 (-77%)

Phase 4: Drift Detection
  Features with drift: 19 / 30 (63.3%)

  Top drifted features:
  Feature   KS Statistic   Drift?
  V1        20.723         YES
  V2        17.424         YES
  V4        17.209         YES
  V3        17.127         YES

Phase 5: Severity
  Score: 0.62 (HIGH)
  Explanation: 19 of 30 features drifted. Retraining recommended.

Phase 6: Recommendation
  Action: RETRAIN
  Confidence: 0.85

Phase 8: Alert
  ID: 811f23b9-7a76-40f4-b5ec-9d0aa51031bf
  Severity: HIGH
  Status: pending

  CLI: modelguard alert resolve 811f23b9... retrain --user admin
======================================================================
```

---

## Key Concepts

| Term | Definition |
|------|------------|
| **Baseline** | Statistical snapshot of training data distributions at deployment time. |
| **Data Drift** | P(X) shift - input feature distributions changed. |
| **Prediction Drift** | P(Y\|X) shift - model outputs changed. |
| **Concept Drift** | True P(Y\|X) changed - requires labels to detect. |
| **Model Decay** | Performance degradation from accumulated drift. |

---

## Design Decisions

### Why These Drift Tests?

| Test | Best For | Why |
|------|----------|-----|
| **KS Test** | Continuous | Detects any shape change, low false positives |
| **Chi-Square** | Categorical | Standard for frequency changes |
| **PSI** | Both | Industry standard in credit risk |
| **Wasserstein** | Continuous | Captures shift magnitude, robust to outliers |
| **KL Divergence** | Distributions | Sensitive to tail changes |
| **Jensen-Shannon** | Both | Symmetric, bounded [0,1] |

### Severity Formula

```python
severity = (
    0.4 * drift_percentage +        # % of features drifted
    0.3 * max_feature_drift +       # worst individual drift
    0.2 * prediction_drift +        # output distribution change
    0.1 * importance_weight         # important features affected?
)

# Thresholds
NONE: < 0.1 | LOW: 0.1-0.3 | MEDIUM: 0.3-0.5 | HIGH: 0.5-0.8 | CRITICAL: >= 0.8
```

### Guardrails

| Guardrail | Purpose |
|-----------|---------|
| Validation Gate | New model must not degrade > 2% vs baseline |
| Cooldown Period | Min 24h between retrains |
| Human Approval | HIGH/CRITICAL requires human sign-off |
| Feedback Loop | Human decisions improve future recommendations |

---

## CLI Reference

```bash
# Models
modelguard model register "fraud-detector" --type classification
modelguard model list

# Baselines
modelguard baseline create <model-id> training_data.csv
modelguard baseline list

# Drift
modelguard drift check <baseline-id> production_data.csv

# Alerts
modelguard alert list
modelguard alert resolve <alert-id> retrain --user admin

# Scheduled Jobs
modelguard schedule create "hourly-check" <model-id> --interval 60
modelguard schedule start-daemon

# API Server
modelguard server start --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/models` | POST/GET | Register/list models |
| `/baselines` | POST | Create baseline |
| `/drift/check` | POST | Run drift detection |
| `/alerts` | GET | List alerts |
| `/alerts/{id}/resolve` | POST | Resolve alert |
| `/jobs` | POST/GET | Create/list scheduled jobs |

---

## Architecture

```
Training Data ──▶ Baseline Snapshot
                        │
Production Data ──▶ Compare ──▶ Drift Detection
                                      │
                              Severity Scoring
                                      │
                           Action Recommendation
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
              Create Alert ──▶ Human Review ──▶ Execute Action
                                                       │
                              ┌─────────────────────────┘
                              ▼
                    Retrain ──▶ Validate ──▶ Deploy
```

---

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
  max_degradation: 0.02
```

---

## License

MIT
