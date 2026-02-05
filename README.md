# ModelGuard

Drift detection for ML models. Tells you when your model is seeing weird data.

## Why this exists

Your fraud model worked great in January. By March, it's missing 40% of frauds. Why? The data changed. Transaction amounts shifted. New merchant categories appeared. Your model is still predicting like it's January.

This tool watches for that. It compares production data against your training baseline and yells at you when things drift.

## What it actually does

1. **Stores a baseline** - Stats from your training data (means, distributions, histograms)
2. **Compares new data** - Runs statistical tests (KS, PSI, Chi-square) against baseline
3. **Scores severity** - How bad is it? LOW/MEDIUM/HIGH/CRITICAL
4. **Creates alerts** - So a human can decide what to do
5. **Tracks decisions** - What did you do last time this happened?

That's it. No magic. No "AI-powered insights." Just math.

## What it doesn't do

- **Won't retrain your model** - That's your job. We just tell you when.
- **Won't deploy anything** - No k8s, no cloud integrations. It's a library.
- **Won't detect concept drift** - That requires labels. We only see input distributions.

## Install

```bash
pip install -e .
modelguard init
```

## Quick demo

```bash
python examples/fraud_detection_demo.py
```

Uses synthetic data based on the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). We inject fake drift (simulate a fraud ring attack) and watch the system catch it.

Output:
```
Features drifted:  19 / 30 (63%)
Severity:          HIGH
F1 drop:           0.79 → 0.18
Recommendation:    RETRAIN
```

The 10% fraud rate in the demo is resampled for speed. Real Kaggle data is 0.17% fraud.

## Basic usage

```python
from modelguard.baseline.creator import BaselineCreator
from modelguard.drift.detector import DriftDetector
from modelguard.core.config import load_config

config = load_config()

# Save baseline from training data
creator = BaselineCreator(config)
baseline = creator.create(
    model_id="my-model",
    training_data=X_train,
    predictions=model.predict(X_train),
)

# Later, check production data
detector = DriftDetector(config)
report = detector.detect(baseline, X_production)

if report.data_drift_detected:
    print(f"Drift in {report.drift_percentage:.0f}% of features")
    print(f"Worst offenders: {report.features_with_drift[:3]}")
```

## CLI

```bash
# Register a model
modelguard model register "fraud-detector" --type classification

# Create baseline from CSV
modelguard baseline create <model-id> training_data.csv

# Check new data for drift
modelguard drift check <baseline-id> production_data.csv

# See what needs attention
modelguard alert list

# Mark an alert as handled
modelguard alert resolve <alert-id> retrain --user yourname
```

## The drift tests

| Test | Use for | What it catches |
|------|---------|-----------------|
| KS test | Continuous features | Any shape change |
| Chi-square | Categorical features | Frequency changes |
| PSI | Both | Industry standard, good for binned data |
| Wasserstein | Continuous | How far distributions moved |

We run KS by default. PSI if you want industry-standard reporting. The others are there if you need them.

## Severity scoring

```
score = 0.4 * (% features drifted)
      + 0.3 * (worst single feature)
      + 0.2 * (prediction distribution shift)
      + 0.1 * (important features affected)
```

Thresholds:
- **LOW** (0.1-0.3): Keep an eye on it
- **MEDIUM** (0.3-0.5): Investigate soon
- **HIGH** (0.5-0.8): Probably need to retrain
- **CRITICAL** (0.8+): Something broke

These numbers are tunable. They're defaults that worked okay for us.

## Scheduled monitoring

```bash
# Check every hour
modelguard schedule create "hourly-check" <model-id> --interval 60

# Start the background daemon
modelguard schedule start-daemon
```

Uses APScheduler. Jobs persist to SQLite so they survive restarts.

## Known limitations

1. **No streaming** - Batch only. You give it a DataFrame, it checks it.
2. **No label drift** - We can't detect concept drift without ground truth labels.
3. **SQLite only** - Works fine for single-machine. Not distributed.
4. **No model versioning integration** - Doesn't talk to MLflow/W&B/etc. Yet.

## Project structure

```
src/modelguard/
├── baseline/      # Captures training data stats
├── drift/         # Statistical tests
├── severity/      # Scoring logic
├── actions/       # Recommendation rules
├── human_loop/    # Alert management
├── monitoring/    # Scheduled jobs
├── storage/       # SQLite persistence
├── api/           # FastAPI endpoints
└── cli/           # Typer commands
```

## Config

```yaml
# config/default.yaml
drift:
  numerical_methods: ["ks_test"]
  categorical_methods: ["chi_square"]
  default_threshold: 0.05

severity:
  thresholds:
    low: 0.1
    medium: 0.3
    high: 0.5
    critical: 0.8
```

## FAQ

**Q: Why not use Evidently/WhyLabs/Arize?**

Those are good. Use them if you need dashboards, hosted monitoring, or enterprise features. This is for when you want something simple you can read and modify.

**Q: Can I use this in production?**

The drift detection math is solid (it's scipy under the hood). The alert system works. But this started as a learning project, so audit the code before trusting it with anything important.

**Q: Why Python only?**

Because ML is Python. If you need a REST API, `modelguard server start` gives you one.

## Contributing

Open an issue first. PRs welcome for:
- Bug fixes
- New drift detection methods
- Better documentation
- Real-world usage examples

## License

MIT
