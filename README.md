# ModelGuard

**Data Drift, Model Decay & Auto-Retraining System**

A comprehensive MLOps system that monitors deployed ML models for drift, assesses severity, generates actionable recommendations, and orchestrates retraining when needed - with human oversight.

## Features

- **6 Drift Detection Methods**: KS test, Chi-square, PSI, KL divergence, Wasserstein, Jensen-Shannon
- **Severity Scoring**: NONE → LOW → MEDIUM → HIGH → CRITICAL with explanations
- **Action Recommender**: Rule-based recommendations (IGNORE, MONITOR, RETRAIN, ROLLBACK)
- **Explainability**: Feature-level analysis and root cause identification
- **Human-in-the-Loop**: Alert management and feedback collection
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

# Run the example
python examples/quickstart.py
```

## CLI Usage

```bash
# Register a model
modelguard model register "my-model" --type classification

# Create a baseline
modelguard baseline create <model-id> training_data.csv

# Check for drift
modelguard drift check <baseline-id> production_data.csv

# List alerts
modelguard alert list

# Start API server
modelguard server start
```

## API Endpoints

- `GET /health` - Health check
- `POST /models` - Register model
- `GET /models` - List models
- `POST /baselines` - Create baseline
- `POST /drift/check` - Check for drift
- `GET /alerts` - List alerts
- `POST /alerts/{id}/resolve` - Resolve alert

## Architecture

```
Train → Deploy → Monitor → Detect Drift → Score Severity → Recommend Action → Human Review → Retrain → Repeat
```

## License

MIT
