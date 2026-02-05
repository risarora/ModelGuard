"""FastAPI application for ModelGuard."""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from modelguard import __version__
from modelguard.core.config import get_config
from modelguard.storage.database import get_database, init_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_database()
    yield


app = FastAPI(
    title="ModelGuard API",
    description="Data Drift, Model Decay & Auto-Retraining System",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class HealthResponse(BaseModel):
    status: str
    version: str


class ModelCreate(BaseModel):
    name: str
    version: str = "1.0.0"
    framework: Optional[str] = None
    model_type: Optional[str] = "classification"
    feature_names: Optional[List[str]] = None
    artifact_path: Optional[str] = None


class ModelResponse(BaseModel):
    id: str
    name: str
    version: str
    framework: Optional[str]
    model_type: Optional[str]
    is_active: bool
    created_at: str


class BaselineCreate(BaseModel):
    model_id: str
    feature_statistics: Dict[str, Any]
    prediction_statistics: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None
    sample_size: Optional[int] = None


class BaselineResponse(BaseModel):
    id: str
    model_id: str
    version: int
    is_active: bool
    sample_size: Optional[int]
    created_at: str


class DriftCheckRequest(BaseModel):
    baseline_id: str
    feature_data: Dict[str, List[Any]]
    features: Optional[List[str]] = None


class DriftCheckResponse(BaseModel):
    report_id: str
    data_drift_detected: bool
    drift_percentage: float
    features_with_drift: List[str]
    severity_level: str
    severity_score: float
    recommended_action: str
    urgency: str


class AlertResponse(BaseModel):
    id: str
    model_id: str
    alert_type: str
    severity: str
    urgency: str
    status: str
    created_at: str
    assigned_to: Optional[str]


class AlertResolve(BaseModel):
    decision: str
    decided_by: str
    notes: Optional[str] = None


# ============== Health Endpoints ==============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version=__version__)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "ModelGuard API",
        "version": __version__,
        "docs": "/docs",
    }


# ============== Model Endpoints ==============

@app.post("/models", response_model=ModelResponse)
async def create_model(model: ModelCreate):
    """Register a new model."""
    from modelguard.storage.repositories.model_repo import ModelRepository

    db = get_database()
    with db.session() as session:
        repo = ModelRepository(session)

        # Check for existing
        existing = repo.get_by_name_version(model.name, model.version)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Model {model.name} version {model.version} already exists",
            )

        record = repo.create(
            name=model.name,
            version=model.version,
            framework=model.framework,
            model_type=model.model_type,
            feature_names=model.feature_names,
            artifact_path=model.artifact_path,
        )

        return ModelResponse(
            id=record.id,
            name=record.name,
            version=record.version,
            framework=record.framework,
            model_type=record.model_type,
            is_active=record.is_active,
            created_at=record.created_at.isoformat(),
        )


@app.get("/models", response_model=List[ModelResponse])
async def list_models(
    active_only: bool = Query(True, description="Only return active models"),
    limit: int = Query(100, description="Maximum number of models to return"),
):
    """List registered models."""
    from modelguard.storage.repositories.model_repo import ModelRepository

    db = get_database()
    with db.session() as session:
        repo = ModelRepository(session)
        models = repo.list_all(active_only=active_only, limit=limit)

        return [
            ModelResponse(
                id=m.id,
                name=m.name,
                version=m.version,
                framework=m.framework,
                model_type=m.model_type,
                is_active=m.is_active,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in models
        ]


@app.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str):
    """Get a model by ID."""
    from modelguard.storage.repositories.model_repo import ModelRepository

    db = get_database()
    with db.session() as session:
        repo = ModelRepository(session)
        model = repo.get(model_id)

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        return ModelResponse(
            id=model.id,
            name=model.name,
            version=model.version,
            framework=model.framework,
            model_type=model.model_type,
            is_active=model.is_active,
            created_at=model.created_at.isoformat() if model.created_at else "",
        )


# ============== Baseline Endpoints ==============

@app.post("/baselines", response_model=BaselineResponse)
async def create_baseline(baseline: BaselineCreate):
    """Create a new baseline."""
    from modelguard.storage.repositories.baseline_repo import BaselineRepository

    db = get_database()
    with db.session() as session:
        repo = BaselineRepository(session)
        record = repo.create(
            model_id=baseline.model_id,
            feature_statistics=baseline.feature_statistics,
            prediction_statistics=baseline.prediction_statistics,
            performance_metrics=baseline.performance_metrics,
            sample_size=baseline.sample_size,
        )

        return BaselineResponse(
            id=record.id,
            model_id=record.model_id,
            version=record.version,
            is_active=record.is_active,
            sample_size=record.sample_size,
            created_at=record.created_at.isoformat() if record.created_at else "",
        )


@app.get("/baselines", response_model=List[BaselineResponse])
async def list_baselines(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, description="Maximum number of baselines to return"),
):
    """List baselines."""
    from modelguard.storage.repositories.baseline_repo import BaselineRepository
    from modelguard.storage.models import BaselineRecord

    db = get_database()
    with db.session() as session:
        if model_id:
            repo = BaselineRepository(session)
            baselines = repo.list_for_model(model_id, limit=limit)
        else:
            baselines = session.query(BaselineRecord).order_by(
                BaselineRecord.created_at.desc()
            ).limit(limit).all()

        return [
            BaselineResponse(
                id=b.id,
                model_id=b.model_id,
                version=b.version,
                is_active=b.is_active,
                sample_size=b.sample_size,
                created_at=b.created_at.isoformat() if b.created_at else "",
            )
            for b in baselines
        ]


@app.get("/baselines/{baseline_id}", response_model=BaselineResponse)
async def get_baseline(baseline_id: str):
    """Get a baseline by ID."""
    from modelguard.storage.repositories.baseline_repo import BaselineRepository

    db = get_database()
    with db.session() as session:
        repo = BaselineRepository(session)
        baseline = repo.get(baseline_id)

        if not baseline:
            raise HTTPException(status_code=404, detail="Baseline not found")

        return BaselineResponse(
            id=baseline.id,
            model_id=baseline.model_id,
            version=baseline.version,
            is_active=baseline.is_active,
            sample_size=baseline.sample_size,
            created_at=baseline.created_at.isoformat() if baseline.created_at else "",
        )


# ============== Drift Endpoints ==============

@app.post("/drift/check", response_model=DriftCheckResponse)
async def check_drift(request: DriftCheckRequest):
    """Check for drift in provided data against a baseline."""
    import pandas as pd

    from modelguard.drift.detector import DriftDetector
    from modelguard.severity.scorer import SeverityScorer
    from modelguard.actions.recommender import ActionRecommender
    from modelguard.storage.repositories.baseline_repo import BaselineRepository
    from modelguard.storage.repositories.drift_repo import DriftReportRepository
    from modelguard.core.types import Baseline, FeatureStatistics, PredictionStatistics

    db = get_database()

    # Get baseline
    with db.session() as session:
        baseline_repo = BaselineRepository(session)
        baseline_record = baseline_repo.get(request.baseline_id)

        if not baseline_record:
            raise HTTPException(status_code=404, detail="Baseline not found")

        # Convert to Baseline object
        feature_stats = {}
        for name, stats in baseline_record.feature_statistics.items():
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
                unique_count=stats.get("unique_count"),
                value_counts=stats.get("value_counts"),
                mode=stats.get("mode"),
            )

        pred_stats_dict = baseline_record.prediction_statistics or {}
        pred_stats = PredictionStatistics(
            prediction_type=pred_stats_dict.get("prediction_type", "classification"),
        )

        baseline = Baseline(
            id=baseline_record.id,
            model_id=baseline_record.model_id,
            created_at=baseline_record.created_at,
            feature_statistics=feature_stats,
            prediction_statistics=pred_stats,
            sample_size=baseline_record.sample_size or 0,
        )

    # Convert input data to DataFrame
    current_data = pd.DataFrame(request.feature_data)

    # Run drift detection
    detector = DriftDetector()
    report = detector.detect(baseline, current_data, features=request.features)

    # Calculate severity
    scorer = SeverityScorer()
    severity = scorer.score_report(report)
    report.severity = severity

    # Get recommendation
    recommender = ActionRecommender()
    recommendation = recommender.recommend(severity)
    report.recommendation = recommendation

    # Save report
    with db.session() as session:
        drift_repo = DriftReportRepository(session)
        drift_repo.create(
            model_id=baseline.model_id,
            baseline_id=baseline.id,
            feature_results={k: [r.to_dict() for r in v] for k, v in report.feature_results.items()},
            data_drift_detected=report.data_drift_detected,
            features_with_drift=report.features_with_drift,
            drift_percentage=report.drift_percentage,
            current_sample_size=report.current_sample_size,
            reference_sample_size=report.reference_sample_size,
            severity=severity.to_dict(),
            recommendation=recommendation.to_dict(),
        )

    return DriftCheckResponse(
        report_id=report.id,
        data_drift_detected=report.data_drift_detected,
        drift_percentage=report.drift_percentage,
        features_with_drift=report.features_with_drift,
        severity_level=severity.level.value,
        severity_score=severity.overall_score,
        recommended_action=recommendation.action.value,
        urgency=recommendation.urgency.value,
    )


# ============== Alert Endpoints ==============

@app.get("/alerts", response_model=List[AlertResponse])
async def list_alerts(
    status: Optional[str] = Query(None, description="Filter by status"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    limit: int = Query(100, description="Maximum number of alerts"),
):
    """List alerts."""
    from modelguard.storage.repositories.alert_repo import AlertRepository

    db = get_database()
    with db.session() as session:
        repo = AlertRepository(session)

        if status:
            alerts = repo.list_by_status(status, model_id=model_id, limit=limit)
        elif model_id:
            alerts = repo.list_for_model(model_id, limit=limit)
        else:
            alerts = repo.list_pending(limit=limit)

        return [
            AlertResponse(
                id=a.id,
                model_id=a.model_id,
                alert_type=a.alert_type,
                severity=a.severity,
                urgency=a.urgency,
                status=a.status,
                created_at=a.created_at.isoformat() if a.created_at else "",
                assigned_to=a.assigned_to,
            )
            for a in alerts
        ]


@app.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str):
    """Get an alert by ID."""
    from modelguard.storage.repositories.alert_repo import AlertRepository

    db = get_database()
    with db.session() as session:
        repo = AlertRepository(session)
        alert = repo.get(alert_id)

        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        return AlertResponse(
            id=alert.id,
            model_id=alert.model_id,
            alert_type=alert.alert_type,
            severity=alert.severity,
            urgency=alert.urgency,
            status=alert.status,
            created_at=alert.created_at.isoformat() if alert.created_at else "",
            assigned_to=alert.assigned_to,
        )


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolve: AlertResolve):
    """Resolve an alert."""
    from modelguard.storage.repositories.alert_repo import AlertRepository

    valid_decisions = ["ignore", "monitor", "retrain", "rollback"]
    if resolve.decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision. Must be one of: {', '.join(valid_decisions)}",
        )

    db = get_database()
    with db.session() as session:
        repo = AlertRepository(session)
        alert = repo.resolve(
            alert_id=alert_id,
            decision=resolve.decision,
            decided_by=resolve.decided_by,
            decision_notes=resolve.notes,
        )

        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {"status": "resolved", "alert_id": alert_id, "decision": resolve.decision}
