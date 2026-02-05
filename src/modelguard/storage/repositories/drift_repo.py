"""Repository for drift report operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from modelguard.storage.models import DriftReportRecord, DriftMetricRecord


class DriftReportRepository:
    """Repository for drift report CRUD operations."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        model_id: str,
        baseline_id: str,
        feature_results: Dict[str, Any],
        data_drift_detected: bool = False,
        prediction_drift_detected: bool = False,
        concept_drift_detected: bool = False,
        features_with_drift: Optional[List[str]] = None,
        drift_percentage: float = 0.0,
        current_sample_size: int = 0,
        reference_sample_size: int = 0,
        severity: Optional[Dict[str, Any]] = None,
        recommendation: Optional[Dict[str, Any]] = None,
    ) -> DriftReportRecord:
        """Create a new drift report."""
        record = DriftReportRecord(
            model_id=model_id,
            baseline_id=baseline_id,
            timestamp=datetime.utcnow(),
            feature_results=feature_results,
            data_drift_detected=data_drift_detected,
            prediction_drift_detected=prediction_drift_detected,
            concept_drift_detected=concept_drift_detected,
            features_with_drift=features_with_drift or [],
            drift_percentage=drift_percentage,
            current_sample_size=current_sample_size,
            reference_sample_size=reference_sample_size,
            severity=severity,
            recommendation=recommendation,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, report_id: str) -> Optional[DriftReportRecord]:
        """Get a drift report by ID."""
        return self.session.query(DriftReportRecord).filter(
            DriftReportRecord.id == report_id
        ).first()

    def get_latest(self, model_id: str) -> Optional[DriftReportRecord]:
        """Get the most recent drift report for a model."""
        return self.session.query(DriftReportRecord).filter(
            DriftReportRecord.model_id == model_id
        ).order_by(DriftReportRecord.timestamp.desc()).first()

    def list_for_model(
        self,
        model_id: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[DriftReportRecord]:
        """List drift reports for a model."""
        query = self.session.query(DriftReportRecord).filter(
            DriftReportRecord.model_id == model_id
        )

        if start_time:
            query = query.filter(DriftReportRecord.timestamp >= start_time)
        if end_time:
            query = query.filter(DriftReportRecord.timestamp <= end_time)

        return query.order_by(DriftReportRecord.timestamp.desc()).limit(limit).all()

    def list_with_drift(
        self,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[DriftReportRecord]:
        """List reports where drift was detected."""
        query = self.session.query(DriftReportRecord).filter(
            DriftReportRecord.data_drift_detected == True
        )

        if model_id:
            query = query.filter(DriftReportRecord.model_id == model_id)

        return query.order_by(DriftReportRecord.timestamp.desc()).limit(limit).all()

    def delete(self, report_id: str) -> bool:
        """Delete a drift report."""
        record = self.get(report_id)
        if record is None:
            return False

        self.session.delete(record)
        self.session.flush()
        return True


class DriftMetricRepository:
    """Repository for individual drift metric records."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        model_id: str,
        baseline_id: str,
        window_start: datetime,
        window_end: datetime,
        feature_name: str,
        method_name: str,
        statistic: float,
        threshold: float,
        drift_detected: bool,
        drift_type: str = "data",
        p_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DriftMetricRecord:
        """Create a new drift metric record."""
        record = DriftMetricRecord(
            model_id=model_id,
            baseline_id=baseline_id,
            window_start=window_start,
            window_end=window_end,
            drift_type=drift_type,
            feature_name=feature_name,
            method_name=method_name,
            statistic=statistic,
            p_value=p_value,
            threshold=threshold,
            drift_detected=drift_detected,
            metadata=metadata,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def list_for_feature(
        self,
        model_id: str,
        feature_name: str,
        limit: int = 100,
    ) -> List[DriftMetricRecord]:
        """List drift metrics for a specific feature."""
        return self.session.query(DriftMetricRecord).filter(
            DriftMetricRecord.model_id == model_id,
            DriftMetricRecord.feature_name == feature_name,
        ).order_by(DriftMetricRecord.window_start.desc()).limit(limit).all()

    def list_for_time_range(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[DriftMetricRecord]:
        """List drift metrics for a time range."""
        return self.session.query(DriftMetricRecord).filter(
            DriftMetricRecord.model_id == model_id,
            DriftMetricRecord.window_start >= start_time,
            DriftMetricRecord.window_end <= end_time,
        ).order_by(DriftMetricRecord.window_start.desc()).all()
