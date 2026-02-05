"""Repository for baseline operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from modelguard.storage.models import BaselineRecord


class BaselineRepository:
    """Repository for baseline CRUD operations."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        model_id: str,
        feature_statistics: Dict[str, Any],
        prediction_statistics: Dict[str, Any],
        performance_metrics: Optional[Dict[str, Any]] = None,
        sample_size: Optional[int] = None,
        data_start_date: Optional[datetime] = None,
        data_end_date: Optional[datetime] = None,
        created_by: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BaselineRecord:
        """Create a new baseline record."""
        # Get next version number
        existing = self.list_for_model(model_id)
        version = len(existing) + 1

        # Deactivate previous baselines
        for baseline in existing:
            if baseline.is_active:
                baseline.is_active = False

        record = BaselineRecord(
            model_id=model_id,
            version=version,
            is_active=True,
            feature_statistics=feature_statistics,
            prediction_statistics=prediction_statistics,
            performance_metrics=performance_metrics,
            sample_size=sample_size,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            created_by=created_by,
            notes=notes,
            metadata=metadata,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, baseline_id: str) -> Optional[BaselineRecord]:
        """Get a baseline by ID."""
        return self.session.query(BaselineRecord).filter(
            BaselineRecord.id == baseline_id
        ).first()

    def get_active(self, model_id: str) -> Optional[BaselineRecord]:
        """Get the active baseline for a model."""
        return self.session.query(BaselineRecord).filter(
            BaselineRecord.model_id == model_id,
            BaselineRecord.is_active == True,
        ).first()

    def list_for_model(
        self,
        model_id: str,
        limit: int = 100,
    ) -> List[BaselineRecord]:
        """List all baselines for a model."""
        return self.session.query(BaselineRecord).filter(
            BaselineRecord.model_id == model_id
        ).order_by(BaselineRecord.version.desc()).limit(limit).all()

    def set_active(self, baseline_id: str) -> Optional[BaselineRecord]:
        """Set a baseline as the active one for its model."""
        record = self.get(baseline_id)
        if record is None:
            return None

        # Deactivate other baselines for this model
        self.session.query(BaselineRecord).filter(
            BaselineRecord.model_id == record.model_id,
            BaselineRecord.id != baseline_id,
        ).update({"is_active": False})

        record.is_active = True
        self.session.flush()
        return record

    def delete(self, baseline_id: str) -> bool:
        """Delete a baseline record."""
        record = self.get(baseline_id)
        if record is None:
            return False

        self.session.delete(record)
        self.session.flush()
        return True
