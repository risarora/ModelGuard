"""Repository for scheduled job operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from modelguard.storage.models import ScheduledJobRecord


class ScheduledJobRepository:
    """Repository for scheduled job CRUD operations."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        name: str,
        job_type: str,
        model_id: str,
        schedule_type: str,
        baseline_id: Optional[str] = None,
        interval_minutes: Optional[int] = None,
        cron_expression: Optional[str] = None,
        data_source_type: Optional[str] = None,
        data_source_config: Optional[Dict[str, Any]] = None,
        notify_on_drift: bool = True,
        notification_config: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> ScheduledJobRecord:
        """Create a new scheduled job."""
        record = ScheduledJobRecord(
            name=name,
            job_type=job_type,
            model_id=model_id,
            baseline_id=baseline_id,
            schedule_type=schedule_type,
            interval_minutes=interval_minutes,
            cron_expression=cron_expression,
            data_source_type=data_source_type,
            data_source_config=data_source_config,
            notify_on_drift=notify_on_drift,
            notification_config=notification_config,
            created_by=created_by,
            is_active=True,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, job_id: str) -> Optional[ScheduledJobRecord]:
        """Get a job by ID."""
        return self.session.query(ScheduledJobRecord).filter(
            ScheduledJobRecord.id == job_id
        ).first()

    def get_by_name(self, name: str) -> Optional[ScheduledJobRecord]:
        """Get a job by name."""
        return self.session.query(ScheduledJobRecord).filter(
            ScheduledJobRecord.name == name
        ).first()

    def list_active(self, limit: int = 100) -> List[ScheduledJobRecord]:
        """List active scheduled jobs."""
        return self.session.query(ScheduledJobRecord).filter(
            ScheduledJobRecord.is_active == True
        ).order_by(ScheduledJobRecord.next_run_at).limit(limit).all()

    def list_all(
        self,
        model_id: Optional[str] = None,
        active_only: bool = False,
        limit: int = 100,
    ) -> List[ScheduledJobRecord]:
        """List all scheduled jobs."""
        query = self.session.query(ScheduledJobRecord)

        if model_id:
            query = query.filter(ScheduledJobRecord.model_id == model_id)

        if active_only:
            query = query.filter(ScheduledJobRecord.is_active == True)

        return query.order_by(ScheduledJobRecord.created_at.desc()).limit(limit).all()

    def list_due(self, before: Optional[datetime] = None) -> List[ScheduledJobRecord]:
        """List jobs that are due to run."""
        if before is None:
            before = datetime.utcnow()

        return self.session.query(ScheduledJobRecord).filter(
            ScheduledJobRecord.is_active == True,
            ScheduledJobRecord.next_run_at <= before,
        ).order_by(ScheduledJobRecord.next_run_at).all()

    def update_run_status(
        self,
        job_id: str,
        status: str,
        next_run_at: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> Optional[ScheduledJobRecord]:
        """Update job run status."""
        record = self.get(job_id)
        if record is None:
            return None

        record.last_run_at = datetime.utcnow()
        record.last_run_status = status
        record.run_count = (record.run_count or 0) + 1

        if next_run_at:
            record.next_run_at = next_run_at

        if error:
            record.last_error = error
        else:
            record.last_error = None

        self.session.flush()
        return record

    def activate(self, job_id: str) -> Optional[ScheduledJobRecord]:
        """Activate a job."""
        record = self.get(job_id)
        if record is None:
            return None

        record.is_active = True
        self.session.flush()
        return record

    def deactivate(self, job_id: str) -> Optional[ScheduledJobRecord]:
        """Deactivate a job."""
        record = self.get(job_id)
        if record is None:
            return None

        record.is_active = False
        self.session.flush()
        return record

    def update_schedule(
        self,
        job_id: str,
        schedule_type: Optional[str] = None,
        interval_minutes: Optional[int] = None,
        cron_expression: Optional[str] = None,
    ) -> Optional[ScheduledJobRecord]:
        """Update job schedule."""
        record = self.get(job_id)
        if record is None:
            return None

        if schedule_type:
            record.schedule_type = schedule_type
        if interval_minutes is not None:
            record.interval_minutes = interval_minutes
        if cron_expression is not None:
            record.cron_expression = cron_expression

        self.session.flush()
        return record

    def delete(self, job_id: str) -> bool:
        """Delete a job."""
        record = self.get(job_id)
        if record is None:
            return False

        self.session.delete(record)
        self.session.flush()
        return True
