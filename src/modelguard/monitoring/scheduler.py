"""Scheduled monitoring jobs for automated drift detection."""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from modelguard.core.config import Config, get_config
from modelguard.core.types import (
    Baseline,
    FeatureStatistics,
    PredictionStatistics,
)
from modelguard.drift.detector import DriftDetector
from modelguard.severity.scorer import SeverityScorer
from modelguard.actions.recommender import ActionRecommender
from modelguard.human_loop.alert_manager import AlertManager
from modelguard.storage.database import get_database
from modelguard.storage.repositories.job_repo import ScheduledJobRepository
from modelguard.storage.repositories.baseline_repo import BaselineRepository
from modelguard.storage.repositories.alert_repo import AlertRepository

logger = logging.getLogger(__name__)


class MonitoringScheduler:
    """
    Schedules and executes periodic drift monitoring jobs.

    Supports:
    - Interval-based scheduling (every N minutes)
    - Cron-based scheduling (cron expressions)
    - Multiple data sources (file, database, API)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        data_loader: Optional[Callable[[Dict[str, Any]], pd.DataFrame]] = None,
    ):
        """
        Initialize the monitoring scheduler.

        Args:
            config: Configuration object
            data_loader: Custom function to load production data
        """
        self.config = config or get_config()
        self.data_loader = data_loader
        self.scheduler = BackgroundScheduler()
        self._running = False

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        # Load active jobs from database
        self._load_jobs()

        self.scheduler.start()
        self._running = True
        logger.info("Monitoring scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self.scheduler.shutdown(wait=True)
        self._running = False
        logger.info("Monitoring scheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def _load_jobs(self) -> None:
        """Load active jobs from database and schedule them."""
        db = get_database()
        with db.session() as session:
            repo = ScheduledJobRepository(session)
            jobs = repo.list_active()

            for job in jobs:
                self._schedule_job(
                    job_id=job.id,
                    name=job.name,
                    job_type=job.job_type,
                    model_id=job.model_id,
                    baseline_id=job.baseline_id,
                    schedule_type=job.schedule_type,
                    interval_minutes=job.interval_minutes,
                    cron_expression=job.cron_expression,
                    data_source_type=job.data_source_type,
                    data_source_config=job.data_source_config,
                    notify_on_drift=job.notify_on_drift,
                )

        logger.info(f"Loaded {len(jobs)} scheduled jobs")

    def _schedule_job(
        self,
        job_id: str,
        name: str,
        job_type: str,
        model_id: str,
        baseline_id: Optional[str],
        schedule_type: str,
        interval_minutes: Optional[int],
        cron_expression: Optional[str],
        data_source_type: Optional[str],
        data_source_config: Optional[Dict[str, Any]],
        notify_on_drift: bool = True,
    ) -> None:
        """Schedule a single job."""
        # Create trigger based on schedule type
        if schedule_type == "interval":
            if not interval_minutes:
                logger.error(f"Job {name}: interval_minutes required for interval schedule")
                return
            trigger = IntervalTrigger(minutes=interval_minutes)
        elif schedule_type == "cron":
            if not cron_expression:
                logger.error(f"Job {name}: cron_expression required for cron schedule")
                return
            trigger = CronTrigger.from_crontab(cron_expression)
        else:
            logger.error(f"Job {name}: Unknown schedule type: {schedule_type}")
            return

        # Add job to scheduler
        self.scheduler.add_job(
            func=self._execute_job,
            trigger=trigger,
            id=job_id,
            name=name,
            kwargs={
                "job_id": job_id,
                "job_type": job_type,
                "model_id": model_id,
                "baseline_id": baseline_id,
                "data_source_type": data_source_type,
                "data_source_config": data_source_config,
                "notify_on_drift": notify_on_drift,
            },
            replace_existing=True,
        )

        logger.info(f"Scheduled job: {name} ({schedule_type})")

    def _execute_job(
        self,
        job_id: str,
        job_type: str,
        model_id: str,
        baseline_id: Optional[str],
        data_source_type: Optional[str],
        data_source_config: Optional[Dict[str, Any]],
        notify_on_drift: bool = True,
    ) -> None:
        """Execute a scheduled monitoring job."""
        logger.info(f"Executing job {job_id}")

        db = get_database()
        try:
            # Load production data
            production_data = self._load_production_data(
                data_source_type, data_source_config
            )

            if production_data is None or len(production_data) == 0:
                raise ValueError("No production data available")

            # Get baseline
            with db.session() as session:
                baseline_repo = BaselineRepository(session)

                if baseline_id:
                    baseline_record = baseline_repo.get(baseline_id)
                else:
                    baseline_record = baseline_repo.get_active(model_id)

                if not baseline_record:
                    raise ValueError(f"No baseline found for model {model_id}")

                # Extract baseline data within session
                baseline_data = {
                    "id": baseline_record.id,
                    "model_id": baseline_record.model_id,
                    "created_at": baseline_record.created_at,
                    "sample_size": baseline_record.sample_size,
                    "feature_statistics": dict(baseline_record.feature_statistics),
                }

            # Reconstruct baseline object
            baseline = self._reconstruct_baseline(baseline_data)

            # Run drift detection
            detector = DriftDetector(self.config)
            drift_report = detector.detect(baseline, production_data)

            # Score severity
            scorer = SeverityScorer(self.config)
            severity = scorer.score_report(drift_report)
            drift_report.severity = severity

            # Get recommendation
            recommender = ActionRecommender(self.config)
            recommendation = recommender.recommend(severity)
            drift_report.recommendation = recommendation

            # Create alert if needed
            if notify_on_drift and drift_report.data_drift_detected:
                alert_manager = AlertManager(self.config)
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
                        alert_repo.create(
                            model_id=alert.model_id,
                            alert_type=alert.alert_type,
                            severity=alert.severity.value,
                            urgency=alert.urgency.value,
                            drift_report_id=alert.drift_report_id,
                            drift_summary={
                                "drift_percentage": drift_report.drift_percentage,
                                "features_with_drift": drift_report.features_with_drift,
                            },
                            recommendation={
                                "action": recommendation.action.value,
                                "confidence": recommendation.confidence,
                            },
                        )

                    logger.info(f"Alert created for job {job_id}")

            # Update job status
            next_run = self._calculate_next_run(job_id)
            with db.session() as session:
                job_repo = ScheduledJobRepository(session)
                job_repo.update_run_status(
                    job_id=job_id,
                    status="success",
                    next_run_at=next_run,
                )

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")

            # Update job status with error
            with db.session() as session:
                job_repo = ScheduledJobRepository(session)
                job_repo.update_run_status(
                    job_id=job_id,
                    status="failed",
                    error=str(e),
                )

    def _load_production_data(
        self,
        source_type: Optional[str],
        source_config: Optional[Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        """Load production data from configured source."""
        if self.data_loader:
            return self.data_loader(source_config or {})

        if not source_type or not source_config:
            return None

        if source_type == "file":
            file_path = source_config.get("path")
            file_format = source_config.get("format", "csv")

            if file_format == "csv":
                return pd.read_csv(file_path)
            elif file_format == "parquet":
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

        elif source_type == "database":
            # Placeholder for database loading
            raise NotImplementedError("Database source not yet implemented")

        elif source_type == "api":
            # Placeholder for API loading
            raise NotImplementedError("API source not yet implemented")

        return None

    def _reconstruct_baseline(self, baseline_data: Dict[str, Any]) -> Baseline:
        """Reconstruct Baseline object from stored data."""
        feature_stats = {}
        for name, stats in baseline_data["feature_statistics"].items():
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

        pred_stats = PredictionStatistics(prediction_type="classification")

        return Baseline(
            id=baseline_data["id"],
            model_id=baseline_data["model_id"],
            created_at=baseline_data["created_at"],
            feature_statistics=feature_stats,
            prediction_statistics=pred_stats,
            sample_size=baseline_data["sample_size"] or 0,
        )

    def _calculate_next_run(self, job_id: str) -> Optional[datetime]:
        """Calculate next run time for a job."""
        job = self.scheduler.get_job(job_id)
        if job and job.next_run_time:
            return job.next_run_time
        return None

    def add_job(
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
        created_by: Optional[str] = None,
    ) -> str:
        """
        Add a new scheduled monitoring job.

        Args:
            name: Job name
            job_type: Type of job (drift_check, performance_check)
            model_id: Model to monitor
            schedule_type: 'interval' or 'cron'
            baseline_id: Specific baseline to use (optional)
            interval_minutes: Minutes between runs (for interval type)
            cron_expression: Cron expression (for cron type)
            data_source_type: Type of data source
            data_source_config: Data source configuration
            notify_on_drift: Create alerts on drift detection
            created_by: User creating the job

        Returns:
            Job ID
        """
        db = get_database()
        with db.session() as session:
            repo = ScheduledJobRepository(session)

            # Check for duplicate name
            existing = repo.get_by_name(name)
            if existing:
                raise ValueError(f"Job with name '{name}' already exists")

            # Create job record
            job = repo.create(
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
                created_by=created_by,
            )
            job_id = job.id

        # Schedule if running
        if self._running:
            self._schedule_job(
                job_id=job_id,
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
            )

        return job_id

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        # Remove from scheduler
        if self._running:
            try:
                self.scheduler.remove_job(job_id)
            except Exception:
                pass  # Job might not be scheduled

        # Remove from database
        db = get_database()
        with db.session() as session:
            repo = ScheduledJobRepository(session)
            return repo.delete(job_id)

    def pause_job(self, job_id: str) -> bool:
        """Pause a scheduled job."""
        if self._running:
            try:
                self.scheduler.pause_job(job_id)
            except Exception:
                pass

        db = get_database()
        with db.session() as session:
            repo = ScheduledJobRepository(session)
            result = repo.deactivate(job_id)
            return result is not None

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        if self._running:
            try:
                self.scheduler.resume_job(job_id)
            except Exception:
                pass

        db = get_database()
        with db.session() as session:
            repo = ScheduledJobRepository(session)
            result = repo.activate(job_id)
            return result is not None

    def run_job_now(self, job_id: str) -> None:
        """Trigger immediate execution of a job."""
        db = get_database()
        with db.session() as session:
            repo = ScheduledJobRepository(session)
            job = repo.get(job_id)

            if not job:
                raise ValueError(f"Job not found: {job_id}")

            # Execute job directly
            self._execute_job(
                job_id=job.id,
                job_type=job.job_type,
                model_id=job.model_id,
                baseline_id=job.baseline_id,
                data_source_type=job.data_source_type,
                data_source_config=job.data_source_config,
                notify_on_drift=job.notify_on_drift,
            )

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs."""
        db = get_database()
        with db.session() as session:
            repo = ScheduledJobRepository(session)
            jobs = repo.list_all()
            return [
                {
                    "id": j.id,
                    "name": j.name,
                    "job_type": j.job_type,
                    "model_id": j.model_id,
                    "schedule_type": j.schedule_type,
                    "interval_minutes": j.interval_minutes,
                    "cron_expression": j.cron_expression,
                    "is_active": j.is_active,
                    "last_run_at": j.last_run_at,
                    "next_run_at": j.next_run_at,
                    "last_run_status": j.last_run_status,
                    "run_count": j.run_count,
                }
                for j in jobs
            ]

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a job."""
        db = get_database()
        with db.session() as session:
            repo = ScheduledJobRepository(session)
            job = repo.get(job_id)

            if not job:
                return None

            return job.to_dict()


# Global scheduler instance
_scheduler: Optional[MonitoringScheduler] = None


def get_scheduler() -> MonitoringScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = MonitoringScheduler()
    return _scheduler


def start_scheduler() -> None:
    """Start the global scheduler."""
    get_scheduler().start()


def stop_scheduler() -> None:
    """Stop the global scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
