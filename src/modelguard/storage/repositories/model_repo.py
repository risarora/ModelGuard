"""Repository for model operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from modelguard.storage.models import ModelRecord


class ModelRepository:
    """Repository for model CRUD operations."""

    def __init__(self, session: Session):
        """Initialize with database session."""
        self.session = session

    def create(
        self,
        name: str,
        version: str,
        framework: Optional[str] = None,
        model_type: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        artifact_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelRecord:
        """Create a new model record."""
        record = ModelRecord(
            name=name,
            version=version,
            framework=framework,
            model_type=model_type,
            feature_names=feature_names,
            target_name=target_name,
            artifact_path=artifact_path,
            hyperparameters=hyperparameters,
            metadata=metadata,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get(self, model_id: str) -> Optional[ModelRecord]:
        """Get a model by ID."""
        return self.session.query(ModelRecord).filter(
            ModelRecord.id == model_id
        ).first()

    def get_by_name(self, name: str) -> Optional[ModelRecord]:
        """Get the latest active model by name."""
        return self.session.query(ModelRecord).filter(
            ModelRecord.name == name,
            ModelRecord.is_active == True,
        ).order_by(ModelRecord.created_at.desc()).first()

    def get_by_name_version(
        self,
        name: str,
        version: str,
    ) -> Optional[ModelRecord]:
        """Get a model by name and version."""
        return self.session.query(ModelRecord).filter(
            ModelRecord.name == name,
            ModelRecord.version == version,
        ).first()

    def list_all(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[ModelRecord]:
        """List all models."""
        query = self.session.query(ModelRecord)
        if active_only:
            query = query.filter(ModelRecord.is_active == True)
        return query.order_by(ModelRecord.created_at.desc()).limit(limit).all()

    def list_by_name(self, name: str) -> List[ModelRecord]:
        """List all versions of a model."""
        return self.session.query(ModelRecord).filter(
            ModelRecord.name == name
        ).order_by(ModelRecord.created_at.desc()).all()

    def update(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> Optional[ModelRecord]:
        """Update a model's attributes."""
        record = self.get(model_id)
        if record is None:
            return None

        for key, value in kwargs.items():
            if hasattr(record, key):
                setattr(record, key, value)

        record.updated_at = datetime.utcnow()
        self.session.flush()
        return record

    def set_deployed(
        self,
        model_id: str,
        deployed: bool = True,
    ) -> Optional[ModelRecord]:
        """Mark a model as deployed/undeployed."""
        record = self.get(model_id)
        if record is None:
            return None

        record.deployed_at = datetime.utcnow() if deployed else None
        record.updated_at = datetime.utcnow()
        self.session.flush()
        return record

    def deactivate(self, model_id: str) -> Optional[ModelRecord]:
        """Deactivate a model."""
        return self.update(model_id, is_active=False)

    def delete(self, model_id: str) -> bool:
        """Delete a model record."""
        record = self.get(model_id)
        if record is None:
            return False

        self.session.delete(record)
        self.session.flush()
        return True
