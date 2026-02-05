"""Configuration management for ModelGuard."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from modelguard.core.exceptions import ConfigurationError


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///./modelguard.db"
    echo: bool = False


@dataclass
class ArtifactStorageConfig:
    """Artifact storage configuration."""
    backend: str = "local"
    base_path: str = "./artifacts"


@dataclass
class NumericalStatsConfig:
    """Configuration for numerical feature statistics."""
    percentiles: List[int] = field(
        default_factory=lambda: [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )
    histogram_bins: int = 50


@dataclass
class CategoricalStatsConfig:
    """Configuration for categorical feature statistics."""
    max_categories: int = 100
    rare_threshold: float = 0.01


@dataclass
class BaselineConfig:
    """Baseline creation configuration."""
    numerical: NumericalStatsConfig = field(default_factory=NumericalStatsConfig)
    categorical: CategoricalStatsConfig = field(default_factory=CategoricalStatsConfig)
    retain_sample: bool = True
    sample_size: int = 10000


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    window_size_hours: int = 1
    min_samples: int = 100
    max_samples: int = 100000


@dataclass
class DriftMethodConfig:
    """Configuration for a drift detection method."""
    name: str
    enabled: bool = True
    threshold: float = 0.05
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftConsensusConfig:
    """Configuration for drift consensus strategy."""
    strategy: str = "any"  # any, majority, all
    min_agreement: int = 1


@dataclass
class DriftConfig:
    """Drift detection configuration."""
    numerical_methods: List[DriftMethodConfig] = field(default_factory=list)
    categorical_methods: List[DriftMethodConfig] = field(default_factory=list)
    consensus: DriftConsensusConfig = field(default_factory=DriftConsensusConfig)


@dataclass
class SeverityConfig:
    """Severity scoring configuration."""
    feature_weights: Dict[str, float] = field(default_factory=dict)
    default_weight: float = 1.0
    aggregation: str = "weighted_mean"  # max, mean, weighted_mean
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "none": 0.0,
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5,
            "critical": 0.8,
        }
    )
    impact_multiplier: float = 1.5


@dataclass
class ActionConfig:
    """Action recommendation configuration."""
    allowed_actions: List[str] = field(
        default_factory=lambda: ["ignore", "monitor", "retrain", "rollback"]
    )
    auto_action_enabled: bool = False
    auto_action_max_severity: str = "low"


@dataclass
class HumanLoopConfig:
    """Human-in-the-loop configuration."""
    escalation_timeout_hours: int = 24
    notifications_enabled: bool = False


@dataclass
class RetrainingDataConfig:
    """Retraining data configuration."""
    lookback_days: int = 30
    min_samples: int = 10000
    validation_split: float = 0.2


@dataclass
class RetrainingValidationConfig:
    """Retraining validation configuration."""
    required_improvement: float = 0.0
    max_degradation: float = 0.05


@dataclass
class RetrainingConfig:
    """Retraining configuration."""
    data: RetrainingDataConfig = field(default_factory=RetrainingDataConfig)
    validation: RetrainingValidationConfig = field(
        default_factory=RetrainingValidationConfig
    )
    deployment_strategy: str = "direct"  # direct, canary
    auto_rollback: bool = True


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    """Application configuration."""
    name: str = "ModelGuard"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration class."""
    app: AppConfig = field(default_factory=AppConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    artifact_storage: ArtifactStorageConfig = field(
        default_factory=ArtifactStorageConfig
    )
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    human_loop: HumanLoopConfig = field(default_factory=HumanLoopConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)


def _parse_drift_methods(methods_config: List[Dict]) -> List[DriftMethodConfig]:
    """Parse drift method configurations."""
    methods = []
    for m in methods_config:
        params = {k: v for k, v in m.items() if k not in ("name", "enabled", "threshold")}
        methods.append(
            DriftMethodConfig(
                name=m.get("name", ""),
                enabled=m.get("enabled", True),
                threshold=m.get("threshold", 0.05),
                params=params,
            )
        )
    return methods


def load_config(
    config_path: Optional[str] = None,
    env_file: Optional[str] = None,
) -> Config:
    """
    Load configuration from YAML file and environment variables.

    Args:
        config_path: Path to YAML config file. If None, uses default.
        env_file: Path to .env file. If None, looks for .env in current directory.

    Returns:
        Config object with all settings.
    """
    # Load environment variables
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    # Determine config file path
    if config_path is None:
        # Try common locations
        possible_paths = [
            Path("config/default.yaml"),
            Path("config.yaml"),
            Path.home() / ".modelguard" / "config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    # Load YAML config if exists
    yaml_config: Dict[str, Any] = {}
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {e}")

    # Build config objects
    app_cfg = yaml_config.get("app", {})
    db_cfg = yaml_config.get("database", {})
    artifact_cfg = yaml_config.get("artifact_storage", {})
    baseline_cfg = yaml_config.get("baseline", {})
    monitoring_cfg = yaml_config.get("monitoring", {})
    drift_cfg = yaml_config.get("drift", {})
    severity_cfg = yaml_config.get("severity", {})
    actions_cfg = yaml_config.get("actions", {})
    human_loop_cfg = yaml_config.get("human_loop", {})
    retraining_cfg = yaml_config.get("retraining", {})
    api_cfg = yaml_config.get("api", {})

    # Override with environment variables
    db_url = os.getenv("DATABASE_URL", db_cfg.get("url", "sqlite:///./modelguard.db"))

    # Parse baseline config
    baseline_stats = baseline_cfg.get("statistics", {})
    numerical_stats = baseline_stats.get("numerical", {})
    categorical_stats = baseline_stats.get("categorical", {})

    # Parse drift methods
    drift_methods = drift_cfg.get("methods", {})
    numerical_methods = _parse_drift_methods(drift_methods.get("numerical", []))
    categorical_methods = _parse_drift_methods(drift_methods.get("categorical", []))

    # Default drift methods if none configured
    if not numerical_methods:
        numerical_methods = [
            DriftMethodConfig(name="ks_test", enabled=True, threshold=0.05),
            DriftMethodConfig(name="psi", enabled=True, threshold=0.1, params={"n_bins": 10}),
            DriftMethodConfig(name="wasserstein", enabled=True, threshold=0.1),
            DriftMethodConfig(name="kl_divergence", enabled=True, threshold=0.1),
        ]
    if not categorical_methods:
        categorical_methods = [
            DriftMethodConfig(name="chi_square", enabled=True, threshold=0.05),
            DriftMethodConfig(name="psi", enabled=True, threshold=0.1),
            DriftMethodConfig(name="jensen_shannon", enabled=True, threshold=0.1),
        ]

    consensus_cfg = drift_cfg.get("consensus", {})

    # Parse retraining config
    retrain_data = retraining_cfg.get("data", {})
    retrain_validation = retraining_cfg.get("validation", {})

    return Config(
        app=AppConfig(
            name=app_cfg.get("name", "ModelGuard"),
            environment=os.getenv("MODELGUARD_ENV", app_cfg.get("environment", "development")),
            debug=os.getenv("MODELGUARD_DEBUG", str(app_cfg.get("debug", True))).lower() == "true",
            log_level=os.getenv("MODELGUARD_LOG_LEVEL", app_cfg.get("log_level", "INFO")),
        ),
        database=DatabaseConfig(
            url=db_url,
            echo=db_cfg.get("echo", False),
        ),
        artifact_storage=ArtifactStorageConfig(
            backend=artifact_cfg.get("backend", "local"),
            base_path=artifact_cfg.get("base_path", "./artifacts"),
        ),
        baseline=BaselineConfig(
            numerical=NumericalStatsConfig(
                percentiles=numerical_stats.get("percentiles", [1, 5, 10, 25, 50, 75, 90, 95, 99]),
                histogram_bins=numerical_stats.get("histogram_bins", 50),
            ),
            categorical=CategoricalStatsConfig(
                max_categories=categorical_stats.get("max_categories", 100),
                rare_threshold=categorical_stats.get("rare_threshold", 0.01),
            ),
            retain_sample=baseline_cfg.get("retain_sample", True),
            sample_size=baseline_cfg.get("sample_size", 10000),
        ),
        monitoring=MonitoringConfig(
            window_size_hours=monitoring_cfg.get("window_size_hours", 1),
            min_samples=monitoring_cfg.get("min_samples", 100),
            max_samples=monitoring_cfg.get("max_samples", 100000),
        ),
        drift=DriftConfig(
            numerical_methods=numerical_methods,
            categorical_methods=categorical_methods,
            consensus=DriftConsensusConfig(
                strategy=consensus_cfg.get("strategy", "any"),
                min_agreement=consensus_cfg.get("min_agreement", 1),
            ),
        ),
        severity=SeverityConfig(
            feature_weights=severity_cfg.get("feature_weights", {}),
            default_weight=severity_cfg.get("default_weight", 1.0),
            aggregation=severity_cfg.get("aggregation", "weighted_mean"),
            thresholds=severity_cfg.get("thresholds", {
                "none": 0.0,
                "low": 0.1,
                "medium": 0.3,
                "high": 0.5,
                "critical": 0.8,
            }),
            impact_multiplier=severity_cfg.get("impact_multiplier", 1.5),
        ),
        actions=ActionConfig(
            allowed_actions=actions_cfg.get("allowed_actions", ["ignore", "monitor", "retrain", "rollback"]),
            auto_action_enabled=actions_cfg.get("auto_action", {}).get("enabled", False),
            auto_action_max_severity=actions_cfg.get("auto_action", {}).get("max_severity", "low"),
        ),
        human_loop=HumanLoopConfig(
            escalation_timeout_hours=human_loop_cfg.get("escalation_timeout_hours", 24),
            notifications_enabled=human_loop_cfg.get("notifications", {}).get("enabled", False),
        ),
        retraining=RetrainingConfig(
            data=RetrainingDataConfig(
                lookback_days=retrain_data.get("lookback_days", 30),
                min_samples=retrain_data.get("min_samples", 10000),
                validation_split=retrain_data.get("validation_split", 0.2),
            ),
            validation=RetrainingValidationConfig(
                required_improvement=retrain_validation.get("required_improvement", 0.0),
                max_degradation=retrain_validation.get("max_degradation", 0.05),
            ),
            deployment_strategy=retraining_cfg.get("deployment", {}).get("strategy", "direct"),
            auto_rollback=retraining_cfg.get("deployment", {}).get("auto_rollback", True),
        ),
        api=APIConfig(
            host=os.getenv("API_HOST", api_cfg.get("host", "0.0.0.0")),
            port=int(os.getenv("API_PORT", api_cfg.get("port", 8000))),
            cors_origins=api_cfg.get("cors_origins", ["*"]),
        ),
    )


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
