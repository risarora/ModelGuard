"""Main CLI application for ModelGuard."""

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from modelguard.cli.commands import baseline, drift, alert, model, server

# Create main app
app = typer.Typer(
    name="modelguard",
    help="ModelGuard: Data Drift, Model Decay & Auto-Retraining System",
    add_completion=False,
)

# Add sub-commands
app.add_typer(baseline.app, name="baseline", help="Baseline management commands")
app.add_typer(drift.app, name="drift", help="Drift detection commands")
app.add_typer(alert.app, name="alert", help="Alert management commands")
app.add_typer(model.app, name="model", help="Model management commands")
app.add_typer(server.app, name="server", help="API server commands")

console = Console()


@app.command()
def version():
    """Show ModelGuard version."""
    from modelguard import __version__
    rprint(f"[bold blue]ModelGuard[/bold blue] version [green]{__version__}[/green]")


@app.command()
def init(
    config_path: str = typer.Option(
        "config/default.yaml",
        "--config", "-c",
        help="Path to create config file",
    ),
):
    """Initialize ModelGuard configuration and database."""
    from pathlib import Path
    import yaml

    from modelguard.core.config import load_config
    from modelguard.storage.database import init_database

    # Create config directory if needed
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    if not config_file.exists():
        # Create default config
        default_config = {
            "version": "1.0",
            "app": {
                "name": "ModelGuard",
                "environment": "development",
                "debug": True,
                "log_level": "INFO",
            },
            "database": {
                "url": "sqlite:///./modelguard.db",
            },
            "artifact_storage": {
                "backend": "local",
                "base_path": "./artifacts",
            },
        }
        with open(config_file, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        rprint(f"[green]Created config file:[/green] {config_path}")

    # Initialize database
    config = load_config(str(config_file))
    init_database(config)
    rprint("[green]Database initialized successfully[/green]")

    # Create artifact directory
    artifact_path = Path(config.artifact_storage.base_path)
    artifact_path.mkdir(parents=True, exist_ok=True)
    rprint(f"[green]Artifact directory created:[/green] {artifact_path}")

    rprint("\n[bold green]ModelGuard initialized successfully![/bold green]")


@app.command()
def status():
    """Show system status and summary."""
    from modelguard.core.config import get_config
    from modelguard.storage.database import get_database

    config = get_config()
    db = get_database()

    # Create status table
    table = Table(title="ModelGuard Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Config status
    table.add_row(
        "Configuration",
        "OK",
        f"Environment: {config.app.environment}",
    )

    # Database status
    try:
        with db.session() as session:
            # Quick query to test connection
            session.execute("SELECT 1")
        table.add_row(
            "Database",
            "Connected",
            config.database.url.split("///")[-1] if ":///" in config.database.url else config.database.url,
        )
    except Exception as e:
        table.add_row(
            "Database",
            "[red]Error[/red]",
            str(e)[:50],
        )

    # Count records
    try:
        with db.session() as session:
            from modelguard.storage.models import ModelRecord, BaselineRecord, AlertRecord

            model_count = session.query(ModelRecord).count()
            baseline_count = session.query(BaselineRecord).count()
            alert_count = session.query(AlertRecord).filter(
                AlertRecord.status == "pending"
            ).count()

            table.add_row("Models", str(model_count), "registered models")
            table.add_row("Baselines", str(baseline_count), "active baselines")
            table.add_row("Pending Alerts", str(alert_count), "awaiting review")
    except Exception:
        pass

    console.print(table)


@app.command()
def info():
    """Show detailed system information."""
    from modelguard import __version__
    from modelguard.core.config import get_config
    from modelguard.drift.detector import DriftDetector

    config = get_config()
    detector = DriftDetector(config)

    # System info
    rprint("\n[bold cyan]System Information[/bold cyan]")
    rprint(f"  Version: {__version__}")
    rprint(f"  Environment: {config.app.environment}")
    rprint(f"  Debug Mode: {config.app.debug}")

    # Drift detection methods
    methods = detector.get_available_methods()
    rprint("\n[bold cyan]Drift Detection Methods[/bold cyan]")
    rprint(f"  Numerical: {', '.join(methods['numerical'])}")
    rprint(f"  Categorical: {', '.join(methods['categorical'])}")

    # Severity thresholds
    rprint("\n[bold cyan]Severity Thresholds[/bold cyan]")
    for level, threshold in config.severity.thresholds.items():
        rprint(f"  {level.upper()}: >= {threshold}")

    # Allowed actions
    rprint("\n[bold cyan]Allowed Actions[/bold cyan]")
    for action in config.actions.allowed_actions:
        rprint(f"  - {action}")


if __name__ == "__main__":
    app()
