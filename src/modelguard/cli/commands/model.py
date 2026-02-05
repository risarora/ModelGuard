"""Model management commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(help="Model management commands")
console = Console()


@app.command("register")
def register_model(
    name: str = typer.Argument(..., help="Model name"),
    version: str = typer.Option(
        "1.0.0", "--version", "-v",
        help="Model version",
    ),
    framework: str = typer.Option(
        "sklearn", "--framework", "-f",
        help="ML framework (sklearn, pytorch, etc.)",
    ),
    model_type: str = typer.Option(
        "classification", "--type", "-t",
        help="Model type (classification or regression)",
    ),
    artifact_path: Optional[str] = typer.Option(
        None, "--artifact", "-a",
        help="Path to model artifact",
    ),
):
    """Register a new model."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.model_repo import ModelRepository

    db = get_database()
    with db.session() as session:
        repo = ModelRepository(session)

        # Check if model with same name and version exists
        existing = repo.get_by_name_version(name, version)
        if existing:
            rprint(f"[red]Model {name} version {version} already exists[/red]")
            raise typer.Exit(1)

        model = repo.create(
            name=name,
            version=version,
            framework=framework,
            model_type=model_type,
            artifact_path=artifact_path,
        )
        model_id = model.id

    rprint(f"[green]Model registered successfully![/green]")
    rprint(f"  ID: {model_id}")
    rprint(f"  Name: {name}")
    rprint(f"  Version: {version}")


@app.command("list")
def list_models(
    all_versions: bool = typer.Option(
        False, "--all", "-a",
        help="Show all versions (not just active)",
    ),
    limit: int = typer.Option(
        20, "--limit", "-n",
        help="Maximum number of models to show",
    ),
):
    """List registered models."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.model_repo import ModelRepository

    db = get_database()
    with db.session() as session:
        repo = ModelRepository(session)
        models = repo.list_all(active_only=not all_versions, limit=limit)
        # Extract data within session to avoid DetachedInstanceError
        model_data = [
            {
                "id": m.id,
                "name": m.name,
                "version": m.version,
                "framework": m.framework,
                "model_type": m.model_type,
                "is_active": m.is_active,
                "deployed_at": m.deployed_at,
                "created_at": m.created_at,
            }
            for m in models
        ]

    if not model_data:
        rprint("[yellow]No models found[/yellow]")
        return

    table = Table(title="Registered Models")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Framework")
    table.add_column("Type")
    table.add_column("Active")
    table.add_column("Deployed")
    table.add_column("Created")

    for m in model_data:
        table.add_row(
            m["id"][:8] + "...",
            m["name"],
            m["version"],
            m["framework"] or "-",
            m["model_type"] or "-",
            "[green]Yes[/green]" if m["is_active"] else "[dim]No[/dim]",
            "[green]Yes[/green]" if m["deployed_at"] else "[dim]No[/dim]",
            m["created_at"].strftime("%Y-%m-%d") if m["created_at"] else "-",
        )

    console.print(table)


@app.command("show")
def show_model(
    model_id: str = typer.Argument(..., help="Model ID to show"),
):
    """Show model details."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.model_repo import ModelRepository
    from modelguard.storage.repositories.baseline_repo import BaselineRepository

    db = get_database()
    with db.session() as session:
        model_repo = ModelRepository(session)
        baseline_repo = BaselineRepository(session)

        model = model_repo.get(model_id)

        if not model:
            rprint(f"[red]Model not found: {model_id}[/red]")
            raise typer.Exit(1)

        # Extract data within session to avoid DetachedInstanceError
        model_data = {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "framework": model.framework,
            "model_type": model.model_type,
            "is_active": model.is_active,
            "created_at": model.created_at,
            "deployed_at": model.deployed_at,
            "artifact_path": model.artifact_path,
            "feature_names": model.feature_names,
        }

        # Get active baseline
        active_baseline = baseline_repo.get_active(model_id)
        baseline_data = None
        if active_baseline:
            baseline_data = {
                "id": active_baseline.id,
                "version": active_baseline.version,
                "created_at": active_baseline.created_at,
            }

    rprint(f"\n[bold cyan]Model: {model_data['name']}[/bold cyan]")
    rprint(f"  ID: {model_data['id']}")
    rprint(f"  Version: {model_data['version']}")
    rprint(f"  Framework: {model_data['framework'] or 'Not specified'}")
    rprint(f"  Type: {model_data['model_type'] or 'Not specified'}")
    rprint(f"  Active: {model_data['is_active']}")
    rprint(f"  Created: {model_data['created_at']}")

    if model_data["deployed_at"]:
        rprint(f"  Deployed: {model_data['deployed_at']}")

    if model_data["artifact_path"]:
        rprint(f"  Artifact: {model_data['artifact_path']}")

    if model_data["feature_names"]:
        rprint(f"\n[bold]Features ({len(model_data['feature_names'])}):[/bold]")
        for f in model_data["feature_names"][:10]:
            rprint(f"  - {f}")
        if len(model_data["feature_names"]) > 10:
            rprint(f"  ... and {len(model_data['feature_names']) - 10} more")

    if baseline_data:
        rprint(f"\n[bold]Active Baseline:[/bold]")
        rprint(f"  ID: {baseline_data['id'][:8]}...")
        rprint(f"  Version: {baseline_data['version']}")
        rprint(f"  Created: {baseline_data['created_at']}")


@app.command("deactivate")
def deactivate_model(
    model_id: str = typer.Argument(..., help="Model ID to deactivate"),
):
    """Deactivate a model."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.model_repo import ModelRepository

    db = get_database()
    with db.session() as session:
        repo = ModelRepository(session)
        model = repo.deactivate(model_id)

    if not model:
        rprint(f"[red]Model not found: {model_id}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]Model {model_id[:8]}... deactivated[/green]")


@app.command("delete")
def delete_model(
    model_id: str = typer.Argument(..., help="Model ID to delete"),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Force deletion without confirmation",
    ),
):
    """Delete a model."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.model_repo import ModelRepository

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete model {model_id[:8]}...?")
        if not confirm:
            rprint("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    db = get_database()
    with db.session() as session:
        repo = ModelRepository(session)
        deleted = repo.delete(model_id)

    if not deleted:
        rprint(f"[red]Model not found: {model_id}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]Model {model_id[:8]}... deleted[/green]")
