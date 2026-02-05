"""Baseline management commands."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(help="Baseline management commands")
console = Console()


@app.command("create")
def create_baseline(
    model_id: str = typer.Argument(..., help="Model ID to create baseline for"),
    data_path: str = typer.Argument(..., help="Path to training data (CSV or Parquet)"),
    predictions_path: Optional[str] = typer.Option(
        None, "--predictions", "-p",
        help="Path to predictions file",
    ),
    labels_path: Optional[str] = typer.Option(
        None, "--labels", "-l",
        help="Path to labels file",
    ),
    prediction_type: str = typer.Option(
        "classification", "--type", "-t",
        help="Prediction type: classification or regression",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Path to save baseline JSON",
    ),
):
    """Create a new baseline from training data."""
    import pandas as pd
    import numpy as np

    from modelguard.baseline.creator import BaselineCreator
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.baseline_repo import BaselineRepository

    # Load data
    rprint(f"[cyan]Loading data from {data_path}...[/cyan]")
    if data_path.endswith(".parquet"):
        data = pd.read_parquet(data_path)
    else:
        data = pd.read_csv(data_path)

    rprint(f"  Loaded {len(data)} samples with {len(data.columns)} features")

    # Load predictions
    predictions = None
    if predictions_path:
        rprint(f"[cyan]Loading predictions from {predictions_path}...[/cyan]")
        predictions = np.load(predictions_path) if predictions_path.endswith(".npy") else pd.read_csv(predictions_path).values.flatten()
    else:
        # Generate dummy predictions for baseline
        rprint("[yellow]No predictions provided, using placeholder values[/yellow]")
        if prediction_type == "classification":
            predictions = np.zeros(len(data))
        else:
            predictions = np.zeros(len(data))

    # Load labels
    labels = None
    if labels_path:
        rprint(f"[cyan]Loading labels from {labels_path}...[/cyan]")
        labels = np.load(labels_path) if labels_path.endswith(".npy") else pd.read_csv(labels_path).values.flatten()

    # Create baseline
    rprint("[cyan]Creating baseline...[/cyan]")
    creator = BaselineCreator()
    baseline = creator.create(
        model_id=model_id,
        training_data=data,
        predictions=predictions,
        labels=labels,
        prediction_type=prediction_type,
    )

    rprint(f"[green]Baseline created:[/green] {baseline.id}")
    rprint(f"  Features: {len(baseline.feature_statistics)}")
    rprint(f"  Sample size: {baseline.sample_size}")

    # Save to database
    db = get_database()
    with db.session() as session:
        repo = BaselineRepository(session)
        repo.create(
            model_id=model_id,
            feature_statistics={k: v.to_dict() for k, v in baseline.feature_statistics.items()},
            prediction_statistics=baseline.prediction_statistics.to_dict(),
            performance_metrics=baseline.performance_metrics.to_dict() if baseline.performance_metrics else None,
            sample_size=baseline.sample_size,
        )
    rprint("[green]Baseline saved to database[/green]")

    # Save to file if requested
    if output:
        with open(output, "w") as f:
            json.dump(baseline.to_dict(), f, indent=2, default=str)
        rprint(f"[green]Baseline saved to {output}[/green]")


@app.command("list")
def list_baselines(
    model_id: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Filter by model ID",
    ),
    limit: int = typer.Option(
        20, "--limit", "-n",
        help="Maximum number of baselines to show",
    ),
):
    """List all baselines."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.baseline_repo import BaselineRepository

    db = get_database()
    with db.session() as session:
        repo = BaselineRepository(session)

        if model_id:
            baselines = repo.list_for_model(model_id, limit=limit)
        else:
            # List all baselines
            from modelguard.storage.models import BaselineRecord
            baselines = session.query(BaselineRecord).order_by(
                BaselineRecord.created_at.desc()
            ).limit(limit).all()

    if not baselines:
        rprint("[yellow]No baselines found[/yellow]")
        return

    table = Table(title="Baselines")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Model ID")
    table.add_column("Version")
    table.add_column("Active", justify="center")
    table.add_column("Features")
    table.add_column("Samples")
    table.add_column("Created")

    for b in baselines:
        table.add_row(
            b.id[:8] + "...",
            b.model_id[:8] + "..." if b.model_id else "-",
            str(b.version),
            "[green]Yes[/green]" if b.is_active else "[dim]No[/dim]",
            str(len(b.feature_statistics)) if b.feature_statistics else "-",
            str(b.sample_size) if b.sample_size else "-",
            b.created_at.strftime("%Y-%m-%d %H:%M") if b.created_at else "-",
        )

    console.print(table)


@app.command("show")
def show_baseline(
    baseline_id: str = typer.Argument(..., help="Baseline ID to show"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed statistics",
    ),
):
    """Show details of a specific baseline."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.baseline_repo import BaselineRepository

    db = get_database()
    with db.session() as session:
        repo = BaselineRepository(session)
        baseline = repo.get(baseline_id)

    if not baseline:
        rprint(f"[red]Baseline not found: {baseline_id}[/red]")
        raise typer.Exit(1)

    rprint(f"\n[bold cyan]Baseline: {baseline.id}[/bold cyan]")
    rprint(f"  Model ID: {baseline.model_id}")
    rprint(f"  Version: {baseline.version}")
    rprint(f"  Active: {baseline.is_active}")
    rprint(f"  Sample Size: {baseline.sample_size}")
    rprint(f"  Created: {baseline.created_at}")

    if baseline.feature_statistics:
        rprint(f"\n[bold]Features ({len(baseline.feature_statistics)}):[/bold]")

        table = Table()
        table.add_column("Feature")
        table.add_column("Type")
        table.add_column("Non-Null")
        table.add_column("Key Stats")

        for name, stats in baseline.feature_statistics.items():
            if stats.get("dtype") == "numerical":
                key_stats = f"mean={stats.get('mean', 0):.2f}, std={stats.get('std', 0):.2f}"
            else:
                key_stats = f"unique={stats.get('unique_count', 0)}"

            table.add_row(
                name,
                stats.get("dtype", "-"),
                f"{(1 - stats.get('null_ratio', 0)) * 100:.1f}%",
                key_stats,
            )

        console.print(table)

    if baseline.performance_metrics and verbose:
        rprint("\n[bold]Performance Metrics:[/bold]")
        for key, value in baseline.performance_metrics.items():
            if value is not None:
                rprint(f"  {key}: {value}")
