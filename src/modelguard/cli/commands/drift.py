"""Drift detection commands."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(help="Drift detection commands")
console = Console()


@app.command("check")
def check_drift(
    baseline_id: str = typer.Argument(..., help="Baseline ID to compare against"),
    data_path: str = typer.Argument(..., help="Path to current data (CSV or Parquet)"),
    features: Optional[str] = typer.Option(
        None, "--features", "-f",
        help="Comma-separated list of features to check",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Path to save report JSON",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Show detailed results",
    ),
):
    """Check for drift in current data against a baseline."""
    import pandas as pd

    from modelguard.drift.detector import DriftDetector
    from modelguard.severity.scorer import SeverityScorer
    from modelguard.actions.recommender import ActionRecommender
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.baseline_repo import BaselineRepository
    from modelguard.core.types import Baseline, FeatureStatistics, PredictionStatistics

    # Load baseline
    rprint(f"[cyan]Loading baseline {baseline_id}...[/cyan]")
    db = get_database()
    with db.session() as session:
        repo = BaselineRepository(session)
        baseline_record = repo.get(baseline_id)

    if not baseline_record:
        rprint(f"[red]Baseline not found: {baseline_id}[/red]")
        raise typer.Exit(1)

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
        class_distribution=pred_stats_dict.get("class_distribution"),
        probability_mean=pred_stats_dict.get("probability_mean"),
        probability_std=pred_stats_dict.get("probability_std"),
        mean=pred_stats_dict.get("mean"),
        std=pred_stats_dict.get("std"),
        percentiles=pred_stats_dict.get("percentiles"),
    )

    baseline = Baseline(
        id=baseline_record.id,
        model_id=baseline_record.model_id,
        created_at=baseline_record.created_at,
        feature_statistics=feature_stats,
        prediction_statistics=pred_stats,
        sample_size=baseline_record.sample_size or 0,
    )

    # Load current data
    rprint(f"[cyan]Loading current data from {data_path}...[/cyan]")
    if data_path.endswith(".parquet"):
        current_data = pd.read_parquet(data_path)
    else:
        current_data = pd.read_csv(data_path)

    rprint(f"  Loaded {len(current_data)} samples")

    # Filter features if specified
    feature_list = None
    if features:
        feature_list = [f.strip() for f in features.split(",")]

    # Run drift detection
    rprint("[cyan]Running drift detection...[/cyan]")
    detector = DriftDetector()
    report = detector.detect(baseline, current_data, features=feature_list)

    # Calculate severity
    scorer = SeverityScorer()
    severity = scorer.score_report(report)
    report.severity = severity

    # Get recommendation
    recommender = ActionRecommender()
    recommendation = recommender.recommend(severity)
    report.recommendation = recommendation

    # Display results
    rprint(f"\n[bold]Drift Report[/bold]")
    rprint(f"  Report ID: {report.id}")
    rprint(f"  Timestamp: {report.timestamp}")

    # Overall status
    if report.data_drift_detected:
        rprint(f"  [red]Data Drift Detected[/red]")
    else:
        rprint(f"  [green]No Significant Drift[/green]")

    rprint(f"  Features with drift: {len(report.features_with_drift)} / {len(report.feature_results)}")
    rprint(f"  Drift percentage: {report.drift_percentage:.1f}%")

    # Severity
    severity_color = {
        "none": "green",
        "low": "yellow",
        "medium": "orange3",
        "high": "red",
        "critical": "bold red",
    }.get(severity.level.value, "white")

    rprint(f"\n[bold]Severity Assessment[/bold]")
    rprint(f"  Level: [{severity_color}]{severity.level.value.upper()}[/{severity_color}]")
    rprint(f"  Score: {severity.overall_score:.2f}")
    rprint(f"  Confidence: {severity.confidence:.2f}")

    # Recommendation
    rprint(f"\n[bold]Recommendation[/bold]")
    rprint(f"  Action: [bold]{recommendation.action.value.upper()}[/bold]")
    rprint(f"  Urgency: {recommendation.urgency.value}")
    rprint(f"  Confidence: {recommendation.confidence:.2f}")

    if recommendation.reasoning:
        rprint("  Reasoning:")
        for reason in recommendation.reasoning:
            rprint(f"    - {reason}")

    # Feature details
    if verbose and report.features_with_drift:
        rprint(f"\n[bold]Drifted Features[/bold]")

        table = Table()
        table.add_column("Feature")
        table.add_column("Methods")
        table.add_column("Max Statistic")
        table.add_column("P-Value")

        for feature in report.features_with_drift:
            results = report.feature_results.get(feature, [])
            detected_methods = [r.method_name for r in results if r.drift_detected]
            max_stat = max((r.statistic for r in results), default=0)
            min_pval = min((r.p_value for r in results if r.p_value), default=None)

            table.add_row(
                feature,
                ", ".join(detected_methods),
                f"{max_stat:.4f}",
                f"{min_pval:.4f}" if min_pval else "-",
            )

        console.print(table)

    # Save report if requested
    if output:
        with open(output, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        rprint(f"\n[green]Report saved to {output}[/green]")


@app.command("report")
def show_report(
    report_id: str = typer.Argument(..., help="Report ID to show"),
):
    """Show a drift report."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.drift_repo import DriftReportRepository

    db = get_database()
    with db.session() as session:
        repo = DriftReportRepository(session)
        report = repo.get(report_id)

    if not report:
        rprint(f"[red]Report not found: {report_id}[/red]")
        raise typer.Exit(1)

    rprint(f"\n[bold cyan]Drift Report: {report.id}[/bold cyan]")
    rprint(f"  Model ID: {report.model_id}")
    rprint(f"  Baseline ID: {report.baseline_id}")
    rprint(f"  Timestamp: {report.timestamp}")
    rprint(f"  Data Drift: {'Yes' if report.data_drift_detected else 'No'}")
    rprint(f"  Prediction Drift: {'Yes' if report.prediction_drift_detected else 'No'}")
    rprint(f"  Features with Drift: {report.features_with_drift}")
    rprint(f"  Drift Percentage: {report.drift_percentage:.1f}%")

    if report.severity:
        rprint(f"\n[bold]Severity:[/bold]")
        rprint(f"  Level: {report.severity.get('level', 'unknown')}")
        rprint(f"  Score: {report.severity.get('overall_score', 0):.2f}")

    if report.recommendation:
        rprint(f"\n[bold]Recommendation:[/bold]")
        rprint(f"  Action: {report.recommendation.get('action', 'unknown')}")


@app.command("history")
def drift_history(
    model_id: str = typer.Argument(..., help="Model ID to show history for"),
    limit: int = typer.Option(
        10, "--limit", "-n",
        help="Number of reports to show",
    ),
):
    """Show drift detection history for a model."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.drift_repo import DriftReportRepository

    db = get_database()
    with db.session() as session:
        repo = DriftReportRepository(session)
        reports = repo.list_for_model(model_id, limit=limit)

    if not reports:
        rprint(f"[yellow]No drift reports found for model {model_id}[/yellow]")
        return

    table = Table(title=f"Drift History for {model_id[:12]}...")
    table.add_column("ID", style="cyan")
    table.add_column("Timestamp")
    table.add_column("Drift %")
    table.add_column("Severity")
    table.add_column("Action")

    for r in reports:
        severity = r.severity.get("level", "-") if r.severity else "-"
        action = r.recommendation.get("action", "-") if r.recommendation else "-"

        table.add_row(
            r.id[:8] + "...",
            r.timestamp.strftime("%Y-%m-%d %H:%M") if r.timestamp else "-",
            f"{r.drift_percentage:.1f}%",
            severity,
            action,
        )

    console.print(table)
