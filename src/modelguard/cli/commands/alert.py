"""Alert management commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(help="Alert management commands")
console = Console()


@app.command("list")
def list_alerts(
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by status (pending, acknowledged, resolved, dismissed)",
    ),
    model_id: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Filter by model ID",
    ),
    limit: int = typer.Option(
        20, "--limit", "-n",
        help="Maximum number of alerts to show",
    ),
):
    """List alerts."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.alert_repo import AlertRepository

    db = get_database()
    with db.session() as session:
        repo = AlertRepository(session)

        if status:
            alerts = repo.list_by_status(status, model_id=model_id, limit=limit)
        elif model_id:
            alerts = repo.list_for_model(model_id, limit=limit)
        else:
            alerts = repo.list_pending(limit=limit)

        # Extract data within session to avoid DetachedInstanceError
        alert_data = [
            {
                "id": a.id,
                "model_id": a.model_id,
                "severity": a.severity,
                "urgency": a.urgency,
                "status": a.status,
                "created_at": a.created_at,
                "assigned_to": a.assigned_to,
            }
            for a in alerts
        ]

    if not alert_data:
        rprint("[yellow]No alerts found[/yellow]")
        return

    table = Table(title="Alerts")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Model")
    table.add_column("Severity")
    table.add_column("Urgency")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Assigned")

    severity_colors = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "green",
        "none": "dim",
    }

    for a in alert_data:
        severity_color = severity_colors.get(a["severity"], "white")
        status_display = a["status"]
        if a["status"] == "pending":
            status_display = "[yellow]pending[/yellow]"
        elif a["status"] == "resolved":
            status_display = "[green]resolved[/green]"

        table.add_row(
            a["id"][:8] + "...",
            a["model_id"][:8] + "..." if a["model_id"] else "-",
            f"[{severity_color}]{a['severity']}[/{severity_color}]",
            a["urgency"],
            status_display,
            a["created_at"].strftime("%Y-%m-%d %H:%M") if a["created_at"] else "-",
            a["assigned_to"] or "-",
        )

    console.print(table)


@app.command("show")
def show_alert(
    alert_id: str = typer.Argument(..., help="Alert ID to show"),
):
    """Show details of a specific alert."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.alert_repo import AlertRepository

    db = get_database()
    with db.session() as session:
        repo = AlertRepository(session)
        alert = repo.get(alert_id)

        if not alert:
            rprint(f"[red]Alert not found: {alert_id}[/red]")
            raise typer.Exit(1)

        # Extract data within session to avoid DetachedInstanceError
        alert_data = {
            "id": alert.id,
            "model_id": alert.model_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "urgency": alert.urgency,
            "status": alert.status,
            "created_at": alert.created_at,
            "assigned_to": alert.assigned_to,
            "drift_summary": alert.drift_summary,
            "recommendation": alert.recommendation,
            "decision": alert.decision,
            "decided_by": alert.decided_by,
            "resolved_at": alert.resolved_at,
            "decision_notes": alert.decision_notes,
        }

    severity_colors = {
        "critical": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "green",
        "none": "dim",
    }
    severity_color = severity_colors.get(alert_data["severity"], "white")

    rprint(f"\n[bold cyan]Alert: {alert_data['id']}[/bold cyan]")
    rprint(f"  Model ID: {alert_data['model_id']}")
    rprint(f"  Type: {alert_data['alert_type']}")
    rprint(f"  Severity: [{severity_color}]{alert_data['severity'].upper()}[/{severity_color}]")
    rprint(f"  Urgency: {alert_data['urgency']}")
    rprint(f"  Status: {alert_data['status']}")
    rprint(f"  Created: {alert_data['created_at']}")

    if alert_data["assigned_to"]:
        rprint(f"  Assigned to: {alert_data['assigned_to']}")

    if alert_data["drift_summary"]:
        rprint(f"\n[bold]Drift Summary:[/bold]")
        for key, value in alert_data["drift_summary"].items():
            rprint(f"  {key}: {value}")

    if alert_data["recommendation"]:
        rprint(f"\n[bold]Recommendation:[/bold]")
        rprint(f"  Action: {alert_data['recommendation'].get('action', 'unknown')}")
        if alert_data["recommendation"].get('reasoning'):
            rprint("  Reasoning:")
            for reason in alert_data["recommendation"]['reasoning']:
                rprint(f"    - {reason}")

    if alert_data["decision"]:
        rprint(f"\n[bold]Resolution:[/bold]")
        rprint(f"  Decision: {alert_data['decision']}")
        rprint(f"  Decided by: {alert_data['decided_by']}")
        rprint(f"  Resolved at: {alert_data['resolved_at']}")
        if alert_data["decision_notes"]:
            rprint(f"  Notes: {alert_data['decision_notes']}")


@app.command("resolve")
def resolve_alert(
    alert_id: str = typer.Argument(..., help="Alert ID to resolve"),
    decision: str = typer.Argument(..., help="Decision: ignore, monitor, retrain, rollback"),
    user: str = typer.Option(
        "cli-user", "--user", "-u",
        help="User making the decision",
    ),
    notes: Optional[str] = typer.Option(
        None, "--notes", "-n",
        help="Decision notes",
    ),
):
    """Resolve an alert with a decision."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.alert_repo import AlertRepository

    valid_decisions = ["ignore", "monitor", "retrain", "rollback"]
    if decision not in valid_decisions:
        rprint(f"[red]Invalid decision. Must be one of: {', '.join(valid_decisions)}[/red]")
        raise typer.Exit(1)

    db = get_database()
    with db.session() as session:
        repo = AlertRepository(session)
        alert = repo.resolve(
            alert_id=alert_id,
            decision=decision,
            decided_by=user,
            decision_notes=notes,
        )

    if not alert:
        rprint(f"[red]Alert not found: {alert_id}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]Alert {alert_id[:8]}... resolved with decision: {decision}[/green]")


@app.command("dismiss")
def dismiss_alert(
    alert_id: str = typer.Argument(..., help="Alert ID to dismiss"),
    user: str = typer.Option(
        "cli-user", "--user", "-u",
        help="User dismissing the alert",
    ),
    reason: Optional[str] = typer.Option(
        None, "--reason", "-r",
        help="Reason for dismissal",
    ),
):
    """Dismiss an alert."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.alert_repo import AlertRepository

    db = get_database()
    with db.session() as session:
        repo = AlertRepository(session)
        alert = repo.dismiss(
            alert_id=alert_id,
            decided_by=user,
            reason=reason,
        )

    if not alert:
        rprint(f"[red]Alert not found: {alert_id}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]Alert {alert_id[:8]}... dismissed[/green]")


@app.command("assign")
def assign_alert(
    alert_id: str = typer.Argument(..., help="Alert ID to assign"),
    user: str = typer.Argument(..., help="User to assign to"),
):
    """Assign an alert to a user."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.alert_repo import AlertRepository

    db = get_database()
    with db.session() as session:
        repo = AlertRepository(session)
        alert = repo.assign(alert_id, user)

    if not alert:
        rprint(f"[red]Alert not found: {alert_id}[/red]")
        raise typer.Exit(1)

    rprint(f"[green]Alert {alert_id[:8]}... assigned to {user}[/green]")
