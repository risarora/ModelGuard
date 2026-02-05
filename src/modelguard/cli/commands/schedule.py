"""Scheduled job management commands."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(help="Scheduled monitoring job commands")
console = Console()


@app.command("list")
def list_jobs(
    model_id: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Filter by model ID",
    ),
    active_only: bool = typer.Option(
        False, "--active", "-a",
        help="Show only active jobs",
    ),
):
    """List scheduled monitoring jobs."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.job_repo import ScheduledJobRepository

    db = get_database()
    with db.session() as session:
        repo = ScheduledJobRepository(session)
        jobs = repo.list_all(model_id=model_id, active_only=active_only)

        job_data = [
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
                "last_run_status": j.last_run_status,
                "run_count": j.run_count,
            }
            for j in jobs
        ]

    if not job_data:
        rprint("[yellow]No scheduled jobs found[/yellow]")
        return

    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Schedule")
    table.add_column("Active")
    table.add_column("Last Run")
    table.add_column("Status")
    table.add_column("Runs")

    for j in job_data:
        schedule = ""
        if j["schedule_type"] == "interval":
            schedule = f"Every {j['interval_minutes']}m"
        elif j["schedule_type"] == "cron":
            schedule = j["cron_expression"]

        last_run = j["last_run_at"].strftime("%Y-%m-%d %H:%M") if j["last_run_at"] else "-"

        status_display = j["last_run_status"] or "-"
        if j["last_run_status"] == "success":
            status_display = "[green]success[/green]"
        elif j["last_run_status"] == "failed":
            status_display = "[red]failed[/red]"

        table.add_row(
            j["id"][:8] + "...",
            j["name"],
            j["job_type"],
            schedule,
            "[green]Yes[/green]" if j["is_active"] else "[dim]No[/dim]",
            last_run,
            status_display,
            str(j["run_count"] or 0),
        )

    console.print(table)


@app.command("create")
def create_job(
    name: str = typer.Argument(..., help="Job name"),
    model_id: str = typer.Argument(..., help="Model ID to monitor"),
    schedule_type: str = typer.Option(
        "interval", "--type", "-t",
        help="Schedule type: interval or cron",
    ),
    interval: int = typer.Option(
        60, "--interval", "-i",
        help="Interval in minutes (for interval type)",
    ),
    cron: Optional[str] = typer.Option(
        None, "--cron", "-c",
        help="Cron expression (for cron type)",
    ),
    job_type: str = typer.Option(
        "drift_check", "--job-type", "-j",
        help="Job type: drift_check or performance_check",
    ),
    data_path: Optional[str] = typer.Option(
        None, "--data-path", "-d",
        help="Path to production data file",
    ),
    notify: bool = typer.Option(
        True, "--notify/--no-notify",
        help="Create alerts on drift detection",
    ),
):
    """Create a new scheduled monitoring job."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.job_repo import ScheduledJobRepository
    from modelguard.storage.repositories.model_repo import ModelRepository

    # Validate schedule
    if schedule_type == "cron" and not cron:
        rprint("[red]Cron expression required for cron schedule type[/red]")
        raise typer.Exit(1)

    db = get_database()
    with db.session() as session:
        # Verify model exists
        model_repo = ModelRepository(session)
        model = model_repo.get(model_id)
        if not model:
            rprint(f"[red]Model not found: {model_id}[/red]")
            raise typer.Exit(1)

        # Check for duplicate name
        job_repo = ScheduledJobRepository(session)
        existing = job_repo.get_by_name(name)
        if existing:
            rprint(f"[red]Job with name '{name}' already exists[/red]")
            raise typer.Exit(1)

        # Build data source config
        data_source_type = None
        data_source_config = None
        if data_path:
            data_source_type = "file"
            data_source_config = {
                "path": data_path,
                "format": "csv" if data_path.endswith(".csv") else "parquet",
            }

        # Create job
        job = job_repo.create(
            name=name,
            job_type=job_type,
            model_id=model_id,
            schedule_type=schedule_type,
            interval_minutes=interval if schedule_type == "interval" else None,
            cron_expression=cron if schedule_type == "cron" else None,
            data_source_type=data_source_type,
            data_source_config=data_source_config,
            notify_on_drift=notify,
            created_by="cli-user",
        )
        job_id = job.id

    rprint(f"[green]Scheduled job created successfully![/green]")
    rprint(f"  ID: {job_id}")
    rprint(f"  Name: {name}")
    rprint(f"  Model: {model_id[:8]}...")
    if schedule_type == "interval":
        rprint(f"  Schedule: Every {interval} minutes")
    else:
        rprint(f"  Schedule: {cron}")


@app.command("show")
def show_job(
    job_id: str = typer.Argument(..., help="Job ID to show"),
):
    """Show details of a scheduled job."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.job_repo import ScheduledJobRepository

    db = get_database()
    with db.session() as session:
        repo = ScheduledJobRepository(session)
        job = repo.get(job_id)

        if not job:
            rprint(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(1)

        job_data = job.to_dict()

    rprint(f"\n[bold cyan]Job: {job_data['name']}[/bold cyan]")
    rprint(f"  ID: {job_data['id']}")
    rprint(f"  Type: {job_data['job_type']}")
    rprint(f"  Model ID: {job_data['model_id']}")

    if job_data["baseline_id"]:
        rprint(f"  Baseline ID: {job_data['baseline_id']}")

    rprint(f"\n[bold]Schedule:[/bold]")
    rprint(f"  Type: {job_data['schedule_type']}")
    if job_data["schedule_type"] == "interval":
        rprint(f"  Interval: {job_data['interval_minutes']} minutes")
    else:
        rprint(f"  Cron: {job_data['cron_expression']}")

    rprint(f"\n[bold]Status:[/bold]")
    rprint(f"  Active: {job_data['is_active']}")
    rprint(f"  Run count: {job_data['run_count'] or 0}")
    if job_data["last_run_at"]:
        rprint(f"  Last run: {job_data['last_run_at']}")
        rprint(f"  Last status: {job_data['last_run_status']}")
    if job_data["next_run_at"]:
        rprint(f"  Next run: {job_data['next_run_at']}")
    if job_data["last_error"]:
        rprint(f"  [red]Last error: {job_data['last_error']}[/red]")

    if job_data["data_source_type"]:
        rprint(f"\n[bold]Data Source:[/bold]")
        rprint(f"  Type: {job_data['data_source_type']}")
        if job_data["data_source_config"]:
            for k, v in job_data["data_source_config"].items():
                rprint(f"  {k}: {v}")

    rprint(f"\n  Created: {job_data['created_at']}")
    if job_data["created_by"]:
        rprint(f"  Created by: {job_data['created_by']}")


@app.command("pause")
def pause_job(
    job_id: str = typer.Argument(..., help="Job ID to pause"),
):
    """Pause a scheduled job."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.job_repo import ScheduledJobRepository

    db = get_database()
    with db.session() as session:
        repo = ScheduledJobRepository(session)
        job = repo.deactivate(job_id)

        if not job:
            rprint(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(1)

    rprint(f"[green]Job {job_id[:8]}... paused[/green]")


@app.command("resume")
def resume_job(
    job_id: str = typer.Argument(..., help="Job ID to resume"),
):
    """Resume a paused job."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.job_repo import ScheduledJobRepository

    db = get_database()
    with db.session() as session:
        repo = ScheduledJobRepository(session)
        job = repo.activate(job_id)

        if not job:
            rprint(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(1)

    rprint(f"[green]Job {job_id[:8]}... resumed[/green]")


@app.command("run")
def run_job(
    job_id: str = typer.Argument(..., help="Job ID to run immediately"),
):
    """Run a job immediately."""
    from modelguard.monitoring.scheduler import get_scheduler

    try:
        scheduler = get_scheduler()
        scheduler.run_job_now(job_id)
        rprint(f"[green]Job {job_id[:8]}... executed[/green]")
    except ValueError as e:
        rprint(f"[red]{e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]Job execution failed: {e}[/red]")
        raise typer.Exit(1)


@app.command("delete")
def delete_job(
    job_id: str = typer.Argument(..., help="Job ID to delete"),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Force deletion without confirmation",
    ),
):
    """Delete a scheduled job."""
    from modelguard.storage.database import get_database
    from modelguard.storage.repositories.job_repo import ScheduledJobRepository

    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete job {job_id[:8]}...?")
        if not confirm:
            rprint("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    db = get_database()
    with db.session() as session:
        repo = ScheduledJobRepository(session)
        deleted = repo.delete(job_id)

        if not deleted:
            rprint(f"[red]Job not found: {job_id}[/red]")
            raise typer.Exit(1)

    rprint(f"[green]Job {job_id[:8]}... deleted[/green]")


@app.command("start-daemon")
def start_daemon(
    foreground: bool = typer.Option(
        False, "--foreground", "-f",
        help="Run in foreground (don't daemonize)",
    ),
):
    """Start the monitoring scheduler daemon."""
    from modelguard.monitoring.scheduler import get_scheduler
    import time

    rprint("[cyan]Starting monitoring scheduler...[/cyan]")

    scheduler = get_scheduler()
    scheduler.start()

    rprint("[green]Scheduler started. Monitoring jobs are now running.[/green]")

    if foreground:
        rprint("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            rprint("\n[yellow]Stopping scheduler...[/yellow]")
            scheduler.stop()
            rprint("[green]Scheduler stopped.[/green]")
    else:
        rprint("Scheduler is running in background.")
        rprint("Use 'modelguard schedule stop-daemon' to stop.")


@app.command("stop-daemon")
def stop_daemon():
    """Stop the monitoring scheduler daemon."""
    from modelguard.monitoring.scheduler import stop_scheduler

    stop_scheduler()
    rprint("[green]Scheduler stopped.[/green]")
