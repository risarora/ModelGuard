"""API server commands."""

import typer
from rich import print as rprint

app = typer.Typer(help="API server commands")


@app.command("start")
def start_server(
    host: str = typer.Option(
        "0.0.0.0", "--host", "-h",
        help="Host to bind to",
    ),
    port: int = typer.Option(
        8000, "--port", "-p",
        help="Port to bind to",
    ),
    reload: bool = typer.Option(
        False, "--reload", "-r",
        help="Enable auto-reload for development",
    ),
    workers: int = typer.Option(
        1, "--workers", "-w",
        help="Number of worker processes",
    ),
):
    """Start the ModelGuard API server."""
    import uvicorn

    rprint(f"[cyan]Starting ModelGuard API server...[/cyan]")
    rprint(f"  Host: {host}")
    rprint(f"  Port: {port}")
    rprint(f"  Workers: {workers}")
    rprint(f"  Reload: {reload}")

    uvicorn.run(
        "modelguard.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )
