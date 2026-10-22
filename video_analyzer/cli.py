"""Command-line interface: `video-analyzer analyze <path_or_url>`."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from video_analyzer import __version__, pipeline
from video_analyzer.config import WHISPER_MODELS, Device, OutputFormat, Settings
from video_analyzer.engines.vision import format_timestamp
from video_analyzer.handlers.local_handler import InvalidVideoError
from video_analyzer.handlers.youtube_handler import YouTubeDownloadError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn screen recordings and YouTube videos into token-efficient agent.md briefs, locally.",
)
console = Console()


class RichReporter(pipeline.Reporter):
    def __init__(self, progress: Progress) -> None:
        self.progress = progress
        self.tasks: dict[str, TaskID] = {}
        self.lock = threading.Lock()

    def start(self, task: str, description: str, total: float | None = None) -> None:
        with self.lock:
            self.tasks[task] = self.progress.add_task(description, total=total)

    def update(self, task: str, completed: float, total: float | None = None) -> None:
        with self.lock:
            if task in self.tasks:
                kwargs = {"total": total} if total else {}
                self.progress.update(self.tasks[task], completed=completed, **kwargs)

    def finish(self, task: str) -> None:
        with self.lock:
            if task in self.tasks:
                task_id = self.tasks[task]
                total = self.progress.tasks[task_id].total or 1
                self.progress.update(task_id, total=total, completed=total)

    def info(self, message: str) -> None:
        self.progress.console.print(f"[dim]•[/dim] {escape(message)}")


def _print_version(value: bool) -> None:
    if value:
        console.print(f"video-analyzer {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the version and exit.",
        is_eager=True,
        callback=_print_version,
    ),
) -> None:
    # Legacy Windows code pages (cp1252) cannot encode rich's spinner glyphs or non-Latin titles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def analyze(
    input: str = typer.Argument(..., help="Local video file path or YouTube URL."),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Destination directory [default: ./output/{video_name}]."
    ),
    model: str = typer.Option("tiny", "--model", "-m", help="Whisper model size."),
    device: Device = typer.Option(Device.auto, "--device", "-d", help="Compute device."),
    language: str = typer.Option(
        "auto", "--language", "-l", help="Spoken language (en, ur, hi, es...)."
    ),
    keyframe_threshold: float = typer.Option(
        0.35,
        "--keyframe-threshold",
        min=0.0,
        max=1.0,
        help="Screen-change sensitivity, 0-1 (lower = more keyframes).",
    ),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="Extract on-screen text from keyframes."),
    keep_temp: bool = typer.Option(
        False, "--keep-temp", help="Keep intermediate audio and downloads."
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.all, "--format", help="Which artifacts to write."
    ),
    fast: bool = typer.Option(
        False, "--fast", help="Captions/transcript + keyframes only, no OCR."
    ),
    cookies_from_browser: str | None = typer.Option(
        None,
        "--cookies-from-browser",
        help="YouTube only: reuse this browser's login (firefox, chrome, edge...).",
    ),
    cookies: Path | None = typer.Option(
        None, "--cookies", help="YouTube only: Netscape-format cookies.txt file."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logging."),
) -> None:
    """Analyze a video and write agent.md, summary.md, transcripts and keyframes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    if model not in WHISPER_MODELS:
        raise typer.BadParameter(
            f"choose one of {', '.join(sorted(WHISPER_MODELS))}", param_hint="--model"
        )

    settings = Settings(
        source=input,
        output_dir=output_dir,
        model=model,
        device=device,
        language=language,
        keyframe_threshold=keyframe_threshold,
        ocr=ocr,
        keep_temp=keep_temp,
        format=format,
        fast=fast,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )
    try:
        with progress:
            output = pipeline.run(settings, RichReporter(progress))
    except (InvalidVideoError, YouTubeDownloadError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2) from None
    except Exception as exc:
        if verbose:
            raise
        console.print(
            f"[red]Analysis failed:[/red] {escape(str(exc))}  (re-run with --verbose for details)"
        )
        raise typer.Exit(code=1) from None

    _print_report(output)


def _print_report(output: pipeline.PipelineOutput) -> None:
    result = output.result
    table = Table(title=f"[bold]{result.video_title}[/bold]", show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Duration", format_timestamp(result.duration_seconds))
    table.add_row("Language", result.detected_language or "unknown")
    table.add_row("Transcript", f"{len(result.transcript)} segments ({result.transcript_source})")
    table.add_row("Keyframes", str(len(result.keyframes)))
    table.add_row("Steps", str(len(result.action_steps)))
    table.add_row("Est. tokens", f"~{output.token_estimate:,} (text artifacts)")
    table.add_row("Output", str(output.output_dir.resolve()))
    for name in output.files:
        table.add_row("", f"  {name}")
    if output.temp_dir:
        table.add_row("Temp kept", str(output.temp_dir))
    console.print(table)


if __name__ == "__main__":
    app()
