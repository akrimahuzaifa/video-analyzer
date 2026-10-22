"""Render agent.md, summary.md and the transcript files."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined

from video_analyzer import __version__
from video_analyzer.engines.vision import format_timestamp
from video_analyzer.models import AnalysisResult
from video_analyzer.synthesis.aligner import ProcessContext


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token for English text)."""
    return max(1, len(text) // 4) if text else 0


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("video_analyzer", "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
        autoescape=False,  # Markdown output, not HTML
    )
    env.filters["ts"] = format_timestamp
    env.filters["cell"] = _md_cell
    return env


def render_transcript_txt(result: AnalysisResult) -> str:
    lines = [
        f"[{format_timestamp(s.start)} - {format_timestamp(s.end)}] {s.text}"
        for s in result.transcript
    ]
    return "\n".join(lines) + "\n" if lines else ""


def render_transcript_json(result: AnalysisResult) -> str:
    payload = {
        "video_title": result.video_title,
        "source": result.source,
        "duration_seconds": result.duration_seconds,
        "language": result.detected_language,
        "transcript_source": result.transcript_source,
        "segments": [s.model_dump() for s in result.transcript],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render_agent_md(result: AnalysisResult, context: ProcessContext) -> str:
    return (
        _environment()
        .get_template("agent.md.j2")
        .render(result=result, context=context, version=__version__)
    )


def render_summary_md(result: AnalysisResult, context: ProcessContext) -> str:
    return (
        _environment()
        .get_template("summary.md.j2")
        .render(result=result, context=context, version=__version__)
    )


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path
