"""Pipeline settings shared by the CLI and the processing engines."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm"}
WHISPER_MODELS = {"tiny", "base", "small", "medium", "large-v3"}


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class OutputFormat(str, Enum):
    all = "all"
    agent_md = "agent-md"
    transcript = "transcript"
    summary = "summary"


class Settings(BaseModel):
    """Everything one `analyze` run needs. Defaults match docs/SPECIFICATIONS.md."""

    source: str
    output_dir: Path | None = None
    model: str = "tiny"
    device: Device = Device.auto
    language: str = "auto"
    keyframe_threshold: float = Field(0.35, ge=0.0, le=1.0)
    ocr: bool = True
    keep_temp: bool = False
    format: OutputFormat = OutputFormat.all
    fast: bool = False
    cookies_from_browser: str | None = None
    cookies_file: Path | None = None

    # Vision tuning that is not exposed as CLI flags yet.
    sample_fps: float = Field(1.0, gt=0.0)
    min_keyframe_gap: float = Field(2.0, ge=0.0)
    max_keyframes: int = Field(40, ge=1)
    keyframe_max_width: int = Field(1280, ge=160)
    jpeg_quality: int = Field(80, ge=10, le=100)

    # Keeps agent.md inside the token budget; the full speech is in transcript.txt.
    max_step_speech_chars: int = Field(600, ge=80)

    @property
    def language_hint(self) -> str | None:
        return None if self.language == "auto" else self.language

    @property
    def run_ocr(self) -> bool:
        return self.ocr and not self.fast

    def wants(self, artifact: OutputFormat) -> bool:
        return self.format in (OutputFormat.all, artifact)
