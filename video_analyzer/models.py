"""Internal data contract (see docs/GRAPHIFY.md, section 3)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SpeechSegment(BaseModel):
    start: float
    end: float
    text: str
    confidence: float | None = None
    language: str | None = None


class Keyframe(BaseModel):
    index: int
    timestamp: float
    file_path: str  # relative to the output directory, e.g. "keyframes/keyframe_001_01_30.jpg"
    resolution: str
    difference_score: float
    ocr_elements: list[str] = Field(default_factory=list)


class ActionStep(BaseModel):
    step_number: int
    start_time: float
    end_time: float
    action_title: str
    observed_action: str | None = None
    spoken_context: str = ""
    visual_state: Keyframe | None = None
    extracted_fields: dict[str, str] = Field(default_factory=dict)


class VideoSource(BaseModel):
    """A resolved input: a local video file plus optional ready-made captions."""

    video_path: Path | None  # None when only YouTube captions could be fetched
    title: str
    origin: str  # original path or URL as the user typed it
    is_youtube: bool = False
    captions: list[SpeechSegment] | None = None
    caption_language: str | None = None


class AnalysisResult(BaseModel):
    video_title: str
    source: str
    duration_seconds: float
    detected_language: str | None = None
    transcript_source: str = "whisper"  # "whisper" or "youtube-captions"
    transcript: list[SpeechSegment] = Field(default_factory=list)
    keyframes: list[Keyframe] = Field(default_factory=list)
    action_steps: list[ActionStep] = Field(default_factory=list)
    summary: str = ""
