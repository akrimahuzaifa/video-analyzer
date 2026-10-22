"""End-to-end runs on the synthetic video, with Whisper replaced by a stub."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from video_analyzer import pipeline
from video_analyzer.cli import app
from video_analyzer.config import OutputFormat, Settings
from video_analyzer.engines import asr, audio
from video_analyzer.models import SpeechSegment, VideoSource


@pytest.fixture
def fake_speech(monkeypatch):
    segments = [
        SpeechSegment(start=1, end=3, text="This is the dashboard.", language="en"),
        SpeechSegment(start=5, end=7, text="Open the billing page.", language="en"),
        SpeechSegment(start=9, end=11, text="Click submit to confirm the payment.", language="en"),
    ]
    monkeypatch.setattr(audio, "has_audio_stream", lambda path: True)
    monkeypatch.setattr(audio, "extract_audio", lambda video, out: out)
    monkeypatch.setattr(
        asr, "transcribe", lambda *a, **k: asr.TranscriptionResult(segments, "en", 19.0)
    )
    return segments


def test_pipeline_writes_all_artifacts(synthetic_video, tmp_path, fake_speech):
    out = tmp_path / "out"
    output = pipeline.run(Settings(source=str(synthetic_video), output_dir=out, ocr=False))

    for name in ["agent.md", "summary.md", "transcript.txt", "transcript.json"]:
        assert (out / name).is_file(), name
    metadata = json.loads((out / "keyframes" / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata) == len(output.result.keyframes) == 4

    agent_md = (out / "agent.md").read_text(encoding="utf-8")
    assert "Agent Execution Brief: billing_demo" in agent_md
    for keyframe in output.result.keyframes:
        assert f"./{keyframe.file_path}" in agent_md
        assert (out / keyframe.file_path).is_file()
    assert output.result.detected_language == "en"
    assert 0 < output.token_estimate < 10_000


def test_pipeline_transcript_only_skips_vision(synthetic_video, tmp_path, fake_speech):
    out = tmp_path / "out"
    output = pipeline.run(
        Settings(source=str(synthetic_video), output_dir=out, format=OutputFormat.transcript)
    )
    assert set(output.files) == {"transcript.txt", "transcript.json"}
    assert not (out / "keyframes").exists()


def test_pipeline_silent_video(synthetic_video, tmp_path):
    # The synthetic video has no audio track, so ASR must be skipped, not crash.
    output = pipeline.run(Settings(source=str(synthetic_video), output_dir=tmp_path, ocr=False))
    assert output.result.transcript == []
    assert output.result.transcript_source == "none"
    assert len(output.result.action_steps) == 4


def test_keep_temp(synthetic_video, tmp_path, fake_speech):
    output = pipeline.run(
        Settings(source=str(synthetic_video), output_dir=tmp_path, ocr=False, keep_temp=True)
    )
    assert output.temp_dir == tmp_path / "_temp" and output.temp_dir.is_dir()


def test_cli_analyze(synthetic_video, tmp_path, fake_speech):
    result = CliRunner().invoke(
        app, ["analyze", str(synthetic_video), "-o", str(tmp_path), "--no-ocr"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "agent.md").is_file()
    assert "Keyframes" in result.output


def test_cli_rejects_missing_file(tmp_path):
    result = CliRunner().invoke(app, ["analyze", str(tmp_path / "missing.mp4")])
    assert result.exit_code == 2
    assert "not found" in result.output


def test_cli_rejects_unknown_model(synthetic_video):
    result = CliRunner().invoke(app, ["analyze", str(synthetic_video), "--model", "huge"])
    assert result.exit_code != 0


def test_pipeline_captions_only_source(monkeypatch, tmp_path):
    """A YouTube video whose download was blocked still yields a transcript-based brief."""
    captions = [
        SpeechSegment(start=0, end=3, text="Open the settings page.", language="en"),
        SpeechSegment(start=3, end=6, text="Then click save.", language="en"),
    ]
    source = VideoSource(
        video_path=None,
        title="Blocked video",
        origin="https://youtu.be/WoFCSwxMLKA",
        is_youtube=True,
        captions=captions,
        caption_language="en",
    )
    monkeypatch.setattr(pipeline, "resolve_source", lambda *a, **k: source)
    output = pipeline.run(Settings(source=source.origin, output_dir=tmp_path))

    assert output.result.transcript_source == "youtube-captions"
    assert output.result.keyframes == []
    assert len(output.result.action_steps) == 1
    assert "Agent Execution Brief: Blocked video" in (tmp_path / "agent.md").read_text("utf-8")


def test_cli_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0 and "video-analyzer 0.1.0" in result.output
