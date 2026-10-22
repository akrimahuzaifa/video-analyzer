"""Orchestration: resolve source -> (audio track || visual track) -> OCR -> align -> render."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from video_analyzer.config import OutputFormat, Settings
from video_analyzer.engines import asr, audio, ocr, vision
from video_analyzer.handlers import Cookies, resolve_source
from video_analyzer.models import AnalysisResult, Keyframe, SpeechSegment, VideoSource
from video_analyzer.synthesis import aligner, formatters

log = logging.getLogger(__name__)

# EasyOCR language codes for the Whisper languages we expect most often.
OCR_LANGUAGES = {"ur", "hi", "es", "fr", "de", "ar", "pt", "it", "nl", "tr", "ru"}


class Reporter:
    """Progress hooks; the CLI overrides these with rich progress bars. Methods may be
    called from worker threads."""

    def start(self, task: str, description: str, total: float | None = None) -> None: ...
    def update(self, task: str, completed: float, total: float | None = None) -> None: ...
    def finish(self, task: str) -> None: ...
    def info(self, message: str) -> None: ...


@dataclass
class PipelineOutput:
    result: AnalysisResult
    output_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    token_estimate: int = 0
    temp_dir: Path | None = None


def slugify(text: str, fallback: str = "video") -> str:
    slug = re.sub(r"[^\w\-]+", "_", text, flags=re.ASCII).strip("_")
    return slug[:80] or fallback


def run(settings: Settings, reporter: Reporter | None = None) -> PipelineOutput:
    reporter = reporter or Reporter()
    work_dir = Path(tempfile.mkdtemp(prefix="video-analyzer-"))
    try:
        output = _run(settings, reporter, work_dir)
        if settings.keep_temp:
            kept = output.output_dir / "_temp"
            shutil.rmtree(kept, ignore_errors=True)
            shutil.move(str(work_dir), kept)
            output.temp_dir = kept
        return output
    finally:
        if not settings.keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)


def _run(settings: Settings, reporter: Reporter, work_dir: Path) -> PipelineOutput:
    need_vision = settings.format != OutputFormat.transcript
    reporter.start("source", "Resolving input")
    source = resolve_source(
        settings.source,
        work_dir / "download",
        language=settings.language_hint,
        need_video=need_vision,
        cookies=Cookies(settings.cookies_from_browser, settings.cookies_file),
    )
    reporter.finish("source")
    if source.captions:
        reporter.info(
            f"Using existing YouTube captions ({source.caption_language}); skipping Whisper."
        )
    if source.video_path is None and need_vision:
        reporter.info(
            "Video download failed; the brief is built from captions only (no keyframes)."
        )
    need_vision = need_vision and source.video_path is not None

    output_dir = settings.output_dir or Path("output") / slugify(source.title)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        speech_future = pool.submit(_speech_track, source, settings, reporter, work_dir)
        vision_future = (
            pool.submit(_vision_track, source, settings, reporter, output_dir)
            if need_vision
            else None
        )
        transcript, language, speech_duration, transcript_source = speech_future.result()
        keyframes, video_duration = vision_future.result() if vision_future else ([], 0.0)

    if keyframes and settings.run_ocr:
        _ocr_track(keyframes, output_dir, language, settings, reporter)

    duration = max(video_duration, speech_duration, transcript[-1].end if transcript else 0.0)
    steps = aligner.align(transcript, keyframes, duration, settings.max_step_speech_chars)
    context = aligner.extract_context(transcript, keyframes)
    result = AnalysisResult(
        video_title=source.title,
        source=source.origin,
        duration_seconds=round(duration, 2),
        detected_language=language,
        transcript_source=transcript_source,
        transcript=transcript,
        keyframes=keyframes,
        action_steps=steps,
        summary=context.goal,
    )

    output = PipelineOutput(result=result, output_dir=output_dir)
    _write_artifacts(output, context, settings)
    return output


def _speech_track(
    source: VideoSource, settings: Settings, reporter: Reporter, work_dir: Path
) -> tuple[list[SpeechSegment], str | None, float, str]:
    if source.captions:
        return source.captions, source.caption_language, 0.0, "youtube-captions"

    if not audio.has_audio_stream(source.video_path):
        reporter.info("No audio track found; the brief will be built from keyframes only.")
        return [], None, 0.0, "none"

    reporter.start("audio", "Extracting audio")
    wav = audio.extract_audio(source.video_path, work_dir / "audio.wav")
    reporter.finish("audio")

    reporter.start("asr", f"Transcribing (whisper {settings.model})")
    transcription = asr.transcribe(
        wav,
        model_size=settings.model,
        device=settings.device,
        language=settings.language_hint,
        on_progress=lambda position, total: reporter.update("asr", position, total),
    )
    reporter.update("asr", transcription.duration, transcription.duration)
    reporter.finish("asr")
    return transcription.segments, transcription.language, transcription.duration, "whisper"


def _vision_track(
    source: VideoSource, settings: Settings, reporter: Reporter, output_dir: Path
) -> tuple[list[Keyframe], float]:
    keyframe_dir = output_dir / "keyframes"
    for stale in keyframe_dir.glob("keyframe_*.jpg"):  # from an earlier run into the same folder
        stale.unlink()

    reporter.start("vision", "Detecting keyframes")
    keyframes, duration = vision.detect_keyframes(
        source.video_path,
        output_dir,
        threshold=settings.keyframe_threshold,
        sample_fps=settings.sample_fps,
        min_gap=settings.min_keyframe_gap,
        max_keyframes=settings.max_keyframes,
        max_width=settings.keyframe_max_width,
        jpeg_quality=settings.jpeg_quality,
        on_progress=lambda position, total: reporter.update("vision", position, total or None),
    )
    reporter.finish("vision")
    return keyframes, duration


def _ocr_track(
    keyframes: list[Keyframe],
    output_dir: Path,
    language: str | None,
    settings: Settings,
    reporter: Reporter,
) -> None:
    if not ocr.ocr_available():
        reporter.info("OCR skipped: EasyOCR is not installed (pip install 'video-analyzer[ocr]').")
        return
    languages = ["en"] + ([language] if language in OCR_LANGUAGES else [])
    use_gpu = asr.resolve_compute(settings.device)[0] == "cuda"
    reporter.start("ocr", "Reading on-screen text", total=len(keyframes))
    try:
        ocr.annotate_keyframes(
            keyframes,
            output_dir,
            languages=languages,
            use_gpu=use_gpu,
            on_progress=lambda done: reporter.update("ocr", done),
        )
    except Exception as exc:  # noqa: BLE001 - OCR is an enrichment, never fatal
        log.warning("OCR failed: %s", exc)
        reporter.info(f"OCR failed and was skipped: {exc}")
    reporter.finish("ocr")


def _write_artifacts(
    output: PipelineOutput, context: aligner.ProcessContext, settings: Settings
) -> None:
    result, out = output.result, output.output_dir
    files = output.files
    briefs: list[str] = []

    if settings.wants(OutputFormat.transcript):
        txt = formatters.render_transcript_txt(result)
        files["transcript.txt"] = formatters.write_text(out / "transcript.txt", txt)
        files["transcript.json"] = formatters.write_text(
            out / "transcript.json", formatters.render_transcript_json(result)
        )
        briefs.append(txt)
    if settings.wants(OutputFormat.summary):
        summary = formatters.render_summary_md(result, context)
        files["summary.md"] = formatters.write_text(out / "summary.md", summary)
        briefs.append(summary)
    if settings.wants(OutputFormat.agent_md):
        agent_md = formatters.render_agent_md(result, context)
        files["agent.md"] = formatters.write_text(out / "agent.md", agent_md)
        briefs.append(agent_md)
    if result.keyframes:
        files["keyframes/"] = vision.write_metadata(result.keyframes, out / "keyframes").parent

    output.token_estimate = sum(formatters.estimate_tokens(text) for text in briefs)
