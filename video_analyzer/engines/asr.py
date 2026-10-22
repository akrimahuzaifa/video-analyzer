"""Speech-to-text with faster-whisper, GPU when available and CPU int8 otherwise."""

from __future__ import annotations

import logging
import math
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from video_analyzer.config import Device
from video_analyzer.models import SpeechSegment

log = logging.getLogger(__name__)


def load_wav(path: Path) -> np.ndarray:
    """Read the 16-bit mono WAV written by engines.audio as float32 in [-1, 1].

    Passing samples directly keeps faster-whisper from decoding the file with PyAV, whose API
    changes between major versions have broken faster-whisper's own loader before.
    """
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError(f"{path.name} must be 16-bit mono PCM")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


@dataclass
class TranscriptionResult:
    segments: list[SpeechSegment]
    language: str | None
    duration: float


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def resolve_compute(device: Device) -> tuple[str, str]:
    """Map the CLI device choice to a (device, compute_type) pair faster-whisper accepts."""
    if device == Device.mps:
        log.warning("faster-whisper has no MPS backend; using CPU instead.")
        return "cpu", "int8"
    if device == Device.cuda or (device == Device.auto and _cuda_available()):
        return "cuda", "float16"
    return "cpu", "int8"


def transcribe(
    audio_path: Path,
    model_size: str = "tiny",
    device: Device = Device.auto,
    language: str | None = None,
    on_progress: Callable[[float, float], None] | None = None,
) -> TranscriptionResult:
    """Transcribe a WAV file. `on_progress` receives (position, total) in seconds."""
    target, compute_type = resolve_compute(device)
    try:
        return _transcribe_on(audio_path, model_size, target, compute_type, language, on_progress)
    except Exception as exc:  # noqa: BLE001 - CUDA DLL problems often surface mid-decode
        if target == "cpu":
            raise
        log.warning("Whisper failed on %s (%s); retrying on CPU int8.", target, exc)
        return _transcribe_on(audio_path, model_size, "cpu", "int8", language, on_progress)


def _transcribe_on(
    audio_path: Path,
    model_size: str,
    target: str,
    compute_type: str,
    language: str | None,
    on_progress: Callable[[float, float], None] | None,
) -> TranscriptionResult:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=target, compute_type=compute_type)
    raw_segments, info = model.transcribe(
        load_wav(audio_path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    segments: list[SpeechSegment] = []
    for seg in raw_segments:  # a generator: decoding happens while iterating
        text = seg.text.strip()
        if text:
            segments.append(
                SpeechSegment(
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=text,
                    confidence=round(math.exp(seg.avg_logprob), 3),
                    language=info.language,
                )
            )
        if on_progress:
            on_progress(seg.end, info.duration)

    return TranscriptionResult(segments=segments, language=info.language, duration=info.duration)
