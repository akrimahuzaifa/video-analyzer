"""Audio extraction with the ffmpeg binary bundled by imageio-ffmpeg (no system ffmpeg needed)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class AudioExtractionError(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - fall back to a system install
        system = shutil.which("ffmpeg")
        if system is None:
            raise AudioExtractionError(
                "No ffmpeg found. Install imageio-ffmpeg or put ffmpeg on PATH."
            ) from None
        return system


def has_audio_stream(video_path: Path) -> bool:
    # `ffmpeg -i` with no output exits non-zero but still prints the stream list to stderr.
    probe = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(video_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return re.search(r"Stream #\d+:\d+.*: Audio:", probe.stderr) is not None


def extract_audio(video_path: Path, out_path: Path, sample_rate: int = 16000) -> Path:
    """Write a mono 16-bit PCM WAV at `sample_rate`, the input format Whisper expects."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le",
        str(out_path),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0 or not out_path.exists():
        raise AudioExtractionError(f"ffmpeg failed to extract audio: {result.stderr.strip()}")
    return out_path
