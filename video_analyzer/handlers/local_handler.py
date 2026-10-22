"""Local video files: validate before any heavy processing starts."""

from __future__ import annotations

from pathlib import Path

from video_analyzer.config import SUPPORTED_VIDEO_EXTENSIONS
from video_analyzer.models import VideoSource


class InvalidVideoError(ValueError):
    pass


def load_local_video(path: str | Path) -> VideoSource:
    video_path = Path(path).expanduser()
    if not video_path.is_file():
        raise InvalidVideoError(f"Video file not found: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise InvalidVideoError(
            f"Unsupported file type '{video_path.suffix}'. Use one of: {supported}"
        )

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened() or not capture.read()[0]:
            raise InvalidVideoError(f"Could not decode any video frames from {video_path}")
    finally:
        capture.release()

    return VideoSource(video_path=video_path.resolve(), title=video_path.stem, origin=str(path))
