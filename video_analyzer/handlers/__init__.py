"""Input handlers: turn a CLI source argument into a VideoSource."""

from __future__ import annotations

from pathlib import Path

from video_analyzer.handlers.local_handler import load_local_video
from video_analyzer.handlers.youtube_handler import Cookies, is_youtube_url, load_youtube_video
from video_analyzer.models import VideoSource


def resolve_source(
    source: str,
    work_dir: Path,
    language: str | None = None,
    use_captions: bool = True,
    need_video: bool = True,
    cookies: Cookies | None = None,
) -> VideoSource:
    if is_youtube_url(source):
        return load_youtube_video(
            source,
            work_dir,
            language=language,
            use_captions=use_captions,
            need_video=need_video,
            cookies=cookies,
        )
    return load_local_video(source)


__all__ = ["Cookies", "resolve_source", "is_youtube_url", "load_local_video", "load_youtube_video"]
