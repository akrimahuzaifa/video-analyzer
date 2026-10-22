"""YouTube links: reuse existing captions, download a capped-resolution copy for keyframes."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from video_analyzer.models import SpeechSegment, VideoSource

log = logging.getLogger(__name__)

_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host not in _YOUTUBE_HOSTS:
        return None
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path == "/watch":
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    else:
        # /shorts/<id>, /embed/<id>, /live/<id>, /v/<id>
        parts = [p for p in parsed.path.split("/") if p]
        candidate = (
            parts[1] if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"} else ""
        )
    return candidate if _VIDEO_ID.match(candidate) else None


def is_youtube_url(source: str) -> bool:
    return extract_video_id(source) is not None


def fetch_captions(
    video_id: str, language: str | None = None
) -> tuple[list[SpeechSegment], str] | None:
    """Return existing captions, preferring manual over auto-generated ones. None if unavailable."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcripts = YouTubeTranscriptApi().list(video_id)
        if language:
            transcript = transcripts.find_transcript([language])
        else:
            available = list(transcripts)
            manual = [t for t in available if not t.is_generated]
            transcript = (manual or available)[0]
        fetched = transcript.fetch()
    except Exception as exc:  # noqa: BLE001 - any failure just means "fall back to Whisper"
        log.info("No usable YouTube captions for %s: %s", video_id, exc)
        return None

    segments = [
        SpeechSegment(
            start=s.start,
            end=s.start + s.duration,
            text=s.text.replace("\n", " ").strip(),
            language=transcript.language_code,
        )
        for s in fetched
        if s.text.strip()
    ]
    return (segments, transcript.language_code) if segments else None


class YouTubeDownloadError(RuntimeError):
    pass


class _YtDlpLogger:
    """Route yt-dlp's console output into logging; failures still surface as exceptions."""

    def debug(self, msg: str) -> None:
        log.debug(msg)

    def info(self, msg: str) -> None:
        log.debug(msg)

    def warning(self, msg: str) -> None:
        log.info(msg)

    def error(self, msg: str) -> None:
        log.debug(msg)


@dataclass
class Cookies:
    """Optional YouTube authentication, needed when YouTube asks to 'confirm you're not a bot'."""

    from_browser: str | None = None  # e.g. "firefox", "chrome", "edge"
    file: Path | None = None  # Netscape-format cookies.txt

    def ydl_options(self) -> dict:
        options: dict = {}
        if self.from_browser:
            options["cookiesfrombrowser"] = (self.from_browser,)
        if self.file:
            options["cookiefile"] = str(self.file)
        return options


def download_video(
    url: str, work_dir: Path, max_height: int = 720, cookies: Cookies | None = None
) -> tuple[Path, str]:
    """Download the video (with audio) into work_dir and return (path, title)."""
    import imageio_ffmpeg
    import yt_dlp

    work_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "outtmpl": str(work_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "logger": _YtDlpLogger(),
        "noprogress": True,
        **(cookies or Cookies()).ydl_options(),
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except yt_dlp.utils.DownloadError as exc:
        message = str(exc).removeprefix("ERROR: ").strip()
        if "sign in" in message.lower() or "not a bot" in message.lower():
            message = (
                "YouTube is asking to confirm you're not a bot. Re-run with "
                "--cookies-from-browser firefox (or chrome/edge) or --cookies cookies.txt. "
                "VPNs and shared IPs trigger this more often."
            )
        raise YouTubeDownloadError(message) from exc
    if not path.exists():
        # The merge step may have changed the extension.
        matches = sorted(work_dir.glob(f"{info['id']}.*"))
        if not matches:
            raise RuntimeError(f"yt-dlp finished but no file was found for {url}")
        path = matches[0]
    return path, info.get("title") or info["id"]


def load_youtube_video(
    url: str,
    work_dir: Path,
    language: str | None = None,
    use_captions: bool = True,
    need_video: bool = True,
    cookies: Cookies | None = None,
) -> VideoSource:
    """Fetch captions and the video. Without a video, the brief is built from captions alone:
    skipped when `need_video` is False and captions exist, or when the download is blocked."""
    video_id = extract_video_id(url)
    if video_id is None:
        raise ValueError(f"Not a YouTube video URL: {url}")

    captions = fetch_captions(video_id, language) if use_captions else None
    video_path: Path | None = None
    title = video_id
    if need_video or not captions:
        try:
            video_path, title = download_video(url, work_dir, cookies=cookies)
        except YouTubeDownloadError as exc:
            if not captions:
                raise
            log.warning("Video download failed, continuing with captions only: %s", exc)
    return VideoSource(
        video_path=video_path,
        title=title,
        origin=url,
        is_youtube=True,
        captions=captions[0] if captions else None,
        caption_language=captions[1] if captions else None,
    )
