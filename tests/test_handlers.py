from __future__ import annotations

from pathlib import Path

import pytest

from video_analyzer.handlers import Cookies, resolve_source, youtube_handler
from video_analyzer.handlers.local_handler import InvalidVideoError, load_local_video
from video_analyzer.handlers.youtube_handler import extract_video_id, is_youtube_url
from video_analyzer.models import SpeechSegment


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/WoFCSwxMLKA",
        "https://youtu.be/WoFCSwxMLKA?t=42",
        "https://www.youtube.com/watch?v=WoFCSwxMLKA",
        "https://youtube.com/watch?v=WoFCSwxMLKA&list=PL123",
        "https://m.youtube.com/watch?v=WoFCSwxMLKA",
        "https://www.youtube.com/shorts/WoFCSwxMLKA",
        "https://www.youtube.com/embed/WoFCSwxMLKA",
        "https://www.youtube.com/live/WoFCSwxMLKA",
    ],
)
def test_extract_video_id(url):
    assert extract_video_id(url) == "WoFCSwxMLKA"


@pytest.mark.parametrize(
    "source",
    [
        "./recordings/billing_demo.mp4",
        r"C:\videos\demo.mkv",
        "https://vimeo.com/123456",
        "https://www.youtube.com/channel/UC123",
        "https://youtu.be/short",
        "https://notyoutube.com/watch?v=WoFCSwxMLKA",
    ],
)
def test_non_youtube_sources(source):
    assert not is_youtube_url(source)


def test_local_video_missing(tmp_path):
    with pytest.raises(InvalidVideoError, match="not found"):
        load_local_video(tmp_path / "nope.mp4")


def test_local_video_unsupported_extension(tmp_path):
    path = tmp_path / "clip.avi"
    path.write_bytes(b"x")
    with pytest.raises(InvalidVideoError, match="Unsupported"):
        load_local_video(path)


def test_local_video_undecodable(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video")
    with pytest.raises(InvalidVideoError, match="decode"):
        load_local_video(path)


def test_local_video_valid(synthetic_video, tmp_path):
    source = resolve_source(str(synthetic_video), tmp_path)
    assert source.title == "billing_demo"
    assert source.video_path == synthetic_video.resolve()
    assert not source.is_youtube and source.captions is None


CAPTIONS = [SpeechSegment(start=0, end=2, text="Hello there.", language="en")]
URL = "https://youtu.be/WoFCSwxMLKA"


@pytest.fixture
def youtube(monkeypatch):
    """Stub the network: captions and download behaviour are set per test."""
    state = {"captions": (CAPTIONS, "en"), "download": "ok", "downloads": 0}

    def fake_download(url, work_dir, cookies=None):
        state["downloads"] += 1
        state["cookies"] = cookies
        if state["download"] == "blocked":
            raise youtube_handler.YouTubeDownloadError("not a bot")
        return work_dir / "WoFCSwxMLKA.mp4", "Demo title"

    monkeypatch.setattr(youtube_handler, "fetch_captions", lambda vid, lang: state["captions"])
    monkeypatch.setattr(youtube_handler, "download_video", fake_download)
    return state


def test_youtube_with_captions_and_video(youtube, tmp_path):
    cookies = Cookies(from_browser="firefox")
    source = resolve_source(URL, tmp_path, cookies=cookies)
    assert source.is_youtube and source.title == "Demo title"
    assert source.captions == CAPTIONS and source.caption_language == "en"
    assert youtube["cookies"] is cookies


def test_youtube_blocked_download_falls_back_to_captions(youtube, tmp_path):
    youtube["download"] = "blocked"
    source = resolve_source(URL, tmp_path)
    assert source.video_path is None
    assert source.captions == CAPTIONS and source.title == "WoFCSwxMLKA"


def test_youtube_blocked_download_without_captions_raises(youtube, tmp_path):
    youtube["download"], youtube["captions"] = "blocked", None
    with pytest.raises(youtube_handler.YouTubeDownloadError):
        resolve_source(URL, tmp_path)


def test_youtube_transcript_only_skips_download(youtube, tmp_path):
    source = resolve_source(URL, tmp_path, need_video=False)
    assert youtube["downloads"] == 0 and source.video_path is None


def test_cookie_options():
    assert Cookies().ydl_options() == {}
    assert Cookies("edge", Path("c.txt")).ydl_options() == {
        "cookiesfrombrowser": ("edge",),
        "cookiefile": "c.txt",
    }
