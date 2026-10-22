from __future__ import annotations

import json

import numpy as np

from tests.conftest import EXPECTED_SCREEN_STARTS
from video_analyzer.engines import vision


def test_keyframe_filename():
    assert vision.keyframe_filename(1, 90.4) == "keyframe_001_01_30.jpg"
    assert vision.keyframe_filename(12, 3725) == "keyframe_012_62_05.jpg"


def test_format_timestamp():
    assert vision.format_timestamp(5) == "00:05"
    assert vision.format_timestamp(90.9) == "01:30"
    assert vision.format_timestamp(3725) == "01:02:05"


def test_difference_scales_with_changed_area():
    blank = np.full((144, 256), 240, dtype=np.float32)
    assert vision.difference(blank, blank) == 0
    assert vision.difference(blank, 255 - blank) == 1

    cursor = blank.copy()
    cursor[60:66, 100:104] = 0  # a few pixels, like a mouse pointer
    half = blank.copy()
    half[:, :128] = 0  # half the screen, like a side panel opening
    assert vision.difference(blank, cursor) < 0.15
    assert abs(vision.difference(blank, half) - 0.5**0.5) < 0.05


def test_detects_each_distinct_screen_once(synthetic_video, tmp_path):
    keyframes, duration = vision.detect_keyframes(synthetic_video, tmp_path, min_gap=1.0)

    assert abs(duration - 19) <= 1
    starts = [kf.timestamp for kf in keyframes]
    assert len(starts) == len(EXPECTED_SCREEN_STARTS), starts
    # Stamped when the change began, even though the image is taken once the screen settles.
    assert starts == EXPECTED_SCREEN_STARTS

    files = sorted(p.name for p in (tmp_path / "keyframes").glob("*.jpg"))
    assert files == sorted(kf.file_path.split("/")[1] for kf in keyframes)
    assert not list((tmp_path / "keyframes").glob(".candidate_*"))


def test_keyframe_cap_keeps_first_and_strongest(synthetic_video, tmp_path):
    keyframes, _ = vision.detect_keyframes(synthetic_video, tmp_path, min_gap=1.0, max_keyframes=2)
    assert len(keyframes) == 2
    assert keyframes[0].timestamp == 0
    assert [kf.index for kf in keyframes] == [1, 2]
    assert len(list((tmp_path / "keyframes").glob("*.jpg"))) == 2


def test_high_threshold_keeps_only_first_frame(synthetic_video, tmp_path):
    keyframes, _ = vision.detect_keyframes(synthetic_video, tmp_path, threshold=0.99)
    assert [kf.timestamp for kf in keyframes] == [0]


def test_write_metadata(synthetic_video, tmp_path):
    keyframes, _ = vision.detect_keyframes(synthetic_video, tmp_path)
    path = vision.write_metadata(keyframes, tmp_path / "keyframes")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data[0]["timecode"] == "00:00"
    assert data[0]["file_path"].startswith("keyframes/keyframe_001_")
