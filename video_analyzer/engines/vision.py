"""Keyframe detection: keep one stable frame per meaningful screen change.

Frames are sampled at `sample_fps` and compared, as small blurred grayscale thumbnails, against the
last kept keyframe. The difference score is sqrt(fraction of 16px grid cells that changed): 0 means
identical, 1 an entirely different screen. Whole-image metrics such as SSIM are dominated by the
unchanged whitespace of typical UIs, so a full page navigation scores only ~0.1; the square root of
the changed area gives a usable range instead. Rough values: cursor movement ~0.08, typing into a
field ~0.25, opening a dropdown ~0.35, navigating to another page 0.4-0.6, a modal with backdrop
~0.95. When the score crosses the threshold, the frame is not saved
immediately: we wait until consecutive samples stop changing, so that scrolls, fades and animations
are captured in their settled state. Near-duplicates of any earlier keyframe (pHash distance) are
dropped.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from video_analyzer.models import Keyframe

THUMB_WIDTH = 256
CELL_SIZE = 16  # thumbnail pixels per grid cell
PIXEL_DELTA = 25  # grayscale change that counts as "changed" (above compression noise)
CELL_CHANGED_FRACTION = 0.03  # share of changed pixels that marks a whole cell as changed
STABLE_EPSILON = 0.15  # max change between consecutive samples for a screen to count as settled
MAX_SETTLE_SECONDS = 3.0  # give up waiting for a settled screen after this long
DUPLICATE_HAMMING = 4  # pHash bit distance at or below which two keyframes count as the same screen


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def keyframe_filename(index: int, timestamp: float) -> str:
    minutes, secs = divmod(int(timestamp), 60)
    return f"keyframe_{index:03d}_{minutes:02d}_{secs:02d}.jpg"


def thumbnail(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height = max(1, round(gray.shape[0] * THUMB_WIDTH / gray.shape[1]))
    small = cv2.resize(gray, (THUMB_WIDTH, height), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (3, 3), 0).astype(np.float32)


def difference(a: np.ndarray, b: np.ndarray) -> float:
    """sqrt of the fraction of grid cells that changed between two same-sized thumbnails."""
    changed = (cv2.absdiff(a, b) > PIXEL_DELTA).astype(np.float32)
    rows, cols = max(1, changed.shape[0] // CELL_SIZE), max(1, changed.shape[1] // CELL_SIZE)
    per_cell = cv2.resize(changed, (cols, rows), interpolation=cv2.INTER_AREA)
    return float(np.sqrt((per_cell > CELL_CHANGED_FRACTION).mean()))


def phash(thumb: np.ndarray) -> int:
    small = cv2.resize(thumb, (32, 32), interpolation=cv2.INTER_AREA)
    low_freq = cv2.dct(small)[:8, :8].flatten()
    bits = low_freq > np.median(low_freq[1:])
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class _Candidate:
    timestamp: float
    score: float
    temp_path: Path
    resolution: str


def _save_jpeg(frame: np.ndarray, path: Path, max_width: int, quality: int) -> str:
    height, width = frame.shape[:2]
    if width > max_width:
        frame = cv2.resize(
            frame, (max_width, round(height * max_width / width)), interpolation=cv2.INTER_AREA
        )
    # cv2.imwrite cannot handle non-ASCII paths on Windows; encode in memory instead.
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode keyframe {path.name}")
    path.write_bytes(buffer.tobytes())
    return f"{width}x{height}"


def detect_keyframes(
    video_path: Path,
    output_dir: Path,
    threshold: float = 0.35,
    sample_fps: float = 1.0,
    min_gap: float = 2.0,
    max_keyframes: int = 40,
    max_width: int = 1280,
    jpeg_quality: int = 80,
    on_progress: Callable[[float, float], None] | None = None,
) -> tuple[list[Keyframe], float]:
    """Write keyframes to `output_dir/keyframes/` and return them plus the video duration."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if not 0 < fps < 1000:  # some webm files report 0 or garbage
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    est_duration = frame_count / fps if frame_count > 0 else 0.0
    step = max(1, round(fps / sample_fps))

    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[_Candidate] = []
    hashes: list[int] = []
    last_kept: np.ndarray | None = None
    last_kept_time = -1e9
    previous: np.ndarray | None = None
    pending_since: float | None = None
    pending_score = 0.0
    frame_index = -1
    timestamp = 0.0

    def keep(frame: np.ndarray, thumb: np.ndarray, at: float, score: float) -> None:
        """Save the settled `frame`, stamped with `at`, the moment the change began."""
        nonlocal last_kept, last_kept_time
        last_kept, last_kept_time = thumb, timestamp
        frame_hash = phash(thumb)
        if any(hamming(frame_hash, h) <= DUPLICATE_HAMMING for h in hashes):
            return  # back on a screen we already captured
        hashes.append(frame_hash)
        temp_path = keyframe_dir / f".candidate_{len(candidates):05d}.jpg"
        resolution = _save_jpeg(frame, temp_path, max_width, jpeg_quality)
        candidates.append(_Candidate(at, score, temp_path, resolution))

    try:
        while True:
            frame_index += 1
            if frame_index % step:
                if not capture.grab():  # skip without decoding
                    break
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            timestamp = frame_index / fps
            thumb = thumbnail(frame)
            if on_progress:
                on_progress(timestamp, est_duration)

            if last_kept is None:
                keep(frame, thumb, timestamp, 1.0)
            else:
                score = difference(thumb, last_kept)
                if pending_since is None and score >= threshold:
                    pending_since, pending_score = timestamp, score
                if pending_since is not None:
                    pending_score = max(pending_score, score)
                    settled = previous is not None and difference(thumb, previous) < STABLE_EPSILON
                    waited_too_long = timestamp - pending_since >= MAX_SETTLE_SECONDS
                    if (settled or waited_too_long) and timestamp - last_kept_time >= min_gap:
                        if score >= threshold:  # the change did not revert while settling
                            keep(frame, thumb, pending_since, pending_score)
                        pending_since = None
            previous = thumb
    finally:
        capture.release()

    duration = max(est_duration, timestamp)
    keyframes = _finalize(candidates, keyframe_dir, max_keyframes)
    return keyframes, duration


def _finalize(
    candidates: list[_Candidate], keyframe_dir: Path, max_keyframes: int
) -> list[Keyframe]:
    """Apply the keyframe cap (first frame always kept, then highest scores) and rename files."""
    chosen = candidates
    if len(candidates) > max_keyframes:
        by_score = sorted(candidates[1:], key=lambda c: c.score, reverse=True)
        chosen = [candidates[0], *by_score[: max_keyframes - 1]]
        chosen.sort(key=lambda c: c.timestamp)

    for candidate in candidates:
        if candidate not in chosen:
            candidate.temp_path.unlink(missing_ok=True)

    keyframes: list[Keyframe] = []
    for index, candidate in enumerate(chosen, start=1):
        name = keyframe_filename(index, candidate.timestamp)
        candidate.temp_path.replace(keyframe_dir / name)
        keyframes.append(
            Keyframe(
                index=index,
                timestamp=round(candidate.timestamp, 2),
                file_path=f"keyframes/{name}",
                resolution=candidate.resolution,
                difference_score=round(candidate.score, 3),
            )
        )
    return keyframes


def write_metadata(keyframes: list[Keyframe], keyframe_dir: Path) -> Path:
    path = keyframe_dir / "metadata.json"
    payload = [{**kf.model_dump(), "timecode": format_timestamp(kf.timestamp)} for kf in keyframes]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
