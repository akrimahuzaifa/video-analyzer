from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

FPS = 10
SIZE = (640, 360)  # width, height


def _screen(title: str, rows: int, modal: bool = False) -> np.ndarray:
    """A fake application screen: header bar, title, a table-ish list, optional modal dialog."""
    frame = np.full((SIZE[1], SIZE[0], 3), 245, dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (SIZE[0], 40), (90, 60, 30), -1)
    cv2.putText(frame, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i in range(rows):
        y = 70 + i * 32
        cv2.rectangle(frame, (20, y), (620, y + 24), (220, 220, 220), 1)
        cv2.putText(frame, f"{title} row {i + 1}: value {i * 37}", (28, y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)  # fmt: skip
    if modal:
        cv2.rectangle(frame, (0, 0), SIZE, (0, 0, 0), -1)  # dimmed backdrop
        cv2.rectangle(frame, (140, 80), (500, 280), (255, 255, 255), -1)
        cv2.putText(
            frame, "Confirm payment?", (170, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        cv2.rectangle(frame, (330, 220), (470, 260), (60, 160, 60), -1)
        cv2.putText(frame, "Submit", (360, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


# (screen, seconds). The return to "Dashboard" must be recognised as a duplicate.
TIMELINE = [
    (_screen("Dashboard", 8), 4),
    (_screen("Billing", 5), 4),
    (_screen("Billing", 5, modal=True), 4),
    (_screen("Dashboard", 8), 3),
    (_screen("Reports", 3), 4),
]
EXPECTED_SCREEN_STARTS = [0, 4, 8, 15]  # seconds at which a new, unseen screen appears


def write_video(path: Path, timeline=TIMELINE) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    assert writer.isOpened(), "OpenCV could not open an mp4v writer"
    for frame, seconds in timeline:
        for _ in range(int(seconds * FPS)):
            writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    return write_video(tmp_path_factory.mktemp("video") / "billing_demo.mp4")
