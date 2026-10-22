"""Optional on-screen text extraction with EasyOCR (`pip install video-analyzer[ocr]`)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from video_analyzer.models import Keyframe

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.4
MAX_ELEMENTS_PER_FRAME = 40  # keeps agent.md inside the token budget


def ocr_available() -> bool:
    try:
        import easyocr  # noqa: F401
    except ImportError:
        return False
    return True


def annotate_keyframes(
    keyframes: list[Keyframe],
    output_dir: Path,
    languages: list[str] | None = None,
    use_gpu: bool = True,
    on_progress: Callable[[int], None] | None = None,
) -> bool:
    """Fill `ocr_elements` on each keyframe in place. Returns False if OCR is unavailable."""
    if not keyframes:
        return True
    if not ocr_available():
        log.warning(
            "EasyOCR is not installed; skipping OCR. Install: pip install 'video-analyzer[ocr]'"
        )
        return False

    import easyocr

    reader = easyocr.Reader(languages or ["en"], gpu=use_gpu, verbose=False)
    for done, keyframe in enumerate(keyframes, start=1):
        image_bytes = (output_dir / keyframe.file_path).read_bytes()
        results = reader.readtext(image_bytes, paragraph=False)
        # Reading order: top-to-bottom, then left-to-right, using each box's top-left corner.
        results.sort(key=lambda r: (round(r[0][0][1] / 12), r[0][0][0]))
        elements: list[str] = []
        for _box, text, confidence in results:
            text = text.strip()
            if confidence >= MIN_CONFIDENCE and len(text) > 1 and text not in elements:
                elements.append(text)
        keyframe.ocr_elements = elements[:MAX_ELEMENTS_PER_FRAME]
        if on_progress:
            on_progress(done)
    return True
