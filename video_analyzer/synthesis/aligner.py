"""Correlate speech segments with keyframes and pull out structure with plain heuristics.

Everything here is deterministic and local: no LLM is involved. The heuristics are deliberately
conservative (English keywords plus language-neutral patterns such as URLs and `Label: value`
pairs); the full transcript is always written alongside so nothing is lost when they miss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from video_analyzer.models import ActionStep, Keyframe, SpeechSegment

URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\"')\]]+"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|io|net|org|app|dev|cloud|ai)(?:/[^\s<>\"')\]]*)?\b",
    re.IGNORECASE,
)
FIELD_PATTERN = re.compile(r"^\s*([A-Za-z][\w /#().-]{1,40}?)\s*[:=]\s*(\S.{0,60})$")

# First match wins, so more specific actions come first.
UI_ACTIONS: list[tuple[str, re.Pattern[str]]] = [
    (action, re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE))
    for action, words in [
        ("Upload", ["upload(?:s|ed|ing)?", "attach(?:es|ed|ing)?", "drag and drop"]),
        ("Download", ["download(?:s|ed|ing)?", "export(?:s|ed|ing)?"]),
        ("Input", ["type", "typing", "enter", "fill (?:in|out)", "paste", "input"]),
        (
            "Select",
            ["select(?:s|ed|ing)?", "choose", "pick", "dropdown", "check the box", "toggle"],
        ),
        ("Click", ["click(?:s|ed|ing)?", "press(?:es|ed|ing)?", "tap", "submit", "save"]),
        ("Navigate", ["open(?:s|ed|ing)?", "go to", "navigate", "switch to", "log ?in", "sign in"]),
    ]
]

ROLE_KEYWORDS = (
    "admin", "administrator", "agent", "analyst", "customer", "developer", "manager",
    "operator", "reviewer", "sales rep", "setter", "supervisor", "support", "user",
)  # fmt: skip

LOGIC_MARKERS = re.compile(
    r"(\bif\b.+\b(then|we|you)\b|\botherwise\b|\bunless\b|\bequals?\b|\bminus\b|\bplus\b|"
    r"\bmultipl(y|ied)\b|\bdivided by\b|\bpercent(age)?\b|\btotal\b.+\b(is|=)\b|[=<>]\s*\$?\d)",
    re.IGNORECASE,
)


def segment_midpoint(segment: SpeechSegment) -> float:
    return (segment.start + segment.end) / 2


def join_speech(segments: list[SpeechSegment]) -> str:
    return " ".join(s.text.strip() for s in segments if s.text.strip())


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + " …"


def detect_ui_action(text: str) -> str | None:
    for action, pattern in UI_ACTIONS:
        if pattern.search(text):
            return action
    return None


def extract_fields(lines: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in lines:
        match = FIELD_PATTERN.match(line)
        if match and not URL_PATTERN.search(match.group(1)):
            fields.setdefault(match.group(1).strip(), match.group(2).strip())
    return fields


def extract_urls(texts: list[str]) -> list[str]:
    seen: list[str] = []
    for text in texts:
        for url in URL_PATTERN.findall(text):
            url = url.rstrip(".,;:")
            if url.lower() not in (s.lower() for s in seen):
                seen.append(url)
    return seen


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।۔])\s+", text) if s.strip()]


def make_title(speech: str, keyframe: Keyframe | None, fallback: str) -> str:
    if speech:
        words = split_sentences(speech)[0].split()
        title = " ".join(words[:10])
        return title.rstrip(".,;:!?") + (" …" if len(words) > 10 else "")
    if keyframe and keyframe.ocr_elements:
        return keyframe.ocr_elements[0][:80]
    return fallback


def align(
    transcript: list[SpeechSegment],
    keyframes: list[Keyframe],
    duration: float,
    max_speech_chars: int = 600,
) -> list[ActionStep]:
    """One step per keyframe, spanning until the next keyframe. Speech goes to the step that
    contains its midpoint; speech before the first keyframe joins the first step."""
    if not keyframes:
        if not transcript:
            return []
        speech = join_speech(transcript)
        return [
            ActionStep(
                step_number=1,
                start_time=0.0,
                end_time=round(duration, 2),
                action_title=make_title(speech, None, "Full video"),
                observed_action=detect_ui_action(speech),
                spoken_context=truncate(speech, max_speech_chars),
            )
        ]

    boundaries = (
        [0.0] + [kf.timestamp for kf in keyframes[1:]] + [max(duration, keyframes[-1].timestamp)]
    )
    buckets: list[list[SpeechSegment]] = [[] for _ in keyframes]
    for segment in transcript:
        mid = segment_midpoint(segment)
        index = next(
            (i for i in range(len(keyframes)) if mid < boundaries[i + 1]), len(keyframes) - 1
        )
        buckets[index].append(segment)

    steps: list[ActionStep] = []
    for i, (keyframe, segments) in enumerate(zip(keyframes, buckets, strict=True)):
        speech = join_speech(segments)
        steps.append(
            ActionStep(
                step_number=i + 1,
                start_time=round(boundaries[i], 2),
                end_time=round(boundaries[i + 1], 2),
                action_title=make_title(speech, keyframe, f"Screen {i + 1}"),
                observed_action=detect_ui_action(speech),
                spoken_context=truncate(speech, max_speech_chars),
                visual_state=keyframe,
                extracted_fields=extract_fields(keyframe.ocr_elements),
            )
        )
    return steps


@dataclass
class ProcessContext:
    goal: str
    applications: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    business_rules: list[str] = field(default_factory=list)


def extract_context(
    transcript: list[SpeechSegment], keyframes: list[Keyframe], max_rules: int = 15
) -> ProcessContext:
    speech = join_speech(transcript)
    sentences = split_sentences(speech)

    goal = truncate(" ".join(sentences[:2]), 300) if sentences else "Not stated in narration."
    ocr_lines = [line for kf in keyframes for line in kf.ocr_elements]
    applications = extract_urls(ocr_lines + [speech])[:10]

    lowered = speech.lower()
    roles = [
        role.title() for role in ROLE_KEYWORDS if re.search(rf"\b{re.escape(role)}s?\b", lowered)
    ]

    rules: list[str] = []
    for sentence in sentences:
        if LOGIC_MARKERS.search(sentence) and sentence not in rules:
            rules.append(truncate(sentence, 240))
        if len(rules) >= max_rules:
            break
    return ProcessContext(goal=goal, applications=applications, roles=roles, business_rules=rules)
