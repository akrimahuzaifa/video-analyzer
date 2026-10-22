from __future__ import annotations

import json

from video_analyzer.models import AnalysisResult, Keyframe, SpeechSegment
from video_analyzer.synthesis import aligner, formatters


def seg(start, end, text):
    return SpeechSegment(start=start, end=end, text=text, language="en")


def kf(index, timestamp, ocr=()):
    return Keyframe(
        index=index,
        timestamp=timestamp,
        file_path=f"keyframes/keyframe_{index:03d}_00_{int(timestamp):02d}.jpg",
        resolution="1280x720",
        difference_score=0.5,
        ocr_elements=list(ocr),
    )


TRANSCRIPT = [
    seg(0.5, 3.0, "Welcome, today we process a customer refund in the billing portal."),
    seg(3.2, 6.0, "First open billing.example.com and log in as the support agent."),
    seg(10.5, 13.0, "Click the refund button."),
    seg(13.1, 16.0, "If the invoice is older than 30 days then the manager must approve it."),
    seg(21.0, 24.0, "Type the amount and press submit."),
]
KEYFRAMES = [
    kf(1, 0.0, ["Billing Portal", "Customer: Jane Doe", "Phone Number: (830) 496-0634"]),
    kf(2, 10.0, ["Refund", "Amount = $40.00"]),
    kf(3, 20.0),
]


def test_align_assigns_speech_by_midpoint():
    steps = aligner.align(TRANSCRIPT, KEYFRAMES, duration=25.0)
    assert [s.step_number for s in steps] == [1, 2, 3]
    assert [(s.start_time, s.end_time) for s in steps] == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
    assert "billing portal" in steps[0].spoken_context
    assert steps[1].spoken_context.startswith("Click the refund button.")
    assert steps[2].spoken_context == "Type the amount and press submit."
    assert steps[2].visual_state.index == 3


def test_align_detects_actions_and_fields():
    steps = aligner.align(TRANSCRIPT, KEYFRAMES, duration=25.0)
    assert steps[0].observed_action == "Navigate"
    assert steps[1].observed_action == "Click"
    assert steps[2].observed_action == "Input"
    assert steps[0].extracted_fields == {
        "Customer": "Jane Doe",
        "Phone Number": "(830) 496-0634",
    }
    assert steps[1].extracted_fields == {"Amount": "$40.00"}


def test_align_without_keyframes_makes_single_step():
    steps = aligner.align(TRANSCRIPT, [], duration=25.0)
    assert len(steps) == 1 and steps[0].visual_state is None
    assert aligner.align([], [], duration=5.0) == []


def test_align_truncates_long_speech():
    long = [seg(0, 5, "word " * 500)]
    steps = aligner.align(long, [kf(1, 0.0)], duration=5.0, max_speech_chars=100)
    assert len(steps[0].spoken_context) <= 102
    assert steps[0].spoken_context.endswith("…")


def test_ui_action_uses_word_boundaries():
    assert aligner.detect_ui_action("This prototype is ready") is None
    assert aligner.detect_ui_action("Now we upload the CSV") == "Upload"


def test_extract_context():
    context = aligner.extract_context(TRANSCRIPT, KEYFRAMES)
    assert context.goal.startswith("Welcome, today we process a customer refund")
    assert "billing.example.com" in context.applications
    assert {"Agent", "Customer", "Manager", "Support"} <= set(context.roles)
    assert any("older than 30 days" in rule for rule in context.business_rules)


def make_result():
    steps = aligner.align(TRANSCRIPT, KEYFRAMES, duration=25.0)
    return AnalysisResult(
        video_title="Refund demo",
        source="refund.mp4",
        duration_seconds=25.0,
        detected_language="en",
        transcript=TRANSCRIPT,
        keyframes=KEYFRAMES,
        action_steps=steps,
    )


def test_render_agent_md():
    result = make_result()
    text = formatters.render_agent_md(result, aligner.extract_context(TRANSCRIPT, KEYFRAMES))
    for heading in [
        "# Agent Execution Brief: Refund demo",
        "## 1. System & Process Context",
        "## 2. Chronological Action Breakdown",
        "## 3. Business Logic, Formulas & Decision Trees",
        "## 4. Recommended Automation Implementation Prompt",
    ]:
        assert heading in text
    assert "### Step 2: Click the refund button (00:10 - 00:20)" in text
    assert "![Step 1](./keyframes/keyframe_001_00_00.jpg)" in text
    assert "| Phone Number | (830) 496-0634 |" in text
    assert formatters.estimate_tokens(text) < 2000


def test_render_summary_md():
    result = make_result()
    text = formatters.render_summary_md(result, aligner.extract_context(TRANSCRIPT, KEYFRAMES))
    assert text.startswith("# Summary: Refund demo")
    assert "1. **00:00**:" in text


def test_render_transcripts():
    result = make_result()
    txt = formatters.render_transcript_txt(result)
    assert txt.splitlines()[0] == (
        "[00:00 - 00:03] Welcome, today we process a customer refund in the billing portal."
    )
    data = json.loads(formatters.render_transcript_json(result))
    assert data["language"] == "en" and len(data["segments"]) == len(TRANSCRIPT)


def test_markdown_cells_are_escaped():
    assert formatters._md_cell("a|b\nc") == "a\\|b c"
