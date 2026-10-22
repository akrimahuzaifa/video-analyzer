# Video-Analyzer: Technical Specifications & Functional Requirements

## 1. Functional Scope

Video-Analyzer provides a unified CLI and optional lightweight web UI to ingest, analyze, transcribe, extract UI state transitions, and produce AI-agent documentation from any video source.

---

## 2. CLI Interface & Usage

### 2.1 Basic Commands
```bash
# Analyze a local screen recording
video-analyzer analyze ./recordings/billing_demo.mp4

# Analyze a YouTube video (public or unlisted)
video-analyzer analyze "https://youtu.be/WoFCSwxMLKA"

# Specify output directory and model size
video-analyzer analyze ./demo.mp4 --output-dir ./output/demo --model base --language auto

# Run quick mode (captures subtitles + keyframes only, skips deep OCR)
video-analyzer analyze ./demo.mp4 --fast
```

### 2.2 CLI Options & Flags
| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input` | `str` (Positional) | Required | Local file path or YouTube URL |
| `--output-dir`, `-o` | `str` | `./output/{video_name}` | Destination directory for artifacts |
| `--model`, `-m` | `str` | `tiny` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `--device`, `-d` | `str` | `auto` | Compute device (`cpu`, `cuda`, `mps`) |
| `--language`, `-l` | `str` | `auto` | Spoken language override (`en`, `ur`, `hi`, `es`, etc.) |
| `--keyframe-threshold` | `float` | `0.35` | Perceptual difference threshold for scene cut detection |
| `--ocr / --no-ocr` | `bool` | `True` | Extract on-screen text from detected keyframes |
| `--keep-temp` | `bool` | `False` | Preserve intermediate extracted audio & uncompressed frames |
| `--format` | `str` | `all` | Specific output format (`all`, `agent-md`, `transcript`, `summary`) |

---

## 3. Output Artifact Specification

When processing completes, the output directory contains:

```text
output/demo/
├── agent.md                  # High-density technical brief for AI Agents
├── summary.md                # Executive & business summary
├── transcript.txt            # Plaintext timestamped transcript
├── transcript.json           # Machine-readable transcript with segments & word timestamps
└── keyframes/                # Cleaned, unique UI screenshots
    ├── keyframe_00_01_30.jpg # Keyframe at 01:30 with metadata
    ├── keyframe_00_07_30.jpg
    └── metadata.json         # Timestamps, OCR text, and screen descriptions
```

### 3.1 Structure of `agent.md`
`agent.md` follows a strict markdown format optimized for consumption by AI agents (e.g. Claude Code, Cursor, GPT-4, Antigravity):

```markdown
# Agent Execution Brief: [Video Title / Subject]

## 1. System & Process Context
- **Primary Goal:** [Extracted Goal]
- **Target Applications / Portals:** [URLs, Tools, Systems observed]
- **User Roles:** [Roles mentioned in narration, e.g. Setter, BA, Sales Rep]

## 2. Chronological Action Breakdown
### Step 1: [Action Name] ([MM:SS] - [MM:SS])
- **Source Screen:** [URL or Application Title]
- **Observed UI Action:** [Click, Input, Select, Upload, Download]
- **Spoken Instructions (Voiceover):** "[Relevant transcribed speech]"
- **Inputs & Fields:**
  | Field Label | Value Type | Example Value from Video |
  | :--- | :--- | :--- |
  | Phone Number | String | (830) 496-0634 |
- **Keyframe Reference:** ![Step 1](./keyframes/keyframe_00_01_30.jpg)

## 3. Business Logic, Formulas & Decision Trees
- [Formula 1: e.g. Plan Savings = Current Plan - New Plan]
- [Decision Rule: e.g. If status has dollar sign -> extract payoff amount]

## 4. Recommended Automation Implementation Prompt
[A drop-in prompt that a developer or agent can execute to build the automation code immediately]
```

---

## 4. Token Budget Optimization Strategy

Passing video or image sequences to frontier LLMs consumes immense token budgets:
- **1 minute of 1080p video (Gemini API native):** ~15,000 to 25,000 tokens.
- **30-minute video:** ~450,000 to 750,000 tokens (often exceeding free/tier-1 quotas).
- **Video-Analyzer Output:**
  - `transcript.txt`: ~2,500 - 4,000 tokens.
  - `summary.md`: ~500 tokens.
  - `agent.md`: ~3,000 - 5,000 tokens.
  - Total token load: **< 10,000 tokens** (Over **95% reduction** in token consumption!).

---

## 5. Extensibility & Future Integrations

- **Direct Export to IDE Agents**: Integration with `.agents/skills/` or Claude Code commands.
- **Whisper Subtitle Cache**: For YouTube URLs, automatically detect if manual or auto-generated captions already exist via `youtube-transcript-api` before initiating heavy audio transcription.
- **Custom Domain Profiles**: Add pluggable regex / heuristic extractors for specific industries (Telecom, Healthcare, Salesforce, QuickBooks, AWS Console).
