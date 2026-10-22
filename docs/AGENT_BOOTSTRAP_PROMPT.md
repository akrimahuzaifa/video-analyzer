# Autonomous Agent Bootstrap Prompt: Video-Analyzer

> **Instructions for the AI Coding Agent**:
> You are building **Video-Analyzer**, an open-source, local-first engine that analyzes local/private video files and YouTube links, producing a high-density, token-efficient `agent.md`, `summary.md`, `transcript.txt`, and keyframes.
> Read this document completely, inspect `docs/ARCHITECTURE.md`, `docs/SPECIFICATIONS.md`, and `docs/GRAPHIFY.md`, and build the project from scratch in this workspace.

---

## 1. Project Mission & High-Level Requirements

Build a clean, robust, modern Python CLI tool that:
1. **Ingests**:
   - Local video files (`.mp4`, `.mkv`, `.mov`, `.webm`).
   - Public or unlisted YouTube URLs (via `yt-dlp` and `youtube-transcript-api`).
2. **Processes Locally (Zero Privacy Leaks)**:
   - Audio extraction to 16kHz WAV using `imageio-ffmpeg` or local ffmpeg.
   - Multilingual transcription using `faster-whisper` (support Urdu, Hindi, Spanish, English, etc.).
   - Scene cut & UI change detection using OpenCV (drop duplicates, keep significant UI transitions).
   - Optional lightweight OCR on keyframes using `EasyOCR` or `PaddleOCR`.
3. **Outputs**:
   - `transcript.txt`: Chronological timestamped transcript.
   - `transcript.json`: Full segment metadata.
   - `summary.md`: High-level business overview.
   - `agent.md`: Highly structured AI-agent execution brief containing step-by-step actions, UI inputs, formulas, and embedded keyframe images.
   - `keyframes/`: Folder containing numbered, clean JPEG screenshots.
4. **Token Budget Target**:
   - Keep total token consumption for an hour-long video under 10k tokens (95% reduction compared to raw multimodal video tokens).

---

## 2. Target File Structure to Implement

```text
video-analyzer/
├── pyproject.toml              # Modern Python packaging (Poetry, Flit, or Hatch)
├── requirements.txt            # Core dependencies
├── README.md                   # Open-source public README
├── video_analyzer/             # Main package
│   ├── __init__.py
│   ├── cli.py                  # Typer or Click CLI interface
│   ├── config.py               # Settings & Pydantic config
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── local_handler.py    # Local video processing
│   │   └── youtube_handler.py  # YouTube streaming & caption cache
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── audio.py            # FFmpeg extraction & normalizer
│   │   ├── asr.py              # faster-whisper wrapper
│   │   ├── vision.py           # Scene detection & pHash filter
│   │   └── ocr.py              # Keyframe text & table extraction
│   ├── synthesis/
│   │   ├── __init__.py
│   │   ├── aligner.py          # Correlates speech with keyframes
│   │   └── formatters.py       # Generates agent.md, summary.md, transcript
│   └── templates/
│       ├── agent.md.j2         # Jinja2 template for agent.md
│       └── summary.md.j2       # Jinja2 template for summary.md
└── tests/
    ├── test_handlers.py
    └── test_synthesis.py
```

---

## 3. Implementation Steps for the Agent

1. **Environment & Scaffolding**:
   - Create `pyproject.toml` or `requirements.txt`:
     - `faster-whisper>=1.0.0`
     - `imageio-ffmpeg>=0.5.0`
     - `yt-dlp>=2024.0.0`
     - `youtube-transcript-api>=0.6.0`
     - `opencv-python>=4.9.0`
     - `typer>=0.9.0`
     - `rich>=13.0.0`
     - `jinja2>=3.1.0`
     - `pydantic>=2.5.0`
2. **Audio & ASR Module**:
   - Write `video_analyzer/engines/audio.py` to extract 16kHz mono audio without disk bloat.
   - Write `video_analyzer/engines/asr.py` with automatic language detection, fallback to CPU int8 compute.
3. **Keyframe & Vision Module**:
   - Write `video_analyzer/engines/vision.py`:
     - Compute frame difference thresholding (SSIM or difference norm).
     - Save unique frames as `keyframe_{index:03d}_{mm}_{ss}.jpg`.
4. **Synthesis & Jinja2 Templates**:
   - Align timestamp segments from ASR with the closest keyframes.
   - Render `agent.md` using the Jinja2 template, grouping contiguous speech under the active keyframe.
5. **CLI Interface**:
   - Expose `video-analyzer analyze <path_or_url>` with clean `rich` progress bars.
