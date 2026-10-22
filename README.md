# 🎥 Video-Analyzer: Local Video-to-Agent Intelligence Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Local First](https://img.shields.io/badge/Privacy-100%25%20Local-green.svg)](#privacy)
[![Token Saver](https://img.shields.io/badge/Tokens-95%25%20Reduction-brightgreen.svg)](#token-savings)

> **Transform screen recordings and tutorial videos into token-efficient, developer-ready `agent.md` specifications for AI coding assistants (Claude Code, Cursor, ChatGPT, Antigravity) — 100% locally with zero cloud dependencies.**

---

## 💡 The Problem

1. **Astronomical Token Costs**: Passing a 30-minute demonstration video or hundreds of high-res screenshots to multimodal LLMs burns **300k–700k tokens** in a single prompt, instantly blowing through free/starter quotas.
2. **Privacy Risks**: Enterprise software demos often show confidential customer records, credentials, and internal endpoints. Uploading unlisted videos to YouTube or third-party cloud analyzers creates serious compliance violations.
3. **Agent Confusion**: Raw transcripts lack UI visual context; raw video lacks actionable structure.

## ✨ The Solution

**Video-Analyzer** runs locally on your laptop or server:
- 🎙️ **Multilingual ASR**: Local transcription with `faster-whisper` (detects English, Urdu, Hindi, Spanish, and code-switching).
- 🖼️ **Perceptual Scene Cuts**: Intelligent keyframe detection that only captures actual UI changes, forms, modals, and clicks.
- 📑 **`agent.md` Artifact**: Produces a structured Markdown brief with timestamped actions, data schemas, decision logic, and linked keyframes — consuming **< 8,000 tokens** (95%+ reduction!).

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/your-username/video-analyzer.git
cd video-analyzer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .              # core: Whisper ASR, keyframes, YouTube
pip install -e ".[ocr]"       # optional: on-screen text via EasyOCR (pulls in PyTorch)
```

No system ffmpeg is needed: the `imageio-ffmpeg` package ships its own binary.

### Usage
```bash
# Analyze a local screen recording
video-analyzer analyze ./recordings/billing_demo.mp4

# Analyze a YouTube video
video-analyzer analyze "https://youtu.be/WoFCSwxMLKA"

# Bigger Whisper model, forced language, custom output folder
video-analyzer analyze ./demo.mp4 --model small --language ur -o ./output/demo

# Quick mode: transcript/captions + keyframes, no OCR
video-analyzer analyze ./demo.mp4 --fast
```

Run `video-analyzer analyze --help` for all options (see also [SPECIFICATIONS.md](./docs/SPECIFICATIONS.md#22-cli-options--flags)).

**GPU:** Whisper uses CUDA automatically when available and falls back to CPU (int8) if the CUDA 12
cuBLAS/cuDNN libraries are missing, so a run never fails because of the GPU setup.

**YouTube "confirm you're not a bot":** YouTube sometimes blocks downloads, especially from VPNs and
shared IPs. Re-run with `--cookies-from-browser firefox` (or `chrome`, `edge`) or `--cookies cookies.txt`.
If captions could still be fetched, the brief is built from them without keyframes.

### Development
```bash
pip install -e ".[dev]"
pytest          # unit + end-to-end tests on a generated video; no network or model download
ruff check . && ruff format --check .
```

---

## 📂 Output Artifacts

```text
output/billing_demo/
├── agent.md                  # High-density technical brief for AI Agents
├── summary.md                # Executive business summary
├── transcript.txt            # Plaintext timestamped transcript
├── transcript.json           # Machine-readable transcript
└── keyframes/                # Extracted UI keyframes
    ├── keyframe_001_01_30.jpg
    └── metadata.json
```

---

## 📚 Documentation

Detailed architectural and developer specifications are available in the [`docs/`](./docs/) directory:
- [System Architecture](./docs/ARCHITECTURE.md)
- [Technical Specifications](./docs/SPECIFICATIONS.md)
- [Component Topology & Graphify](./docs/GRAPHIFY.md)
- [AI Agent Bootstrap Prompt](./docs/AGENT_BOOTSTRAP_PROMPT.md)

---

## 🛡️ Privacy Guarantee

Video-Analyzer processes everything locally on your machine. No video frames, audio tracks, or customer data are ever transmitted to any third-party cloud service. Intermediate audio and frames are purged immediately after execution.

---

## 📄 License
MIT License. Open-source for the developer and AI agent community.
