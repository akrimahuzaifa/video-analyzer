# Video-Analyzer: System Architecture & Design

## 1. Vision & Core Philosophy

**Video-Analyzer** is an open-source, local-first engine that transforms screen recordings, tutorials, and demonstration videos (local private files or public YouTube links) into highly structured, token-efficient intelligence artifacts for AI coding agents (Claude Code, Cursor, ChatGPT, Gemini, Antigravity).

### Why this project exists:
1. **Token Exhaustion on Free/Starter Tiers**: Passing raw video files or hundreds of 1080p frames to multimodal LLMs rapidly consumes 100k-500k tokens per query, instantly exhausting context windows and triggering rate limits. Video-Analyzer extracts the exact semantic essence into a compressed, high-density gent.md saving 85-95% of tokens.
2. **Zero Privacy Leaks (Local-First)**: Enterprise screen recordings frequently contain confidential customer PII, customer billing data, passwords, and internal URLs. Users should not be forced to upload unlisted/public videos to YouTube or third-party servers. All computation (audio extraction, Whisper ASR, scene detection, OCR) runs 100% locally on the user\'s machine.
3. **Multilingual & Code-Switching Intelligence**: Real-world walkthroughs often involve bilingual narration (Urdu/Hindi/Spanish mixed with English technical terms). Video-Analyzer uses Whisper\'s multilingual models with phonetic recovery to preserve technical terminology.
4. **Intelligent Keyframe Reduction**: Rather than naive interval sampling (e.g. 1 frame every second = 1800 images), Video-Analyzer uses scene-cut detection, perceptual hashing (pHash), and structural similarity (SSIM) to capture *only* meaningful UI changes, modals, and user clicks (~20-40 keyframes).

---

## 2. High-Level Architecture

`mermaid
graph TD
    subgraph Inputs [Input Sources]
        LocalVideo[Local Video File: .mp4, .mkv, .mov, .webm]
        YTVideo[YouTube URL / Video ID]
    end

    subgraph Ingestion [Ingestion & Extraction Engine]
        YTDLP[yt-dlp: Stream / Subtitle / Video Fetcher]
        FFmpeg[imageio-ffmpeg / system ffmpeg: Audio & Frame Demuxer]
        LocalVideo --> FFmpeg
        YTVideo --> YTDLP
        YTDLP --> FFmpeg
    end

    subgraph ProcessingPipeline [Parallel Processing Pipeline]
        subgraph AudioTrack [Audio & Speech Pipeline]
            AudioPrep[Audio Normalizer: 16kHz Mono WAV]
            ASR[faster-whisper / whisper.cpp: Multilingual Speech-to-Text]
            AudioPrep --> ASR
        end

        subgraph VisualTrack [Visual & UI Pipeline]
            SceneDetect[Scene Cut & Perceptual Hash Filter]
            OCR[PaddleOCR / EasyOCR / Tesseract: UI Text & Form Extractor]
            CropScale[UI Element & Modal Cropper]
            SceneDetect --> CropScale
            CropScale --> OCR
        end

        FFmpeg --> AudioPrep
        FFmpeg --> SceneDetect
    end

    subgraph SynthesisEngine [Artifact Generation Engine]
        Aligner[Timestamp-to-Visual Synchronizer]
        TemplateGen[Jinja2 Agent Markdown Engine]
        ASR --> Aligner
        OCR --> Aligner
        SceneDetect --> Aligner
        Aligner --> TemplateGen
    end

    subgraph OutputArtifacts [Generated Deliverables]
        OutTranscript[transcript.txt / transcript.json]
        OutSummary[summary.md]
        OutAgent[agent.md: Optimized for AI Agents]
        OutKeyframes[keyframes/: Minimal Keyframes with Annotations]
        TemplateGen --> OutTranscript
        TemplateGen --> OutSummary
        TemplateGen --> OutAgent
        TemplateGen --> OutKeyframes
    end
`

---

## 3. Component Breakdown

### 3.1 Ingestion Layer (ingestion/)
- **Local Ingest**: Accepts local file paths, validates video codec, resolution, and audio channels using fprobe / OpenCV.
- **YouTube Ingest**: Uses yt-dlp and youtube-transcript-api. If YouTube automated subtitles already exist in the target language, it downloads them in 2 seconds without requiring audio transcription! If not, it streams the best audio-only format directly into the ASR pipeline.

### 3.2 Speech-to-Text Engine (	ranscription/)
- Powered by aster-whisper (CTranslate2) or whisper.cpp for ultra-fast local CPU/GPU inference.
- Auto-detects spoken languages (Urdu, Hindi, Spanish, English, etc.).
- Produces normalized JSON with word-level or sentence-level timestamps [{start: float, end: float, text: str}].

### 3.3 Visual & UI Extractor (ision/)
- **Keyframe Extraction**:
  - Uses OpenCV content-aware difference (frame differencing) and pHash to drop duplicate / static frames.
  - Generates ~1 keyframe per meaningful screen transition (new tab, modal open, dropdown selection, button click).
- **OCR & Field Extraction**:
  - Lightweight OCR on keyframe regions to extract button labels, table column headers, form inputs, URLs, and status badges.

### 3.4 Artifact Generator (generator/)
- **	ranscript.txt**: Raw chronological transcript with timestamps.
- **summary.md**: Executive summary (What is this video? What is the goal? What systems are involved?).
- **gent.md**: The crowning deliverable. A structured technical brief designed specifically for LLMs:
  - Step-by-step numbered instructions.
  - Exact input field names, button labels, and expected outputs.
  - Data mapping tables (e.g. Source Field -> Target CRM Field).
  - Business logic & edge-case rules extracted from voiceover.
  - File/code templates for immediate implementation.

---

## 4. Privacy & Ephemeral Storage Contract
1. **Zero Data Retention**: Uploaded/local videos are read directly from local disk. Intermediate audio (.wav) and uncompressed frames in ./tmp are purged immediately upon generation unless --keep-temp is specified.
2. **Local Execution**: No video or audio is ever sent to external cloud APIs by default. The entire pipeline works completely offline on Windows, Linux, and macOS.
