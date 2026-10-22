# GRAPHIFY.md: Component Topology & Data Flow Graph

This document provides a visual knowledge graph and architectural model for the **Video-Analyzer** system.

---

## 1. System Component Graph

```mermaid
graph TD
    subgraph CLI [CLI & Entrypoint Layer]
        Main[cli.py: Click / Typer CLI]
        Config[config.py: Global & Pipeline Settings]
        Main --> Config
    end

    subgraph IngestionLayer [Input Ingestion & Demuxer]
        LocalHandler[handlers/local_video.py]
        YTHandler[handlers/youtube_video.py]
        FFmpegWrapper[utils/ffmpeg_util.py]
        Main --> LocalHandler
        Main --> YTHandler
        LocalHandler --> FFmpegWrapper
        YTHandler --> FFmpegWrapper
    end

    subgraph ExtractionLayer [Parallel Processing Pipelines]
        subgraph AudioPipeline [Audio Track Extraction]
            AudioExtractor[utils/audio_extractor.py: 16kHz Mono]
            WhisperEngine[engines/whisper_asr.py: faster-whisper]
            AudioExtractor --> WhisperEngine
        end

        subgraph VisionPipeline [Vision & Screen Extraction]
            KeyframeDetector[engines/scene_detector.py: pHash / SSIM]
            OCREngine[engines/ocr_extractor.py: PaddleOCR / EasyOCR]
            KeyframeDetector --> OCREngine
        end
    end

    subgraph SynthesisLayer [Semantic Alignment & Formatting]
        Synthesizer[synthesis/timeline_aligner.py]
        AgentGenerator[synthesis/agent_md_generator.py]
        SummaryGenerator[synthesis/summary_generator.py]
        TranscriptFormatter[synthesis/transcript_formatter.py]

        WhisperEngine --> Synthesizer
        KeyframeDetector --> Synthesizer
        OCREngine --> Synthesizer

        Synthesizer --> AgentGenerator
        Synthesizer --> SummaryGenerator
        Synthesizer --> TranscriptFormatter
    end

    subgraph ArtifactStore [Output Artifacts]
        OutAgent[agent.md]
        OutSummary[summary.md]
        OutTranscript[transcript.txt / transcript.json]
        OutKeyframes[keyframes/*.jpg]

        AgentGenerator --> OutAgent
        SummaryGenerator --> OutSummary
        TranscriptFormatter --> OutTranscript
        KeyframeDetector --> OutKeyframes
    end
```

---

## 2. Data Flow & Sequential Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Calling Agent
    participant CLI as cli.py
    participant Handler as Input Handlers (Local/YT)
    participant ASR as Speech-to-Text (faster-whisper)
    participant Vision as Scene Detector & OCR
    participant Aligner as Timeline Aligner
    participant Gen as Artifact Generator

    User->>CLI: video-analyzer analyze <source>
    CLI->>Handler: Resolve source (detect local file vs YouTube URL)

    alt is YouTube URL
        Handler->>Handler: Check existing captions (youtube-transcript-api)
        Handler->>Handler: Stream audio/video via yt-dlp
    else is Local Video
        Handler->>Handler: Validate file codec and audio stream
    end

    par Audio Processing
        Handler->>ASR: Extract 16kHz WAV & transcribe
        ASR-->>Aligner: Timestamped speech segments [{start, end, text, lang}]
    and Visual Processing
        Handler->>Vision: Detect scene cuts & perceptual changes
        Vision->>Vision: Filter out static/duplicate frames
        Vision->>Vision: Run lightweight OCR on extracted keyframes
        Vision-->>Aligner: Visual states [{timestamp, keyframe_path, ocr_text}]
    end

    Aligner->>Aligner: Correlate speech with UI keyframes by timestamp
    Aligner->>Gen: Correlated timeline data
    Gen->>Gen: Render agent.md via template
    Gen->>Gen: Render summary.md and transcript.txt
    Gen-->>User: Output directory ready with all deliverables
```

---

## 3. Data Contract / Internal Schemas

```mermaid
classDiagram
    class SpeechSegment {
        +float start
        +float end
        +str text
        +float confidence
        +str language
    }

    class Keyframe {
        +float timestamp
        +str file_path
        +str resolution
        +float difference_score
        +List~str~ ocr_elements
    }

    class ActionStep {
        +int step_number
        +float start_time
        +float end_time
        +str action_title
        +str spoken_context
        +Keyframe visual_state
        +Dict~str, str~ extracted_fields
    }

    class AnalysisResult {
        +str video_title
        +float duration_seconds
        +str detected_language
        +List~SpeechSegment~ transcript
        +List~Keyframe~ keyframes
        +List~ActionStep~ action_steps
        +str summary
        +save(output_dir)
    }

    AnalysisResult *-- SpeechSegment
    AnalysisResult *-- Keyframe
    AnalysisResult *-- ActionStep
    ActionStep --> Keyframe
```
