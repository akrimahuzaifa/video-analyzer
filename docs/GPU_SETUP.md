# GPU Setup (NVIDIA CUDA)

Video-Analyzer transcribes with `faster-whisper` (CTranslate2). With `--device auto` (the default) it
uses an NVIDIA GPU when one is usable and falls back to CPU (int8) otherwise. A broken GPU setup never
fails a run; it only makes it slower.

| Whisper model | Use on CPU | Use on GPU |
| :--- | :--- | :--- |
| `tiny`, `base` | fine | fine |
| `small` | slow on long videos | recommended default on a GPU |
| `medium`, `large-v3` | impractical for long videos | best accuracy (Urdu/Hindi, code-switching) |

As a rough guide, the 25-second test clip took 8.0 s on CPU and 3.4 s on an RTX 4060 Laptop GPU with
`--model small`, including model loading. On long videos the gap grows to roughly 5-10x.

---

## 1. Requirements

- An NVIDIA GPU and a recent driver. CTranslate2 4.x needs the **CUDA 12** runtime, which any driver
  reporting `CUDA Version: 12.x` or `13.x` in `nvidia-smi` can run.
- **cuBLAS 12** and **cuDNN 9** libraries. You do **not** need to install the full CUDA Toolkit: the
  `[gpu]` extra installs both from PyPI into your virtual environment.

Check the driver:

```bash
nvidia-smi        # look for "CUDA Version: 12.x" or higher in the header
```

## 2. Install

### Windows

```powershell
pip install -e ".[gpu]"
```

This installs `nvidia-cublas-cu12` (around 500 MB). The Windows `ctranslate2` wheel already ships
`cudnn64_9.dll`. Video-Analyzer adds the `site-packages\nvidia\*\bin` folders to the DLL search path
itself, so no `PATH` changes are needed.

### Linux

```bash
pip install -e ".[gpu]"     # nvidia-cublas-cu12 + nvidia-cudnn-cu12
export LD_LIBRARY_PATH=$(python -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')
```

Add the `export` line to your shell profile or virtualenv `activate` script. The library loader only
reads `LD_LIBRARY_PATH` when the process starts, so the program cannot set it for you.

### macOS

There is no CUDA on macOS, and CTranslate2 has no Metal (MPS) backend. `--device mps` falls back to
CPU with a warning.

## 3. Verify

```bash
python -c "from video_analyzer.engines import asr; print(asr.resolve_compute(asr.Device.auto))"
# ('cuda', 'float16')  -> the GPU will be used
# ('cpu', 'int8')      -> no usable GPU found
```

Then run an analysis with `-v`:

```bash
video-analyzer analyze ./demo.mp4 --model small -v
```

If you see `Whisper failed on cuda (...); retrying on CPU int8.`, the GPU was detected but a library
is missing. The message names the library:

| Message mentions | Fix |
| :--- | :--- |
| `cublas64_12.dll` / `libcublas.so.12` | `pip install "nvidia-cublas-cu12>=12.4,<13"` (Linux: also set `LD_LIBRARY_PATH`) |
| `cudnn*` | Linux: `pip install "nvidia-cudnn-cu12>=9,<10"` and set `LD_LIBRARY_PATH` |
| `CUDA driver version is insufficient` | Update the NVIDIA driver |
| `out of memory` | Use a smaller `--model`, or `--device cpu` |

## 4. Forcing a device

```bash
video-analyzer analyze ./demo.mp4 --device cuda   # use the GPU, still falls back to CPU on error
video-analyzer analyze ./demo.mp4 --device cpu    # never touch the GPU
```

OCR (`[ocr]` extra, EasyOCR/PyTorch) also uses the GPU when Whisper resolves to CUDA, but PyTorch
needs its own CUDA build: see the install selector at https://pytorch.org/get-started/locally/.
