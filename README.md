# OhPossum Video Processor (op-vp)

This repository contains the codebase for the OhPossum video processor board. The current target hardware is:
- Raspberry Pi 5
- Raspberry Pi AI HAT+ (Hailo 8 NPU) for inference hardware acceleration
- Raspberry Pi Camera Module 3

There are also future plans to support a dual-camera pipeline, using a Raspberry Pi Camera Module 3 NoIR for low light and night vision.

---

## Overview

Although this project targets the specific hardware from above, this codebase is setup to enable local testing. This is achieved via the swappable `Engine` and `VideoSource` used by the main inference pipeline. On local hardware (e.g. laptop/desktop), there are both accelerated and non-accelerated options for the `Engine` (CPU/GPU) and a local video file can be used as a replacement for a camera stream.

```text
op-vp/
├── meson.build
├── meson.options                          # Cross-platform build profile options
├── download_model.py                      # Model download/export util script
│
├── src/
│   ├── main.cpp
│   │
│   ├── engine/                            # --- Inference Engines ---
│   │   ├── base_engine.hpp                # Abstract base interface
│   │   ├── engine_factory.hpp/cpp
│   │   ├── cpu_engine.hpp/cpp             # CPU driver
│   │   ├── gpu_engine.hpp/cpp             # ONNX Runtime CUDA driver
│   │   ├── hailo_engine.hpp/cpp           # HAILO NPU driver
│   │   └── ...                            # Any other future drivers
│   │
│   └── video_source/                      # --- Video Ingestion ---
│       ├── base_source.hpp                # Abstract base interface
│       ├── video_source_factory.hpp/cpp
│       ├── file_source.hpp/cpp            # Real-time (metadata-paced) video file source
│       └── camera_source.hpp/cpp          # RPi5 libcamera driver
│
└── media/
    └── car_occlusion.mp4                  # Testing sample video
```

---

## Quick Start

### 1. Requirements and Dependencies
Ensure your environment has Meson, Ninja, OpenCV 4, and Boost Program Options installed natively.

For NVIDIA GPU execution, your environment requires the **CUDA Toolkit** matching your system architecture along with **cuDNN**.

### 2. Download and Export Models
The repository includes a Python script to download and export YOLO models. Run this to pull and compile target models into ONNX format:

```bash
uv run download_model.py --model yolov8n
```

### 3. Build and Execute
Configure and compile the application for target hardware using Meson (defaults to rpi5):

```bash
meson setup builddir
meson compile -C builddir
```

For non-target hardware testing:
```bash
meson setup builddir -Dplatform=desktop
meson compile -C builddir
```

To run (this example is commonly used for GPU desktop testing):
```bash
./builddir/opvp --engine gpu --source file --input media/car_occlusion.mp4
```
