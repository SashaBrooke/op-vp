# OhPossum Video Processor

This repository contains code for the OhPossum video processor board. This is implemented as a Python application that executes real-time object detection and tracking using YOLO, while handling USB Serial and Ethernet communications (system controls and control system data exchange).

## System Layout
- `src/main.py`: Main entry executable handling parameter passing, enum decoding, and the core processing loop.
- `src/engine/`: Package containing isolated hardware inference backends managed via `EngineFactory`.
- `src/video_source/`: Package containing stream capture variations managed via `VideoSourceFactory`.
- `media/`: Local storage folder for integration testing media containers.

---

## Execution Guide

Run your commands from the root directory using the `uv` environment coordinator. Your dependencies are locked and read entirely from `uv.lock`.

### 1. Local Testing (Laptop Default Mode)
Runs the strict GPU pipeline processing a sample local file using its default internal weight targeting (`yolov8n.pt`). It will instantly throw an error if a valid CUDA installation is missing.
```bash
uv run src/main.py --engine gpu --source file --input media/sample.mp4
```

### 2. CPU Emergency Fallback Testing
If your laptop lacks an Nvidia GPU, use the CPU engine variant to debug scripts locally:
```bash
uv run src/main.py --engine cpu --source file --input media/sample.mp4
```

### 3. Deploying onto the Target Raspberry Pi 5
When deploying onto hardware containing the Raspberry Pi Camera Module 3 and the Hailo-8L AI Acceleration Module, execute using this syntax configuration:
```bash
libcamerify uv run src/main.py --engine hailo --source camera --cam_idx 0
```
*(Note: `libcamerify` is used to smoothly map the Pi Camera Module 3's complex libcamera format directly into standard V4L2 pipelines for OpenCV).*

---

## Command Arguments Parameter Reference

| Flag | Options | Default | Functional Description |
| :--- | :--- | :--- | :--- |
| `--engine` | `gpu`, `cpu`, `hailo` | **`gpu`** | Targets the hardware acceleration engine. `gpu` requires valid CUDA; `hailo` targets the RPi5 AI Module. |
| `--source` | `file`, `camera` | **`file`** | Selects frame stream ingestion interface method. |
| `--input` | *String Path* | **`media/sample.mp4`** | Target file path configuration string used exclusively when `--source file` is active. |
| `--model` | *String Path* | **`None`** | Manual overrides configuration file string path. Omit flag entirely to enforce clean internal engine class defaults. |
| `--cam_idx` | *Integer* | **`0`** | System V4L2 reference tracking index mapped when using `--source camera`. |
