#!/usr/bin/env uv run
# /// script
# dependencies = [
#     "ultralytics>=8.3.0",
#     "onnx>=1.16.0",
#     "onnxruntime>=1.19.0",
#     "onnxslim>=0.1.90",
# ]
# ///

import argparse
import os
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="OPVP Model Downloader and ONNX Exporter"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="Target YOLOv8 model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Target folder"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Target naming definitions
    pt_model_name = f"{args.model}.pt"
    onnx_model_name = f"{args.model}.onnx"
    final_onnx_target = output_path / onnx_model_name

    print(f"[Downloader] Initializing workspace allocation for {args.model}...")

    try:
        from ultralytics import YOLO

        # Fetch weights from Ultralytics assets server
        print(f"[Downloader] Pulling weights: {pt_model_name}")
        model = YOLO(pt_model_name)

        # Export to ONNX format
        print(f"[Exporter] Transforming to ONNX...")
        # format='onnx' transforms inputs into standard continuous tensor formats
        # imgsz=640 fixes internal execution dims to match the C++ preprocessing loops
        exported_path_str = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
        exported_path = Path(exported_path_str)

        # Clean up
        if exported_path.exists() and exported_path != final_onnx_target:
            if final_onnx_target.exists():
                final_onnx_target.unlink()
            exported_path.rename(final_onnx_target)

        if Path(pt_model_name).exists():
            Path(pt_model_name).unlink()

        print("\n-----------------------------------------------------------------")
        print(f"[Success] Model compiled and verified: {final_onnx_target}")
        print("-----------------------------------------------------------------")

    except Exception as e:
        print(f"\n[Fatal Failure] Conversion pipeline failed to compile: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
