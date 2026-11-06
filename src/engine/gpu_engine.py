import torch
from ultralytics import YOLO
import numpy as np
from engine.base_engine import BaseInferenceEngine

class GPUEngine(BaseInferenceEngine):
    """Dedicated GPU engine utilizing local CUDA infrastructure. Throws error if unavailable."""
    
    DEFAULT_MODEL = "yolov8n.pt"

    def __init__(self, model_path: str = None):
        self.model_path = model_path if model_path else self.DEFAULT_MODEL
        self.model = None

    def initialize(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[FATAL] GPUEngine initialization failed! No CUDA-capable GPU detected. "
                "Ensure Nvidia drivers and CUDA toolkits are correctly configured."
            )
        print(f"[Engine] Initializing GPU Engine with CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[Engine] Loading weights from target path: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to("cuda")

    def track_frame(self, frame: np.ndarray) -> tuple[list[list[int]], list[int], list[str]]:
        # KEY FIX: Standard assignment is restored, but tracker is forced to ByteTrack
        results = self.model.track(frame, persist=True, device="cuda", verbose=False, tracker="bytetrack.yaml")
        
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return [], [], []

        # Restored to your clean, original block format
        frame_result = results[0]
        boxes = frame_result.boxes.xyxy.int().cpu().tolist()
        track_ids = frame_result.boxes.id.int().cpu().tolist()
        names = self.model.names
        labels = [names[c] for c in frame_result.boxes.cls.int().cpu().tolist()]

        return boxes, track_ids, labels
