from ultralytics import YOLO
import numpy as np
from engine.base_engine import BaseInferenceEngine

class CPUEngine(BaseInferenceEngine):
    """Fallback local engine running strictly on host CPU layout for development laptops."""
    
    DEFAULT_MODEL = "yolov8n.pt"

    def __init__(self, model_path: str = None):
        self.model_path = model_path if model_path else self.DEFAULT_MODEL
        self.model = None

    def initialize(self) -> None:
        print(f"[Engine] Initializing CPU Engine with weights: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to("cpu")

    def track_frame(self, frame: np.ndarray) -> tuple[list[list[int]], list[int], list[str]]:
        results = self.model.track(frame, persist=True, device="cpu", verbose=False)
        
        # Extract the first frame result from the batch list
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return [], [], []

        frame_result = results[0]
        boxes = frame_result.boxes.xyxy.int().tolist()
        track_ids = frame_result.boxes.id.int().tolist()
        names = self.model.names
        labels = [names[c] for c in frame_result.boxes.cls.int().tolist()]

        return boxes, track_ids, labels
