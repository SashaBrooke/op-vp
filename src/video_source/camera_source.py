import cv2
import numpy as np
from video_source.base_source import BaseVideoSource

class CameraSource(BaseVideoSource):
    """Binds camera pipelines (V4L2 for USB webcams or rpicam-apps/libcamerify for Module 3)."""
    
    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None

    def start(self) -> None:
        print(f"[Source] Attaching to camera device hardware index: {self.index}")
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeWarning(f"Hardware camera device index {self.index} could not be initialized.")

    def read_frame(self) -> tuple[bool, np.ndarray]:
        if self.cap is None:
            return False, np.empty((0,))
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            print("[Source] Camera hardware links disconnected.")
