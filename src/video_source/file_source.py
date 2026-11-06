import cv2
import queue
import threading
import time
import numpy as np
from video_source.base_source import BaseVideoSource

class FileSource(BaseVideoSource):
    """Handles parsing and asynchronous pre-loading of compressed local media containers."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=128)  # Buffer up to 128 frames in RAM
        self.running = False
        self.worker_thread = None

    def _reader_worker(self):
        """Background worker that continuously decodes frames into RAM."""
        while self.running:
            if not self.frame_queue.full():
                success, frame = self.cap.read()
                if not success:
                    self.running = False
                    break
                self.frame_queue.put((True, frame))
            else:
                time.sleep(0.001)  # Prevent CPU spinning when buffer is full

    def start(self) -> None:
        print(f"[Source] Opening media file source path: {self.file_path}")
        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Failed to open video file at path: {self.file_path}")
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.worker_thread.start()

    def read_frame(self) -> tuple[bool, np.ndarray]:
        # If the background thread stopped and the queue is empty, the video is finished
        if not self.running and self.frame_queue.empty():
            return False, np.empty((0,))
        
        try:
            # Grab the next pre-loaded frame from RAM instantly
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, np.empty((0,))

    def release(self) -> None:
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            print("[Source] Async media file source closed.")
