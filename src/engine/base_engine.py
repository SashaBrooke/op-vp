from abc import ABC, abstractmethod
import numpy as np

class BaseInferenceEngine(ABC):
    """Abstract interface defining required behaviors for all hardware backends."""

    @abstractmethod
    def initialize(self) -> None:
        """Allocate hardware resources and load the model weights into memory."""
        pass

    @abstractmethod
    def track_frame(self, frame: np.ndarray) -> tuple[list[list[int]], list[int], list[str]]:
        """
        Executes simultaneous object detection and tracking on a single frame.
        Returns: (bounding_boxes, tracking_ids, class_labels)
        """
        pass
