from abc import ABC, abstractmethod
import numpy as np

class BaseVideoSource(ABC):
    """Abstract interface defining continuous frame ingestion workflows."""

    @abstractmethod
    def start(self) -> None:
        """Prepare context managers, open file descriptors, or bind camera hardware."""
        pass

    @abstractmethod
    def read_frame(self) -> tuple[bool, np.ndarray]:
        """Retrieves the newest frame matrix. Returns (Status flag, Frame image)"""
        pass

    @abstractmethod
    def release(self) -> None:
        """Close hardware handles and clear processing buffers safely."""
        pass
