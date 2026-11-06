from enum import Enum
from typing import Optional
from video_source.base_source import BaseVideoSource
from video_source.file_source import FileSource
from video_source.camera_source import CameraSource

class SourceType(Enum):
    """Strongly typed enum defining the supported video feed strategies."""
    FILE = "file"
    CAMERA = "camera"

class VideoSourceFactory:
    """Handles context initialization and device bindings for incoming frames."""

    @staticmethod
    def create_source(
        source_type: SourceType, 
        file_path: Optional[str] = None, 
        camera_idx: int = 0
    ) -> BaseVideoSource:
        """
        Factory method to instantiate a video ingest pipeline.
        Returns an implementation matching the BaseVideoSource contract.
        """
        if not isinstance(source_type, SourceType):
            raise TypeError(f"Expected SourceType enum value, got {type(source_type)}")

        if source_type == SourceType.FILE:
            if not file_path:
                raise ValueError("A valid file path string must be provided for FILE source generation.")
            return FileSource(file_path=file_path)
            
        elif source_type == SourceType.CAMERA:
            return CameraSource(index=camera_idx)
            
        else:
            raise ValueError(f"Unsupported stream source configuration: {source_type}")
