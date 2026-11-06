from enum import Enum
from typing import Optional
from engine.base_engine import BaseInferenceEngine
from engine.cpu_engine import CPUEngine
from engine.gpu_engine import GPUEngine
from engine.hailo_engine import HailoEngine

class EngineType(Enum):
    """Strongly typed enum defining the supported hardware backends."""
    GPU = "gpu"
    CPU = "cpu"
    HAILO = "hailo"

class EngineFactory:
    """Handles runtime initialization and resource allocation for inference backends."""
    
    @staticmethod
    def create_engine(engine_type: EngineType, model_path: Optional[str] = None) -> BaseInferenceEngine:
        """
        Factory method to instantiate a specific execution engine.
        Delegates to internal class defaults if model_path is omitted.
        """
        if not isinstance(engine_type, EngineType):
            raise TypeError(f"Expected EngineType enum value, got {type(engine_type)}")

        if engine_type == EngineType.GPU:
            return GPUEngine(model_path=model_path)
            
        elif engine_type == EngineType.CPU:
            return CPUEngine(model_path=model_path)
            
        elif engine_type == EngineType.HAILO:
            return HailoEngine(model_path=model_path)
            
        else:
            raise ValueError(f"Unsupported engine type implementation: {engine_type}")
