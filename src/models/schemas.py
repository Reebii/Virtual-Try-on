from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from enum import Enum

class ViewType(str, Enum):
    FRONT = "front"
    BACK = "back"
    SIDE = "side"
    CLOSE_UP = "close-up"

class ImageAnalysisRequest(BaseModel):
    image_paths: List[Path]
    prompt: Optional[str] = None

class ImageGenerationRequest(BaseModel):
    description: str
    view_type: ViewType
    output_path: Path

class VirtualTryOnResponse(BaseModel):
    success: bool
    generated_images: List[Path]
    analysis: Optional[str] = None
    error: Optional[str] = None