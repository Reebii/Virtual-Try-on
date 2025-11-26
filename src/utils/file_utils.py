import os
import shutil
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageOps
import io

from src.config.settings import settings
from src.utils.logger import logger

class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def validate_image_path(image_path: Path) -> Tuple[bool, str]:
        """Validate image file path and properties."""
        try:
            if not image_path.exists():
                return False, f"File does not exist: {image_path}"
            
            if not image_path.is_file():
                return False, f"Path is not a file: {image_path}"
            
            # Check file extension
            if image_path.suffix.lower() not in settings.ALLOWED_EXTENSIONS:
                return False, f"Unsupported file format: {image_path.suffix}"
            
            # Check file size
            file_size = image_path.stat().st_size
            if file_size > settings.MAX_IMAGE_SIZE:
                return False, f"File too large: {file_size} bytes"
            
            # Verify it's a valid image
            with Image.open(image_path) as img:
                img.verify()
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"
    
    @staticmethod
    def get_images_from_directory(directory: Path) -> List[Path]:
        """Get all valid images from directory."""
        images = []
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return images
        
        for ext in settings.ALLOWED_EXTENSIONS:
            pattern = f"*{ext}" if not ext.startswith("*") else ext
            images.extend(directory.glob(pattern))
        
        # Validate each image
        valid_images = []
        for img_path in images:
            is_valid, message = FileUtils.validate_image_path(img_path)
            if is_valid:
                valid_images.append(img_path)
            else:
                logger.warning(f"Invalid image {img_path}: {message}")
        
        return valid_images
    
    @staticmethod
    def prepare_output_directory() -> bool:
        """Create and prepare output directory."""
        try:
            settings.OUTPUT_DIR.mkdir(exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            return False
    
    @staticmethod
    def resize_image_if_needed(image_path: Path, max_dimension: int = 2048) -> Image.Image:
        """Resize image if it exceeds maximum dimensions."""
        try:
            with Image.open(image_path) as img:
                if max(img.size) <= max_dimension:
                    return img.copy()
                
                # Calculate new dimensions maintaining aspect ratio
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                
                return img.resize(new_size, Image.Resampling.LANCZOS)
                
        except Exception as e:
            logger.error(f"Failed to resize image {image_path}: {e}")
            raise