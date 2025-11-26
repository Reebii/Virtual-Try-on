#!/usr/bin/env python3
"""
Virtual Try-On System - Single Composite Output
Generates ONE image with 4 views: front, back, side, close-up
Same model (kid), same garment across all views
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import google.generativeai as genai

from src.config.settings import settings
from src.utils.logger import logger
from src.utils.file_utils import FileUtils
from src.services.gemini_service import ImageGenerationService
from src.models.schemas import VirtualTryOnResponse, ViewType


class VirtualTryOnApp:
    """Main application class for virtual try-on system."""
    
    def __init__(self):
        self.image_service = ImageGenerationService()
        self.results: Dict[str, Any] = {}
    
    def run(self, garment_images: Optional[List[Path]] = None) -> VirtualTryOnResponse:
        """
        Execute the virtual try-on pipeline - generates single composite image.
        
        Args:
            garment_images: List of garment image paths (optional)
        """
        logger.info("Starting single composite image generation...")
        
        try:
            # Step 1: Find all subdirectories with images
            subdirs_with_images = self._find_subdirectories_with_images(settings.INPUT_DIR)
            
            if not subdirs_with_images:
                return VirtualTryOnResponse(
                    success=False,
                    generated_images=[],
                    error="No subdirectories with images found"
                )
            
            all_generated_images = []
            
            # Process each subdirectory
            for subdir in subdirs_with_images:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing directory: {subdir.name}")
                logger.info(f"{'='*60}")
                
                # Get garment images from this subdirectory
                garment_images = self._get_images_from_directory(subdir)
                
                if len(garment_images) < 1:
                    logger.warning(f"No images found in {subdir.name}, skipping...")
                    continue
                
                logger.info(f"Found {len(garment_images)} garment images:")
                for img in garment_images:
                    logger.info(f"  - {img.name}")
                
                # Create output directory within the same subdirectory
                output_dir = subdir / "output"
                output_dir.mkdir(exist_ok=True)
                logger.info(f"Output directory: {output_dir}")
                
                # Map garment images to views
                garment_map = self._map_garments_to_views(garment_images)
                
                # Generate composite image for this subdirectory
                composite_path = self._generate_composite_image(
                    garment_map,
                    garment_images,
                    output_dir
                )
                
                if composite_path:
                    all_generated_images.append(composite_path)
                    logger.info(f"✅ Generated composite for {subdir.name}")
                else:
                    logger.warning(f"⚠️ Failed to generate composite for {subdir.name}")
            
            if not all_generated_images:
                return VirtualTryOnResponse(
                    success=False,
                    generated_images=[],
                    error="Failed to generate any composite images"
                )
            
            logger.info(f"\n✅ Successfully generated {len(all_generated_images)} composite images!")
            
            return VirtualTryOnResponse(
                success=True,
                generated_images=all_generated_images
            )
            
        except Exception as e:
            logger.error(f"Virtual try-on pipeline failed: {e}")
            return VirtualTryOnResponse(
                success=False,
                generated_images=[],
                error=str(e)
            )
    
    def _setup_directories(self) -> bool:
        """Setup required directories."""
        try:
            settings.INPUT_DIR.mkdir(exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            return False
    
    def _find_subdirectories_with_images(self, parent_dir: Path) -> List[Path]:
        """Find all subdirectories that contain image files."""
        subdirs = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        try:
            # Check all subdirectories
            for item in parent_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name != 'output':
                    # Check if this directory has any images
                    has_images = any(
                        f.suffix.lower() in image_extensions 
                        for f in item.iterdir() 
                        if f.is_file()
                    )
                    if has_images:
                        subdirs.append(item)
                        logger.info(f"Found subdirectory with images: {item.name}")
            
            return subdirs
        except Exception as e:
            logger.error(f"Error finding subdirectories: {e}")
            return []
    
    def _get_images_from_directory(self, directory: Path) -> List[Path]:
        """Get all image files from a specific directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        images = []
        
        try:
            for file in directory.iterdir():
                if file.is_file() and file.suffix.lower() in image_extensions:
                    images.append(file)
            return sorted(images)
        except Exception as e:
            logger.error(f"Error reading images from {directory}: {e}")
            return []
    
    def _map_garments_to_views(self, garment_images: List[Path]) -> Dict[str, Path]:
        """Map garment images to appropriate views based on filename."""
        garment_map = {}
        default_image = garment_images[0]
        
        view_keywords = {
            ViewType.FRONT: ['front', 'forward'],
            ViewType.BACK: ['back', 'rear'],
            ViewType.SIDE: ['side', 'lateral', 'profile'],
            ViewType.CLOSE_UP: ['close', 'detail', 'closeup', 'zoom']
        }
        
        for view_type in [ViewType.FRONT, ViewType.BACK, ViewType.SIDE, ViewType.CLOSE_UP]:
            matched = False
            
            for img in garment_images:
                img_name_lower = img.stem.lower()
                
                for keyword in view_keywords[view_type]:
                    if keyword in img_name_lower:
                        garment_map[view_type.value] = img
                        logger.info(f"Matched {img.name} to {view_type.value} view")
                        matched = True
                        break
                
                if matched:
                    break
            
            if not matched:
                garment_map[view_type.value] = default_image
                logger.info(f"Using default {default_image.name} for {view_type.value} view")
        
        return garment_map
    
    def _generate_composite_image(
        self,
        garment_map: Dict[str, Path],
        all_garment_images: List[Path],
        output_dir: Path
    ) -> Optional[Path]:
        """
        Generate single composite image with all 4 views.
        Uses ONE consistent prompt to ensure same model across all views.
        Sends ALL images in the subdirectory as context to Gemini.
        
        Args:
            garment_map: Mapping of view types to garment image paths
            all_garment_images: All image files in the subdirectory
            output_dir: Directory where the composite image should be saved
        """
        logger.info("🎨 Generating single composite image with 4 views...")

        try:
            garment_images_pil: List[Image.Image] = []
            used_paths = set()

            # 1. Add one primary image per view (front, side, back, close-up)
            view_order = [ViewType.FRONT, ViewType.SIDE, ViewType.BACK, ViewType.CLOSE_UP]

            for view in view_order:
                garment_path = garment_map.get(view.value)
                if garment_path and garment_path not in used_paths:
                    img = self.image_service._prepare_image(garment_path)
                    if img:
                        garment_images_pil.append(img)
                        used_paths.add(garment_path)
                        logger.info(f"Prepared {garment_path.name} for {view.value}")
                    else:
                        logger.warning(f"Failed to prepare image {garment_path} for {view.value}")

            # 2. Add ALL remaining images from the subdirectory as extra context
            for img_path in all_garment_images:
                if img_path not in used_paths:
                    img = self.image_service._prepare_image(img_path)
                    if img:
                        garment_images_pil.append(img)
                        used_paths.add(img_path)
                        logger.info(f"Prepared extra context image {img_path.name}")
                    else:
                        logger.warning(f"Failed to prepare extra context image {img_path}")

            if not garment_images_pil:
                logger.error("No garment images could be prepared")
                return None

            # Build comprehensive prompt for all views
            prompt = self._build_composite_prompt(garment_map)

            # Create content with prompt and all garment images
            content_parts = [prompt] + garment_images_pil

            logger.info("🎨 Calling API to generate composite image with ALL context images...")
            
            # 👉 Here we'll use top_p, top_k and seed
            response = self.image_service.model.generate_content(
                content_parts,
                generation_config={
                    'temperature': 0.15,   # lower for strong consistency
                    'top_p': 0.8,
                    'top_k': 40,
                    'candidate_count': 1,
                },
            )

            # Save the composite image in the output directory
            output_path = output_dir / "composite_all_views.png"
            

            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data:
                                image_data = part.inline_data.data

                                if isinstance(image_data, bytes):
                                    with open(output_path, 'wb') as f:
                                        f.write(image_data)
                                else:
                                    import base64
                                    image_bytes = base64.b64decode(image_data)
                                    with open(output_path, 'wb') as f:
                                        f.write(image_bytes)

                                logger.info(f"✅ Composite image saved to: {output_path}")
                                
                                # Use the new pipeline for complete processing
                                pipeline_results = self.image_service.process_composite_pipeline(
                                    garment_images=all_garment_images,
                                    composite_path=output_path,
                                    output_dir=output_dir,
                                    scale_factor=4,
                                    strength="high"
                                )
                                
                                # Log the pipeline results
                                if pipeline_results.get('upscaled_composite'):
                                    logger.info(f"✅ Upscaled composite: {pipeline_results['upscaled_composite']}")
                                
                                if pipeline_results.get('cropped_views'):
                                    logger.info(f"✅ Cropped views: {len(pipeline_results['cropped_views'])} images")
                                    for view_name, path in pipeline_results['cropped_views'].items():
                                        logger.info(f"   - {view_name}: {path.name}")
                                
                                if pipeline_results.get('zucchini_views'):
                                    logger.info(f"✅ Zucchini refined views: {len(pipeline_results['zucchini_views'])} images")
                                    for view_name, path in pipeline_results['zucchini_views'].items():
                                        logger.info(f"   - {view_name}: {path.name}")
                                
                                return output_path

            logger.warning("⚠️ No image in response")
            return None

        except Exception as e:
            logger.error(f"❌ Composite generation failed: {e}")
            return None

    def _build_composite_prompt(self, garment_map: Dict[str, Path]) -> str:
        """Build prompt for single composite image with all 4 views."""
        
        return """Create a professional fashion catalog page showing ONE CHILD MODEL wearing the same garment in 4 different views.

CRITICAL LAYOUT REQUIREMENT:

Create a 2x2 grid layout showing:
- TOP-LEFT QUADRANT: FULL-BODY FRONT VIEW
  * Child facing camera directly
  * Both arms visible, natural standing pose
  * Entire garment visible from front

- TOP-RIGHT QUADRANT: FULL-BODY SIDE PROFILE  
  * Child in 90-degree side view
  * Profile face visible, side silhouette clear
  * Garment side details and fit shown

- BOTTOM-LEFT QUADRANT: FULL-BODY BACK VIEW
  * Child facing away from camera
  * Back of garment completely visible
  * Back design details, tags, closures shown

- BOTTOM-RIGHT QUADRANT: GARMENT CLOSE-UP DETAIL
  * Focus on fabric texture, stitching, patterns
  * Show collar, sleeves, or unique design elements
  * Macro-style photography of garment details

ABSOLUTE REQUIREMENTS - CONSISTENCY:
- Use the SAME CHILD MODEL in all 4 views
- Keep the child's FACE identical across front, side views
- Keep the child's BODY TYPE identical (same height, build, proportions)
- Keep the child's SKIN TONE identical across all views
- Keep the child's HAIR identical (same style, color, length)
- Keep the child's AGE appearance identical (same age in all views)

GARMENT REQUIREMENTS - 100% ACCURACY:
- Use the EXACT garment shown in the reference images
- Match garment COLOR precisely - same shade, same tone in all views
- Match garment PATTERN exactly - same design, same print
- Match garment STYLE exactly - same cut, same fit, same silhouette
- Keep all DESIGN DETAILS - buttons, zippers, pockets, logos exactly as shown
- The garment must look IDENTICAL across all 4 views, just from different angles

MODEL SPECIFICATIONS:
- Child model (age-appropriate for the garment)
- Professional, natural poses appropriate for each view
- Same child, same outfit, different angles only
- Clean white studio background for all views
- Professional fashion photography lighting

TECHNICAL REQUIREMENTS:
- High resolution composite image
- NO text labels or words on the image
- Clean, minimal presentation without any text overlays
- Photorealistic rendering
- Sharp focus on garment details
- Professional fashion catalog quality
- 2048x2048 or larger resolution

OUTPUT: Single composite image showing 4 views of the SAME child model wearing the EXACT same garment."""


def main():
    """Main entry point."""
    print("🚀 Virtual Try-On System Starting...")
    print("👧 Single Composite Mode: 4 views, same kid model, 100% accurate garment")
    print("📁 Processing subdirectories with images...")
    print(f"Input Directory: {settings.INPUT_DIR}")
    print("-" * 50)
    
    app = VirtualTryOnApp()
    result = app.run()
    
    print("\n" + "=" * 50)
    if result.success:
        print("✅ Composite Images Generated Successfully!")
        print(f"📁 Generated {len(result.generated_images)} composite image(s):")
        for img_path in result.generated_images:
            print(f"   ✨ {img_path.name}")
            print(f"   📍 {img_path}")
    else:
        print("❌ Generation Failed!")
        print(f"Error: {result.error}")
    
    print(f"\n📋 Check '{settings.BASE_DIR / 'virtual_tryon.log'}' for detailed logs.")
    print(f"💡 Tip: Output files are saved in 'output' folder within each subdirectory")


if __name__ == "__main__":
    main()