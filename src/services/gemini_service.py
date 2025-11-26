import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import base64
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional
import requests
from pathlib import Path
import io



from src.config.settings import settings
from src.utils.logger import logger
from src.utils.file_utils import FileUtils


class GeminiService:
    """Service for interacting with Gemini AI models using Google Generative AI SDK."""
    
    def __init__(self):
        self.model = None
        self.configured = False
        self.configure()
    

    def configure(self) -> bool:
        """Configure Gemini API client."""
        try:
            if not settings.GEMINI_API_KEY:
                logger.error("GEMINI_API_KEY not found in environment variables")
                return False
            
            # Configure the API key
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Initialize the model
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            self.configured = True
            
            logger.info("✅ Gemini API configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            return False
    
    def _prepare_image_for_gemini(self, image_path: Path) -> Optional[Image.Image]:
        """Prepare image for Gemini API."""
        try:
            img = FileUtils.resize_image_if_needed(image_path)
            
            # Convert image to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to prepare image {image_path}: {e}")
            return None


class ImageGenerationService:
    """Service for virtual try-on using Gemini image generation."""
    
    def __init__(self):
        self.model = None
        self.configured = False
        self.configure()
    
    def configure(self) -> bool:
        """Configure the client for image generation."""
        try:
            if not settings.GEMINI_API_KEY:
                logger.error("❌ GEMINI_API_KEY not found")
                return False
            
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Initialize image generation model
            try:
                self.model = genai.GenerativeModel('gemini-3-pro-image-preview')
                self.configured = True
                logger.info("✅ Gemini image generation initialized successfully!")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to initialize model: {e}")
                logger.error("Note: Image generation requires billing to be enabled")
                self.configured = False
                return False
                
        except Exception as e:
            logger.error(f"Failed to configure image generation: {e}")
            return False
    
    def _prepare_image(self, image_path: Path) -> Optional[Image.Image]:
        """Prepare image for API."""
        try:
            img = FileUtils.resize_image_if_needed(image_path)
            
            # Convert to RGB
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to prepare image {image_path}: {e}")
            return None
    
    def refine_with_zucchini_brand(
        self,
        input_path: Path,
        output_path: Path,
    ) -> bool:
        """
        Apply Zucchini brand master image processing to refine the background
        while preserving the model and garment 100% unchanged.
        
        Args:
            input_path: Path to the source image
            output_path: Path where the refined image will be saved
        
        Returns:
            True if success, False otherwise
        """
        if not self.configured or not self.model:
            logger.error("❌ ImageGenerationService not configured; cannot refine")
            return False

        logger.info(f"🎨 Applying Zucchini brand refinement to {input_path.name}")

        try:
            # Load the image
            orig_img: Optional[Image.Image] = self._prepare_image(input_path)
            if orig_img is None:
                logger.error(f"❌ Failed to load image for refinement: {input_path}")
                return False

            # Zucchini Brand Master Image Processing Prompt
            zucchini_prompt = """Zucchini Brand Master Image Processing

I. Core Preservation Instructions (Non-Negotiable)
1. Garment Integrity: CRUCIALLY, keep the model and ALL clothing (garment design, pattern, color, and fabric texture) 100% UNCHANGED. Ensure the output image accurately reflects the colors and details of the input garment.
2. Model Features: Preserve the model's appearance, posture, and body language EXACTLY as shown, applying only SUBTLE refinements toward the Zucchini posing style.

II. Environment and Background
1. Setting: Replace the background with a minimalist, high-fashion urban or architectural environment. Use elements like: stark concrete walls, polished industrial facades, clean street corners, or architecturally interesting building columns.
2. Atmosphere: The environment should embody 'Street-Style Chic' and 'Confident Minimalism'. The background should complement the look without being distracting.

III. Lighting and Mood
1. Lighting Style: Apply soft, diffused, and highly directional lighting, creating a subtle, mood-driven aesthetic.
2. Shadows: Introduce natural, soft shadows that emphasize the model's form and add depth to the architecture. Avoid harsh midday sunlight or flat, direct flash.
3. Color Palette: The overall image tone should lean toward a refined palette of soft neutrals (creams, blacks, taupes), with the garment color being the primary focal point.

IV. Model Direction and Posing
1. Posture & Energy: SUBTLY refine the model's pose to be 'Grounded, effortless' and intentional. Shoulders should be relaxed yet open.
2. Expression: The model should project 'quiet dominance' with a direct, self-assured attitude ('Pretty with Edge').
3. Hair & Makeup: Ensure the hair is sleek yet effortless. Skin should appear fresh and radiant with bold, defined features.

V. Framing and Technical Output
1. Framing: Maintain Portrait orientation. Frame the shot to allow for an editor to crop the final image to a 60–90% Frame Fill of the model/garment.
2. Output Quality: High resolution, photorealistic, professional fashion photography quality.

NEGATIVE PROMPT (To Prevent Errors):
No props, no busy background textures, no distortion, no color change to garment, no cartoon styles, no AI artifacts, no changes to the model's clothing or body.

CRITICAL: The garment and model MUST remain identical. ONLY the background and subtle lighting refinements should change."""

            # Generate refined image
            response = self.model.generate_content(
                [zucchini_prompt, orig_img],
                generation_config={
                    "temperature": 0.05,
                    "top_p": 0.9,
                    "top_k": 40,
                    "candidate_count": 1,
                },
            )

            # Extract and save the refined image
            refined_img = None
            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                data = part.inline_data.data
                                img_bytes = data if isinstance(data, bytes) else base64.b64decode(data)
                                try:
                                    refined_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    refined_img.save(output_path, "PNG", quality=98)
                                    logger.info(f"✅ Zucchini brand refinement saved to: {output_path}")
                                    return True
                                except Exception as e:
                                    logger.error(f"❌ Failed to decode refined image: {e}")
                                    return False

            logger.warning("⚠️ No refined image from model")
            return False

        except Exception as e:
            logger.error(f"❌ Zucchini brand refinement failed: {e}")
            return False

    def process_composite_pipeline(
        self,
        composite_path: Path,
        output_dir: Path,
        scale_factor: int = 4,
        strength: str = "high"
    ) -> Dict[str, Path]:
        """
        Complete pipeline: Upscale composite image and crop into 4 Zucchini-style individual views.
        
        Args:
            composite_path: Path to the composite image
            output_dir: Directory where all outputs will be saved
            scale_factor: Upscaling factor
            strength: Enhancement strength
        
        Returns:
            Dictionary with paths to all generated images
        """
        logger.info("🚀 Starting composite pipeline: Upscale → Crop → Zucchini Refinement")
        
        results = {
            'original_composite': composite_path,
            'upscaled_composite': None,
            'cropped_views': {},
            'zucchini_views': {}
        }
        
        try:
            # Step 1: Upscale the composite image
            upscaled_path = output_dir / "composite_all_views_upscaled.png"
            if self.upscale_image(composite_path, upscaled_path, scale_factor, strength, use_ai_upscaling=True):
                results['upscaled_composite'] = upscaled_path
                logger.info("✅ Composite upscaling completed")
            else:
                logger.error("❌ Composite upscaling failed")
                return results
            
            # Step 2: Crop the upscaled composite into 4 individual views
            cropped_views = self._crop_composite_to_individual_views(upscaled_path, output_dir)
            results['cropped_views'] = cropped_views
            logger.info("✅ Composite cropping completed")
            
            # Step 3: Apply Zucchini refinement to each individual view
            for view_name, cropped_path in cropped_views.items():
                zucchini_path = output_dir / f"zucchini_{view_name}.png"
                if self.refine_with_zucchini_brand(cropped_path, zucchini_path):
                    results['zucchini_views'][view_name] = zucchini_path
                    logger.info(f"✅ Zucchini refinement for {view_name} completed")
                else:
                    logger.warning(f"⚠️ Zucchini refinement for {view_name} failed")
            
            logger.info("🎉 Composite pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Composite pipeline failed: {e}")
            return results

    def _crop_composite_to_individual_views(self, composite_path: Path, output_dir: Path) -> Dict[str, Path]:
        """
        Crop the 2x2 composite image into 4 individual view images.
        Each cropped image will be resized to 1080x1440 resolution (portrait format).
        
        Args:
            composite_path: Path to the composite upscaled image
            output_dir: Directory where individual view images should be saved
        
        Returns:
            Dictionary mapping view names to file paths
        """
        logger.info("✂️ Cropping composite image into individual views...")
        
        cropped_paths = {}
        
        try:
            # Open the composite image
            composite_img = Image.open(composite_path)
            width, height = composite_img.size
            
            logger.info(f"Composite image size: {width}x{height}")
            
            # Calculate crop dimensions (2x2 grid)
            crop_width = width // 2
            crop_height = height // 2
            
            # Define the crop boxes for each view (left, upper, right, lower)
            crop_boxes = {
                "front": (0, 0, crop_width, crop_height),                    # Top-left
                "side": (crop_width, 0, width, crop_height),                 # Top-right
                "back": (0, crop_height, crop_width, height),                # Bottom-left
                "closeup": (crop_width, crop_height, width, height)          # Bottom-right
            }
            
            # Target size for Zucchini-style portrait images
            target_size = (1080, 1440)  # Portrait format for fashion photography
            
            # Crop and save each view
            for view_name, box in crop_boxes.items():
                # Crop the image
                cropped = composite_img.crop(box)
                
                # Resize to target resolution (1080x1440) - portrait format
                resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
                
                # Save the individual view
                output_path = output_dir / f"view_{view_name}.png"
                resized.save(output_path, "PNG", quality=99)
                
                cropped_paths[view_name] = output_path
                logger.info(f"✅ Saved {view_name} view: {output_path} ({target_size[0]}x{target_size[1]})")
            
            logger.info("✂️ Successfully cropped all 4 views!")
            return cropped_paths
            
        except Exception as e:
            logger.error(f"❌ Failed to crop composite image: {e}")
            return cropped_paths

    def upscale_image(
        self,
        input_path: Path,
        output_path: Path,
        scale_factor: int = 3,
        strength: str = "high",
        use_ai_upscaling: bool = True,
    ) -> bool:
        """
        Intelligent upscaling using gemini-3-pro-image-preview for quality preservation.
        
        Args:
            input_path:  Path to the source image
            output_path: Path where the upscaled image will be saved
            scale_factor: How much to multiply resolution by (2, 3, 4, etc)
            strength:    Enhancement intensity ("low", "medium", "high")
            use_ai_upscaling: If True, uses AI model for smart upscaling; if False, uses simple Pillow resize
        
        Returns:
            True if success, False otherwise
        """
        if not self.configured or not self.model:
            logger.error("❌ ImageGenerationService not configured; cannot upscale")
            return False

        try:
            orig_img: Optional[Image.Image] = self._prepare_image(input_path)
            if orig_img is None:
                logger.error(f"❌ Failed to load image for upscaling: {input_path}")
                return False
        except Exception as e:
            logger.error(f"❌ Error opening image for upscaling: {e}")
            return False

        orig_w, orig_h = orig_img.size
        target_w = orig_w * scale_factor
        target_h = orig_h * scale_factor

        logger.info(
            f"🔍 {'AI-powered' if use_ai_upscaling else 'Standard'} upscaling {input_path.name} from "
            f"{orig_w}x{orig_h} to {target_w}x{target_h}"
        )

        if not use_ai_upscaling:
            # Simple Pillow upscaling
            try:    
                upscaled_img = orig_img.resize((target_w, target_h), Image.LANCZOS)
                upscaled_img.save(output_path, "PNG", quality=98)
                logger.info(f"✅ Standard upscaled image saved to: {output_path}")
                return True
            except Exception as e:
                logger.error(f"❌ Failed to save upscaled image: {e}")
                return False

        # AI-powered upscaling with gemini-3-pro-image-preview
        if strength == "low":
            detail_text = (
                f"Upscale this image to {target_w}x{target_h} resolution. "
                "Gently enhance clarity and reduce noise. "
                "Preserve all original details, composition, colors, and style exactly as shown."
            )
        elif strength == "medium":
            detail_text = (
                f"Upscale this image to {target_w}x{target_h} resolution. "
                "Enhance sharpness and clarity while preserving all details. "
                "Maintain the exact composition, colors, garment details, and model features."
            )
        else:  # "high"
            detail_text = (
                f"Upscale this image to {target_w}x{target_h} resolution with maximum detail enhancement. "
                "Sharpen garment textures, fabric details, and edges with precision. "
                "Preserve the model's face, skin tone, and all garment colors EXACTLY as shown. "
                "Maintain perfect color accuracy, composition, and lighting. "
                "Do not add or remove any elements."
            )

        prompt = f"Professional image upscaling task.\n\n{detail_text}\n\nOutput resolution: {target_w}x{target_h} pixels.\nMaintain photorealistic quality."

        try:
            response = self.model.generate_content(
                [prompt, orig_img],
                generation_config={
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "candidate_count": 1,
                },
            )
        except Exception as e:
            logger.error(f"❌ AI upscaling failed, falling back to standard resize: {e}")
            upscaled_img = orig_img.resize((target_w, target_h), Image.LANCZOS)
            upscaled_img.save(output_path, "PNG", quality=98)
            return True

        upscaled_img = None
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            data = part.inline_data.data
                            img_bytes = data if isinstance(data, bytes) else base64.b64decode(data)
                            try:
                                upscaled_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                
                                # Verify the output size is correct, if not resize it
                                if upscaled_img.size != (target_w, target_h):
                                    logger.warning(
                                        f"⚠️ AI model returned {upscaled_img.size[0]}x{upscaled_img.size[1]} "
                                        f"instead of {target_w}x{target_h}, applying final resize..."
                                    )
                                    upscaled_img = upscaled_img.resize((target_w, target_h), Image.LANCZOS)
                                
                                upscaled_img.save(output_path, "PNG", quality=98)
                                logger.info(
                                    f"✅ AI-upscaled image saved to: {output_path} "
                                    f"({upscaled_img.size[0]}x{upscaled_img.size[1]})"
                                )
                                return True
                            except Exception as e:
                                logger.error(f"❌ Failed to decode upscaled image: {e}")

        if upscaled_img is None:
            logger.warning("⚠️ No upscaled image from AI model; using standard resize")
            upscaled_img = orig_img.resize((target_w, target_h), Image.LANCZOS)
            upscaled_img.save(output_path, "PNG", quality=98)
            return True

        return False
    
    def generate_image(self, description: str, view_type: str, output_path: Path, 
                      resolution: str = "2K", model_image_path: Optional[Path] = None,
                      garment_image_path: Optional[Path] = None) -> bool:
        """
        Generate image - supports text-to-image, virtual try-on, and garment-to-model.
        
        Args:
            description: Text description (for text-to-image mode)
            view_type: Type of view (front, back, side, close-up)
            output_path: Path to save the generated image
            resolution: Image resolution - "1K", "2K", or "4K" (default: "2K")
            model_image_path: Optional model image for virtual try-on
            garment_image_path: Optional garment image for virtual try-on or garment-to-model
        """
        if not self.configured or not self.model:
            logger.warning("⚠️ Model not configured, creating placeholder...")
            return self._create_placeholder_image(description or "Garment", view_type, output_path)
        
        # Check mode: virtual try-on (both images) vs garment-to-model (garment only)
        if model_image_path and garment_image_path:
            return self._virtual_tryon(model_image_path, garment_image_path, view_type, output_path, resolution)
        elif garment_image_path:
            return self._garment_to_model(garment_image_path, view_type, output_path, resolution)
        
        # Regular text-to-image generation
        logger.info(f"🎨 Generating {view_type} view ({resolution})...")
        
        try:
            prompt = self._build_prompt(description, view_type, resolution)
            
            logger.info("🍌 Calling image generation API...")
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.2,
                    'candidate_count': 1,
                }
            )
            
            # Extract image from response
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
                                    image_bytes = base64.b64decode(image_data)
                                    with open(output_path, 'wb') as f:
                                        f.write(image_bytes)
                                
                                logger.info(f"✅ Successfully generated {view_type} view!")
                                logger.info(f"💾 Saved to: {output_path}")
                                return True
            
            logger.warning("⚠️ No image data in response")
            return self._create_placeholder_image(description, view_type, output_path)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Generation failed: {error_msg}")
            
            if "403" in error_msg or "billing" in error_msg.lower():
                logger.error("💳 Billing not enabled. Enable billing at: https://console.cloud.google.com/billing")
            elif "404" in error_msg:
                logger.error("❌ Model not available. Make sure you have access to Gemini image generation.")
            elif "429" in error_msg:
                logger.error("⏱️ Rate limit exceeded. Wait a moment and try again.")
            elif "SAFETY" in error_msg.upper():
                logger.error("🛡️ Content blocked by safety filters. Try modifying your prompt.")
            
            logger.info("Creating placeholder image instead...")
            return self._create_placeholder_image(description, view_type, output_path)
    
    def _garment_to_model(self, garment_image_path: Path, view_type: str, 
                         output_path: Path, resolution: str = "2K") -> bool:
        """
        Generate model wearing the garment (garment-only mode).
        """
        logger.info(f"👗 Generating model wearing garment - {view_type} view ({resolution})...")
        
        try:
            # Prepare garment image
            garment_img = self._prepare_image(garment_image_path)
            
            if not garment_img:
                logger.error("Failed to prepare garment image")
                return self._create_placeholder_image("Garment Model", view_type, output_path)
            
            # Build prompt for garment-to-model
            prompt = self._build_garment_to_model_prompt(view_type, resolution)
            
            # Generate with garment image
            content_parts = [prompt, garment_img]
            
            logger.info("🎨 Generating model with garment...")
            response = self.model.generate_content(
                content_parts,
                generation_config={
                    'temperature': 0.1,
                    'candidate_count': 1,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Extract and save image
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
                                    image_bytes = base64.b64decode(image_data)
                                    with open(output_path, 'wb') as f:
                                        f.write(image_bytes)
                                
                                logger.info(f"✅ Model generation successful! Saved to: {output_path}")
                                return True
            
            logger.warning("⚠️ No image in response")
            return self._create_placeholder_image("Garment Model", view_type, output_path)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Model generation failed: {error_msg}")
            
            if "403" in error_msg or "billing" in error_msg.lower():
                logger.error("💳 Enable billing at: https://console.cloud.google.com/billing")
            elif "429" in error_msg:
                logger.error("⏱️ Rate limit exceeded. Wait and try again.")
            
            return self._create_placeholder_image("Garment Model", view_type, output_path)
    
    def _build_garment_to_model_prompt(self, view_type: str, resolution: str) -> str:
        """Build prompt for generating model wearing the garment."""
        
        view_instructions = {
            "front": "front view, facing camera",
            "back": "back view, showing back of garment",
            "side": "side profile view",
            "close-up": "close-up detail view"
        }
        
        view_text = view_instructions.get(view_type.lower(), "front view")
        
        return f"""Generate a professional fashion photograph of a model wearing this exact garment.

CRITICAL REQUIREMENTS - GARMENT ACCURACY:
- Use the EXACT garment shown in the image
- Match the garment's COLOR precisely - same shade, same tone
- Match the garment's PATTERN exactly - same design, same print, same details
- Match the garment's STYLE exactly - same cut, same fit, same silhouette
- Match the garment's FABRIC TEXTURE - same material appearance
- Keep all DESIGN DETAILS - buttons, zippers, pockets, logos, stitching exactly as shown
- Maintain the garment's PROPORTIONS and FIT STYLE

MODEL & COMPOSITION:
- Professional fashion model, appropriate for the garment type
- {view_text}
- Natural, professional pose
- Clean white studio background
- Professional fashion photography lighting
- High-quality commercial photography style
- Model should complement but not overshadow the garment

TECHNICAL:
- Photorealistic rendering
- Sharp focus on garment details
- Accurate colors and textures
- Resolution: {resolution}
- Commercial fashion catalog quality

The garment MUST look exactly as shown in the reference image. Generate a professional fashion photograph."""
    
    def _virtual_tryon(self, model_image_path: Path, garment_image_path: Path, 
                      view_type: str, output_path: Path, resolution: str = "2K") -> bool:
        """
        Generate virtual try-on: put the garment on the model.
        """
        logger.info(f"👕 Generating virtual try-on - {view_type} view ({resolution})...")
        
        try:
            # Prepare images
            model_img = self._prepare_image(model_image_path)
            garment_img = self._prepare_image(garment_image_path)
            
            if not model_img or not garment_img:
                logger.error("Failed to prepare images")
                return self._create_placeholder_image("Virtual Try-On", view_type, output_path)
            
            # Build simple prompt
            prompt = self._build_tryon_prompt(view_type, resolution)
            
            # Generate with both images
            content_parts = [prompt, model_img, garment_img]
            
            logger.info("🎨 Generating virtual try-on...")
            response = self.model.generate_content(
                content_parts,
                generation_config={
                    'temperature': 0.25,  # Lower temp for more consistency
                    'candidate_count': 1,
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Extract and save image
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
                                    image_bytes = base64.b64decode(image_data)
                                    with open(output_path, 'wb') as f:
                                        f.write(image_bytes)
                                
                                logger.info(f"✅ Virtual try-on successful! Saved to: {output_path}")
                                return True
            
            logger.warning("⚠️ No image in response")
            return self._create_placeholder_image("Virtual Try-On", view_type, output_path)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Virtual try-on failed: {error_msg}")
            
            if "403" in error_msg or "billing" in error_msg.lower():
                logger.error("💳 Enable billing at: https://console.cloud.google.com/billing")
            elif "429" in error_msg:
                logger.error("⏱️ Rate limit exceeded. Wait and try again.")
            
            return self._create_placeholder_image("Virtual Try-On", view_type, output_path)
    
    def _build_tryon_prompt(self, view_type: str, resolution: str) -> str:
        """Build virtual try-on prompt that preserves face, body, and garment exactly."""
        
        view_instructions = {
            "front": "front view",
            "back": "back view",
            "side": "side view",
            "close-up": "close-up view"
        }
        
        view_text = view_instructions.get(view_type.lower(), "front view")
        
        return f"""Virtual try-on task: Place the garment from the second image onto the person in the first image.

CRITICAL REQUIREMENTS:
- Keep the person's FACE exactly as shown in the first image - same face, same expression, same features
- Keep the person's BODY exactly as shown - same body type, same pose, same proportions
- Keep the GARMENT exactly as shown in the second image - same color, same pattern, same design, same fit style
- Maintain the person's SKIN TONE exactly as in the first image
- Preserve the person's HAIR exactly as shown in the first image
- Keep BACKGROUND consistent with the first image

VIEW: {view_text}

Only change: Replace the original clothing with the new garment while keeping everything else identical.
The result should look like the same person simply changed their clothes.
Photorealistic output, seamless integration."""
    
    def _build_prompt(self, description: str, view_type: str, resolution: str) -> str:
        """Build optimized prompt for text-to-image generation."""
        
        resolution_config = {
            "1K": "1024x1024 resolution",
            "2K": "2048x2048 resolution, high detail",
            "4K": "4096x4096 resolution, ultra high detail, maximum quality"
        }
        
        res_instruction = resolution_config.get(resolution, "2048x2048 resolution")
        
        view_specific_instructions = {
            "front": "Full-body front view, centered, symmetrical composition, facing camera directly, professional modeling pose, clear facial features",
            "back": "Full-body back view, complete back details visible, garment back design clearly shown, centered composition",
            "side": "Full-body side profile view, garment silhouette and side details visible, professional pose, face profile clear",
            "close-up": "Detailed close-up focusing on fabric texture, stitching details, garment features, macro photography style"
        }
        
        view_instruction = view_specific_instructions.get(
            view_type.lower(), 
            "Full-body professional photography view"
        )
        
        return f"""Professional fashion photography in white studio setting.

VIEW: {view_instruction}

SUBJECT DESCRIPTION:
{description}

REQUIREMENTS:
- Resolution: {res_instruction}
- Background: Clean white studio backdrop
- Lighting: Professional studio lighting, soft natural shadows
- Style: High-end fashion catalog photography
- Model: Professional pose, natural expression, clear facial features
- Garment: Accurate to description - exact colors, patterns, textures, fit
- Composition: Centered, well-balanced, full subject visible
- Quality: Photorealistic, sharp focus, accurate details

Create a professional fashion photograph with accurate garment representation and natural model appearance."""

    def _create_placeholder_image(self, description: str, view_type: str, output_path: Path) -> bool:
        """Create a professional placeholder image when generation fails."""
        try:
            # Create a high-quality placeholder
            img = Image.new('RGB', (1024, 1024), color=(250, 250, 250))
            draw = ImageDraw.Draw(img)
            
            # Add gradient background effect
            for i in range(1024):
                color_val = int(250 - (i / 1024) * 30)
                draw.line([(0, i), (1024, i)], fill=(color_val, color_val, color_val + 5))
            
            # Add decorative frame
            draw.rectangle([40, 40, 984, 984], outline=(255, 193, 7), width=4)
            draw.rectangle([50, 50, 974, 974], outline=(255, 215, 0), width=2)
            
            # Draw banana icon
            center_x, center_y = 512, 280
            # Simple banana shape
            draw.ellipse([center_x-80, center_y-40, center_x+80, center_y+40], 
                        fill=(255, 223, 0), outline=(255, 193, 7), width=3)
            
            # Add text content
            title = "Image Generation Preview"
            view_text = f"{view_type.upper()} VIEW"
            
            # Use default font
            try:
                font_large = ImageFont.truetype("arial.ttf", 36)
                font_medium = ImageFont.truetype("arial.ttf", 28)
                font_small = ImageFont.truetype("arial.ttf", 18)
            except:
                font_large = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # Draw title
            draw.text((512, 400), title, fill=(50, 50, 50), anchor="mm", font=font_large)
            draw.text((512, 470), view_text, fill=(255, 152, 0), anchor="mm", font=font_medium)
            
            # Add description (wrapped)
            desc_preview = description[:100] + "..." if len(description) > 100 else description
            words = desc_preview.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 50:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            y_position = 550
            for line in lines[:4]:
                draw.text((512, y_position), line, fill=(80, 80, 80), anchor="mm", font=font_small)
                y_position += 35
            
            # Add footer
            draw.text((512, 800), "Placeholder Image - Enable Billing for Real Generation", 
                     fill=(150, 150, 150), anchor="mm", font=font_small)
            draw.text((512, 840), "💰 Cost: $0.134 per 1K/2K image | $0.24 per 4K image", 
                     fill=(150, 150, 150), anchor="mm", font=font_small)
            draw.text((512, 880), "🔗 Enable at: console.cloud.google.com/billing", 
                     fill=(150, 150, 150), anchor="mm", font=font_small)
            
            # Save
            img.save(str(output_path), 'PNG', quality=95)
            logger.info(f"✅ Created placeholder for {view_type} view")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create placeholder: {e}")
            return False