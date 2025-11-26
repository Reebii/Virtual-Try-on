import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings configuration."""
    
    def __init__(self):
        # API Configuration with better validation
        self.GEMINI_API_KEY = self._get_and_validate_api_key()
        
        # Use the new Gemini 3 Pro model
        self.GEMINI_MODEL = "gemini-3-pro-preview"
        
        # Path Configuration
        self.BASE_DIR = Path(__file__).parent.parent.parent
        self.INPUT_DIR = self.BASE_DIR / "inputs"
        self.OUTPUT_DIR = self.BASE_DIR / "outputs"
        
        # Image Configuration
        self.ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]
        self.MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
        self.OUTPUT_RESOLUTION = "3:4"
        
        # API Configuration
        self.REQUEST_TIMEOUT = 30
        self.RATE_LIMIT_DELAY = 2.0
        self.MAX_RETRIES = 3
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    def _get_and_validate_api_key(self) -> str:
        """Get and validate the API key."""
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Please create a '.env' file in the project root with:\n"
                "GEMINI_API_KEY=your_actual_api_key_here\n"
                "Get your API key from: https://aistudio.google.com/app/apikey"
            )
        
        # Remove any quotes or whitespace
        api_key = api_key.strip().strip('"').strip("'")
        
        # Basic validation
        if not api_key.startswith('AIza'):
            raise ValueError(
                f"API key format appears incorrect. Should start with 'AIza'.\n"
                f"Your key starts with: {api_key[:4] if len(api_key) >= 4 else 'too short'}\n"
                "Get a valid API key from: https://aistudio.google.com/app/apikey"
            )
        
        if len(api_key) < 20:
            raise ValueError(
                f"API key appears too short. Length: {len(api_key)} characters\n"
                "Get a valid API key from: https://aistudio.google.com/app/apikey"
            )
        
        print(f"✅ API key loaded: {api_key[:10]}...{api_key[-5:]}")
        return api_key
    
    def validate(self) -> bool:
        """Validate all settings."""
        # Check if directories can be created
        try:
            self.INPUT_DIR.mkdir(exist_ok=True)
            self.OUTPUT_DIR.mkdir(exist_ok=True)
            return True
        except Exception as e:
            raise ValueError(f"Failed to create directories: {e}")


# Create settings instance
try:
    settings = Settings()
    settings.validate()
    print("✅ Settings loaded successfully!")
    print(f"📝 Using Gemini model: {settings.GEMINI_MODEL}")
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    exit(1)