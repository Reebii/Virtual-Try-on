#!/usr/bin/env python3
"""
Debug script to check API key issues
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

def debug_api_key():
    print("🔍 Debugging Gemini API Key...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found in current directory")
        print("Current directory:", Path.cwd())
        return
    
    print("✅ .env file exists")
    
    # Read API key directly from .env file
    with open('.env', 'r') as f:
        env_content = f.read()
        print("📄 .env file content:")
        print(env_content)
        print("-" * 50)
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Available environment variables:")
        for key, value in os.environ.items():
            if 'GEMINI' in key or 'API' in key or 'KEY' in key:
                print(f"  {key}: {value}")
        return
    
    print(f"✅ GEMINI_API_KEY found: {api_key[:10]}...{api_key[-5:]}")
    print(f"📏 API Key length: {len(api_key)} characters")
    
    # Check API key format
    if api_key.startswith('AIza'):
        print("✅ API key format looks correct (starts with 'AIza')")
    else:
        print("❌ API key format may be incorrect (should start with 'AIza')")
    
    # Test the API key
    print("\n🧪 Testing API key with Gemini...")
    try:
        genai.configure(api_key=api_key)
        
        # Try to list models
        models = genai.list_models()
        print("✅ API key is VALID! Successfully connected to Gemini API")
        print(f"📋 Available models: {len(list(models))} models found")
        
    except Exception as e:
        print(f"❌ API key test FAILED: {e}")
        
        # Provide specific solutions based on error
        error_str = str(e)
        if "API_KEY_INVALID" in error_str:
            print("\n🔧 SOLUTIONS:")
            print("1. Get a NEW API key from: https://aistudio.google.com/app/apikey")
            print("2. Make sure you're using the correct Google account")
            print("3. Check if the API key has proper permissions")
            print("4. Ensure the API key is copied completely (no extra spaces)")
        
        elif "quota" in error_str.lower():
            print("\n🔧 SOLUTION: Wait for quota reset or upgrade your plan")
        
        elif "permission" in error_str.lower():
            print("\n🔧 SOLUTION: Enable the Gemini API in Google Cloud Console")

if __name__ == "__main__":
    debug_api_key()