# 🎯 Virtual Try-On AI System!

**AI-Powered Fashion Catalog Generation · Multi-View Consistency · Professional Studio Quality**

*Generate stunning 4-view fashion composites with perfect model & garment consistency*

</div>

## 🚀 Overview

**Virtual Try-On AI System** is a cutting-edge AI solution that automatically generates professional fashion catalog images featuring the **same child model** wearing the **exact same garment** across **four different views** in a single composite image.

 
 ✨ Key Features

- **🎨 Single Composite Output**: 2x2 grid with front, side, back, and close-up views
- **👧 Perfect Consistency**: Same model, same garment across all views
- **🏷️ Zucchini Brand Styling**: Professional fashion photography refinement
- **📁 Batch Processing**: Automatically processes multiple garment directories
- **🔄 Complete Pipeline**: Upscaling → Cropping → Brand refinement
- **🎯 E-commerce Ready**: Professional quality for online catalogs

## 🎪 Demo

### Input → Output Workflow


📁 Input Directory/
├── 👗 garment_front.jpg
├── 👗 garment_side.jpg
├── 👗 garment_back.jpg
└── 👗 garment_detail.jpg

🎯 AI Processing
│
├── 🎨 Composite Generation (2x2 Grid)
├── 🔍 AI Upscaling (4x Resolution)
├── ✂️ Smart Cropping (4 Individual Views)
└── 🏷️ Zucchini Brand Refinement

📁 Output Directory/
├── 🖼️ composite_all_views.png
├── 🖼️ composite_all_views_upscaled.png
├── 👁️ view_front.png (1080x1440)
├── 👁️ view_side.png (1080x1440)
├── 👁️ view_back.png (1080x1440)
├── 👁️ view_closeup.png (1080x1440)
├── 🏷️ zucchini_front.png
├── 🏷️ zucchini_side.png
├── 🏷️ zucchini_back.png
└── 🏷️ zucchini_closeup.png




## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Google Gemini API Key
- 4GB+ RAM

### Quick Setup

# Clone repository
git clone ....
cd virtual-try-on-ai

# Create virtual environment
python -m venv venv

source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env


# Edit .env with your Gemini API key

# .env file
GEMINI_API_KEY=AIzaxxx.........

LOG_LEVEL=INFO
MAX_IMAGE_SIZE=10485760
REQUEST_TIMEOUT=30
RATE_LIMIT_DELAY=2.0
MAX_RETRIES=XX


# Run the virtual try-on system
python main.py



