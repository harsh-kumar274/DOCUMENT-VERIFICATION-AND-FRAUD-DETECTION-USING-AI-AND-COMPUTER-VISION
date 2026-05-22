import os
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pan-verification")

# General Settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "PAN_CARD_TEMPLATE.jpeg")

# Tesseract OCR Configuration
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"D:\Program Files\Tesseract-OCR\tesseract.exe"
]

TESSERACT_CMD = None
USE_MOCK_OCR = True

# Detect Tesseract Installation
for path in TESSERACT_PATHS:
    if os.path.exists(path):
        TESSERACT_CMD = path
        USE_MOCK_OCR = False
        logger.info(f"Tesseract OCR found at: {path}")
        break

if USE_MOCK_OCR:
    logger.warning("Tesseract OCR binary not found. Standard OCR will run in MOCK mode using templates.")

# Preprocessing Settings
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Fraud Scoring Weights (Total must sum up to 1.0)
WEIGHT_FEATURE_MISMATCH = 0.30
WEIGHT_TAMPERING = 0.25
WEIGHT_OCR_INVALID = 0.20
WEIGHT_DOC_DETECTION_RISK = 0.15
WEIGHT_IMAGE_QUALITY_RISK = 0.10

# Decision Bands
BAND_GENUINE_MAX = 0.30     # Up to 30% is likely genuine
BAND_REVIEW_MAX = 0.60      # 31% to 60% needs manual review
                            # 61%+ is suspicious/likely fraudulent
