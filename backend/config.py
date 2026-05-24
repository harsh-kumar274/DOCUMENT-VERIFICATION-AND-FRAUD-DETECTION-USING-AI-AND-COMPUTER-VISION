import os
import shutil
import logging
import platform

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pan-verification")

# General Settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "PAN_CARD_TEMPLATE.jpeg")

# ---------------------------------------------------------------------------
# Tesseract OCR Configuration
# ---------------------------------------------------------------------------
# Candidate paths to scan — ordered by likelihood.
# The auto-detection loop below sets TESSERACT_CMD and USE_MOCK_OCR
# automatically, so you should NOT need to edit this file manually.
# If your Tesseract is installed in a non-standard location, add its path
# to the list below.
# ---------------------------------------------------------------------------
# Candidate paths to scan — ordered by platform likelihood.
# macOS (Homebrew), Linux, then Windows.
TESSERACT_PATHS = [
    # macOS (Homebrew)
    "/opt/homebrew/bin/tesseract",
    "/usr/local/bin/tesseract",
    # Linux
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    # Windows
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"D:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\tools\Tesseract-OCR\tesseract.exe",
]

# --- Auto-detect Tesseract installation ---
TESSERACT_CMD = None
USE_MOCK_OCR = True  # Assume unavailable until found

# Method 1: Try shutil.which() — finds tesseract if it's on the system PATH
_which_result = shutil.which("tesseract")
if _which_result:
    TESSERACT_CMD = _which_result
    USE_MOCK_OCR = False
    logger.info(f"[Tesseract] Found via PATH: {TESSERACT_CMD}")

# Method 2: Fall back to scanning known paths
if USE_MOCK_OCR:
    for path in TESSERACT_PATHS:
        if os.path.exists(path):
            TESSERACT_CMD = path
            USE_MOCK_OCR = False
            logger.info(f"[Tesseract] Found at known path: {TESSERACT_CMD}")
            break

if USE_MOCK_OCR:
    logger.warning(
        "[Tesseract] OCR binary not found on PATH or in any known path. "
        "Falling back to MOCK mode. "
        "Install Tesseract: brew install tesseract (macOS) / "
        "apt install tesseract-ocr (Linux) / "
        "https://github.com/UB-Mannheim/tesseract/wiki (Windows). "
        "Or add its path to TESSERACT_PATHS in backend/config.py"
    )
else:
    logger.info(f"[Tesseract] Real OCR enabled: {TESSERACT_CMD} (mock={USE_MOCK_OCR})")

# Preprocessing Settings
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Fraud Scoring Weights (Total must sum up to 1.0)
WEIGHT_FEATURE_MISMATCH    = 0.30
WEIGHT_TAMPERING           = 0.25
WEIGHT_OCR_INVALID         = 0.20
WEIGHT_DOC_DETECTION_RISK  = 0.15
WEIGHT_IMAGE_QUALITY_RISK  = 0.10

# Decision Bands
BAND_GENUINE_MAX = 0.30     # Up to 30% is likely genuine
BAND_REVIEW_MAX  = 0.60     # 31% to 60% needs manual review
                            # 61%+ is suspicious/likely fraudulent
