import cv2
import numpy as np
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import logger
from backend.preprocessing import run_preprocessing_pipeline
from backend.cv_pipeline import run_full_verification_pipeline

app = FastAPI(
    title="AI-Powered PAN Card Verification & Tampering Detection API",
    description="Full-stack CV API to verify PAN cards and detect digital manipulations.",
    version="1.0.0"
)

# Enable CORS for cross-origin local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root folders
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# 1. Health Check
@app.get("/health")
def health_check():
    """Confirms API status and server sanity."""
    return {
        "status": "ok",
        "service": "pan-verification-api"
    }

# 2. Phase 1 Image Preprocessing Endpoint
@app.post("/api/phase-1/preprocess")
async def preprocess_image(file: UploadFile = File(...)):
    """
    Receives an image and returns Phase 1 intermediate steps and quality analysis.
    """
    # Verify file extensions
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format. Upload JPG, PNG, or WebP.")
        
    try:
        # Read file bytes
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Decoded image is empty.")
            
        # Run preprocessing
        result = run_preprocessing_pipeline(img)
        return result
    except Exception as e:
        logger.error(f"Preprocessing endpoint crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

# 3. End-to-End Verification Pipeline Endpoint
@app.post("/api/verify")
async def verify_pan_card(file: UploadFile = File(...)):
    """
    Orchestrates the entire 9-phase computer vision verification and TAMPERING detection pipeline.
    """
    # Verify file extensions
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format. Upload JPG, PNG, or WebP.")
        
    try:
        # Read file bytes
        file_bytes = await file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Decoded image is empty.")
            
        # Execute the unified orchestrator
        report = run_full_verification_pipeline(img)
        if not report["success"]:
            raise HTTPException(status_code=500, detail=report["error"])
            
        return report
    except Exception as e:
        logger.error(f"Verify endpoint crashed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification error: {str(e)}")

# Serve Static Web UI Files
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    
    @app.get("/")
    def read_root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    logger.warning(f"Frontend directory '{FRONTEND_DIR}' does not exist yet. Web server will launch in API-only mode.")
