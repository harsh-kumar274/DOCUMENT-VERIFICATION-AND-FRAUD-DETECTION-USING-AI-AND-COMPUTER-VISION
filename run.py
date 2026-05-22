import sys
import os
import subprocess

# Ensure the root project path is in the system path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print an elegant dashboard launch banner."""
    print("=" * 70)
    print("    ANTIGRAVITY PAN-VERIFY: COMPUTER VISION FRAUD DETECTION ENGINE    ")
    print("=" * 70)
    print("  [Backend]  FastAPI REST Server (Fast OpenCV & Numpy Pipeline)")
    print("  [Frontend] Premium Glassmorphic Web Dashboard (HTML5/CSS3/JS)")
    print("=" * 70)

def check_dependencies():
    """Verify standard modules are present before launching."""
    required = ["fastapi", "uvicorn", "numpy", "cv2", "pytesseract"]
    missing = []
    
    for lib in required:
        try:
            if lib == "cv2":
                import cv2
            else:
                __import__(lib)
        except ImportError:
            missing.append(lib)
            
    if missing:
        print(f"[*] Missing Python dependencies: {', '.join(missing)}")
        print("[*] Installing requirements automatically using pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("[+] Successfully installed dependencies!\n")
        except Exception as e:
            print(f"[-] Auto-installation failed: {str(e)}")
            print("[!] Please run: pip install -r requirements.txt manually.\n")
            sys.exit(1)
    else:
        print("[+] Python dependency checks: OK.")

def check_ocr_environment():
    """Confirm Tesseract OCR environment details."""
    from backend.config import TESSERACT_CMD, USE_MOCK_OCR
    if USE_MOCK_OCR:
        print("[!] Tesseract OCR not found in local system paths.")
        print("[!] The backend will operate in OCR MOCK MODE.")
        print("[!]   - Standard image metrics, Canny/Contour detection, RANSAC perspective,")
        print("[!]     ORB feature matching, and tampering heatmaps will run ACTUAL CV.")
        print("[!]   - OCR text fields will simulate realistic parsed values.")
        print("[!]   - Complete pipelines remain 100% testable and runnable immediately!\n")
    else:
        print(f"[+] Tesseract OCR integration configured successfully: {TESSERACT_CMD}\n")

if __name__ == "__main__":
    print_banner()
    check_dependencies()
    check_ocr_environment()
    
    # Defer import until after pip installs have run above
    import uvicorn
    
    print("[*] Launching FastAPI Web Application...")
    print("[*] Open your browser and navigate to: http://127.0.0.1:8000")
    print("[*] Press Ctrl+C to terminate.")
    print("-" * 70)
    
    # Launch uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
