# AI-Powered PAN Card Verification & Fraud Detection

A full-stack system that verifies Indian PAN card images and detects tampering using Computer Vision, OCR, and a rule-based fraud decision engine.

## Features

- **9-Phase CV Pipeline** — preprocessing, detection, correction, features, tampering, OCR, validation, decision
- **Multi-Strategy OCR** — tries 5 preprocessing variants per field, picks highest confidence result
- **OCR Post-Correction** — fixes common character confusions (O↔0, I↔1, B↔8, etc.)
- **RANSAC Perspective Correction** — fixes skew, rotation, and perspective distortion
- **Tampering Detection** — ELA analysis, texture checks, noise pattern analysis
- **PAN Validation** — regex check, holder-type code, structural verification
- **Fraud Decision Engine** — weighted score aggregation into a final verdict
- **Glassmorphic Web UI** — dark-themed premium dashboard with real-time results

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core language |
| OpenCV 4.9 | Image processing, edge detection, contours, perspective transforms |
| Tesseract OCR | Text extraction from PAN card fields |
| FastAPI | REST API backend |
| NumPy | Array operations |
| Pillow | Image format conversions |
| Jinja2 | HTML template rendering |
| HTML/CSS/JS | Frontend dashboard |

## Project Structure

```
├── backend/
│   ├── config.py             # Tesseract path auto-detection, system config
│   ├── preprocessing.py      # Phase 1: Image cleanup & quality metrics
│   ├── detection.py          # Phase 2: Document boundary detection
│   ├── correction.py         # Phase 3: RANSAC perspective correction
│   ├── features.py           # Phase 4-5: ORB feature extraction & matching
│   ├── tampering.py          # Phase 6: Forgery detection (ELA)
│   ├── ocr.py                # Phase 7: Multi-strategy Tesseract OCR
│   ├── validation.py         # Phase 8: PAN format validation
│   ├── decision.py           # Phase 9: Fraud decision engine
│   ├── cv_pipeline.py        # End-to-end pipeline orchestrator
│   ├── main.py               # FastAPI routes
│   └── utils.py              # Shared utilities
│
├── frontend/
│   ├── index.html            # Main UI
│   ├── style.css             # Dark glassmorphic theme
│   └── app.js                # Frontend logic & API calls
│
├── templates/
│   └── PAN_CARD_TEMPLATE.jpeg
│
├── requirements.txt
├── run.py                    # Entry point
└── SYSTEM_DESIGN.md          # Architecture docs
```

## Pipeline

```
Upload Image
     │
     ▼
1. Preprocessing ──► Noise reduction, contrast, sharpening
     │
     ▼
2. Detection ──► Edge detection, contour analysis, boundary find
     │
     ▼
3. Correction ──► RANSAC homography, perspective warp to 856×540
     │
     ├──────────────────┬────────────────────┐
     ▼                  ▼                    ▼
4-5. Features     6. Tampering        7. OCR Extraction
  Template match    ELA analysis        PAN, Name, DOB
  Layout check      Texture check       Father's Name
     │                  │                    │
     └──────────────────┼────────────────────┘
                        ▼
               8. PAN Validation ──► Regex, format, structure checks
                        │
                        ▼
               9. Fraud Decision ──► Score aggregation → Verdict
                        │
                        ▼
               ✅ Genuine / 🟡 Manual Review / ❌ Fraudulent
```

## Setup

### 1. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt install tesseract-ocr
```

**Windows:** Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH.

### 2. Install & Run

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

Open **http://127.0.0.1:8000** in your browser.

## Usage

1. Open the dashboard in your browser
2. Upload a PAN card image (drag & drop or click)
3. Watch the pipeline process each phase in real-time
4. Review the extracted OCR data, confidence scores, and final verdict

### API

```bash
curl -X POST http://127.0.0.1:8000/api/verify -F "file=@pan_card.jpg"
```

## Sample Output

```
PAN:        "GRNPP3804H"           conf=86.0%   ✅
Name:       "SIDDHARTH PRAJAPATI"  conf=91.5%   ✅
FatherName: "INDERJEET PRAJAPATI"  conf=65.3%   ⚠️
DOB:        "01/01/2004"           conf=73.8%   ⚠️

Validation: PASS (all structural checks passed)
Decision:   NEEDS MANUAL REVIEW (39.2% fraud probability)
```

## Dependencies

```
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6
opencv-python==4.9.0.80
numpy>=1.26.0,<2.0
pytesseract>=0.3.10
Pillow>=10.0.0
jinja2>=3.1.2
pydantic>=2.0
```

## License

Educational and internal verification purposes only. Comply with data privacy regulations (DPDP Act) when handling real PAN card images.
