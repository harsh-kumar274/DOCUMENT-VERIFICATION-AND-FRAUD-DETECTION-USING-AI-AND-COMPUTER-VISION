# AI-Powered PAN Card Verification and Tampering Detection System

## 📝 Overview
This project is an AI-powered system designed to verify PAN card images and detect possible tampering. It leverages classical computer vision, Optical Character Recognition (OCR), rule-based validation, and a centralized fraud decision engine to authenticate PAN cards.

The system evaluates uploaded PAN card images to ensure clarity, correctness of structure, and validity of the information presented on the card.

## ✨ Features
- **Image Preprocessing:** Enhances image quality for reliable analysis.
- **Document Detection & Perspective Correction:** Identifies the PAN card in an image and corrects orientation and skew.
- **Feature Extraction & Matching:** Verifies that the card matches the expected visual template of a legitimate Indian PAN card.
- **Forgery & Tampering Detection:** Identifies digital or physical alterations using advanced vision techniques.
- **OCR Extraction & Validation:** Extracts textual data (e.g., PAN number, name, DOB) and cross-verifies its validity.
- **Fraud Decision Engine:** Synthesizes all data points to deliver a clear `Genuine` or `Tampered/Fraudulent` verdict.
- **Intuitive Web Interface:** A lightweight HTML/JS frontend for uploading images and reviewing the verification results.

## 📂 Project Structure
```text
FRAUD DETECTION/
│
├── backend/                  # Python backend containing the core logic
│   ├── config.py             # System configuration parameters
│   ├── correction.py         # Perspective correction modules
│   ├── cv_pipeline.py        # Core Computer Vision pipeline integration
│   ├── decision.py           # Final fraud decision engine
│   ├── detection.py          # Document boundary detection
│   ├── features.py           # Feature extraction and template matching
│   ├── main.py               # Backend main entry point / API definition
│   ├── ocr.py                # Optical Character Recognition logic
│   ├── preprocessing.py      # Image cleanup and enhancement
│   ├── tampering.py          # Forgery and tampering detection algorithms
│   ├── utils.py              # Helper functions
│   └── validation.py         # PAN format/regex validation rules
│
├── frontend/                 # Web interface for the system
│   ├── app.js                # Frontend logic and API integration
│   ├── index.html            # Main UI layout
│   └── style.css             # UI styling
│
├── templates/                # Folder for application templates
├── SYSTEM_DESIGN.md          # Detailed architecture and design documentation
├── requirements.txt          # Python dependencies
└── run.py                    # Script to start the entire application
```

## 🚀 Installation & Setup

1. **Clone or Download the Repository:**
   Navigate to the project root directory.

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   Start the application using the entry point script:
   ```bash
   python run.py
   ```
# 🏗️ System Architecture

```text
                        ┌───────────────────────┐
                        │   User Uploads PAN    │
                        │      Card Image       │
                        └──────────┬────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   Image Preprocessing    │
                    │ • Noise Reduction        │
                    │ • Contrast Enhancement   │
                    │ • Image Sharpening       │
                    └──────────┬───────────────┘
                               │
                               ▼
                 ┌──────────────────────────────┐
                 │ Document Detection & Cropping │
                 │ • Edge Detection              │
                 │ • Boundary Identification     │
                 │ • Perspective Correction      │
                 └──────────┬───────────────────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼

┌────────────────┐  ┌──────────────────┐  ┌───────────────────┐
│ OCR Extraction │  │ Feature Matching │  │ Tampering Analysis │
│ • PAN Number   │  │ • Template Match │  │ • Edited Regions   │
│ • Name         │  │ • Layout Check   │  │ • Texture Analysis │
│ • DOB          │  │ • Keypoint Match │  │ • Clone Detection  │
└──────┬─────────┘  └────────┬─────────┘  └─────────┬─────────┘
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │   Validation Engine     │
                 │ • PAN Regex Validation  │
                 │ • OCR Confidence Check  │
                 │ • Data Consistency      │
                 └──────────┬──────────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │   Fraud Decision Engine │
                 │ • Score Aggregation     │
                 │ • Risk Analysis         │
                 │ • Final Classification  │
                 └──────────┬──────────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │  Final Verification Result   │
              │  ✅ Genuine                  │
              │  ❌ Tampered/Fraudulent      │
              └──────────────────────────────┘
```
        
## 🖥️ Usage
1. Open your browser and navigate to the application URL (provided in the terminal after running `run.py`).
2. Upload a clear image of a PAN card.
3. The system will process the image through the pipeline and display the results, including extracted text, tampering confidence, and the final verification decision.

## 📄 License
This project is for educational and internal verification purposes. Make sure to comply with data privacy policies (like DPDP Act) when handling real Personally Identifiable Information (PII) like PAN cards.
