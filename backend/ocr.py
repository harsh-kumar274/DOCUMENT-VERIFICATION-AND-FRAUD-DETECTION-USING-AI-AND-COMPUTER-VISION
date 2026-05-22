import cv2
import numpy as np
import re
from backend.config import TESSERACT_CMD, USE_MOCK_OCR, logger
from backend.utils import image_to_base64

# Configure pytesseract if available and installed
if not USE_MOCK_OCR and TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    pytesseract = None

# Standard coordinates for PAN field crops (on 856x540 normalized canvas)
FIELD_REGIONS = {
    "PAN": {"y1": 340, "y2": 410, "x1": 50, "x2": 580},
    "Name": {"y1": 150, "y2": 230, "x1": 50, "x2": 580},
    "FatherName": {"y1": 220, "y2": 290, "x1": 50, "x2": 580},
    "DOB": {"y1": 280, "y2": 350, "x1": 50, "x2": 380}
}

def clean_ocr_text(text: str) -> str:
    """Strip unnecessary whitespace and clean alphanumeric chars."""
    text = text.replace("\n", " ").strip()
    # Remove excessive symbols
    return re.sub(r'[^\w\s\/\-\:\.\(\)]', '', text).strip()

def run_ocr_extraction(corrected_image: np.ndarray) -> dict:
    """
    Orchestrates Phase 7: OCR Extraction.
    Applies region-specific crops on the normalized canvas.
    If Tesseract is not installed, uses a robust mock fallback engine.
    """
    h, w = corrected_image.shape[:2]
    vis_img = corrected_image.copy()
    
    extracted_fields = {
        "PAN": {"text": "", "confidence": 0.0, "is_mock": False},
        "Name": {"text": "", "confidence": 0.0, "is_mock": False},
        "FatherName": {"text": "", "confidence": 0.0, "is_mock": False},
        "DOB": {"text": "", "confidence": 0.0, "is_mock": False}
    }
    
    # Preprocess image for OCR (Grayscale and Adaptive Thresholding)
    gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    if USE_MOCK_OCR or pytesseract is None:
        logger.info("Running OCR in MOCK mode. Generating realistic values from template geometry.")
        # Generate clean mock OCR data
        extracted_fields["PAN"] = {"text": "ABCPS1234F", "confidence": 98.5, "is_mock": True}
        extracted_fields["Name"] = {"text": "HARSH SHARMA", "confidence": 97.2, "is_mock": True}
        extracted_fields["FatherName"] = {"text": "RAMESH SHARMA", "confidence": 95.0, "is_mock": True}
        extracted_fields["DOB"] = {"text": "21/05/1998", "confidence": 99.0, "is_mock": True}
        
        # Draw bounding boxes around simulated regions on the visual representation
        for field, coords in FIELD_REGIONS.items():
            x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_img, field, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        # Standard extraction using Tesseract
        logger.info("Executing actual Tesseract OCR on region-specific crops.")
        
        for field, coords in FIELD_REGIONS.items():
            x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
            crop = thresh[y1:y2, x1:x2]
            
            # Use specific configs:
            # - PSM 7: Treat the image as a single text line.
            # - PAN should restrict characters to alphanumeric upper-case
            config = "--psm 7"
            if field == "PAN":
                config += " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            elif field == "DOB":
                config += " -c tessedit_char_whitelist=0123456789/.-"
                
            try:
                # Run OCR
                data = pytesseract.image_to_data(crop, config=config, output_type=pytesseract.Output.DICT)
                
                # Consolidate text strings
                words = []
                confidences = []
                for i in range(len(data["text"])):
                    word = data["text"][i].strip()
                    conf = float(data["conf"][i])
                    if word and conf > -1:
                        words.append(word)
                        confidences.append(conf)
                        
                extracted_text = clean_ocr_text(" ".join(words))
                avg_confidence = float(np.mean(confidences)) if confidences else 0.0
                
                extracted_fields[field] = {
                    "text": extracted_text,
                    "confidence": round(avg_confidence, 2),
                    "is_mock": False
                }
                
                # Draw success boundary
                color = (0, 255, 0) if avg_confidence > 70 else (0, 165, 255)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_img, f"{field} ({int(avg_confidence)}%)", (x1, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
            except Exception as e:
                logger.error(f"Failed to OCR field {field}: {str(e)}")
                extracted_fields[field] = {"text": "", "confidence": 0.0, "is_mock": False}
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis_img, f"{field} ERROR", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return {
        "phase": "ocr_extraction",
        "success": True,
        "fields": extracted_fields,
        "ocr_vis_base64": image_to_base64(vis_img),
        "explanation": "Field-specific cropping and OCR applied. " + 
                       ("Simulated OCR results generated (Mock Mode)." if USE_MOCK_OCR else "Text successfully parsed via Tesseract OCR engine.")
    }
