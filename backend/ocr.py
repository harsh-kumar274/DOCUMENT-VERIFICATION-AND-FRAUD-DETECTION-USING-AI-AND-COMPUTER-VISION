"""
Phase 7: Region-Specific OCR Extraction
========================================
Implements a per-field OCR pipeline with dedicated preprocessing for each
PAN card region (PAN number, Name, Father Name, DOB).

Architecture:
    Input PAN Card
        ↓
    Preprocessing
        ↓
    Homography + RANSAC
        ↓
    Perspective Correction
        ↓
    Region-Based OCR  ← This module
        ↓
    Rule-Based Validation
        ↓
    Fraud Decision

Key improvements:
    1.  Dynamic proportional ROI scaling — adapts to any canvas size
    2.  Tight PAN ROI — focused crop, no surrounding label noise
    3.  ROI padding — prevents character boundary clipping
    4.  PAN uses psm 8 (single word) — correct for a 10-char token
    5.  PAN-specific pipeline: 4× upscale → CLAHE → Otsu → sharpen → morph close
    6.  Effective (2,2) morphology kernel — non-trivial, actually helps
    7.  OCR post-correction layer — resolves P↔D, O↔0, I↔1, B↔8 etc.
    8.  OCR reliability states — distinguishes extraction failure vs. uncertainty
    9.  Full debug visualization — raw ROI, resized, thresholded, final OCR input
   10.  Image-derived mock values — different uploads produce different results
"""

import cv2
import itertools
import numpy as np
import re
from backend.config import TESSERACT_CMD, USE_MOCK_OCR, logger
from backend.utils import image_to_base64

# ---------------------------------------------------------------------------
# Tesseract bootstrap
# ---------------------------------------------------------------------------
if not USE_MOCK_OCR and TESSERACT_CMD:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    pytesseract = None

# ---------------------------------------------------------------------------
# OCR confidence threshold — below this validation uses softer rules
# ---------------------------------------------------------------------------
OCR_CONFIDENCE_THRESHOLD = 85.0

# ---------------------------------------------------------------------------
# ROI padding — added to every edge to prevent character boundary clipping
# ---------------------------------------------------------------------------
ROI_PADDING = 10

# ---------------------------------------------------------------------------
# PAN regex for post-correction validation
# ---------------------------------------------------------------------------
_PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# ---------------------------------------------------------------------------
# Valid PAN holder-type codes (4th character)
# ---------------------------------------------------------------------------
_PAN_HOLDER_CODES = frozenset("PCFHATBGLJ")

# ---------------------------------------------------------------------------
# OCR character confusion map for PAN post-correction
# Each key maps to the alternative character(s) that OCR may substitute.
# ---------------------------------------------------------------------------
_OCR_CONFUSIONS: dict[str, list[str]] = {
    # Digit ↔ Letter confusions
    "0": ["O"],
    "O": ["0"],
    "1": ["I", "L"],
    "I": ["1", "L"],
    "L": ["I", "1"],
    "5": ["S"],
    "S": ["5"],
    "8": ["B"],
    "B": ["8"],
    "2": ["Z"],
    "Z": ["2"],
    "6": ["G"],
    "G": ["6"],
    # Same-category confusions (high visual similarity)
    "D": ["P", "O"],
    "P": ["D", "F"],
    "E": ["F"],
    "F": ["E", "P"],
    "Q": ["O", "0"],
    "U": ["V"],
    "V": ["U"],
    "N": ["M"],
    "M": ["N"],
    "C": ["G"],
    "K": ["X"],
    "X": ["K"],
}

# ---------------------------------------------------------------------------
# Field-specific OCR character whitelists
# ---------------------------------------------------------------------------
FIELD_WHITELISTS = {
    "PAN":        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "Name":       "ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
    "FatherName": "ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
    "DOB":        "0123456789/.-",
}




# ---------------------------------------------------------------------------
# 1. Dynamic proportional ROI computation
# ---------------------------------------------------------------------------

def _compute_field_regions(height: int, width: int) -> dict:
    """
    Compute field ROIs as proportions of the actual corrected image size.

    Proportions are calibrated against the standard GoI PAN card layout
    (physical ratio ≈ 1.585, canvas 856×540).  Using ratios instead of
    hardcoded pixel coordinates makes the pipeline robust to homography
    scaling shifts.

    Layout reference (856 × 540 canvas, y=0 is top):
        0–15%   → Logo + "Income Tax Department" header text
        15–40%  → Photo region (right side); empty left zone
        42–54%  → Cardholder Name
        54–66%  → Father's / Spouse Name
        66–76%  → Date of Birth
        74–88%  → "Permanent Account Number" label + PAN alphanumeric
    """
    return {
        # PAN number sits near the bottom of the card, below the label line
        "PAN": {
            "x1": int(width * 0.06),
            "x2": int(width * 0.65),
            "y1": int(height * 0.74),
            "y2": int(height * 0.88),
        },
        # Cardholder Name — left side, roughly middle of card
        "Name": {
            "x1": int(width * 0.06),
            "x2": int(width * 0.68),
            "y1": int(height * 0.42),
            "y2": int(height * 0.54),
        },
        # Father's Name — just below the Name row
        "FatherName": {
            "x1": int(width * 0.06),
            "x2": int(width * 0.68),
            "y1": int(height * 0.54),
            "y2": int(height * 0.66),
        },
        # Date of Birth — below Father's Name
        "DOB": {
            "x1": int(width * 0.06),
            "x2": int(width * 0.50),
            "y1": int(height * 0.66),
            "y2": int(height * 0.76),
        },
    }


# ---------------------------------------------------------------------------
# 2 & 3. ROI extraction with padding
# ---------------------------------------------------------------------------

def _extract_roi(image: np.ndarray, coords: dict) -> np.ndarray:
    """
    Crop the image to the given coords, then expand by ROI_PADDING on all
    sides (clamped to image bounds) to avoid clipping character edges.
    """
    ih, iw = image.shape[:2]
    x1 = max(0, coords["x1"] - ROI_PADDING)
    y1 = max(0, coords["y1"] - ROI_PADDING)
    x2 = min(iw, coords["x2"] + ROI_PADDING)
    y2 = min(ih, coords["y2"] + ROI_PADDING)
    return image[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# 5 & 6. Field-specific preprocessing
# ---------------------------------------------------------------------------

def _preprocess_roi_for_ocr(roi_bgr: np.ndarray, field: str) -> tuple[np.ndarray, dict]:
    """
    Apply field-specific preprocessing to a cropped ROI before OCR.

    PAN field pipeline (optimized for a 10-char uppercase alphanumeric token):
        1. Grayscale
        2. 4× INTER_CUBIC upscale  — finer character resolution
        3. CLAHE contrast enhance   — normalises uneven card printing
        4. Otsu global threshold    — stable binary for uniform backgrounds
        5. Mild unsharp sharpening  — improves edge definition
        6. Morphology close (2,2)   — bridges broken character strokes

    Non-PAN field pipeline (names / DOB — more variable layout):
        1. Grayscale
        2. 2.5× INTER_CUBIC upscale
        3. Adaptive Gaussian threshold  — handles shadow/lighting gradients
        4. Unsharp sharpen
        5. Light Gaussian denoise

    Returns
    -------
    processed_image  — final image fed to Tesseract
    debug_stages     — dict of intermediate stage images for visualization
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    debug: dict[str, np.ndarray] = {"grayscale": gray.copy()}

    if field == "PAN":
        # --- Step 2: 4× upscale ---
        scale = 4.0
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        debug["resized"] = upscaled.copy()

        # --- Step 3: CLAHE ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(upscaled)
        debug["clahe"] = clahe_img.copy()

        # --- Step 4: Otsu global threshold (better than adaptive for PAN) ---
        _, thresh = cv2.threshold(
            clahe_img, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        debug["threshold"] = thresh.copy()

        # --- Step 5: Mild unsharp sharpening ---
        blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=0.8)
        sharpened = cv2.addWeighted(thresh, 1.4, blur, -0.4, 0)

        # --- Step 6: Morphology close (2,2) — bridges broken strokes ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
        debug["final_ocr_input"] = final.copy()

    else:
        # --- Step 2: 2.5× upscale ---
        scale = 2.5
        upscaled = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        debug["resized"] = upscaled.copy()

        # --- Step 3: Adaptive threshold (handles lighting gradients) ---
        thresh = cv2.adaptiveThreshold(
            upscaled, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15, C=8
        )
        debug["threshold"] = thresh.copy()

        # --- Step 4: Unsharp sharpen ---
        blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(thresh, 1.5, blur, -0.5, 0)

        # --- Step 5: Light denoise ---
        final = cv2.GaussianBlur(sharpened, (1, 1), 0)
        debug["final_ocr_input"] = final.copy()

    return final, debug


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str, field: str) -> str:
    """Strip OCR noise and enforce field-appropriate character sets."""
    text = text.replace("\n", " ").strip()

    if field == "PAN":
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
    elif field in ("Name", "FatherName"):
        text = re.sub(r"[^A-Z ]", "", text.upper())
        text = re.sub(r" {2,}", " ", text).strip()
    elif field == "DOB":
        text = re.sub(r"[^0-9/\-.]", "", text)

    return text


# ---------------------------------------------------------------------------
# 7. OCR post-correction layer for PAN
# ---------------------------------------------------------------------------

def _attempt_pan_correction(raw_pan: str) -> tuple[str, bool, str]:
    """
    Try single-character substitutions from _OCR_CONFUSIONS to find a
    candidate that:
        (a) matches the PAN regex [A-Z]{5}[0-9]{4}[A-Z]
        (b) has a valid 4th character (holder type code)

    Strategy:
        - Try each position independently (one substitution at a time).
        - Return the first candidate that satisfies both structural checks.
        - If none found, try two simultaneous substitutions at the most
          confusable positions (positions 3 and 4, i.e. holder and surname).

    Returns
    -------
    (corrected_pan, was_corrected, correction_log)
    """
    if not raw_pan or len(raw_pan) != 10:
        return raw_pan, False, "PAN length != 10; correction skipped."

    # Already valid — nothing to do
    if _PAN_REGEX.match(raw_pan) and raw_pan[3] in _PAN_HOLDER_CODES:
        return raw_pan, False, "PAN already valid; no correction needed."

    log_lines = [f"Raw PAN: '{raw_pan}' — attempting OCR confusion correction."]

    # --- Pass 1: single-character substitutions ---
    for pos, char in enumerate(raw_pan):
        for alt in _OCR_CONFUSIONS.get(char, []):
            candidate = raw_pan[:pos] + alt + raw_pan[pos + 1:]
            if _PAN_REGEX.match(candidate) and candidate[3] in _PAN_HOLDER_CODES:
                log_lines.append(
                    f"  Single-sub correction: pos {pos} '{char}'→'{alt}' → '{candidate}' PASS"
                )
                return candidate, True, "\n".join(log_lines)
            log_lines.append(
                f"  Single-sub: pos {pos} '{char}'→'{alt}' → '{candidate}' FAIL"
            )

    # --- Pass 2: two simultaneous substitutions (positions 3 & 4 priority) ---
    priority_pairs = [
        (3, 4), (0, 3), (0, 4), (1, 3), (2, 3),
    ]
    other_pairs = [
        (a, b) for a, b in itertools.combinations(range(10), 2)
        if (a, b) not in priority_pairs
    ]

    for pos_a, pos_b in priority_pairs + other_pairs:
        char_a, char_b = raw_pan[pos_a], raw_pan[pos_b]
        for alt_a in _OCR_CONFUSIONS.get(char_a, []):
            for alt_b in _OCR_CONFUSIONS.get(char_b, []):
                candidate = (
                    raw_pan[:pos_a] + alt_a +
                    raw_pan[pos_a + 1:pos_b] + alt_b +
                    raw_pan[pos_b + 1:]
                )
                if _PAN_REGEX.match(candidate) and candidate[3] in _PAN_HOLDER_CODES:
                    log_lines.append(
                        f"  Double-sub: pos {pos_a}='{char_a}'→'{alt_a}', "
                        f"pos {pos_b}='{char_b}'→'{alt_b}' → '{candidate}' PASS"
                    )
                    return candidate, True, "\n".join(log_lines)

    log_lines.append("  No valid correction found after exhaustive single+double substitution.")
    return raw_pan, False, "\n".join(log_lines)





# ---------------------------------------------------------------------------
# 4. Real Tesseract per-field pipeline (with psm 8 for PAN)
# ---------------------------------------------------------------------------

def _run_field_ocr(
    corrected_image: np.ndarray,
    field: str,
    coords: dict,
    vis_img: np.ndarray,
) -> dict:
    """
    Extract OCR text from one field ROI.

    8. OCR reliability state is encoded in the return dict:
        ocr_corrected  — True if post-correction was applied to PAN
        low_confidence — True if confidence < OCR_CONFIDENCE_THRESHOLD
        state          — one of: 'ok', 'low_conf', 'corrected', 'empty', 'error'
    """
    ih, iw = corrected_image.shape[:2]

    # 3. Apply padding, clip to image bounds
    x1 = max(0, coords["x1"] - ROI_PADDING)
    y1 = max(0, coords["y1"] - ROI_PADDING)
    x2 = min(iw, coords["x2"] + ROI_PADDING)
    y2 = min(ih, coords["y2"] + ROI_PADDING)

    roi_bgr = corrected_image[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        logger.warning(f"[OCR] Empty ROI for '{field}' — skipping.")
        return {
            "text": "", "confidence": 0.0, "is_mock": False,
            "low_confidence": True, "ocr_corrected": False,
            "correction_log": "", "state": "empty",
            "roi_base64": "", "preprocessed_base64": "",
            "debug_stages": {},
        }

    roi_base64 = image_to_base64(roi_bgr)

    # 5 & 6. Field-specific preprocessing
    processed, debug_stages = _preprocess_roi_for_ocr(roi_bgr, field)
    preprocessed_base64 = image_to_base64(processed)

    # Convert debug stage images to base64 for visualization
    debug_stages_b64 = {
        stage_name: image_to_base64(stage_img)
        for stage_name, stage_img in debug_stages.items()
    }

    # 4. Build Tesseract config
    whitelist = FIELD_WHITELISTS.get(field, "")
    if field == "PAN":
        # psm 8 = single word — correct for a contiguous 10-char token
        psm = "8"
    elif field in ("Name", "FatherName"):
        # psm 6 = uniform block of text — handles multi-word names
        psm = "6"
    else:
        # psm 7 = single text line — good for DOB
        psm = "7"

    config_parts = [f"--psm {psm}", "--oem 3"]
    if whitelist:
        config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
    tess_config = " ".join(config_parts)

    try:
        data = pytesseract.image_to_data(
            processed,
            config=tess_config,
            output_type=pytesseract.Output.DICT
        )

        words, confidences = [], []
        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = float(data["conf"][i])
            if word and conf > -1:
                words.append(word)
                confidences.append(conf)

        raw_text = " ".join(words)
        cleaned = clean_ocr_text(raw_text, field)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        low_conf = avg_conf < OCR_CONFIDENCE_THRESHOLD

        # 7. OCR post-correction layer (PAN only)
        ocr_corrected = False
        correction_log = ""
        if field == "PAN":
            corrected_pan, ocr_corrected, correction_log = _attempt_pan_correction(cleaned)
            if ocr_corrected:
                logger.info(
                    f"[OCR] PAN post-correction applied: '{cleaned}' → '{corrected_pan}'"
                )
            cleaned = corrected_pan

        # 8. Reliability state
        if not cleaned:
            state = "empty"
        elif ocr_corrected:
            state = "corrected"
        elif low_conf:
            state = "low_conf"
        else:
            state = "ok"

        logger.info(
            f"[OCR] '{field}' → '{cleaned}' | conf={avg_conf:.1f}% | "
            f"state={state}"
        )

        # Annotate visualisation overlay on the corrected card image
        color = (72, 199, 116) if avg_conf >= OCR_CONFIDENCE_THRESHOLD else (255, 165, 0)
        if avg_conf < 60:
            color = (60, 80, 220)
        if ocr_corrected:
            color = (200, 130, 255)   # purple = auto-corrected

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label = f"{field}: '{cleaned}' ({avg_conf:.0f}%{'*' if ocr_corrected else ''})"
        cv2.putText(vis_img, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        return {
            "text": cleaned,
            "confidence": round(avg_conf, 2),
            "is_mock": False,
            "low_confidence": low_conf,
            "ocr_corrected": ocr_corrected,
            "correction_log": correction_log,
            "state": state,
            "roi_base64": roi_base64,
            "preprocessed_base64": preprocessed_base64,
            "debug_stages": debug_stages_b64,
        }

    except Exception as exc:
        logger.error(f"[OCR] Tesseract error on '{field}': {exc}")
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 200), 2)
        cv2.putText(vis_img, f"{field} OCR ERROR", (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
        return {
            "text": "", "confidence": 0.0, "is_mock": False,
            "low_confidence": True, "ocr_corrected": False,
            "correction_log": str(exc), "state": "error",
            "roi_base64": roi_base64,
            "preprocessed_base64": preprocessed_base64,
            "debug_stages": debug_stages_b64,
        }


# ---------------------------------------------------------------------------
# Public orchestration entry-point
# ---------------------------------------------------------------------------

def run_ocr_extraction(corrected_image: np.ndarray) -> dict:
    """
    Phase 7 — Region-Specific OCR Extraction.

    Runs a dedicated crop → preprocess → OCR → post-correct pipeline for each
    PAN card field independently.  If Tesseract is unavailable the system falls
    back to an image-fingerprint-seeded mock engine.

    Returns
    -------
    dict with keys:
        phase              — "ocr_extraction"
        success            — True
        fields             — per-field extraction dicts (text, confidence, state …)
        overall_confidence — mean confidence across all fields
        ocr_vis_base64     — annotated card image with ROI overlays
        debug_rois         — per-field debug images (raw + all preprocessing stages)
        is_mock            — True when Tesseract is unavailable
        explanation        — human-readable pipeline summary
    """
    ih, iw = corrected_image.shape[:2]
    vis_img = corrected_image.copy()

    # 1. Compute proportional ROI coordinates for this image's dimensions
    field_regions = _compute_field_regions(ih, iw)

    extracted_fields: dict = {}
    debug_rois: dict = {}

    if USE_MOCK_OCR or pytesseract is None:
        logger.error("[Phase 7] Tesseract OCR binary not found — cannot extract text.")
        # Return empty fields; all states marked 'unavailable' so validation
        # and the decision engine can handle gracefully without fake data.
        for field in field_regions:
            extracted_fields[field] = {
                "text": "", "confidence": 0.0, "is_mock": False,
                "low_confidence": True, "ocr_corrected": False,
                "correction_log": "", "state": "unavailable",
                "roi_base64": "", "preprocessed_base64": "", "debug_stages": {},
            }
            debug_rois[field] = {"roi_base64": "", "preprocessed_base64": "", "stages": {}}

        is_mock = False
        explanation = (
            "ERROR: Tesseract OCR binary not found. "
            "Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki "
            "and ensure it is on the system PATH."
        )

    else:
        logger.info("[Phase 7] Running Tesseract OCR with per-field preprocessing.")
        extracted_fields = {}

        for field, coords in field_regions.items():
            result = _run_field_ocr(corrected_image, field, coords, vis_img)
            extracted_fields[field] = result
            debug_rois[field] = {
                "roi_base64": result.pop("roi_base64", ""),
                "preprocessed_base64": result.pop("preprocessed_base64", ""),
                "stages": result.pop("debug_stages", {}),
            }

        is_mock = False
        any_low = any(f.get("low_confidence") for f in extracted_fields.values())
        any_corrected = any(f.get("ocr_corrected") for f in extracted_fields.values())

        parts = [
            "PAN field: 4× upscale → CLAHE → Otsu → sharpen → morph-close (psm 8). "
            "Name/DOB: 2.5× upscale → adaptive threshold → sharpen (psm 6/7)."
        ]
        if any_corrected:
            parts.append("⚡ OCR post-correction was applied to one or more fields.")
        if any_low:
            parts.append(
                "⚠ One or more fields have low OCR confidence — "
                "validation results may be unreliable."
            )
        else:
            parts.append("All fields extracted with acceptable confidence.")

        explanation = " ".join(parts)

    confidences = [f["confidence"] for f in extracted_fields.values()]
    overall_confidence = round(float(np.mean(confidences)), 2) if confidences else 0.0

    return {
        "phase": "ocr_extraction",
        "success": True,
        "fields": extracted_fields,
        "overall_confidence": overall_confidence,
        "ocr_vis_base64": image_to_base64(vis_img),
        "debug_rois": debug_rois,
        "is_mock": is_mock,
        "explanation": explanation,
    }
