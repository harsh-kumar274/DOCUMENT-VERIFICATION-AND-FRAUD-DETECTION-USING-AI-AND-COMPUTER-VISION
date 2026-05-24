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
    2.  Multi-strategy OCR — tries multiple preprocessing pipelines,
        picks the one with highest Tesseract confidence
    3.  ROI padding — prevents character boundary clipping
    4.  PAN uses psm 8 (single word) — correct for a 10-char token
    5.  OCR post-correction layer — resolves P↔D, O↔0, I↔1, B↔8 etc.
    6.  OCR reliability states — distinguishes extraction failure vs. uncertainty
    7.  Full debug visualization — raw ROI, resized, thresholded, final OCR input
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

# Minimum per-word confidence to include in OCR output
# Tesseract returns -1 for empty/rejected words, and 0 for very low confidence
MIN_WORD_CONFIDENCE = 5.0

# ---------------------------------------------------------------------------
# ROI padding — added to every edge to prevent character boundary clipping
# ---------------------------------------------------------------------------
ROI_PADDING = 12

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
    # Name/FatherName: no whitelist — Tesseract can't handle spaces in whitelist
    # DOB: no whitelist — clean_ocr_text() handles character filtering post-OCR
}




# ---------------------------------------------------------------------------
# 1. Dynamic proportional ROI computation
# ---------------------------------------------------------------------------

def _compute_field_regions(height: int, width: int) -> dict:
    """
    Compute field ROIs as proportions of the actual corrected image size.

    Proportions are calibrated against the standard GoI PAN card layout
    (physical ratio ≈ 1.585, canvas 856×540) by running full-image OCR
    on the template and measuring actual text positions.

    Layout reference (856 × 540 corrected canvas, y=0 is top):
        0–14%   → Logo + "INCOME TAX DEPARTMENT" + "GOVT. OF INDIA" header
        14–20%  → Photo region (left side); Emblem (center)
        20–33%  → "Permanent Account Number Card" label + QR (right)
        35–46%  → **PAN alphanumeric** (e.g. ABCDE1234A) — center-left
        46–50%  → "Name" label (Hindi + English)
        49–59%  → **Cardholder Name** (FIRST NAME MIDDLE NAME SURNAME)
        60–65%  → "Father's Name" label
        63–74%  → **Father's/Spouse Name**
        74–80%  → "Date of Birth" label
        78–86%  → **DOB** (DD/MM/YYYY)
        85–100% → Signature area
    """
    return {
        # PAN number — the large bold text "ABCDE1234A"
        # At y=40-47%, right of the photo, below "Permanent Account Number Card"
        "PAN": {
            "x1": int(width * 0.22),
            "x2": int(width * 0.65),
            "y1": int(height * 0.38),
            "y2": int(height * 0.48),
        },
        # Cardholder Name — the bold English name line only
        # At y=58-63%, skip the "नाम / Name" Hindi label above
        "Name": {
            "x1": int(width * 0.03),
            "x2": int(width * 0.65),
            "y1": int(height * 0.57),
            "y2": int(height * 0.64),
        },
        # Father's Name — the bold English name line only
        # At y=73-78%, skip the Hindi + "Father's Name" label
        "FatherName": {
            "x1": int(width * 0.03),
            "x2": int(width * 0.65),
            "y1": int(height * 0.72),
            "y2": int(height * 0.79),
        },
        # Date of Birth — the DD/MM/YYYY value
        # At y=87-92%, below the "Date of Birth" label
        "DOB": {
            "x1": int(width * 0.03),
            "x2": int(width * 0.25),
            "y1": int(height * 0.86),
            "y2": int(height * 0.93),
        },
    }


# ---------------------------------------------------------------------------
# 2. Multi-strategy preprocessing
# ---------------------------------------------------------------------------

def _create_preprocessing_variants(roi_bgr: np.ndarray, field: str) -> list[tuple[str, np.ndarray]]:
    """
    Create multiple preprocessing variants of an ROI for multi-attempt OCR.
    Returns a list of (strategy_name, processed_image) tuples.
    The best result (highest confidence) will be selected.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    variants = []

    # Determine upscale factor based on ROI size
    # Small ROIs need more upscaling, large ones need less
    if max(h, w) < 50:
        scale = 4.0
    elif max(h, w) < 100:
        scale = 3.0
    elif max(h, w) < 200:
        scale = 2.0
    else:
        scale = 1.5

    upscaled = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # --- Strategy 1: Simple upscale only (let Tesseract handle it) ---
    variants.append(("raw_upscale", upscaled.copy()))

    # --- Strategy 2: CLAHE + Otsu threshold ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(upscaled)
    _, otsu_thresh = cv2.threshold(clahe_img, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("clahe_otsu", otsu_thresh))

    # --- Strategy 3: Adaptive threshold (Gaussian) ---
    # blockSize must be odd and > 1
    block_size = max(15, int(upscaled.shape[0] * 0.1) | 1)
    if block_size % 2 == 0:
        block_size += 1
    adaptive_thresh = cv2.adaptiveThreshold(
        upscaled, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size, C=10
    )
    variants.append(("adaptive_gauss", adaptive_thresh))

    # --- Strategy 4: Inverted threshold (for dark text on light bg) ---
    # Sometimes Tesseract works better with inverted images
    _, inv_thresh = cv2.threshold(clahe_img, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Re-invert to get black text on white
    inv_reinverted = cv2.bitwise_not(inv_thresh)
    variants.append(("inverted_otsu", inv_reinverted))

    # --- Strategy 5: Bilateral filter + Otsu (noise-aware) ---
    denoised = cv2.bilateralFilter(upscaled, d=9, sigmaColor=75, sigmaSpace=75)
    _, denoised_thresh = cv2.threshold(denoised, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("bilateral_otsu", denoised_thresh))

    return variants


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
# OCR post-correction layer for PAN
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
# Single-strategy OCR attempt
# ---------------------------------------------------------------------------

def _try_ocr_on_image(
    processed: np.ndarray,
    field: str,
    tess_config: str,
) -> tuple[str, float, list[str], list[float]]:
    """
    Run Tesseract on a single preprocessed image variant.
    Returns (cleaned_text, avg_confidence, raw_words, word_confidences).
    """
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
            # Only accept words with reasonable confidence
            # Tesseract returns -1 for rejected words, and near-0 for garbage
            if word and conf >= MIN_WORD_CONFIDENCE:
                words.append(word)
                confidences.append(conf)

        raw_text = " ".join(words)
        cleaned = clean_ocr_text(raw_text, field)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return cleaned, avg_conf, words, confidences

    except Exception as exc:
        logger.debug(f"[OCR] Tesseract attempt failed for '{field}': {exc}")
        return "", 0.0, [], []


# ---------------------------------------------------------------------------
# Multi-strategy per-field OCR pipeline
# ---------------------------------------------------------------------------

def _run_field_ocr(
    corrected_image: np.ndarray,
    field: str,
    coords: dict,
    vis_img: np.ndarray,
) -> dict:
    """
    Extract OCR text from one field ROI using multiple preprocessing strategies.
    Picks the strategy that produces the highest-confidence result.
    """
    ih, iw = corrected_image.shape[:2]

    # Apply padding, clip to image bounds
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

    # Build Tesseract configs — try multiple PSM modes per field
    whitelist = FIELD_WHITELISTS.get(field, "")

    # PSM modes to try (primary first, then fallbacks)
    if field == "PAN":
        psm_modes = ["6", "7", "8"]  # block, single line, single word
    elif field in ("Name", "FatherName"):
        psm_modes = ["6", "7"]        # block, single line
    else:
        psm_modes = ["7", "6"]        # single line, block

    # Generate multiple preprocessing variants
    variants = _create_preprocessing_variants(roi_bgr, field)

    # Try each variant × PSM combination and pick the best
    best_text = ""
    best_conf = 0.0
    best_strategy = "none"
    best_processed = None

    for psm in psm_modes:
        config_parts = [f"--psm {psm}", "--oem 3"]
        if whitelist:
            config_parts.append(f"-c tessedit_char_whitelist={whitelist}")
        tess_config = " ".join(config_parts)

        for strategy_name, processed_img in variants:
            cleaned, avg_conf, words, confs = _try_ocr_on_image(
                processed_img, field, tess_config
            )

            # For PAN, prefer results that are closer to 10 chars
            score = avg_conf
            if field == "PAN" and cleaned:
                # Bonus for being close to 10 chars
                len_diff = abs(len(cleaned) - 10)
                if len_diff == 0:
                    score += 20
                elif len_diff <= 2:
                    score += 5

            logger.debug(
                f"[OCR] {field} psm={psm} strategy '{strategy_name}': "
                f"text='{cleaned}' conf={avg_conf:.1f} score={score:.1f}"
            )

            if score > best_conf or (score == best_conf and len(cleaned) > len(best_text)):
                best_text = cleaned
                best_conf = avg_conf
                best_strategy = f"{strategy_name}_psm{psm}"
                best_processed = processed_img

    # Also try without whitelist for PAN if result is poor
    if field == "PAN" and (best_conf < 50 or len(best_text) != 10):
        alt_config = f"--psm {psm} --oem 3"
        for strategy_name, processed_img in variants[:3]:
            cleaned, avg_conf, words, confs = _try_ocr_on_image(
                processed_img, field, alt_config
            )
            cleaned = re.sub(r"[^A-Z0-9]", "", cleaned.upper())
            score = avg_conf
            if len(cleaned) == 10:
                score += 20

            if score > best_conf:
                best_text = cleaned
                best_conf = avg_conf
                best_strategy = f"{strategy_name}_no_wl"
                best_processed = processed_img

    logger.info(
        f"[OCR] {field}: best strategy='{best_strategy}' "
        f"text='{best_text}' conf={best_conf:.1f}%"
    )

    if best_processed is None:
        best_processed = variants[0][1] if variants else roi_bgr

    preprocessed_base64 = image_to_base64(best_processed)
    debug_stages_b64 = {
        name: image_to_base64(img) for name, img in variants[:3]
    }

    # Clamp confidence to valid range
    avg_conf = max(0.0, min(100.0, best_conf))
    low_conf = avg_conf < OCR_CONFIDENCE_THRESHOLD

    # OCR post-correction layer (PAN only)
    ocr_corrected = False
    correction_log = ""
    if field == "PAN" and best_text:
        corrected_pan, ocr_corrected, correction_log = _attempt_pan_correction(best_text)
        if ocr_corrected:
            logger.info(
                f"[OCR] PAN post-correction applied: '{best_text}' → '{corrected_pan}'"
            )
        best_text = corrected_pan

    # Reliability state
    if not best_text:
        state = "empty"
    elif ocr_corrected:
        state = "corrected"
    elif low_conf:
        state = "low_conf"
    else:
        state = "ok"

    # Annotate visualisation overlay on the corrected card image
    color = (72, 199, 116) if avg_conf >= OCR_CONFIDENCE_THRESHOLD else (255, 165, 0)
    if avg_conf < 60:
        color = (60, 80, 220)
    if ocr_corrected:
        color = (200, 130, 255)   # purple = auto-corrected

    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
    label = f"{field}: '{best_text}' ({avg_conf:.0f}%{'*' if ocr_corrected else ''})"
    cv2.putText(vis_img, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    return {
        "text": best_text,
        "confidence": round(avg_conf, 2),
        "is_mock": False,
        "low_confidence": low_conf,
        "ocr_corrected": ocr_corrected,
        "correction_log": correction_log,
        "state": state,
        "strategy_used": best_strategy,
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

    Runs a dedicated crop → multi-strategy preprocess → OCR → post-correct
    pipeline for each PAN card field independently.  If Tesseract is
    unavailable the system returns empty fields.

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
        logger.info("[Phase 7] Running multi-strategy Tesseract OCR.")
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

        strategies_used = [
            f"{field}={f.get('strategy_used', '?')}"
            for field, f in extracted_fields.items()
        ]

        parts = [
            "Multi-strategy OCR: each field tested with 5 preprocessing variants "
            "(raw, CLAHE+Otsu, adaptive, inverted, bilateral). "
            f"Best strategies: {', '.join(strategies_used)}."
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
