import numpy as np
from backend.config import (
    WEIGHT_FEATURE_MISMATCH,
    WEIGHT_TAMPERING,
    WEIGHT_OCR_INVALID,
    WEIGHT_DOC_DETECTION_RISK,
    WEIGHT_IMAGE_QUALITY_RISK,
    BAND_GENUINE_MAX,
    BAND_REVIEW_MAX,
    logger
)

def run_decision_engine(
    image_quality_risk: float,
    doc_detection_risk: float,
    feature_mismatch_score: float,
    tampering_score: float,
    ocr_invalidity_score: float
) -> dict:
    """
    Orchestrates Phase 9: Fraud Decision Engine.
    Combines stage-wise risk signals into a final weighted fraud probability.
    Classifies the outcome and generates user-facing analytical reasoning.
    """
    # 1. Calculate Weighted Fraud Probability
    raw_fraud_score = (
        WEIGHT_FEATURE_MISMATCH * feature_mismatch_score +
        WEIGHT_TAMPERING * tampering_score +
        WEIGHT_OCR_INVALID * ocr_invalidity_score +
        WEIGHT_DOC_DETECTION_RISK * doc_detection_risk +
        WEIGHT_IMAGE_QUALITY_RISK * image_quality_risk
    )
    
    # Clip and normalize
    fraud_probability = float(np.clip(raw_fraud_score, 0.0, 1.0))
    
    # 2. Determine Decision Band
    if fraud_probability <= BAND_GENUINE_MAX:
        decision = "LIKELY GENUINE"
        decision_color = "#10B981" # Green
        explanation = "The document looks authentic. Visually aligns with layout templates, matches standard metadata validation rules, and exhibits no digital tempering footprints."
    elif fraud_probability <= BAND_REVIEW_MAX:
        decision = "NEEDS MANUAL REVIEW"
        decision_color = "#F59E0B" # Amber/Orange
        explanation = "Suspicious markers detected. Mild anomalies found in document borders, OCR extraction validation, or visual textures. Manual verification is recommended."
    else:
        decision = "SUSPICIOUS / LIKELY FORGED"
        decision_color = "#EF4444" # Red
        explanation = "CRITICAL WARNING: High threat level. Structural matching failure, severe tampering signals, or format violations strongly suggest this is a digitally edited or fake document."
        
    # 3. Compile Scorecard with Explanations for Frontend Rendering
    scorecard = [
        {
            "name": "Feature Match Alignment",
            "weight": f"{int(WEIGHT_FEATURE_MISMATCH * 100)}%",
            "score": round(feature_mismatch_score, 2),
            "threat": "High" if feature_mismatch_score > 0.6 else "Medium" if feature_mismatch_score > 0.3 else "Low",
            "description": "Checks physical layout alignment, logo locations, and print geometry against the reference template using ORB & RANSAC."
        },
        {
            "name": "Digital Forgery & Tampering",
            "weight": f"{int(WEIGHT_TAMPERING * 100)}%",
            "score": round(tampering_score, 2),
            "threat": "High" if tampering_score > 0.6 else "Medium" if tampering_score > 0.3 else "Low",
            "description": "Verifies pixel texture variance, edge sharpness transitions, and keypoint copy-move duplicate regions."
        },
        {
            "name": "OCR Validation Consistency",
            "weight": f"{int(WEIGHT_OCR_INVALID * 100)}%",
            "score": round(ocr_invalidity_score, 2),
            "threat": "High" if ocr_invalidity_score > 0.6 else "Medium" if ocr_invalidity_score > 0.3 else "Low",
            "description": "Runs 6 rule-based checks on OCR-extracted fields only: PAN regex format, holder type (4th char), surname initial (5th char vs OCR name), digit sequence (chars 6–9), series character (10th char), and DOB format validity."
        },
        {
            "name": "Boundary Corner Detection",
            "weight": f"{int(WEIGHT_DOC_DETECTION_RISK * 100)}%",
            "score": round(doc_detection_risk, 2),
            "threat": "High" if doc_detection_risk > 0.4 else "Low",
            "description": "Evaluates rectangular card aspect ratio and boundary reliability via Canny edge detection, morphological closing, contour analysis, ApproxPolyDP, and aspect-ratio filtering."
        },
        {
            "name": "Image Quality & Clarity",
            "weight": f"{int(WEIGHT_IMAGE_QUALITY_RISK * 100)}%",
            "score": round(image_quality_risk, 2),
            "threat": "High" if image_quality_risk > 0.4 else "Low",
            "description": "Assesses upload brightness, contrast levels, sensor noise, and camera blur."
        }
    ]
    
    logger.info(f"Fraud Decision computed: {decision} ({fraud_probability * 100:.1f}%)")
    
    return {
        "phase": "fraud_decision",
        "success": True,
        "fraud_probability": round(fraud_probability, 2),
        "decision": decision,
        "color": decision_color,
        "scorecard": scorecard,
        "explanation": explanation
    }
