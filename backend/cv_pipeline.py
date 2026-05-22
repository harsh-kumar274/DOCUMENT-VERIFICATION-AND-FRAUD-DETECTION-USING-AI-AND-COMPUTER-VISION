import cv2
import traceback
from backend.config import logger
from backend.preprocessing import run_preprocessing_pipeline
from backend.detection import detect_document_boundary
from backend.correction import correct_perspective
from backend.features import run_feature_matching
from backend.tampering import run_tampering_detection
from backend.ocr import run_ocr_extraction
from backend.validation import run_pan_validation
from backend.decision import run_decision_engine

def run_full_verification_pipeline(image) -> dict:
    """
    Orchestrates the entire end-to-end PAN Card Fraud Verification Pipeline.
    Runs Phase 1 through Phase 9 sequentially, compiling a traceable audit trail.
    """
    try:
        # Load image if it's a file path
        if isinstance(image, str):
            from backend.utils import file_to_image
            img = file_to_image(image)
        else:
            img = image.copy()
            
        from backend.utils import resize_image
        # Normalize size to make processing speeds uniform and reliable
        img = resize_image(img, 1280, 720)
        
        # --- Stage 1: Preprocessing ---
        prep_res = run_preprocessing_pipeline(img)
        if not prep_res["success"]:
            raise ValueError("Image preprocessing failed.")
            
        # Extract preprocessed outputs
        # Find step base64 image (gray is the standard for detection)
        # We can convert back base64 grayscale to CV image for subsequent stages
        from backend.utils import base64_to_image
        
        gray_step = next(step for step in prep_res["steps"] if step["name"] == "grayscale")
        gray_img = base64_to_image(gray_step["image_base64"])
        
        # --- Stage 2: Document Detection ---
        det_res = detect_document_boundary(img, gray_img)
        
        # --- Stage 3: Perspective Correction ---
        corners = det_res["corners"]
        corr_res = correct_perspective(img, corners)
        
        corrected_img = base64_to_image(corr_res["corrected_base64"])
        
        # --- Stage 4 & 5: Feature Extraction & Matching ---
        feat_res = run_feature_matching(corrected_img)
        
        # --- Stage 6: Tampering Detection ---
        tamp_res = run_tampering_detection(corrected_img)
        
        # --- Stage 7: OCR Extraction ---
        ocr_res = run_ocr_extraction(corrected_img)
        
        # --- Stage 8: PAN Validation ---
        val_res = run_pan_validation(ocr_res)
        
        # --- Stage 9: Fraud Decision Engine ---
        dec_res = run_decision_engine(
            image_quality_risk=prep_res["metrics"]["quality_risk"],
            doc_detection_risk=det_res["detection_risk"],
            feature_mismatch_score=feat_res["mismatch_score"],
            tampering_score=tamp_res["tampering_score"],
            ocr_invalidity_score=val_res["ocr_invalidity_score"]
        )
        
        # Compile complete structured report
        return {
            "success": True,
            "preprocessing": prep_res,
            "document_detection": det_res,
            "perspective_correction": corr_res,
            "feature_matching": feat_res,
            "tampering_detection": tamp_res,
            "ocr_extraction": ocr_res,
            "pan_validation": val_res,
            "decision_engine": dec_res
        }
        
    except Exception as e:
        logger.error(f"End-to-end pipeline crashed: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": f"Verification pipeline encountered an internal error: {str(e)}"
        }
