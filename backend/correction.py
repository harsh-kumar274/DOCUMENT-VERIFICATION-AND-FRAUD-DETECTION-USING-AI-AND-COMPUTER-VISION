import cv2
import numpy as np
from backend.config import logger
from backend.utils import image_to_base64

def correct_perspective(image: np.ndarray, corners: list) -> dict:
    """
    Orchestrates Phase 3 Perspective Correction.
    Uses Homography estimation with RANSAC to map the 4 card corners
    to a standardized flat top-down view (dimension 856x540).
    """
    try:
        # Standard PAN card pixel dimensions maintaining correct physical ratio (1.585)
        target_width = 856
        target_height = 540
        
        # Source coordinates from detection corners (ordered: TL, TR, BR, BL)
        src_pts = np.array(corners, dtype="float32")
        
        # Target coordinates in destination space
        dst_pts = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype="float32")
        
        # Compute Homography Matrix using RANSAC
        # RANSAC is useful when mapping coordinates to eliminate small coordinate drift outliers
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            raise ValueError("Homography matrix estimation failed.")
            
        # Apply perspective warp
        corrected_img = cv2.warpPerspective(image, H, (target_width, target_height))
        
        # Calculate deskew status details
        status_flat = status.flatten() if status is not None else []
        num_inliers = int(np.sum(status_flat))
        logger.info(f"Perspective warping successfully completed. Inliers: {num_inliers}/4 corners.")
        
        return {
            "phase": "perspective_correction",
            "success": True,
            "corrected_base64": image_to_base64(corrected_img),
            "explanation": "Homography transformation calculated via RANSAC. Perspective skew and rotation corrected to standard 856x540 layout."
        }
    except Exception as e:
        logger.error(f"Perspective correction failed: {str(e)}")
        # If warping fails, return original resized as standard dimensions
        fallback_img = cv2.resize(image, (856, 540))
        return {
            "phase": "perspective_correction",
            "success": False,
            "corrected_base64": image_to_base64(fallback_img),
            "explanation": f"Failed to perform perspective warp: {str(e)}. Applied standard aspect fallback crop."
        }
