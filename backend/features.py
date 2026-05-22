import cv2
import numpy as np
import os
from backend.config import TEMPLATE_PATH, logger
from backend.utils import image_to_base64

def run_feature_matching(corrected_image: np.ndarray) -> dict:
    """
    Orchestrates Phase 4 & 5 Feature Extraction & Matching.
    1. Extracts ORB features from the corrected card and the reference template.
    2. Performs Brute-Force Hamming matching.
    3. Uses RANSAC on the matches to verify structural alignment.
    4. Calculates a mismatch metric score.
    """
    try:
        # 1. Load Reference Template
        if not os.path.exists(TEMPLATE_PATH):
            raise FileNotFoundError(f"Reference template not found at {TEMPLATE_PATH}")
            
        template_img = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_COLOR)
        if template_img is None:
            raise ValueError(f"Could not read reference template image from {TEMPLATE_PATH}")
            
        # Normalize template dimensions to match corrected image (856x540)
        target_w, target_h = 856, 540
        template_img = cv2.resize(template_img, (target_w, target_h))
        
        # Convert both to grayscale
        gray_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Extract ORB Keypoints & Descriptors
        orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        
        kp_template, des_template = orb.detectAndCompute(gray_template, None)
        kp_corrected, des_corrected = orb.detectAndCompute(gray_corrected, None)
        
        if des_template is None or des_corrected is None:
            raise ValueError("Could not extract ORB descriptors from one or both images.")
            
        # 3. Match Features using Brute-Force Hamming Distance
        # ORB is a binary descriptor, so cv2.NORM_HAMMING is appropriate
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_template, des_corrected)
        
        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep top matches
        top_matches = matches[:150]
        
        # 4. Geometry Verification with RANSAC Homography
        mismatch_score = 1.0 # Default to 100% mismatch
        success = False
        explanation = ""
        matches_rendered_b64 = ""
        ransac_inlier_ratio = 0.0
        
        if len(top_matches) >= 8:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_corrected[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)
            
            # Find homography between the two sets of feature points using RANSAC
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if mask is not None:
                inliers = int(np.sum(mask))
                total = len(mask)
                ransac_inlier_ratio = inliers / total
                
                # A high inlier ratio means the spatial distribution of matched features is structurally aligned
                # We map this to mismatch score: higher inlier ratio -> lower mismatch score
                mismatch_score = float(np.clip(1.0 - ransac_inlier_ratio, 0.0, 1.0))
                success = True
                
                # Highlight inlier matches
                inlier_matches = [top_matches[i] for i in range(len(top_matches)) if mask[i]]
                
                # Draw the structural matches
                matched_vis = cv2.drawMatches(
                    template_img, kp_template, 
                    corrected_image, kp_corrected, 
                    inlier_matches[:50], None, 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                matches_rendered_b64 = image_to_base64(matched_vis)
                explanation = f"Layout matched template with {inliers} structurally aligned inliers. Match confidence is {ransac_inlier_ratio * 100:.1f}%."
            else:
                explanation = "Feature matching failed to find geometrical structure consistency."
        else:
            explanation = "Extremely poor visual match. Inadequate structural feature correlation found."
            
        return {
            "phase": "feature_matching",
            "success": success,
            "mismatch_score": round(mismatch_score, 2),
            "ransac_inlier_ratio": round(ransac_inlier_ratio, 2),
            "num_matches": len(top_matches),
            "matches_base64": matches_rendered_b64,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Feature matching execution failed: {str(e)}")
        return {
            "phase": "feature_matching",
            "success": False,
            "mismatch_score": 1.0,
            "ransac_inlier_ratio": 0.0,
            "num_matches": 0,
            "matches_base64": "",
            "explanation": f"Error running feature matching pipeline: {str(e)}"
        }
