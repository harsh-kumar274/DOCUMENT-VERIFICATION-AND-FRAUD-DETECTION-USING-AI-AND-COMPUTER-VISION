import cv2
import numpy as np
from backend.config import logger
from backend.utils import image_to_base64

def detect_copy_move(image_gray: np.ndarray, max_points: int = 500) -> tuple:
    """
    Detect copy-move tampering by self-matching keypoints in the same image.
    Finds identical/highly similar feature descriptors in different spatial locations.
    """
    try:
        # Detect ORB keypoints and descriptors
        orb = cv2.ORB_create(nfeatures=max_points, scaleFactor=1.2)
        kp, des = orb.detectAndCompute(image_gray, None)
        
        if des is None or len(kp) < 10:
            return 0.0, []
            
        # Self-matching descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Match each descriptor to its k=2 nearest neighbors in the same descriptor set
        matches = bf.knnMatch(des, des, k=3)
        
        suspicious_matches = []
        duplicate_count = 0
        
        for m in matches:
            if len(m) < 3:
                continue
            best, second, third = m
            
            # The best match will be the descriptor itself (distance = 0)
            # The second best match is what we check. Enforce Lowe's ratio on second and third
            if second.distance < 0.65 * third.distance:
                # Retrieve spatial coordinates
                pt1 = np.array(kp[best.queryIdx].pt)
                pt2 = np.array(kp[second.trainIdx].pt)
                
                # Verify they are not the same keypoint and are separated by a minimum physical distance
                dist = np.linalg.norm(pt1 - pt2)
                if dist > 35:
                    suspicious_matches.append((pt1, pt2))
                    duplicate_count += 1
                    
        # Normalize copy-move score based on number of duplicated clusters
        copy_move_score = float(np.clip(duplicate_count / 15.0, 0.0, 1.0))
        return copy_move_score, suspicious_matches
    except Exception as e:
        logger.error(f"Copy-Move detection error: {str(e)}")
        return 0.0, []

def analyze_texture_and_edges(image_color: np.ndarray, image_gray: np.ndarray) -> tuple:
    """
    Analyzes local texture variance and unnatural edge boundaries.
    Generates a visual anomaly heatmap highlighting suspicious editing regions.
    """
    h, w = image_gray.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # --- 1. Edge Inconsistency (Canny Edge Gradient Check) ---
    # Digital pastes introduce sharp step edges
    edges = cv2.Canny(image_gray, 80, 200)
    
    # Dilate edges to analyze surrounding blocks
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    
    # --- 2. Local Texture Variance ---
    # Divide the image into a grid of 16x16 blocks
    block_size = 16
    grid_h = h // block_size
    grid_w = w // block_size
    
    variances = np.zeros((grid_h, grid_w))
    
    for r in range(grid_h):
        for c in range(grid_w):
            y1, y2 = r * block_size, (r + 1) * block_size
            x1, x2 = c * block_size, (c + 1) * block_size
            
            block = image_gray[y1:y2, x1:x2]
            variances[r, c] = np.var(block)
            
    # Standardize local variance using Median Absolute Deviation (robust to outliers)
    median_var = np.median(variances)
    mad_var = np.median(np.abs(variances - median_var)) + 1e-5
    
    # Anomalous blocks have extremely high or low variance outliers compared to natural ink bleeding
    variance_anomalies = np.abs(variances - median_var) / mad_var
    
    # Populate the pixel-level anomaly heatmap
    for r in range(grid_h):
        for c in range(grid_w):
            y1, y2 = r * block_size, (r + 1) * block_size
            x1, x2 = c * block_size, (c + 1) * block_size
            
            # Map block anomaly score
            score = variance_anomalies[r, c]
            
            # Amplify score in regions containing high edge density (suspicious text boundaries)
            edge_ratio = np.sum(dilated_edges[y1:y2, x1:x2] > 0) / (block_size**2)
            if edge_ratio > 0.15:
                score *= 1.8
                
            heatmap[y1:y2, x1:x2] = score

    # Normalize heatmap between 0 and 1
    cv2.GaussianBlur(heatmap, (21, 21), 0, dst=heatmap)
    min_val, max_val = heatmap.min(), heatmap.max()
    if max_val > min_val:
        heatmap = (heatmap - min_val) / (max_val - min_val)
        
    # Threshold heatmap to identify high-risk edit locations
    tamper_mask = heatmap > 0.65
    tampering_pixel_ratio = np.sum(tamper_mask) / (h * w)
    
    # Texture/Edge score is proportional to anomalous area
    texture_edge_score = float(np.clip(tampering_pixel_ratio * 12.0, 0.0, 1.0))
    
    # Generate Heatmap Color Overlay for visualization
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend color heatmap onto original image (70% original, 30% heatmap)
    blend_vis = cv2.addWeighted(image_color, 0.7, heatmap_color, 0.3, 0)
    
    return texture_edge_score, blend_vis

def run_tampering_detection(corrected_image: np.ndarray) -> dict:
    """
    Orchestrates Phase 6 Forgery and Tampering Detection.
    Runs self-matching copy-move detectors and local gradient/variance analysis.
    Returns combined tampering scores and a highlighted visual overlay.
    """
    try:
        gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Copy-Move Forgery Detection
        copy_move_score, suspicious_clusters = detect_copy_move(gray)
        
        # 2. Local Texture/Edge Gradient Inconsistency Analysis
        texture_edge_score, heatmap_img = analyze_texture_and_edges(corrected_image, gray)
        
        # Draw Copy-Move indicators on top of the Heatmap visual
        output_vis = heatmap_img.copy()
        for pt1, pt2 in suspicious_clusters:
            p1 = tuple(pt1.astype(int))
            p2 = tuple(pt2.astype(int))
            # Draw line connecting copied and pasted areas
            cv2.line(output_vis, p1, p2, (0, 255, 0), 2)
            cv2.circle(output_vis, p1, 6, (0, 0, 255), -1)
            cv2.circle(output_vis, p2, 6, (255, 0, 0), -1)
            
        # Combine metrics into single tampering threat score (Weighted)
        # Texture/Edge variance (60%), Copy-Move clusters (40%)
        combined_score = 0.60 * texture_edge_score + 0.40 * copy_move_score
        combined_score = float(np.clip(combined_score, 0.0, 1.0))
        
        success = True
        explanation = "Tampering analysis finished. "
        if combined_score > 0.60:
            explanation += "CRITICAL WARNING: High texture variance anomalies and edge spikes detected around print boundaries. Strong likelihood of digital manipulation."
        elif combined_score > 0.30:
            explanation += "Suspicious texture inconsistencies found. Image has mild signals of editing."
        else:
            explanation += "No significant anomalies found. Texture gradient and pixel clusters are consistent."
            
        return {
            "phase": "tampering_detection",
            "success": success,
            "tampering_score": round(combined_score, 2),
            "copy_move_score": round(copy_move_score, 2),
            "texture_edge_score": round(texture_edge_score, 2),
            "heatmap_base64": image_to_base64(output_vis),
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Tampering detection execution failed: {str(e)}")
        return {
            "phase": "tampering_detection",
            "success": False,
            "tampering_score": 0.0,
            "copy_move_score": 0.0,
            "texture_edge_score": 0.0,
            "heatmap_base64": "",
            "explanation": f"Tampering verification error: {str(e)}"
        }
