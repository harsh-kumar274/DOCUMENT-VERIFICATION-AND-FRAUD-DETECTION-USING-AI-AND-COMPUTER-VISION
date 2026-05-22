import cv2
import numpy as np
from backend.config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, logger
from backend.utils import image_to_base64

def estimate_noise(image_gray: np.ndarray) -> float:
    """Estimate the noise standard deviation using a fast edge-avoiding method."""
    H, W = image_gray.shape
    # Compute Laplacian-like high-pass filter
    M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    sigma = np.sum(np.abs(cv2.filter2D(image_gray, -1, np.array(M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
    return float(np.round(sigma, 2))

def get_fft_spectrum(image_gray: np.ndarray) -> np.ndarray:
    """Generate a log-scaled FFT magnitude spectrum for visualization."""
    dft = np.fft.fft2(image_gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-8)
    
    # Normalize to 0-255 range for rendering
    min_val, max_val = magnitude_spectrum.min(), magnitude_spectrum.max()
    if max_val > min_val:
        magnitude_spectrum = 255 * (magnitude_spectrum - min_val) / (max_val - min_val)
    return magnitude_spectrum.astype(np.uint8)

def suppress_fft_periodic_noise(image_gray: np.ndarray) -> np.ndarray:
    """
    Remove periodic scanner or screen grid lines using Frequency Domain Notch Filtering.
    Identifies strong, high-frequency symmetric peaks in the power spectrum and zeroes them out.
    """
    try:
        rows, cols = image_gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # 2D Fast Fourier Transform
        f = np.fft.fft2(image_gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        
        # Calculate magnitude spectrum to locate noise peaks
        magnitude = np.abs(fshift)
        
        # Look for local maxima outside the central low-frequency area
        # Mask out the center (DC component and low frequencies)
        center_mask = np.ones((rows, cols), dtype=np.uint8)
        cv2.circle(center_mask, (ccol, crow), 15, 0, -1)
        
        masked_magnitude = magnitude * center_mask
        mean_val = np.mean(masked_magnitude)
        std_val = np.std(masked_magnitude)
        
        # Find points that are heavily prominent (e.g., > 5 std deviations above mean)
        threshold = mean_val + 5 * std_val
        peaks = np.where(masked_magnitude > threshold)
        
        filtered_shift = fshift.copy()
        
        # Apply notch filter at peak positions (zeroing out a small radius)
        num_peaks_suppressed = 0
        for r, c in zip(peaks[0], peaks[1]):
            # Avoid cleaning too close to center
            dist_to_center = np.sqrt((r - crow)**2 + (c - ccol)**2)
            if dist_to_center > 20:
                cv2.circle(filtered_shift, (c, r), 3, 0, -1)
                num_peaks_suppressed += 1
                
        if num_peaks_suppressed > 0:
            logger.info(f"FFT periodic noise filter suppressed {num_peaks_suppressed} noise spikes.")
            
        # Inverse FFT
        f_ishift = np.fft.ifftshift(filtered_shift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and clip back to valid gray range
        return np.clip(img_back, 0, 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"FFT periodic noise suppression failed: {str(e)}. Returning original gray.")
        return image_gray

def run_preprocessing_pipeline(image: np.ndarray) -> dict:
    """
    Run the entire Image Preprocessing pipeline (Phase 1).
    Returns a dictionary with base64 images of each step and quality metrics.
    """
    # 1. Base Dimensions & Normalized Copy
    h, w = image.shape[:2]
    
    # 2. Grayscale Conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. CLAHE Local Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    enhanced = clahe.apply(gray)
    
    # 4. Bilateral Noise Removal (preserving sharp edges and text)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 5. FFT Frequency Domain Filter for scanner periodic line removal
    fft_spectrum_before = get_fft_spectrum(gray)
    fft_filtered = suppress_fft_periodic_noise(denoised)
    fft_spectrum_after = get_fft_spectrum(fft_filtered)
    
    # 6. OCR-Ready Adaptive Thresholding
    thresholded = cv2.adaptiveThreshold(
        fft_filtered, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 8
    )
    
    # Compute metrics
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise_est = estimate_noise(gray)
    
    # Formulate recommendations based on standard quality bands
    recomm = "Image quality is excellent for PAN validation."
    risk_score = 0.0
    
    if blur_score < 100:
        recomm = "WARNING: Image is highly blurry. Corner detection & OCR might fail."
        risk_score += 0.5
    elif blur_score < 250:
        recomm = "Note: Image is slightly blurry. Proceeding with caution."
        risk_score += 0.2
        
    if brightness < 60:
        recomm += " Also, the image is very dark. CLAHE has been applied to normalize."
        risk_score += 0.3
    elif brightness > 220:
        recomm += " Also, the image is highly overexposed/washed out."
        risk_score += 0.3
        
    risk_score = min(risk_score, 1.0)
    
    return {
        "phase": "image_preprocessing",
        "success": True,
        "metrics": {
            "width": w,
            "height": h,
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "blur_score": round(blur_score, 2),
            "noise_estimate": noise_est,
            "quality_risk": risk_score
        },
        "steps": [
            {
                "name": "original",
                "description": "Original raw image uploaded by user.",
                "image_base64": image_to_base64(image)
            },
            {
                "name": "grayscale",
                "description": "Converted to single-channel intensity for base processing.",
                "image_base64": image_to_base64(gray)
            },
            {
                "name": "contrast_enhanced",
                "description": "CLAHE applied to balance localized lighting and amplify border contrast.",
                "image_base64": image_to_base64(enhanced)
            },
            {
                "name": "denoised",
                "description": "Bilateral filter applied to eliminate camera noise while protecting text margins.",
                "image_base64": image_to_base64(denoised)
            },
            {
                "name": "fft_spectrum",
                "description": "2D Fast Fourier Transform power spectrum (revealing periodic frequencies).",
                "image_base64": image_to_base64(fft_spectrum_before)
            },
            {
                "name": "fft_filtered",
                "description": "Inverse FFT after notch-reject filter to suppress grid lines and texture.",
                "image_base64": image_to_base64(fft_filtered)
            },
            {
                "name": "thresholded",
                "description": "Adaptive Gaussian binarization optimized for high Tesseract character recognition.",
                "image_base64": image_to_base64(thresholded)
            }
        ],
        "recommendation": recomm
    }
