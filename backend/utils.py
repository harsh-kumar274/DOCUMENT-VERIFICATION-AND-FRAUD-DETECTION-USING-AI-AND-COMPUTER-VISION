import cv2
import numpy as np
import base64
import os
from backend.config import logger

def resize_image(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """Resize image preserving aspect ratio if dimensions exceed maximums."""
    h, w = image.shape[:2]
    scale = 1.0
    
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h} (scale factor: {scale:.2f})")
    
    return image

def image_to_base64(image: np.ndarray, format_str: str = ".jpg") -> str:
    """Convert an OpenCV image to a base64 encoded string."""
    try:
        success, encoded_img = cv2.imencode(format_str, image)
        if not success:
            raise ValueError("Failed to encode image")
        base64_data = base64.b64encode(encoded_img).decode("utf-8")
        ext = format_str.replace(".", "")
        return f"data:image/{ext};base64,{base64_data}"
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        return ""

def base64_to_image(base64_str: str) -> np.ndarray:
    """Convert a base64 encoded string back to an OpenCV image."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decoded image is None")
        return img
    except Exception as e:
        logger.error(f"Error decoding base64 to image: {str(e)}")
        raise ValueError(f"Invalid image format: {str(e)}")

def file_to_image(filepath: str) -> np.ndarray:
    """Read an image file robustly."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {filepath}")
    return img
