import cv2
import numpy as np
from backend.config import logger
from backend.utils import image_to_base64


# ---------------------------------------------------------------------------
# Utility: Point ordering
# ---------------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points into a canonical order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

    Args:
        pts: numpy array of shape (4, 2).

    Returns:
        Ordered numpy array of shape (4, 2) as float32.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-Left  → smallest (x + y) sum
    # Bottom-Right → largest (x + y) sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-Right → smallest (y - x) diff, i.e. largest (x - y)
    # Bottom-Left → largest (y - x) diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# ---------------------------------------------------------------------------
# Stage 1 – Image Enhancement for Detection
# ---------------------------------------------------------------------------

def _enhance_for_detection(image_gray: np.ndarray) -> np.ndarray:
    """
    Apply a dedicated enhancement chain tuned for card-boundary detection.

    Steps:
        1. Gaussian Blur  – suppress high-frequency sensor noise before edge detection.
        2. Histogram Equalization (CLAHE-style global fallback using cv2.equalizeHist)
           – lift contrast on dark / unevenly lit card surfaces so that the card
             border becomes a strong edge relative to the background.

    Returns:
        Enhanced single-channel uint8 image.
    """
    # 1. Gaussian Blur  (kernel 5×5, σ auto-calculated)
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # 2. Global Histogram Equalization for maximum edge contrast
    #    (We intentionally use global equalisation here rather than CLAHE because
    #     we want the global card-to-background contrast boosted, not local patches.)
    equalized = cv2.equalizeHist(blurred)

    return equalized


# ---------------------------------------------------------------------------
# Stage 2 – Canny + Morphological Closing
# ---------------------------------------------------------------------------

def _canny_and_close(enhanced: np.ndarray) -> np.ndarray:
    """
    Detect edges with Canny and then close small gaps so that the card
    outline forms a fully connected contour.

    Steps:
        1. Canny Edge Detection – precise, thin edges at the card boundary.
        2. Morphological Closing – bridge tiny gaps caused by reflections /
           shadows on the card edge.

    Returns:
        Binary uint8 edge map.
    """
    # 1. Canny – auto thresholds via Otsu's method on the enhanced image
    #    Lower threshold  = 0.5 × upper;  upper derived from Otsu.
    otsu_thresh, _ = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    lower = max(0, int(0.5 * otsu_thresh))
    upper = int(otsu_thresh)
    edges = cv2.Canny(enhanced, lower, upper)

    # 2. Morphological Closing – close small breaks in the card border lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


# ---------------------------------------------------------------------------
# Stage 3 – Contour Detection → ApproxPolyDP → Aspect-Ratio Filtering
# ---------------------------------------------------------------------------

def _find_card_contour(
    closed_edges: np.ndarray,
    image_area: int,
    image_h: int,
    image_w: int,
) -> np.ndarray | None:
    """
    Find the best quadrilateral contour that represents the PAN card.

    Pipeline:
        1. cv2.findContours on the closed edge map (external contours only).
        2. Sort contours by area descending; inspect the top-N candidates.
        3. ApproxPolyDP – simplify each contour to a polygon.
        4. Aspect-Ratio Filtering – accept only quadrilaterals whose width/height
           ratio is within the expected PAN card range (≈ 1.3 – 2.1) and that
           cover at least 8 % of the image area.

    Returns:
        numpy array of shape (4, 2) with the four card corners, or None.
    """
    contours, _ = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Sort by area descending; only inspect the largest 10 candidates
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        # ApproxPolyDP – epsilon controls how tightly the polygon follows the contour.
        # 2 % of arc-length is a standard heuristic for rectangles.
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) != 4:
            # Not a quadrilateral – try with a slightly looser epsilon
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) != 4:
                continue

        corners_candidate = approx.reshape(4, 2).astype(np.float32)

        # ---- Aspect-Ratio Filter ----
        ordered = order_points(corners_candidate)
        tl, tr, br, bl = ordered

        width_top    = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left  = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        avg_w = (width_top + width_bottom) / 2.0
        avg_h = (height_left + height_right) / 2.0

        if avg_h < 1e-6:
            continue

        ratio = avg_w / avg_h

        # PAN card physical ratio ≈ 85.6 mm / 53.98 mm ≈ 1.585
        # Accept anything between 1.3 and 2.1 to handle moderate perspective distortion
        aspect_ok = 1.3 <= ratio <= 2.1

        # Area filter: contour must cover at least 8 % of the image
        area_ratio = cv2.contourArea(approx) / (image_area + 1e-6)
        area_ok = area_ratio >= 0.08

        if aspect_ok and area_ok:
            logger.info(
                f"Card contour accepted – aspect ratio: {ratio:.3f}, "
                f"area coverage: {area_ratio:.2%}"
            )
            return ordered  # Already ordered [TL, TR, BR, BL]

    return None


# ---------------------------------------------------------------------------
# Main public entry-point
# ---------------------------------------------------------------------------

def detect_document_boundary(image: np.ndarray, image_gray: np.ndarray) -> dict:
    """
    Phase 2 – Document Detection.

    New pipeline (replaces Harris corner detector):
        ┌─────────────────────────────────┐
        │  Grayscale image (from Phase 1) │
        └──────────────┬──────────────────┘
                       ↓
              Image Enhancement
          (Gaussian Blur + Hist EQ)
                       ↓
             Canny Edge Detection
                       ↓
          Morphological Closing (7×7)
                       ↓
            Contour Detection (external)
                       ↓
                 ApproxPolyDP
                       ↓
         Aspect-Ratio Filtering (1.3–2.1)
                       ↓
          Four ordered card corners [TL,TR,BR,BL]

    ORB feature extraction and matching is performed AFTER perspective
    correction in Phase 4 (features.py) and is not part of this stage.

    Returns:
        dict with keys: phase, success, method_used, detection_risk,
                        corners (list[list[float]]), preview_base64, explanation,
                        debug_steps (list of intermediate base64 images).
    """
    h, w = image.shape[:2]
    image_area = h * w
    preview_img = image.copy()
    debug_steps: list[dict] = []

    success = False
    corners: np.ndarray | None = None
    method = "None"
    detection_risk = 0.0

    # ------------------------------------------------------------------
    # Stage 1: Image Enhancement (Gaussian Blur + Histogram Equalization)
    # ------------------------------------------------------------------
    try:
        enhanced = _enhance_for_detection(image_gray)
        debug_steps.append({
            "name": "detection_enhanced",
            "description": (
                "Gaussian Blur (5×5) applied to suppress sensor noise, followed by "
                "global Histogram Equalization to maximise card-to-background contrast."
            ),
            "image_base64": image_to_base64(enhanced),
        })
    except Exception as exc:
        logger.error(f"Enhancement step failed: {exc}")
        enhanced = image_gray.copy()

    # ------------------------------------------------------------------
    # Stage 2: Canny Edge Detection + Morphological Closing
    # ------------------------------------------------------------------
    try:
        closed_edges = _canny_and_close(enhanced)
        debug_steps.append({
            "name": "detection_edges",
            "description": (
                "Canny edge detection (auto-thresholds via Otsu) to locate the precise "
                "card boundary, followed by 7×7 morphological closing to bridge edge gaps."
            ),
            "image_base64": image_to_base64(closed_edges),
        })
    except Exception as exc:
        logger.error(f"Canny/closing step failed: {exc}")
        # Hard fallback: plain Canny with fixed thresholds
        closed_edges = cv2.Canny(image_gray, 50, 150)

    # ------------------------------------------------------------------
    # Stage 3: Contour → ApproxPolyDP → Aspect-Ratio Filtering
    # ------------------------------------------------------------------
    try:
        corners = _find_card_contour(closed_edges, image_area, h, w)
        if corners is not None:
            success = True
            method = "Contour + ApproxPolyDP + Aspect-Ratio Filter"
            logger.info("PAN card boundary detected via contour pipeline.")
    except Exception as exc:
        logger.error(f"Contour detection step failed: {exc}")

    # ------------------------------------------------------------------
    # Fallback: full-image boundary with padding
    # ------------------------------------------------------------------
    if not success or corners is None:
        logger.warning(
            "Contour pipeline could not locate clear card boundary. "
            "Applying full-image fallback with 5% padding."
        )
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        corners = np.array([
            [pad_x,     pad_y    ],
            [w - pad_x, pad_y    ],
            [w - pad_x, h - pad_y],
            [pad_x,     h - pad_y],
        ], dtype=np.float32)
        method = "Full-Image Fallback"
        # High risk – we did not detect a distinct document boundary
        detection_risk = 0.5
        success = False
    else:
        # Re-check aspect ratio for risk scoring (corners are already ordered)
        tl, tr, br, bl = corners
        avg_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        avg_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        ratio = avg_w / (avg_h + 1e-8)

        if ratio < 1.2 or ratio > 2.0:
            detection_risk = 0.4
            logger.warning(f"Detected card border aspect ratio is abnormal: {ratio:.3f}")
        else:
            detection_risk = 0.0

    # ------------------------------------------------------------------
    # Draw corners and boundary on preview image
    # ------------------------------------------------------------------
    corner_colors = [
        (0, 255,   0),   # TL – green
        (255, 165,  0),  # TR – orange
        (0,   0,  255),  # BR – blue
        (255,  0,   0),  # BL – red
    ]
    labels = ["TL", "TR", "BR", "BL"]

    for i, (corner, color) in enumerate(zip(corners, corner_colors)):
        cx, cy = int(corner[0]), int(corner[1])
        cv2.circle(preview_img, (cx, cy), 12, color, -1)
        cv2.putText(
            preview_img, labels[i],
            (cx - 18, cy - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

    cv2.polylines(preview_img, [corners.astype(int)], True, (0, 255, 255), 4)

    return {
        "phase": "document_detection",
        "success": success,
        "method_used": method,
        "detection_risk": detection_risk,
        "corners": corners.tolist(),
        "preview_base64": image_to_base64(preview_img),
        "debug_steps": debug_steps,
        "explanation": (
            f"Document boundary located using {method}." if success
            else (
                "Warning: Contour-based card detection could not verify a clear "
                "rectangular document boundary. Default full-frame perspective applied."
            )
        ),
    }
