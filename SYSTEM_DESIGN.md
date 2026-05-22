py# AI-Powered PAN Card Verification and Tampering Detection System using Computer Vision

## 1. Project Goal

This project verifies a PAN card image and detects possible tampering by combining classical computer vision, OCR, rule-based validation, and a final fraud decision engine.

The system must answer four questions:

1. Is the uploaded image clear enough for analysis?
2. Does the image contain a properly detected and corrected PAN card?
3. Does the card visually match the expected PAN card structure/template?
4. Are the extracted PAN details valid and consistent with the visual evidence?

## 2. High-Level Pipeline

```text
Input PAN Card Image
        ↓
Image Preprocessing
        ↓
Document Detection
        ↓
Perspective Correction
        ↓
Feature Extraction
        ↓
Feature Matching
        ↓
Forgery/Tampering Detection
        ↓
OCR Extraction
        ↓
PAN Validation
        ↓
Fraud Decision
```

## 3. System Architecture

The project is split into a frontend and a backend.

### 3.1 Frontend Responsibilities

The frontend is responsible for user interaction and result explanation.

Main duties:

- Upload PAN card image.
- Show original image preview.
- Trigger backend verification pipeline.
- Show each phase result separately.
- Display visual outputs such as preprocessed image, detected document boundary, corrected card, matched features, OCR boxes, and fraud score.
- Explain why a card is accepted, suspicious, or rejected.
- Let the user trace each decision back to the pipeline stage that produced it.

Expected frontend screens:

- Upload screen.
- Phase-wise analysis screen.
- OCR extraction screen.
- Final fraud decision screen.
- Optional history/report screen.

### 3.2 Backend Responsibilities

The backend performs all computer vision and decision logic.

Main duties:

- Receive image uploads.
- Validate file type and image quality.
- Run image preprocessing.
- Detect card boundary.
- Correct perspective.
- Extract and match features.
- Detect tampering signals.
- Run OCR.
- Validate PAN format.
- Combine all scores into final fraud probability.
- Return a structured JSON response with image artifacts and scores.

Recommended backend stack:

- Python
- FastAPI
- OpenCV
- NumPy
- pytesseract
- scikit-image, optional
- scikit-learn, optional for scoring experiments

### 3.3 Data Flow Between Frontend and Backend

```text
Frontend
  └── Uploads image as multipart/form-data
        ↓
Backend API
  └── Runs phase pipeline
        ↓
Backend response
  ├── Processed image previews as base64
  ├── Extracted OCR values
  ├── Stage-wise confidence scores
  ├── Fraud probability
  └── Human-readable explanations
        ↓
Frontend
  └── Displays traceable verification report
```

## 4. Backend API Design

### 4.1 Health Check

```http
GET /health
```

Purpose:

- Confirms the backend is running.

Response:

```json
{
  "status": "ok",
  "service": "pan-verification-api"
}
```

### 4.2 Phase 1 Image Preprocessing

```http
POST /api/phase-1/preprocess
```

Input:

- `file`: uploaded PAN card image.

Output:

```json
{
  "phase": "image_preprocessing",
  "metrics": {
    "width": 1280,
    "height": 720,
    "brightness": 132.4,
    "contrast": 48.2,
    "blur_score": 294.7,
    "noise_estimate": 5.8
  },
  "steps": [
    {
      "name": "grayscale",
      "description": "Converted image to grayscale for intensity analysis.",
      "image_base64": "..."
    }
  ],
  "recommendation": "Image quality is acceptable for document detection."
}
```

### 4.3 Full Verification, Future Endpoint

```http
POST /api/verify
```

This endpoint will eventually run all phases and return the complete fraud report.

## 5. Phase 1: Image Preprocessing

### 5.1 Goal

Improve the uploaded image so later stages such as edge detection, feature extraction, OCR, and tampering detection become more reliable.

Poor image quality causes:

- Weak card boundaries.
- Incorrect corner detection.
- Missing ORB keypoints.
- Bad OCR extraction.
- False tampering alerts.

### 5.2 Inputs

Accepted inputs:

- JPG
- JPEG
- PNG
- WebP, optional

Image cases:

- Clear mobile photo.
- Dark image.
- Low contrast scan.
- Tilted image.
- Noisy image.
- Blurry image.
- Image with shadows.
- Image with periodic scanner noise.

### 5.3 Outputs

Phase 1 should produce:

- Original normalized image.
- Grayscale image.
- Contrast-enhanced image.
- Denoised image.
- Frequency-filtered image.
- OCR-ready thresholded image.
- Image quality metrics.
- Recommendation for the next stage.

### 5.4 Techniques and Why They Are Used

#### 5.4.1 Grayscale Conversion

What it does:

- Converts the RGB/BGR image into a single intensity channel.

Why:

- Most low-level CV operations such as edge detection, thresholding, blur detection, and contour tracing work better on grayscale images.

Used by:

- Document detection (Gaussian Blur → Histogram Equalization → Canny → Morphological Closing → Contour → ApproxPolyDP).
- OCR preprocessing.
- Blur and brightness analysis.

#### 5.4.2 Histogram Equalization

Problem solved:

- Dark images.
- Low contrast images.
- Poor visibility of text and borders.

Technique:

- Use CLAHE, Contrast Limited Adaptive Histogram Equalization, instead of only global histogram equalization.

Why CLAHE:

- Global histogram equalization can over-amplify noise.
- CLAHE improves local contrast while limiting excessive brightness changes.

Useful cases:

- PAN card photo taken in a dim room.
- Image captured under shadow.
- Faded scan.

Risk:

- If applied too aggressively, it can make compression artifacts stronger.

Mitigation:

- Use moderate clip limit.
- Keep both original and enhanced image for comparison.

#### 5.4.3 Noise Removal

Problem solved:

- Salt-and-pepper noise.
- Camera sensor noise.
- Small compression artifacts.

Techniques:

- Median filter for salt-and-pepper noise.
- Gaussian filter for smooth random noise.
- Bilateral filter, optional, for preserving edges.

Why:

- OCR and edge detection are sensitive to noise.
- Denoising avoids false keypoints during ORB extraction.

Useful cases:

- Low-light mobile photo.
- Scanned image with speckles.
- Compressed WhatsApp-forwarded image.

Risk:

- Too much smoothing can blur text edges.

Mitigation:

- Use small kernel sizes.
- Keep denoising configurable.

#### 5.4.4 Frequency Domain Filtering

Problem solved:

- Periodic scan lines.
- Repeating printer/scanner texture.
- Structured noise patterns.

Technique:

- Apply Fourier transform.
- Identify dominant periodic noise frequencies.
- Suppress those frequency components.
- Apply inverse Fourier transform.

Why this is academically strong:

- It directly connects Fourier transform syllabus concepts to a real document verification problem.
- Many student projects skip frequency domain processing, so this gives the project a stronger CV foundation.

Useful cases:

- Scanned PAN card with horizontal or vertical line patterns.
- Printed card photographed from a screen.
- Repeated texture noise from low-quality printers.

Risk:

- Incorrect filtering can remove real text strokes.

Mitigation:

- Suppress only strong frequency peaks away from the image center.
- Use frequency filtering as an optional enhancement, not the only image used.

#### 5.4.5 OCR-Ready Thresholding

Problem solved:

- OCR engines perform better when text is separated from background.

Technique:

- Adaptive thresholding.

Why:

- PAN card images may have uneven lighting.
- Adaptive thresholding handles local brightness variations better than one global threshold.

Useful cases:

- Shadows on one side of the card.
- Light reflection.
- Uneven mobile camera exposure.

## 6. Phase 2: Document Detection

### 6.1 Goal

Locate the PAN card region inside the uploaded image.

### 6.2 Main Technique: Contour-Based Corner Detection

What it does:

- Locates the four corners of the PAN card by tracing its closed rectangular boundary through a chained edge-analysis pipeline.

Why this approach was adopted:

- Harris corner detection failed in practice: it produces tens of candidate corners scattered across text glyphs, logo patterns, and card artwork rather than reliably returning the four true document corners.
- A contour-based approach explicitly targets the largest closed boundary in the image, which — after proper edge enhancement — corresponds to the card outline.

Pipeline:

```text
Grayscale image (from Phase 1)
        ↓
Image Enhancement
  • Gaussian Blur (5×5) — suppress sensor / JPEG noise
  • Histogram Equalization — maximise card-to-background contrast
        ↓
Canny Edge Detection
  • Upper threshold via Otsu's method
  • Lower threshold = 0.5 × upper
        ↓
Morphological Closing (7×7 rect kernel, 2 iterations)
  • Bridges edge gaps caused by reflections and shadows
        ↓
Contour Detection (cv2.RETR_EXTERNAL)
  • Only external contours; avoids internal card text
        ↓
ApproxPolyDP
  • ε = 2 % of arc-length → quadrilateral simplification
  • Retry at ε = 4 % if 4-vertex polygon not found
        ↓
Aspect-Ratio Filtering
  • Accept quadrilaterals with width/height ∈ [1.3, 2.1]
  • PAN card physical ratio ≈ 1.585 (85.6 mm / 54 mm)
  • Must cover ≥ 8 % of image area
        ↓
Four ordered corners [TL, TR, BR, BL]
```

Post-detection (Phase 4 onwards):

- ORB feature extraction and matching is performed **after** perspective correction, not in this stage.
- This cleanly separates geometric correction from appearance-based analysis.

### 6.3 Cases and Techniques

Clear rectangular card on plain background:

- Canny + contour pipeline directly yields a tight 4-point polygon.
- Aspect-ratio filter confirms the shape as a card.

Busy or textured background:

- Morphological closing ensures the card border remains a dominant closed contour.
- Sorting contours by area and inspecting only the top-10 candidates avoids false positives from background clutter.

Low contrast boundary (dark card on dark background):

- Histogram equalization before Canny lifts the relative edge strength at the card border.
- If the boundary still cannot be found, the system falls back to a 5%-padded full-image rectangle and flags a high detection risk (0.5).

Partially visible card:

- ApproxPolyDP will not produce a clean quadrilateral; the aspect-ratio filter rejects such candidates.
- Fallback rectangle is applied and the result is flagged for manual review.

Moderately tilted / perspective-distorted card:

- The contour follows the true projected shape of the card, so the four corners are still geometrically correct for the subsequent homography step.

## 7. Phase 3: Perspective Correction

### 7.1 Goal

Convert a tilted PAN card image into a straight top-down view.

### 7.2 Main Technique: Homography with RANSAC

What it does:

- Homography maps points from the tilted document plane to a rectangular target plane.
- RANSAC removes incorrect corner or feature matches.

Pipeline:

```text
Detected corner points
        ↓
Estimate homography matrix
        ↓
Use RANSAC to reject outliers
        ↓
Apply perspective transform
        ↓
Aligned PAN card image
```

### 7.3 Why RANSAC Matters

Real images may contain:

- Wrong corner detections.
- Background corners.
- Card shadows.
- Reflections.

RANSAC improves robustness by selecting the transformation that agrees with the strongest set of points.

## 8. Phase 4: Feature Extraction

### 8.1 Goal

Extract visual keypoints and descriptors from the corrected PAN card.

### 8.2 Recommended Technique: ORB

Why ORB:

- Fast.
- Free to use.
- Works well for real-time and student projects.
- Suitable for matching logos, layout structure, and printed features.

Comparison:

| Algorithm | Strength | Limitation |
| --- | --- | --- |
| SIFT | Very robust | Slower |
| SURF | Fast and robust | Patent/licensing concerns in older OpenCV builds |
| ORB | Fast and practical | Less robust than SIFT in extreme cases |

### 8.3 Extracted Features

ORB can extract keypoints from:

- Government emblem/logo area.
- Text regions.
- Card layout boundaries.
- Signature-like patterns.
- Printed symbols.

## 9. Phase 5: Feature Matching

### 9.1 Goal

Compare the uploaded PAN card against an official template/reference.

### 9.2 Matching Techniques

Brute Force Matching:

- Easy to implement.
- Good for first version.
- Works with ORB using Hamming distance.

K-D Tree:

- Useful for faster matching with floating-point descriptors such as SIFT.

LSH:

- Good approximate matching option for large descriptor sets.
- Useful as a project enhancement.

### 9.3 Fraud Signals

Feature matching may reveal:

- Template mismatch.
- Modified text area.
- Missing logo or changed placement.
- Distorted card structure.
- Region-level mismatch around PAN number, name, or DOB.

## 10. Phase 6: Forgery and Tampering Detection

### 10.1 Goal

Detect visual evidence that the card image has been edited.

### 10.2 Copy-Move Forgery Detection

Problem:

- Fraudster copies one region and pastes it elsewhere.

Technique:

- Extract ORB or SIFT features.
- Match features within the same image.
- Detect suspicious duplicate clusters.

Output:

- Duplicate region candidates.
- Copy-move confidence score.

### 10.3 Texture Inconsistency

Problem:

- Edited regions often have different texture or compression patterns.

Technique:

- HOG descriptors.
- Local Binary Patterns, optional.
- Local variance analysis.

Output:

- Texture anomaly heatmap.
- Texture inconsistency score.

### 10.4 Edge Inconsistency

Problem:

- Editing creates unnatural boundaries around pasted text.

Technique:

- Canny edge detection.
- Gradient magnitude analysis.
- Edge density comparison around text fields.

Output:

- Suspicious boundary map.
- Edge anomaly score.

## 11. Phase 7: OCR Extraction

### 11.1 Goal

Extract important text fields from the corrected PAN card.

OCR engine:

- Tesseract OCR.

Fields:

- PAN number.
- Name.
- Father name, if visible.
- Date of birth.

### 11.2 OCR Preprocessing

Use outputs from Phase 1:

- Contrast-enhanced image.
- OCR-ready thresholded image.
- Denoised image.

Possible OCR strategies:

- Full-card OCR.
- Region-specific OCR after template alignment.
- Multiple preprocessing attempts and choose best confidence.

## 12. Phase 8: PAN Validation

### 12.1 Goal

Validate whether OCR-extracted PAN card fields conform to official Income Tax Department format rules.

Important scope constraint:

- All checks operate exclusively on OCR-extracted values from the same uploaded PAN card.
- The visual reference template used in Phase 4 and 5 is a structural alignment aid only.
- No template identity data (template name, template DOB, template surname) is used or referenced here.

PAN format:

```text
ABCDE1234F
```

### 12.2 Validation Pipeline

```text
OCR Extraction (Phase 7)
        ↓
Extract PAN Number
        ↓
Extract Name / Surname
        ↓
Extract DOB
        ↓
Rule-Based Validation Checks (1–6)
```

### 12.3 Validation Checks

#### Check 1: PAN Regex Format Validation

Validates the full PAN string against the official pattern:

```regex
^[A-Z]{5}[0-9]{4}[A-Z]{1}$
```

Failure penalty: 0.50 (hard failure — card likely fake if basic format is wrong).

#### Check 2: Holder Type Validation (4th Character)

The 4th character encodes the taxpayer entity category:

| Code | Entity |
| --- | --- |
| P | Individual (Personal) |
| C | Company |
| F | Firm / Limited Liability Partnership |
| H | Hindu Undivided Family (HUF) |
| A | Association of Persons (AOP) |
| T | Trust |
| B | Body of Individuals (BOI) |
| G | Government Agency |
| L | Local Authority |
| J | Artificial Juridical Person |

Failure penalty: 0.15.

#### Check 3: Surname Initial Validation (5th Character)

For Indian PAN cards the 5th character must equal the first letter of the cardholder's surname.

Extraction rule:

- OCR name field: full name from the uploaded card, e.g. HARSH SHARMA.
- Surname: last word of the OCR name, e.g. SHARMA.
- Initial: first letter of surname, e.g. S.
- PAN 5th character must equal this initial.

Example:

```text
OCR Name : HARSH SHARMA
Surname  : SHARMA   →   Initial: S
PAN      : ABCDS1234F   →   5th char: S   →   PASS
```

Source: OCR-extracted name from the same uploaded card only. No template comparison.

Failure penalty: 0.25. Warning (no penalty) if name field is empty.

#### Check 4: Sequential Digits Validation (Characters 6–9)

Characters at positions 6, 7, 8, 9 must all be decimal digits 0–9.

Failure penalty: 0.10.

#### Check 5: Series / Checksum Character Validation (10th Character)

The 10th character is the PAN series identifier issued by the Income Tax Department. It must be an uppercase letter A–Z.

Note: A publicly standardised algorithmic checksum formula for PAN does not exist. This check enforces the structural constraint (must be a single uppercase letter).

Failure penalty: 0.10.

#### Check 6: DOB Format and Date Sanity Validation

Accepted formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY.

Additional sanity checks:

- Year must be between 1900 and the current year.
- Day and month values must be calendar-valid.

Failure penalty: 0.15.

## 13. Phase 9: Fraud Decision Engine

### 13.1 Goal

Combine evidence from all stages into a final fraud probability.

### 13.2 Example Signals

| Check | Result | Impact |
| --- | --- | --- |
| OCR PAN format valid | Yes | Reduces fraud probability |
| Feature mismatch | High | Increases fraud probability |
| Texture anomaly | Yes | Increases fraud probability |
| Duplicate regions | Yes | Increases fraud probability |
| Blur score poor | Yes | Lowers confidence |
| Template alignment good | Yes | Reduces fraud probability |

### 13.3 Suggested Scoring Model

Initial rule-based score:

```text
fraud_score =
  0.30 * feature_mismatch_score +
  0.25 * tampering_score +
  0.20 * ocr_invalidity_score +
  0.15 * document_detection_risk +
  0.10 * image_quality_risk
```

Decision bands:

| Fraud Probability | Decision |
| --- | --- |
| 0% - 30% | Likely genuine |
| 31% - 60% | Needs manual review |
| 61% - 100% | Suspicious / likely fraudulent |

## 14. Traceability Matrix

| Project Stage | Syllabus Topic | Purpose |
| --- | --- | --- |
| Image preprocessing | Histogram equalization | Improve contrast |
| Image preprocessing | Gaussian and median filtering | Remove noise |
| Image preprocessing | Fourier transform | Remove periodic scan noise |
| Document detection | Canny edges, morphological closing, contour detection, ApproxPolyDP, aspect-ratio filter | Locate and extract the four card corners robustly |
| Perspective correction | RANSAC | Remove outliers while estimating homography |
| Feature extraction | ORB, SIFT, SURF | Extract keypoints and descriptors |
| Feature matching | Brute Force, K-D Tree, LSH | Compare uploaded card with template |
| Tampering detection | HOG, gradients, edge detection | Detect edited regions |
| OCR | Tesseract | Extract PAN details |
| Validation | Regex | Validate PAN format |
| Decision engine | Weighted scoring | Produce fraud probability |

## 15. Development Roadmap

### Milestone 1: Phase 1 Working Slice

Deliver:

- Backend upload endpoint.
- Image preprocessing pipeline.
- Frontend upload page.
- Display preprocessed images and quality metrics.

### Milestone 2: Document Detection

Deliver:

- Contour-based detection pipeline (Gaussian Blur → Histogram Equalization → Canny → Morphological Closing → Contour → ApproxPolyDP → Aspect-Ratio Filter).
- Document boundary overlay with labeled corners (TL, TR, BR, BL).
- Intermediate debug images for each detection stage.
- Full-image fallback with detection-risk scoring.

### Milestone 3: Perspective Correction

Deliver:

- Homography estimation.
- RANSAC-based correction.
- Straightened PAN card output.

### Milestone 4: Feature Extraction and Matching

Deliver:

- ORB keypoint extraction.
- Brute Force matching against template.
- Match score and visualization.

### Milestone 5: Tampering Detection

Deliver:

- Copy-move detection.
- Texture inconsistency score.
- Edge anomaly score.

### Milestone 6: OCR and PAN Validation

Deliver:

- Tesseract OCR extraction.
- PAN regex validation.
- Field-level confidence.

### Milestone 7: Fraud Decision Report

Deliver:

- Combined score.
- Decision band.
- Explainable report.

## 16. Required Inputs From User Later

The project can start without these, but the following will be needed for stronger accuracy:

- One or more sample PAN card images for testing, with sensitive details masked if needed.
- A clean reference/template image for feature matching.
- Decision threshold preference: strict, balanced, or lenient.
- Whether the final project should be a local demo, deployed web app, or college submission package.

