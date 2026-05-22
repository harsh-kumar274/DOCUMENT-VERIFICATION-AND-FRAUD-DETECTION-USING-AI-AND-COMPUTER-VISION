import re
import numpy as np
from datetime import datetime
from backend.config import logger

# ---------------------------------------------------------------------------
# Official Income Tax Department PAN Card Holder Type Codes (4th character)
# ---------------------------------------------------------------------------
PAN_HOLDER_TYPES = {
    "P": "Individual (Personal)",
    "C": "Company",
    "F": "Firm / Limited Liability Partnership",
    "H": "Hindu Undivided Family (HUF)",
    "A": "Association of Persons (AOP)",
    "T": "Trust",
    "B": "Body of Individuals (BOI)",
    "G": "Government Agency",
    "L": "Local Authority",
    "J": "Artificial Juridical Person",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_pan_regex(pan_str: str) -> tuple:
    """
    Check 1 — PAN Regex Format Validation.
    Official PAN format: [A-Z]{5}[0-9]{4}[A-Z]{1}
    Returns (passed: bool, message: str, score_penalty: float)
    """
    pan_regex = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
    if re.match(pan_regex, pan_str):
        return True, f"PAN '{pan_str}' matches the official format [A-Z]{{5}}[0-9]{{4}}[A-Z].", 0.0
    else:
        detail = f"Expected pattern ABCDE1234F — got '{pan_str}'."
        return False, f"PAN format is invalid. {detail}", 0.50


def _check_holder_type(pan_str: str) -> tuple:
    """
    Check 2 — Holder Type Validation (4th character).
    Validates the taxpayer category encoded in position 4.
    Returns (passed: bool, message: str, score_penalty: float)
    """
    if len(pan_str) < 4:
        return False, "PAN too short to extract holder-type character.", 0.15

    char4 = pan_str[3]
    holder_desc = PAN_HOLDER_TYPES.get(char4)
    if holder_desc:
        return (
            True,
            f"4th character '{char4}' is a valid taxpayer category: {holder_desc}.",
            0.0,
        )
    else:
        return (
            False,
            f"4th character '{char4}' is not a recognised Income Tax entity code. "
            f"Valid codes: {', '.join(PAN_HOLDER_TYPES.keys())}.",
            0.15,
        )


def _check_surname_initial(pan_str: str, name_str: str) -> tuple:
    """
    Check 3 — Surname Initial Validation (5th character).
    For individuals (4th char == 'P') the 5th PAN character must equal the
    first letter of the cardholder's surname extracted from the OCR name field.

    Indian PAN naming convention:
        Full name on card: FIRST [MIDDLE] SURNAME
        Surname initial  → last word of the name → its first letter

    Example:
        Name: HARSH SHARMA  →  Surname: SHARMA  →  Initial: S
        PAN : ABCDS1234F    →  5th char: S       →  PASS

    Note: This validation uses ONLY the OCR-extracted name from the SAME
    uploaded card. No template identity data is used or referenced here.

    Returns (passed: bool, message: str, score_penalty: float)
    """
    if len(pan_str) < 5:
        return False, "PAN too short to extract surname-initial character.", 0.0

    pan_5th = pan_str[4]

    if not name_str:
        return (
            None,  # None = WARNING (not hard failure)
            "Cardholder name field is empty — surname initial check skipped.",
            0.0,
        )

    name_parts = name_str.upper().split()
    if not name_parts:
        return (
            None,
            "Name field could not be tokenised — surname initial check skipped.",
            0.0,
        )

    # Last word in the full name is the surname (standard Indian PAN convention)
    surname = name_parts[-1]
    surname_initial = surname[0]

    if surname_initial == pan_5th:
        return (
            True,
            f"5th character '{pan_5th}' matches the surname initial of '{surname}' (from OCR name: '{name_str}').",
            0.0,
        )
    else:
        return (
            False,
            f"Mismatch: 5th PAN character is '{pan_5th}' but OCR surname "
            f"'{surname}' starts with '{surname_initial}'. "
            f"(OCR name field: '{name_str}')",
            0.25,
        )


def _check_sequential_digits(pan_str: str) -> tuple:
    """
    Check 4 — Sequential Digits Validation (characters 6–9).
    Positions 5–8 (0-indexed) must be exactly 4 decimal digits.
    Returns (passed: bool, message: str, score_penalty: float)
    """
    if len(pan_str) < 9:
        return False, "PAN too short to verify digit sequence (positions 6–9).", 0.10

    digit_segment = pan_str[5:9]   # characters at positions 6,7,8,9 (1-indexed)
    if digit_segment.isdigit():
        return (
            True,
            f"Characters 6–9 '{digit_segment}' are all valid numeric digits.",
            0.0,
        )
    else:
        non_digits = [c for c in digit_segment if not c.isdigit()]
        return (
            False,
            f"Characters 6–9 '{digit_segment}' contain non-digit character(s): "
            f"{non_digits}. All four must be numeric.",
            0.10,
        )


def _check_checksum_character(pan_str: str) -> tuple:
    """
    Check 5 — Checksum / Series Character Validation (10th character).
    The 10th character (position index 9) must be an uppercase letter A–Z.
    The Income Tax Department uses this as an issuing series identifier.
    Full algorithmic checksum verification is not publicly standardised;
    this check enforces the structural constraint (must be [A-Z]).

    Returns (passed: bool, message: str, score_penalty: float)
    """
    if len(pan_str) < 10:
        return False, "PAN too short to verify the 10th (series/checksum) character.", 0.10

    char10 = pan_str[9]
    if char10.isalpha() and char10.isupper():
        return (
            True,
            f"10th character '{char10}' is a valid uppercase letter (series/checksum character).",
            0.0,
        )
    else:
        return (
            False,
            f"10th character '{char10}' must be an uppercase letter [A–Z]. "
            "This is the PAN issuing series identifier.",
            0.10,
        )


def _check_dob(dob_str: str) -> tuple:
    """
    Check 6 — DOB Format and Date Sanity Validation.
    Accepted formats: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    Returns (passed: bool, message: str, score_penalty: float)
    """
    if not dob_str:
        return False, "Date of Birth field is empty.", 0.15

    dob_str = dob_str.strip()

    # Match pattern
    patterns = [
        r"^(\d{2})/(\d{2})/(\d{4})$",
        r"^(\d{2})-(\d{2})-(\d{4})$",
        r"^(\d{2})\.(\d{2})\.(\d{4})$",
    ]
    format_matched = any(re.match(p, dob_str) for p in patterns)
    if not format_matched:
        return (
            False,
            f"DOB '{dob_str}' does not match any accepted format. "
            "Expected DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YYYY.",
            0.15,
        )

    # Parse and sanity-check the actual date values
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            date_obj = datetime.strptime(dob_str, fmt)
            current_year = datetime.now().year
            if date_obj.year < 1900 or date_obj.year > current_year:
                return (
                    False,
                    f"DOB year '{date_obj.year}' is out of acceptable range "
                    f"(1900 – {current_year}).",
                    0.15,
                )
            age = current_year - date_obj.year
            return (
                True,
                f"DOB '{dob_str}' is valid. Approximate age: {age} years.",
                0.0,
            )
        except ValueError:
            continue

    return (
        False,
        f"DOB '{dob_str}' contains invalid calendar values (e.g. day > 31 or month > 12).",
        0.15,
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_pan_validation(ocr_results: dict) -> dict:
    """
    Phase 8 — Rule-Based PAN Format Validation.

    All checks are performed EXCLUSIVELY on OCR-extracted values from the
    uploaded PAN card.  The visual reference template used in Phase 4/5 is a
    structural alignment aid only; no template identity data is used here.

    Validation pipeline:
        OCR Extraction
            ↓
        Extract PAN Number
            ↓
        Extract Name / Surname
            ↓
        Extract DOB
            ↓
        1. PAN Regex Validation          [A-Z]{5}[0-9]{4}[A-Z]
        2. Holder Type Validation        4th character → taxpayer category
        3. Surname Initial Validation    5th character vs OCR surname initial
        4. Sequential Digits Validation  characters 6–9 must be digits
        5. Checksum Character Validation 10th character must be [A-Z]
        6. DOB Format Validation         DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY

    Returns:
        dict with keys: phase, success, ocr_invalidity_score, checks, explanation
    """
    fields = ocr_results.get("fields", {})

    pan_str  = fields.get("PAN",  {}).get("text", "").upper().strip().replace(" ", "")
    name_str = fields.get("Name", {}).get("text", "").upper().strip()
    dob_str  = fields.get("DOB",  {}).get("text", "").strip()

    logger.info(
        f"Running Phase 8 PAN Validation | PAN='{pan_str}' | "
        f"Name='{name_str}' | DOB='{dob_str}'"
    )

    validation_checks = []
    ocr_invalidity_score = 0.0

    # ------------------------------------------------------------------
    # Check 1: PAN Regex Format
    # ------------------------------------------------------------------
    passed, msg, penalty = _check_pan_regex(pan_str)
    ocr_invalidity_score += penalty
    validation_checks.append({
        "check": "PAN Format Regex",
        "status": "PASS" if passed else "FAIL",
        "message": msg,
    })

    # ------------------------------------------------------------------
    # Checks 2–5 only make sense if PAN has the expected length
    # ------------------------------------------------------------------
    if len(pan_str) == 10:
        # Check 2: Holder Type
        passed, msg, penalty = _check_holder_type(pan_str)
        ocr_invalidity_score += penalty
        validation_checks.append({
            "check": "Holder Type (4th Character)",
            "status": "PASS" if passed else "FAIL",
            "message": msg,
        })

        # Check 3: Surname Initial
        result, msg, penalty = _check_surname_initial(pan_str, name_str)
        ocr_invalidity_score += penalty
        if result is None:
            status = "WARNING"
        elif result:
            status = "PASS"
        else:
            status = "FAIL"
        validation_checks.append({
            "check": "Surname Initial (5th Character)",
            "status": status,
            "message": msg,
        })

        # Check 4: Sequential Digits (chars 6–9)
        passed, msg, penalty = _check_sequential_digits(pan_str)
        ocr_invalidity_score += penalty
        validation_checks.append({
            "check": "Sequential Digits (Chars 6–9)",
            "status": "PASS" if passed else "FAIL",
            "message": msg,
        })

        # Check 5: Checksum / Series Character (10th char)
        passed, msg, penalty = _check_checksum_character(pan_str)
        ocr_invalidity_score += penalty
        validation_checks.append({
            "check": "Series / Checksum Character (10th)",
            "status": "PASS" if passed else "FAIL",
            "message": msg,
        })

    else:
        # PAN length is wrong — add a single length failure and skip char-level checks
        ocr_invalidity_score += 0.20
        validation_checks.append({
            "check": "PAN Length",
            "status": "FAIL",
            "message": (
                f"PAN has {len(pan_str)} characters instead of the required 10. "
                "Character-level checks (holder type, surname, digits, checksum) skipped."
            ),
        })

    # ------------------------------------------------------------------
    # Check 6: DOB Format and Sanity
    # ------------------------------------------------------------------
    passed, msg, penalty = _check_dob(dob_str)
    ocr_invalidity_score += penalty
    validation_checks.append({
        "check": "DOB Format & Date Validity",
        "status": "PASS" if passed else "FAIL",
        "message": msg,
    })

    # ------------------------------------------------------------------
    # Compile result
    # ------------------------------------------------------------------
    ocr_invalidity_score = float(np.clip(ocr_invalidity_score, 0.0, 1.0))
    hard_failures = [c for c in validation_checks if c["status"] == "FAIL"]
    success = len(hard_failures) == 0

    if success:
        explanation = (
            "All PAN validation checks passed. The extracted fields conform to "
            "official Income Tax Department format rules."
        )
    else:
        fail_names = ", ".join(c["check"] for c in hard_failures)
        explanation = (
            f"Validation failed on: {fail_names}. "
            "Mismatched structures or format anomalies detected in the OCR-extracted fields."
        )

    logger.info(
        f"Phase 8 complete — success={success}, invalidity_score={ocr_invalidity_score:.2f}"
    )

    return {
        "phase": "pan_validation",
        "success": success,
        "ocr_invalidity_score": round(ocr_invalidity_score, 2),
        "checks": validation_checks,
        "explanation": explanation,
    }
