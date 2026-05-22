"""
Phase 8: Rule-Based PAN Format Validation with OCR Confidence Awareness
=========================================================================
Performs structural validation of PAN card fields extracted by Phase 7 OCR.

Key improvements over naive strict validation:
  - All checks are OCR-confidence aware — low-confidence extractions produce
    WARNING status instead of hard FAIL, avoiding false positives from OCR noise
  - Check messages clearly distinguish actual format violations from OCR
    uncertainty (character misread, e.g. P↔D, O↔0, I↔1, B↔8, S↔E)
  - Validation only runs after successful perspective correction and ROI
    extraction; missing/empty OCR output is treated as low-confidence warning
  - Each check reports the raw OCR character alongside the validation verdict
    so operators can inspect exactly what Tesseract returned
"""

import re
import numpy as np
from datetime import datetime
from backend.config import logger

# ---------------------------------------------------------------------------
# OCR confidence threshold imported from ocr.py constant
# (duplicated here to avoid circular import)
# ---------------------------------------------------------------------------
OCR_CONFIDENCE_THRESHOLD = 85.0

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

# Common OCR misread pairs for informative error messages
_COMMON_MISREADS = {
    "D": ["P"], "P": ["D"],
    "0": ["O"], "O": ["0"],
    "1": ["I", "L"], "I": ["1", "L"], "L": ["I", "1"],
    "8": ["B"], "B": ["8"],
    "S": ["5", "E"], "5": ["S"], "E": ["S"],
    "G": ["C", "6"], "C": ["G"],
    "Z": ["2"], "2": ["Z"],
    "Q": ["O", "0"], "U": ["V"], "V": ["U"],
}


def _ocr_misread_hint(char: str) -> str:
    """Return a human-readable hint about possible OCR misreads for a char."""
    alternatives = _COMMON_MISREADS.get(char.upper(), [])
    if alternatives:
        alt_str = ", ".join(f"'{a}'" for a in alternatives)
        return f" Common OCR misread: '{char}' may have been confused with {alt_str}."
    return ""


def _confidence_aware_status(passed: bool, field_confidence: float,
                               penalty: float) -> tuple:
    """
    Determine check status considering OCR confidence.

    If the check fails BUT the confidence is below the threshold, the failure
    is downgraded to WARNING (possible OCR misread, not definite fraud).

    Returns: (status_str, effective_penalty)
    """
    if passed:
        return "PASS", 0.0
    if field_confidence < OCR_CONFIDENCE_THRESHOLD:
        # Downgrade to WARNING — lower penalty weight
        return "LOW_CONF_WARNING", penalty * 0.40
    return "FAIL", penalty


# ---------------------------------------------------------------------------
# Individual validation checks
# ---------------------------------------------------------------------------

def _check_pan_regex(pan_str: str, pan_conf: float) -> tuple:
    """
    Check 1 — PAN Regex Format Validation.
    Format: [A-Z]{5}[0-9]{4}[A-Z]{1}
    """
    pan_regex = r"^[A-Z]{5}[0-9]{4}[A-Z]$"
    passed = bool(re.match(pan_regex, pan_str))

    if passed:
        return "PASS", (
            f"PAN '{pan_str}' matches the official format [A-Z]{{5}}[0-9]{{4}}[A-Z]."
        ), 0.0

    # Build informative failure / warning message
    detail = f"Expected ABCDE1234F format — OCR returned '{pan_str}'."
    if pan_conf < OCR_CONFIDENCE_THRESHOLD:
        status = "LOW_CONF_WARNING"
        penalty = 0.20   # half of full penalty
        msg = (
            f"PAN format invalid. {detail} "
            f"⚠ OCR confidence is {pan_conf:.1f}% (below {OCR_CONFIDENCE_THRESHOLD:.0f}% threshold). "
            "This may be caused by OCR misreading characters — not a definitive fraud signal."
        )
    else:
        status = "FAIL"
        penalty = 0.50
        msg = f"PAN format is invalid. {detail}"

    return status, msg, penalty


def _check_holder_type(pan_str: str, pan_conf: float) -> tuple:
    """
    Check 2 — Holder Type (4th character).
    """
    if len(pan_str) < 4:
        return "FAIL", "PAN too short to extract holder-type character.", 0.15

    char4 = pan_str[3]
    holder_desc = PAN_HOLDER_TYPES.get(char4)

    if holder_desc:
        return "PASS", (
            f"4th character '{char4}' is a valid taxpayer category: {holder_desc}."
        ), 0.0

    # Failed — build OCR-aware message
    hint = _ocr_misread_hint(char4)
    status, penalty = _confidence_aware_status(False, pan_conf, 0.15)[:2]

    if status == "LOW_CONF_WARNING":
        msg = (
            f"OCR extracted 4th character as '{char4}' — not a recognised "
            f"Income Tax entity code. Confidence: {pan_conf:.1f}% (low).{hint} "
            f"Valid codes: {', '.join(PAN_HOLDER_TYPES.keys())}. "
            "Validation marked as uncertain, not as definitive fraud."
        )
    else:
        msg = (
            f"4th character '{char4}' is not a recognised Income Tax entity code.{hint} "
            f"Valid codes: {', '.join(PAN_HOLDER_TYPES.keys())}."
        )

    return status, msg, penalty


def _check_surname_initial(pan_str: str, name_str: str,
                            pan_conf: float, name_conf: float) -> tuple:
    """
    Check 3 — Surname Initial (5th character) vs OCR name field.
    For Indian PAN: Full name = FIRST [MIDDLE] SURNAME
    The 5th PAN character must equal the first letter of the surname.
    """
    if len(pan_str) < 5:
        return "FAIL", "PAN too short to extract surname-initial character.", 0.0

    pan_5th = pan_str[4]

    if not name_str:
        return (
            "WARNING",
            "Cardholder name field is empty — surname initial check skipped.",
            0.0,
        )

    name_parts = name_str.upper().split()
    if not name_parts:
        return (
            "WARNING",
            "Name field could not be tokenised — surname initial check skipped.",
            0.0,
        )

    surname = name_parts[-1]
    surname_initial = surname[0]

    if surname_initial == pan_5th:
        return (
            "PASS",
            f"5th character '{pan_5th}' matches surname initial of '{surname}' "
            f"(OCR name: '{name_str}').",
            0.0,
        )

    # Mismatch — evaluate confidence of both fields
    hint_pan = _ocr_misread_hint(pan_5th)
    hint_name = _ocr_misread_hint(surname_initial)

    both_low = pan_conf < OCR_CONFIDENCE_THRESHOLD or name_conf < OCR_CONFIDENCE_THRESHOLD
    status, penalty = _confidence_aware_status(False, min(pan_conf, name_conf), 0.25)[:2]

    if status == "LOW_CONF_WARNING":
        msg = (
            f"Surname initial mismatch: PAN 5th char is '{pan_5th}' but OCR surname "
            f"'{surname}' starts with '{surname_initial}'.{hint_pan}{hint_name} "
            f"PAN confidence: {pan_conf:.1f}%, Name confidence: {name_conf:.1f}%. "
            "One or both fields are low-confidence — mismatch may be an OCR artefact."
        )
    else:
        msg = (
            f"Mismatch: 5th PAN character is '{pan_5th}' but OCR surname "
            f"'{surname}' starts with '{surname_initial}'.{hint_pan} "
            f"(OCR name field: '{name_str}')"
        )

    return status, msg, penalty


def _check_sequential_digits(pan_str: str, pan_conf: float) -> tuple:
    """
    Check 4 — Characters 6–9 must be exactly 4 decimal digits.
    """
    if len(pan_str) < 9:
        return "FAIL", "PAN too short to verify digit sequence (positions 6–9).", 0.10

    digit_segment = pan_str[5:9]
    if digit_segment.isdigit():
        return "PASS", f"Characters 6–9 '{digit_segment}' are all valid numeric digits.", 0.0

    non_digits = [c for c in digit_segment if not c.isdigit()]
    hints = "".join(_ocr_misread_hint(c) for c in non_digits)
    status, penalty = _confidence_aware_status(False, pan_conf, 0.10)[:2]

    if status == "LOW_CONF_WARNING":
        msg = (
            f"Characters 6–9 '{digit_segment}' contain non-digit(s): {non_digits}.{hints} "
            f"OCR confidence is {pan_conf:.1f}% — non-digits may be misread letters."
        )
    else:
        msg = (
            f"Characters 6–9 '{digit_segment}' contain non-digit(s): {non_digits}.{hints} "
            "All four must be numeric."
        )

    return status, msg, penalty


def _check_checksum_character(pan_str: str, pan_conf: float) -> tuple:
    """
    Check 5 — 10th character must be an uppercase letter [A-Z].
    """
    if len(pan_str) < 10:
        return "FAIL", "PAN too short to verify the 10th (series/checksum) character.", 0.10

    char10 = pan_str[9]
    if char10.isalpha() and char10.isupper():
        return (
            "PASS",
            f"10th character '{char10}' is a valid uppercase letter (series/checksum character).",
            0.0,
        )

    hint = _ocr_misread_hint(char10)
    status, penalty = _confidence_aware_status(False, pan_conf, 0.10)[:2]

    if status == "LOW_CONF_WARNING":
        msg = (
            f"10th character '{char10}' must be [A–Z].{hint} "
            f"OCR confidence is {pan_conf:.1f}% — may be an OCR misread."
        )
    else:
        msg = (
            f"10th character '{char10}' must be an uppercase letter [A–Z].{hint} "
            "This is the PAN issuing series identifier."
        )

    return status, msg, penalty


def _check_dob(dob_str: str, dob_conf: float) -> tuple:
    """
    Check 6 — DOB format and date sanity.
    Accepted: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
    """
    if not dob_str:
        if dob_conf < OCR_CONFIDENCE_THRESHOLD:
            return (
                "LOW_CONF_WARNING",
                f"DOB field is empty. OCR confidence was {dob_conf:.1f}% — "
                "field may not have been readable.",
                0.06,
            )
        return "FAIL", "Date of Birth field is empty.", 0.15

    dob_str = dob_str.strip()

    patterns = [
        r"^(\d{2})/(\d{2})/(\d{4})$",
        r"^(\d{2})-(\d{2})-(\d{4})$",
        r"^(\d{2})\.(\d{2})\.(\d{4})$",
    ]
    format_matched = any(re.match(p, dob_str) for p in patterns)

    if not format_matched:
        status, penalty = _confidence_aware_status(False, dob_conf, 0.15)[:2]
        if status == "LOW_CONF_WARNING":
            msg = (
                f"DOB '{dob_str}' does not match DD/MM/YYYY format. "
                f"OCR confidence: {dob_conf:.1f}% — extraction may be incomplete or noisy."
            )
        else:
            msg = (
                f"DOB '{dob_str}' does not match any accepted format. "
                "Expected DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YYYY."
            )
        return status, msg, penalty

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            date_obj = datetime.strptime(dob_str, fmt)
            current_year = datetime.now().year
            if date_obj.year < 1900 or date_obj.year > current_year:
                status, penalty = _confidence_aware_status(False, dob_conf, 0.15)[:2]
                return (
                    status,
                    f"DOB year '{date_obj.year}' is out of range (1900–{current_year}).",
                    penalty,
                )
            age = current_year - date_obj.year
            return "PASS", f"DOB '{dob_str}' is valid. Approximate age: {age} years.", 0.0
        except ValueError:
            continue

    status, penalty = _confidence_aware_status(False, dob_conf, 0.15)[:2]
    return (
        status,
        f"DOB '{dob_str}' contains invalid calendar values (e.g. day > 31 or month > 12).",
        penalty,
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_pan_validation(ocr_results: dict) -> dict:
    """
    Phase 8 — Rule-Based PAN Format Validation.

    Validates OCR-extracted fields against Income Tax Department structural
    rules, with full awareness of per-field OCR confidence scores.

    Checks:
        1. PAN Regex             — [A-Z]{5}[0-9]{4}[A-Z]
        2. Holder Type           — 4th character → taxpayer entity code
        3. Surname Initial       — 5th char vs OCR surname initial
        4. Sequential Digits     — chars 6–9 must be digits
        5. Series/Checksum Char  — 10th char must be [A-Z]
        6. DOB Format & Sanity   — DD/MM/YYYY, range 1900–current year

    Status values:
        PASS              — check passed
        FAIL              — check failed with high OCR confidence (real error)
        LOW_CONF_WARNING  — check failed but OCR confidence is low
                            (may be OCR misread, not definitive fraud)
        WARNING           — structural skip (e.g. empty name field)
    """
    fields = ocr_results.get("fields", {})
    is_mock = ocr_results.get("is_mock", False)

    # Extract text, confidence, and correction metadata for each field
    pan_data    = fields.get("PAN",        {})
    name_data   = fields.get("Name",       {})
    father_data = fields.get("FatherName", {})
    dob_data    = fields.get("DOB",        {})

    pan_str  = pan_data.get("text", "").upper().strip().replace(" ", "")
    name_str = name_data.get("text", "").upper().strip()
    dob_str  = dob_data.get("text", "").strip()

    pan_conf    = float(pan_data.get("confidence", 0.0))
    name_conf   = float(name_data.get("confidence", 0.0))
    dob_conf    = float(dob_data.get("confidence", 0.0))

    pan_corrected  = bool(pan_data.get("ocr_corrected", False))
    pan_state      = pan_data.get("state", "ok")

    logger.info(
        f"[Phase 8] PAN Validation | PAN='{pan_str}' (conf={pan_conf:.1f}% "
        f"corrected={pan_corrected} state={pan_state}) | "
        f"Name='{name_str}' (conf={name_conf:.1f}%) | DOB='{dob_str}' (conf={dob_conf:.1f}%)"
    )

    validation_checks = []
    ocr_invalidity_score = 0.0

    # ------------------------------------------------------------------
    # Check 1: PAN Regex Format
    # ------------------------------------------------------------------
    status, msg, penalty = _check_pan_regex(pan_str, pan_conf)
    ocr_invalidity_score += penalty
    validation_checks.append({"check": "PAN Format Regex", "status": status, "message": msg})

    # ------------------------------------------------------------------
    # Checks 2–5: Character-level checks (only if PAN is 10 chars)
    # ------------------------------------------------------------------
    if len(pan_str) == 10:

        # Check 2: Holder Type (4th char)
        status, msg, penalty = _check_holder_type(pan_str, pan_conf)
        ocr_invalidity_score += penalty
        validation_checks.append({"check": "Holder Type (4th Character)", "status": status, "message": msg})

        # Check 3: Surname Initial (5th char vs OCR name)
        status, msg, penalty = _check_surname_initial(pan_str, name_str, pan_conf, name_conf)
        ocr_invalidity_score += penalty
        validation_checks.append({"check": "Surname Initial (5th Character)", "status": status, "message": msg})

        # Check 4: Digit sequence (chars 6–9)
        status, msg, penalty = _check_sequential_digits(pan_str, pan_conf)
        ocr_invalidity_score += penalty
        validation_checks.append({"check": "Sequential Digits (Chars 6–9)", "status": status, "message": msg})

        # Check 5: Series/Checksum character (10th char)
        status, msg, penalty = _check_checksum_character(pan_str, pan_conf)
        ocr_invalidity_score += penalty
        validation_checks.append({"check": "Series / Checksum Character (10th)", "status": status, "message": msg})

    else:
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
    # Check 6: DOB Format & Sanity
    # ------------------------------------------------------------------
    status, msg, penalty = _check_dob(dob_str, dob_conf)
    ocr_invalidity_score += penalty
    validation_checks.append({"check": "DOB Format & Date Validity", "status": status, "message": msg})

    # ------------------------------------------------------------------
    # Compile result
    # ------------------------------------------------------------------
    ocr_invalidity_score = float(np.clip(ocr_invalidity_score, 0.0, 1.0))

    hard_failures    = [c for c in validation_checks if c["status"] == "FAIL"]
    low_conf_issues  = [c for c in validation_checks if c["status"] == "LOW_CONF_WARNING"]
    warnings         = [c for c in validation_checks if c["status"] == "WARNING"]

    success = len(hard_failures) == 0

    correction_note = (
        " PAN OCR post-correction was applied (character confusion substitution)."
        if pan_corrected else ""
    )

    if success and not low_conf_issues:
        explanation = (
            "All PAN validation checks passed. The extracted fields conform to "
            "official Income Tax Department format rules." + correction_note
        )
    elif success and low_conf_issues:
        lc_names = ", ".join(c["check"] for c in low_conf_issues)
        explanation = (
            f"No definitive validation failures. However, {len(low_conf_issues)} "
            f"check(s) could not be confirmed due to low OCR confidence: {lc_names}. "
            "Results may be unreliable — consider re-scanning with better image quality."
            + correction_note
        )
    else:
        fail_names = ", ".join(c["check"] for c in hard_failures)
        explanation = (
            f"Validation failed on: {fail_names}. "
            "Format anomalies detected in the OCR-extracted fields."
            + correction_note
        )

    if is_mock:
        explanation = "[MOCK MODE] " + explanation

    logger.info(
        f"[Phase 8] Complete — success={success}, "
        f"hard_failures={len(hard_failures)}, "
        f"low_conf_warnings={len(low_conf_issues)}, "
        f"invalidity_score={ocr_invalidity_score:.2f}"
    )

    return {
        "phase": "pan_validation",
        "success": success,
        "ocr_invalidity_score": round(ocr_invalidity_score, 2),
        "checks": validation_checks,
        "hard_failure_count": len(hard_failures),
        "low_conf_warning_count": len(low_conf_issues),
        "pan_ocr_corrected": pan_corrected,
        "explanation": explanation,
    }
