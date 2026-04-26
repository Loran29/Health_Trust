"""
Evidence extraction for TrustMap India.

For each capability claim in an assessment, retrieves the actual quoted text
from data/facilities_clean.parquet that supports or contradicts it.

No LLM calls — pure field extraction and keyword matching.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

RAW_FACILITIES_PATH = Path("data/facilities_clean.parquet")
ASSESSMENTS_PATH = Path("backend/data/assessments_llm.parquet")

# Raw fields mined for evidence, in descending specificity
_EVIDENCE_FIELDS = ["specialties", "procedure", "equipment", "capability", "description", "facilitytypeid"]

# Keywords that indicate each capability in free text
_CAP_KEYWORDS: dict[str, list[str]] = {
    "emergency":     ["emergency", "casualty", "trauma", "24/7", "always open", "24x7"],
    "icu":           ["icu", "intensive care", "critical care", "ventilator"],
    "surgery":       ["surgery", "surgical", "operation theatre", "OT", "laparoscop", "cardiacSurgery"],
    "obstetrics":    ["obstetric", "maternity", "delivery", "gynecolog", "gynaecolog", "c-section", "caesarean", "reproductiveEndocrinology"],
    "dialysis":      ["dialysis", "kidney", "renal", "nephrology"],
    "oncology":      ["oncolog", "cancer", "chemotherapy", "radiation", "medicalOncology"],
    "cardiology":    ["cardiology", "cardiac", "cardiothoracic", "heart"],
    "anesthesia":    ["anesthesia", "anaesthesia", "anesthesiology"],
    "pediatrics":    ["pediatric", "paediatric", "children", "child care", "neonatal"],
    "mental_health": ["mental health", "psychiatry", "psychiatrist", "psychology"],
    "dentistry":     ["dental", "dentist", "dentistry", "orthodontic", "endodontics", "periodontics", "RCT", "root canal", "aestheticDentistry"],
    "primary_care":  ["general medicine", "familyMedicine", "family medicine", "internalMedicine", "internal medicine", "primary care", "general practice", "OPD", "outpatient"],
    "ophthalmology": ["ophthalmology", "eye", "retina", "cataract", "vision"],
    "orthopedics":   ["orthopedic", "orthopaedic", "bone", "joint", "fracture", "spine"],
    "dermatology":   ["dermatology", "dermatologist", "skin"],
    "runs_24_7":     ["24/7", "24 hour", "always open", "round the clock", "24x7"],
}


def _val_to_str(val: Any) -> str:
    if val is None:
        return ""
    # numpy arrays and pandas Series
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        return ", ".join(str(v) for v in val if v is not None)
    s = str(val)
    return "" if s.lower() in {"nan", "none"} else s


def _snippet_around(text: str, keyword: str, window: int = 80) -> str:
    """Return a window of text around the first keyword hit."""
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return ""
    start = max(0, idx - 20)
    end = min(len(text), idx + len(keyword) + window)
    chunk = text[start:end].strip()
    if start > 0:
        chunk = "..." + chunk
    if end < len(text):
        chunk = chunk + "..."
    return chunk


def _keyword_evidence(field_texts: dict[str, str], cap: str) -> tuple[str, str]:
    """
    Search field_texts for keywords matching cap.
    Returns (snippet, source_field) or ("", "").
    """
    keywords = _CAP_KEYWORDS.get(cap, [cap.replace("_", " ")])
    for field in _EVIDENCE_FIELDS:
        text = field_texts.get(field, "")
        if not text:
            continue
        for kw in keywords:
            snip = _snippet_around(text, kw)
            if snip:
                return snip, field
    return "", ""


def _find_raw_row(facility_id: str, raw_df: pd.DataFrame) -> pd.Series | None:
    """
    Match a raw facilities row to a facility_id using the same slug logic
    that api.py uses (name + city → slugify).
    """
    def make_fid(name: Any, city: Any) -> str:
        n = "" if not name or str(name).lower() in {"nan", "none"} else str(name)
        c = "" if not city or str(city).lower() in {"nan", "none"} else str(city)
        return re.sub(r"[^a-z0-9]+", "-", f"{n} {c}".lower()).strip("-")

    for _, row in raw_df.iterrows():
        if make_fid(row.get("name"), row.get("address_city")) == facility_id:
            return row
    return None


def get_evidence_snippets(
    facility_id: str,
    raw_df: pd.DataFrame | None = None,
    asmt_df: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """
    For each capability claim on the facility, return a dict keyed by
    capability name:

        {
            "dentistry": {
                "status": "confirmed",
                "evidence_snippet": "periodontics, endodontics, dentistry, aestheticDentistry",
                "source_field": "specialties",
                "contradiction_note": "..."   # only when contradicted
            },
            ...
        }

    Evidence priority:
    1. LLM-stored evidence_snippet (already quoted from raw data)
    2. Direct lookup of the LLM-named field in raw parquet
    3. Keyword search across all evidence fields
    """
    if raw_df is None:
        raw_df = pd.read_parquet(RAW_FACILITIES_PATH)
    if asmt_df is None:
        asmt_df = pd.read_parquet(ASSESSMENTS_PATH)

    # Assessment row
    asmt_rows = asmt_df[asmt_df["facility_id"] == facility_id]
    if asmt_rows.empty:
        return {}
    asmt = asmt_rows.iloc[0]

    # Raw facility row
    raw = _find_raw_row(facility_id, raw_df)

    # Build field-text cache from raw
    field_texts: dict[str, str] = {}
    if raw is not None:
        for field in _EVIDENCE_FIELDS:
            field_texts[field] = _val_to_str(raw.get(field, ""))

    # Parse claims and contradictions
    try:
        claims: list[dict] = json.loads(asmt.get("capability_claims", "[]") or "[]")
    except (json.JSONDecodeError, TypeError):
        claims = []

    try:
        contras: list[dict] = json.loads(asmt.get("contradictions", "[]") or "[]")
    except (json.JSONDecodeError, TypeError):
        contras = []

    # Index contradictions by capability mention
    contra_notes: dict[str, str] = {}
    for c in contras:
        note = (c.get("why_contradictory") or c.get("reason") or "").strip()
        if not note:
            continue
        note_lower = note.lower()
        for cap in _CAP_KEYWORDS:
            if cap in note_lower or cap.replace("_", " ") in note_lower:
                if cap not in contra_notes:
                    contra_notes[cap] = note[:250]

    result: dict[str, dict] = {}
    for claim in claims:
        cap = claim.get("capability", "")
        if not cap:
            continue
        status = claim.get("status", "unknown")
        llm_field = claim.get("evidence_field", "")
        llm_snippet = (claim.get("evidence_snippet") or "").strip()

        # --- Priority 1: LLM snippet is already actual raw text ---
        snippet = llm_snippet
        source = llm_field

        # --- Priority 2: Re-read the exact raw field the LLM cited ---
        if raw is not None and llm_field and llm_field in field_texts:
            raw_text = field_texts[llm_field]
            if raw_text:
                # If the LLM snippet is a substring of the raw field, use
                # the fuller raw text (up to 200 chars) for more context
                if llm_snippet and llm_snippet[:30].lower() in raw_text.lower():
                    snippet = raw_text[:200]
                elif not llm_snippet:
                    snippet = raw_text[:200]

        # --- Priority 3: Keyword search if still empty ---
        if not snippet and raw is not None:
            snippet, source = _keyword_evidence(field_texts, cap)

        entry: dict[str, Any] = {
            "status": status,
            "evidence_snippet": snippet or "(no raw text found)",
            "source_field": source or "unknown",
        }

        # Attach contradiction note for contradicted/disputed caps
        if status in ("contradicted", "disputed"):
            note = contra_notes.get(cap, "")
            if not note:
                # Fallback: search all contradiction texts
                for c in contras:
                    full = (c.get("why_contradictory") or c.get("reason") or "")
                    if cap in full.lower() or cap.replace("_", " ") in full.lower():
                        note = full[:250]
                        break
            if note:
                entry["contradiction_note"] = note

        result[cap] = entry

    return result
