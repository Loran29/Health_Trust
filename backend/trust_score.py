from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ASSESSMENTS_PATH = Path("backend/data/assessments_llm.parquet")
FACILITIES_PATH = Path("data/facilities_clean.parquet")

SEVERITY_DEDUCTIONS = {1: 5, 2: 10, 3: 15, 4: 20, 5: 25}

DEMO_FACILITIES = [
    ("1000 Smiles Dental Clinic", "Hyderabad"),
    ("Krishna Homeopathy Research Hospital", "Jaipur"),
    ("Aastha Children Hospital", "Dehri"),
    ("City Health Clinic", "Guwahati"),
    ("7 Star Healthcare", "Delhi"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def make_facility_id(name: str, city: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{name} {city}".lower())
    return slug.strip("-")


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>", "nat"}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    # numpy arrays have tolist() but so do numpy scalars — check for sequence length safely
    if hasattr(value, "__len__") and not isinstance(value, str):
        try:
            return len(value) > 0
        except TypeError:
            pass
    return str(value).strip().lower() not in {"", "nan", "none", "null", "<na>", "nat"}


def _parse_json_list(value: Any) -> list[dict]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _equipment_strings(raw_row: pd.Series) -> list[str]:
    eq = raw_row.get("equipment")
    if eq is None:
        return []
    items: list[Any] = eq if isinstance(eq, list) else (eq.tolist() if hasattr(eq, "tolist") else [eq])
    return [str(i).lower() for i in items if not _is_null(i)]


def _safe_int(value: Any) -> int | None:
    if _is_null(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _active_capabilities(claims: list[dict]) -> set[str]:
    return {c["capability"] for c in claims if c.get("status") in {"confirmed", "inferred"}}


# ---------------------------------------------------------------------------
# 1. Internal consistency score
# ---------------------------------------------------------------------------

def internal_consistency_score(assessment: dict[str, Any]) -> float:
    """Start at 100, deduct by severity for each contradiction. Floor 0."""
    contradictions = _parse_json_list(assessment.get("contradictions"))
    score = 100.0
    for c in contradictions:
        try:
            severity = int(c.get("severity", 0))
        except (TypeError, ValueError):
            severity = 0
        score -= SEVERITY_DEDUCTIONS.get(severity, 0)
    return max(0.0, score)


# ---------------------------------------------------------------------------
# 2. Capability plausibility score
# ---------------------------------------------------------------------------

def capability_plausibility_score(assessment: dict[str, Any], raw_row: pd.Series) -> float:
    """Deduct 15 pts for each active capability claim that fails its prerequisite."""
    claims = _parse_json_list(assessment.get("capability_claims"))
    active = _active_capabilities(claims)
    equipment = _equipment_strings(raw_row)
    description = str(raw_row.get("description") or "").lower()
    num_doctors = _safe_int(raw_row.get("numberdoctors"))

    deductions = 0

    if "icu" in active:
        if not (
            any("ventilator" in e for e in equipment)
            or any("critical care" in e for e in equipment)
        ):
            deductions += 15

    if "surgery" in active:
        anesthesia_active = "anesthesia" in active
        has_theatre = any("operating" in e or "theatre" in e for e in equipment)
        if not (anesthesia_active and has_theatre):
            deductions += 15

    if "dialysis" in active:
        if not any("dialysis" in e for e in equipment):
            deductions += 15

    if "oncology" in active:
        if not (
            any("chemotherapy" in e or "radiation" in e for e in equipment)
            or "chemotherapy" in description
            or "radiation" in description
        ):
            deductions += 15

    if "obstetrics" in active:
        if num_doctors is None or num_doctors < 2:
            deductions += 15

    if "emergency" in active and "runs_24_7" in active:
        if not (
            (num_doctors is not None and num_doctors >= 5)
            or "24" in description
        ):
            deductions += 15

    return max(0.0, 100.0 - deductions)


# ---------------------------------------------------------------------------
# 3. Activity signal score
# ---------------------------------------------------------------------------

def _recency_points(recency: Any) -> int:
    if _is_null(recency):
        return 0
    # Parquet stores this as a pandas Timestamp
    if isinstance(recency, (pd.Timestamp, datetime)):
        dt = recency
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        years_ago = (datetime.now(tz=timezone.utc) - dt).days / 365.25
        if years_ago <= 1:
            return 30
        if years_ago <= 2:
            return 15
        return 0
    # Fallback: try parsing a string
    s = str(recency).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s[:19], fmt).replace(tzinfo=timezone.utc)
            years_ago = (datetime.now(tz=timezone.utc) - dt).days / 365.25
            if years_ago <= 1:
                return 30
            if years_ago <= 2:
                return 15
            return 0
        except ValueError:
            continue
    return 0


def activity_signal_score(raw_row: pd.Series) -> float:
    """0-100 from recency, social presence, logo, and follower count."""
    score = 0.0

    score += _recency_points(raw_row.get("recency_of_page_update"))

    social_count = _safe_int(raw_row.get("distinct_social_media_presence_count")) or 0
    score += min(30, social_count * 10)

    logo = raw_row.get("custom_logo_presence")
    if not _is_null(logo):
        try:
            if float(logo) != 0:
                score += 20
        except (TypeError, ValueError):
            if str(logo).lower() in {"true", "yes", "1"}:
                score += 20

    followers = _safe_int(raw_row.get("engagement_metrics_n_followers"))
    if followers is not None:
        if followers > 100:
            score += 20
        elif followers > 0:
            score += 10

    return min(100.0, score)


# ---------------------------------------------------------------------------
# 4. Completeness score
# ---------------------------------------------------------------------------

def completeness_score(raw_row: pd.Series) -> float:
    """(filled_fields / 7) * 100 for phone, address, specialties, equipment,
    capability, number_of_doctors, capacity."""
    phone = raw_row.get("officialphone") if _has_value(raw_row.get("officialphone")) else raw_row.get("phone_numbers")
    fields = {
        "phone": phone,
        "address": raw_row.get("address_city") or raw_row.get("address_line1"),
        "specialties": raw_row.get("specialties"),
        "equipment": raw_row.get("equipment"),
        "capability": raw_row.get("capability"),
        "number_of_doctors": raw_row.get("numberdoctors"),
        "capacity": raw_row.get("capacity"),
    }
    count_filled = sum(1 for v in fields.values() if _has_value(v))
    return (count_filled / 7) * 100.0


# ---------------------------------------------------------------------------
# 5. Overall score
# ---------------------------------------------------------------------------

def compute_overall(subscores: dict[str, float]) -> float:
    """Weighted average: 35% consistency, 30% plausibility, 15% activity, 20% completeness."""
    return (
        subscores["internal_consistency"] * 0.35
        + subscores["capability_plausibility"] * 0.30
        + subscores["activity_signal"] * 0.15
        + subscores["completeness"] * 0.20
    )


# ---------------------------------------------------------------------------
# 6. Confidence interval
# ---------------------------------------------------------------------------

def confidence_interval(subscores: dict[str, float]) -> tuple[float, float]:
    """Width narrows as completeness grows. Clamped to [0, 100]."""
    overall = compute_overall(subscores)
    width = 30 - (subscores["completeness"] * 0.25)
    low = max(0.0, overall - width / 2)
    high = min(100.0, overall + width / 2)
    return (low, high)


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def score_row(assessment: dict[str, Any], raw_row: pd.Series) -> dict[str, Any]:
    ic = internal_consistency_score(assessment)
    cp = capability_plausibility_score(assessment, raw_row)
    act = activity_signal_score(raw_row)
    comp = completeness_score(raw_row)
    subscores = {
        "internal_consistency": ic,
        "capability_plausibility": cp,
        "activity_signal": act,
        "completeness": comp,
    }
    overall = compute_overall(subscores)
    ci = confidence_interval(subscores)
    return {
        "facility_name": assessment.get("facility_name", ""),
        "city": assessment.get("city", ""),
        "internal_consistency": round(ic, 1),
        "capability_plausibility": round(cp, 1),
        "activity_signal": round(act, 1),
        "completeness": round(comp, 1),
        "overall_trust_score": round(overall, 1),
        "confidence_interval": f"[{round(ci[0], 1)}, {round(ci[1], 1)}]",
    }


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 35)
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print(df.to_string(index=False))


def main() -> None:
    assessments = pd.read_parquet(ASSESSMENTS_PATH)
    facilities = pd.read_parquet(FACILITIES_PATH)

    facilities["facility_id"] = facilities.apply(
        lambda r: make_facility_id(
            "" if _is_null(r.get("name")) else str(r["name"]),
            "" if _is_null(r.get("address_city")) else str(r["address_city"]),
        ),
        axis=1,
    )

    merged = assessments.merge(facilities, on="facility_id", how="inner", suffixes=("_asmt", "_raw"))
    print(f"Assessments: {len(assessments):,} | Facilities: {len(facilities):,} | Joined: {len(merged):,}")

    # Score first 20
    first20: list[dict[str, Any]] = []
    for _, row in merged.head(20).iterrows():
        first20.append(score_row(row.to_dict(), row))
    _print_table("First 20 Facilities — Trust Scores", first20)

    # Demo facilities
    demo_results: list[dict[str, Any]] = []
    missing: list[str] = []
    for demo_name, demo_city in DEMO_FACILITIES:
        mask = (
            merged["facility_name"].str.contains(demo_name, case=False, na=False)
            & (merged["city"].str.lower() == demo_city.lower())
        )
        matches = merged[mask]
        if matches.empty:
            missing.append(f"{demo_name} / {demo_city}")
            continue
        row = matches.iloc[0]
        demo_results.append(score_row(row.to_dict(), row))

    if demo_results:
        _print_table("Demo Facilities — Full Scores", demo_results)

    if missing:
        print(f"\nWARNING — demo facilities not found in joined data: {missing}")


if __name__ == "__main__":
    main()
